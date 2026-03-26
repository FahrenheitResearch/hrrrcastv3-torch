from __future__ import annotations

import argparse
import contextlib
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import xarray as xr
from skimage.exposure import match_histograms

from .graph import build_torch_model_from_keras_archive
from .runtime import HRRRCastDiffusionRunner


logger = logging.getLogger(__name__)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _src_dir() -> Path:
    return _repo_root() / "src"


def _ensure_src_on_path() -> None:
    src_dir = str(_src_dir())
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)


def _resolve_repo_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return (_repo_root() / path).resolve()


class TorchPreprocessedDataLoader:
    def __init__(self, preprocessed_file: str):
        self.preprocessed_file = preprocessed_file
        self.data = None
        self.metadata = None
        self._load_data()

    def _load_data(self) -> None:
        if not Path(self.preprocessed_file).exists():
            raise FileNotFoundError(f"Preprocessed data file not found: {self.preprocessed_file}")

        self.data = np.load(self.preprocessed_file)
        self.metadata = {
            "init_year": str(self.data["init_year"]),
            "init_month": str(self.data["init_month"]),
            "init_day": str(self.data["init_day"]),
            "init_hh": str(self.data["init_hh"]),
            "init_datetime": str(self.data["init_datetime"]),
            "pl_vars": self.data["pl_vars"].tolist(),
            "sfc_vars": self.data["sfc_vars"].tolist(),
            "levels": self.data["levels"].tolist(),
            "grid_height": int(self.data["grid_height"]),
            "grid_width": int(self.data["grid_width"]),
            "downsample_factor": int(self.data["downsample_factor"]),
            "norm_file": str(self.data["norm_file"]),
        }

    def get_model_input(self) -> np.ndarray:
        return self.data["model_input"]

    def get_coordinates(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.data["lats"], self.data["lons"]

    def get_init_datetime(self) -> datetime:
        return datetime.fromisoformat(self.metadata["init_datetime"])


def load_torch_checkpoint(
    checkpoint_path: str | Path,
    *,
    device: str | torch.device = "cpu",
    dtype: Optional[torch.dtype] = None,
) -> torch.nn.Module:
    checkpoint_path = Path(checkpoint_path)
    payload = torch.load(checkpoint_path, map_location="cpu")
    keras_archive = _resolve_repo_path(payload["keras_archive"])
    model = build_torch_model_from_keras_archive(keras_archive)
    model.load_state_dict(payload["state_dict"])
    if dtype is None:
        dtype = torch.float32
    model.to(device=device, dtype=dtype)
    model.eval()
    return model


class TorchWeatherForecaster:
    def __init__(
        self,
        network: torch.nn.Module,
        data_loader_hrrr: TorchPreprocessedDataLoader,
        data_loader_gfs: TorchPreprocessedDataLoader,
        *,
        members: List[int],
        batch_size: int,
        device: str | torch.device,
        dtype: torch.dtype,
        pmm_alpha: float = 0.7,
        use_nudging: bool = True,
        tile_size: tuple[int, int] | None = None,
        tile_halo: int = 32,
        allow_full_frame: bool = False,
    ):
        _ensure_src_on_path()
        import utils  # noqa: WPS433

        self.utils = utils
        self.device = torch.device(device)
        self.dtype = dtype
        self.network = network
        self.runner = HRRRCastDiffusionRunner(network)
        self.data_loader_hrrr = data_loader_hrrr
        self.data_loader_gfs = data_loader_gfs
        self.metadata = data_loader_hrrr.metadata
        self.members = members
        self.num_members = len(members)
        self.batch_size = batch_size
        self.pmm_alpha = pmm_alpha
        self.use_nudging = use_nudging and len(members) > 1
        self.tile_size = tile_size
        self.tile_halo = tile_halo
        self.allow_full_frame = allow_full_frame

        self.LOG_TRANSFORM_VARS = ["VIS", "APCP", "HGTCC", "CAPE"]
        self.NEG_LOG_TRANSFORM_VARS = ["CIN"]

        model_input_hrrr = data_loader_hrrr.get_model_input()
        model_input_gfs = data_loader_gfs.get_model_input()
        self.input_shape = model_input_hrrr.shape
        self.predicted_channels = len(self.metadata["pl_vars"]) * len(self.metadata["levels"]) + len(self.metadata["sfc_vars"])
        self.gfs_channels = model_input_gfs.shape[-1]
        self.static_channels = max(model_input_hrrr.shape[-1] - self.predicted_channels, 0)

        norm_file = _resolve_repo_path(self.metadata["norm_file"])
        ds_norm = xr.open_dataset(norm_file)
        self._init_channel_stats(ds_norm)
        ds_norm.close()

        nlat = model_input_hrrr.shape[1]
        nlon = model_input_hrrr.shape[2]
        self.member_noise: Dict[int, torch.Tensor] = {}
        for member in members:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(member)
            noise = torch.randn(
                (1, nlat, nlon, self.predicted_channels),
                generator=generator,
                dtype=self.dtype,
            )
            self.member_noise[member] = noise.to(self.device)

        if self.tile_size is None and not self.allow_full_frame:
            raise RuntimeError(
                "Full-frame diffusion sampling is disabled by default. "
                "Set a tile size or pass allow_full_frame=True explicitly."
            )

    def _autocast_context(self):
        if self.device.type != "cuda":
            return contextlib.nullcontext()
        if self.dtype not in {torch.float16, torch.bfloat16}:
            return contextlib.nullcontext()
        return torch.autocast(device_type="cuda", dtype=self.dtype)

    def estimate_step_tensor_gib(self) -> float:
        h, w = self.input_shape[1], self.input_shape[2]
        channels = 328
        bytes_per_value = torch.finfo(self.dtype).bits // 8 if self.dtype.is_floating_point else 4
        total_bytes = h * w * channels * bytes_per_value
        return total_bytes / (1024**3)

    def _compute_pmm_mean(self, member_outputs: Dict[int, np.ndarray], method: int = 2) -> Tuple[np.ndarray, List[int]]:
        from compute_pmm import compute_PMM  # noqa: WPS433

        if not member_outputs:
            raise ValueError("member_outputs is empty; cannot compute PMM")

        pl_vars = self.metadata["pl_vars"]
        sfc_vars = self.metadata["sfc_vars"]
        levels = self.metadata["levels"]
        num_pl_channels = len(pl_vars) * len(levels)

        pmm_channels = []
        for var_name in ["REFC", "APCP"]:
            if var_name in sfc_vars:
                pmm_channels.append(num_pl_channels + sfc_vars.index(var_name))

        if not pmm_channels:
            first_arr = next(iter(member_outputs.values()))
            if first_arr.ndim == 4:
                first_arr = first_arr[0]
            ny, nx = first_arr.shape[:2]
            return np.empty((ny, nx, 0), dtype=first_arr.dtype), []

        stack = np.stack(
            [member_outputs[m][0] if member_outputs[m].ndim == 4 else member_outputs[m] for m in sorted(member_outputs)],
            axis=0,
        )
        pmm_results = []
        valid_channels = []
        for channel_idx in pmm_channels:
            if channel_idx >= stack.shape[-1]:
                continue
            channel_stack = np.transpose(stack[:, :, :, channel_idx], (1, 2, 0))
            da = xr.DataArray(channel_stack, dims=("latitude", "longitude", "member"), coords={"member": np.arange(stack.shape[0])})
            pmm_results.append(compute_PMM(da, method=method).values)
            valid_channels.append(channel_idx)
        if not pmm_results:
            return np.empty((stack.shape[1], stack.shape[2], 0), dtype=stack.dtype), []
        return np.stack(pmm_results, axis=-1), valid_channels

    def _nudge_members_toward_pmm(
        self,
        member_outputs: Dict[int, np.ndarray],
        pmm_values: np.ndarray,
        pmm_channels: List[int],
        alpha: float,
    ) -> Dict[int, np.ndarray]:
        if not pmm_channels or pmm_values.shape[-1] == 0:
            return member_outputs

        nudged_outputs: Dict[int, np.ndarray] = {}
        for member, arr in member_outputs.items():
            arr2 = arr[0] if arr.ndim == 4 else arr
            nudged = arr2.copy()
            for i, channel_idx in enumerate(pmm_channels):
                if channel_idx >= arr2.shape[-1] or i >= pmm_values.shape[-1]:
                    continue
                member_channel = arr2[:, :, channel_idx]
                pmm_channel = pmm_values[:, :, i]
                blended_channel = alpha * member_channel + (1.0 - alpha) * pmm_channel
                nudged[:, :, channel_idx] = match_histograms(blended_channel, member_channel, channel_axis=None)
            nudged_outputs[member] = nudged[None, ...]
        return nudged_outputs

    def _init_channel_stats(self, ds_norm: xr.Dataset) -> None:
        fallback_mins_raw, fallback_maxs_raw = self.get_variable_bounds()
        raw_means: List[float] = []
        raw_stds: List[float] = []
        raw_mins: List[float] = []
        raw_maxs: List[float] = []
        channel_idx = 0

        for var in self.metadata["pl_vars"]:
            if var not in ds_norm.variables:
                for _ in self.metadata["levels"]:
                    raw_means.append(0.0)
                    raw_stds.append(1.0)
                    raw_mins.append(float(fallback_mins_raw[channel_idx]))
                    raw_maxs.append(float(fallback_maxs_raw[channel_idx]))
                    channel_idx += 1
                continue
            stats = ds_norm[var].values
            nlev_stats = stats.shape[1] if stats.ndim > 1 else 1
            for i, _level in enumerate(self.metadata["levels"]):
                if i < nlev_stats:
                    stat_mean = float(stats[0, i])
                    stat_std = float(stats[1, i]) if float(stats[1, i]) != 0 else 1.0
                    stat_min = float(stats[2, i]) if stats.shape[0] > 2 else float(fallback_mins_raw[channel_idx])
                    stat_max = float(stats[3, i]) if stats.shape[0] > 3 else float(fallback_maxs_raw[channel_idx])
                else:
                    stat_mean = 0.0
                    stat_std = 1.0
                    stat_min = float(fallback_mins_raw[channel_idx])
                    stat_max = float(fallback_maxs_raw[channel_idx])
                raw_means.append(stat_mean)
                raw_stds.append(stat_std)
                raw_mins.append(stat_min if not np.isnan(stat_min) else float(fallback_mins_raw[channel_idx]))
                raw_maxs.append(stat_max if not np.isnan(stat_max) else float(fallback_maxs_raw[channel_idx]))
                channel_idx += 1

        for var in self.metadata["sfc_vars"]:
            if var not in ds_norm.variables:
                raw_means.append(0.0)
                raw_stds.append(1.0)
                raw_mins.append(float(fallback_mins_raw[channel_idx]))
                raw_maxs.append(float(fallback_maxs_raw[channel_idx]))
                channel_idx += 1
                continue
            stats = ds_norm[var].values
            stat_mean = float(np.nanmean(stats[0]))
            stat_std = float(np.nanmean(stats[1])) if np.nanmean(stats[1]) != 0 else 1.0
            stat_min = float(np.nanmean(stats[2])) if stats.shape[0] > 2 else float(fallback_mins_raw[channel_idx])
            stat_max = float(np.nanmean(stats[3])) if stats.shape[0] > 3 else float(fallback_maxs_raw[channel_idx])
            raw_means.append(stat_mean)
            raw_stds.append(stat_std)
            raw_mins.append(stat_min if not np.isnan(stat_min) else float(fallback_mins_raw[channel_idx]))
            raw_maxs.append(stat_max if not np.isnan(stat_max) else float(fallback_maxs_raw[channel_idx]))
            channel_idx += 1

        self.channel_means = np.array(raw_means, dtype=np.float32)
        self.channel_stds = np.array(raw_stds, dtype=np.float32)
        self.channel_mins = (np.array(raw_mins, dtype=np.float32) - self.channel_means) / self.channel_stds
        self.channel_maxs = (np.array(raw_maxs, dtype=np.float32) - self.channel_means) / self.channel_stds

    @staticmethod
    def compute_time_features(init_times_np, lead_times_np) -> np.ndarray:
        if not isinstance(init_times_np, (list, np.ndarray)):
            init_times_np = [init_times_np]
        if lead_times_np is not None and not isinstance(lead_times_np, (list, np.ndarray)):
            lead_times_np = [lead_times_np]
        time_coord = pd.to_datetime(init_times_np)
        if lead_times_np is not None:
            time_coord += pd.to_timedelta(lead_times_np, unit="h")
        hours = pd.DatetimeIndex(time_coord).hour.astype(np.float32)
        doy = pd.DatetimeIndex(time_coord).dayofyear.astype(np.float32)
        v4 = (time_coord >= np.datetime64("2021-03-23T00")).astype(np.float32)
        v3 = ((time_coord >= np.datetime64("2018-07-12T00")) & (time_coord < np.datetime64("2021-03-23T00"))).astype(np.float32)
        return np.stack(
            [
                np.sin(2 * np.pi * hours / 24.0).astype(np.float32),
                np.cos(2 * np.pi * hours / 24.0).astype(np.float32),
                np.sin(2 * np.pi * doy / 365.0).astype(np.float32),
                np.cos(2 * np.pi * doy / 365.0).astype(np.float32),
                v4.astype(np.float32),
                v3.astype(np.float32),
            ],
            axis=-1,
        )

    def date_encoding_tensor(self, init_times_np, lead_times_np) -> torch.Tensor:
        enc = self.compute_time_features(init_times_np, lead_times_np)
        enc = torch.as_tensor(enc, dtype=self.dtype, device=self.device)
        batch_size = enc.shape[0]
        enc = enc.reshape(batch_size, 1, 1, 6).expand(batch_size, self.input_shape[1], self.input_shape[2], 6)
        return enc

    def denormalize(self, output: np.ndarray) -> np.ndarray:
        if output.ndim == 3:
            output = output[None, ...]
        c_out = output.shape[-1]
        means = self.channel_means[:c_out][None, None, None, :]
        stds = self.channel_stds[:c_out][None, None, None, :]
        return np.squeeze(output * stds + means)

    def _apply_inverse_transforms(self, ds: xr.Dataset) -> xr.Dataset:
        from transform import inverse_log_transform_array, inverse_neg_log_transform_array  # noqa: WPS433

        for var in self.LOG_TRANSFORM_VARS:
            if var in ds.variables:
                ds[var].values[:] = inverse_log_transform_array(ds[var].values)
        for var in self.NEG_LOG_TRANSFORM_VARS:
            if var in ds.variables:
                ds[var].values[:] = inverse_neg_log_transform_array(ds[var].values)
        return ds

    def create_xarray_dataset(self, init_datetime: datetime, times: List[int], lats: np.ndarray, lons: np.ndarray, data: np.ndarray) -> xr.Dataset:
        data_vars = {}
        var_index = 0
        pl_vars = self.metadata["pl_vars"]
        sfc_vars = self.metadata["sfc_vars"]
        levels = self.metadata["levels"]

        for pl_var in pl_vars:
            pl_data = np.transpose(data[..., var_index : var_index + len(levels)], (0, 3, 1, 2))
            data_vars[pl_var] = xr.DataArray(
                np.expand_dims(pl_data, 0),
                dims=("time", "lead_time", "level", "latitude", "longitude"),
                coords={
                    "time": [init_datetime],
                    "lead_time": ("lead_time", times, {"units": "hours"}),
                    "level": ("level", levels, {"units": "hPa"}),
                    "latitude": (("latitude", "longitude"), lats),
                    "longitude": (("latitude", "longitude"), lons),
                },
                name=pl_var,
            )
            var_index += len(levels)

        for sfc_var in sfc_vars:
            sfc_data = data[..., var_index]
            data_vars[sfc_var] = xr.DataArray(
                np.expand_dims(sfc_data, 0),
                dims=("time", "lead_time", "latitude", "longitude"),
                coords={
                    "time": [init_datetime],
                    "lead_time": ("lead_time", times, {"units": "hours"}),
                    "latitude": (("latitude", "longitude"), lats),
                    "longitude": (("latitude", "longitude"), lons),
                },
                name=sfc_var,
            )
            var_index += 1

        return xr.Dataset(data_vars)

    def build_single_hour_dataset(
        self,
        init_datetime: datetime,
        hour: int,
        lats: np.ndarray,
        lons: np.ndarray,
        forecast_norm: np.ndarray,
    ) -> xr.Dataset:
        from diagnostics import compute_diagnostics  # noqa: WPS433

        denorm = self.denormalize(forecast_norm)
        if denorm.ndim == 3:
            denorm = denorm[None, ...]
        ds_hour = self.create_xarray_dataset(init_datetime, [hour], lats, lons, denorm)

        for cname in ["LAND", "OROG"]:
            raw_key = f"{cname}_raw"
            if raw_key in self.data_loader_hrrr.data.files and cname not in ds_hour:
                cvals = self.data_loader_hrrr.data[raw_key].astype(np.float32)
                const_4d = np.tile(cvals[None, None, :, :], (1, 1, 1, 1))
                ds_hour[cname] = xr.DataArray(
                    const_4d,
                    dims=("time", "lead_time", "latitude", "longitude"),
                    coords={
                        "time": [init_datetime],
                        "lead_time": ("lead_time", [hour], {"units": "hours"}),
                        "latitude": (("latitude", "longitude"), lats),
                        "longitude": (("latitude", "longitude"), lons),
                    },
                    name=cname,
                )

        ds_hour = self._apply_inverse_transforms(ds_hour)
        return compute_diagnostics(ds_hour)

    def write_single_hour_netcdf(
        self,
        init_datetime: datetime,
        hour: int,
        ds_hour: xr.Dataset,
        output_dir: str,
        member: Union[int, str],
    ) -> str:
        date_str = f"{self.metadata['init_year']}{self.metadata['init_month']}{self.metadata['init_day']}/{self.metadata['init_hh']}"
        outdir = Path(output_dir) / date_str
        outdir.mkdir(parents=True, exist_ok=True)
        mem_str = "avg" if str(member) == "avg" else f"mem{int(member)}"
        nc_path = outdir / f"hrrrcast_{mem_str}_f{hour:02d}.nc"
        ds_hour.to_netcdf(nc_path)
        return str(nc_path)

    def write_single_hour_grib2(
        self,
        init_datetime: datetime,
        hour: int,
        ds_hour: xr.Dataset,
        output_dir: str,
        member: Union[int, str],
    ) -> None:
        from nc2grib import Netcdf2Grib  # noqa: WPS433

        date_str = f"{self.metadata['init_year']}{self.metadata['init_month']}{self.metadata['init_day']}/{self.metadata['init_hh']}"
        outdir = Path(output_dir) / date_str
        outdir.mkdir(parents=True, exist_ok=True)
        converter = Netcdf2Grib()
        ds_hour = ds_hour.assign_coords(lead_time=("lead_time", [hour]))
        converter.save_grib2(init_datetime, ds_hour, member, outdir)

    def get_variable_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        raw_bounds = {
            "UGRD": (-120, 120),
            "VGRD": (-120, 120),
            "VVEL": (-30, 30),
            "TMP": (180, 340),
            "HGT": (-600, 20000),
            "SPFH": (0, 0.05),
            "PRES": (50000, 110000),
            "MSLMA": (50000, 110000),
            "REFC": (0, 80),
            "T2M": (180, 340),
            "UGRD10M": (-100, 100),
            "VGRD10M": (-100, 100),
            "UGRD80M": (-100, 100),
            "VGRD80M": (-100, 100),
            "D2M": (180, 340),
            "TCDC": (0, 100),
            "LCDC": (0, 100),
            "MCDC": (0, 100),
            "HCDC": (0, 100),
            "VIS": (0, 100000),
            "APCP": (0, 500),
            "HGTCC": (0, 20000),
            "CAPE": (0, 7000),
            "CIN": (-2000, 0),
        }
        mins = []
        maxs = []
        num_levels = len(self.metadata["levels"])
        for index, var in enumerate(raw_bounds):
            vmin, vmax = raw_bounds[var]
            if var in self.LOG_TRANSFORM_VARS:
                vmin = np.log1p(vmin)
                vmax = np.log1p(vmax)
            elif var in self.NEG_LOG_TRANSFORM_VARS:
                vmin = np.sign(vmin) * np.log1p(abs(vmin))
                vmax = np.sign(vmax) * np.log1p(abs(vmax))
            repeat = num_levels if index < 6 else 1
            for _ in range(repeat):
                mins.append(vmin)
                maxs.append(vmax)
        return np.array(mins, dtype=np.float32), np.array(maxs, dtype=np.float32)

    def _build_initial_input(self) -> torch.Tensor:
        hrrr_input = self.data_loader_hrrr.get_model_input()
        gfs_input = self.data_loader_gfs.get_model_input()
        nlat = hrrr_input.shape[1]
        nlon = hrrr_input.shape[2]
        hrrr_tensor = torch.as_tensor(hrrr_input, dtype=self.dtype, device=self.device)
        gfs_tensor = torch.as_tensor(gfs_input[:1], dtype=self.dtype, device=self.device)
        rand_channel = torch.ones((1, nlat, nlon, self.predicted_channels), dtype=self.dtype, device=self.device)
        date_channel = torch.ones((1, nlat, nlon, 6), dtype=self.dtype, device=self.device)
        step_channel = torch.ones((1, nlat, nlon, 1), dtype=self.dtype, device=self.device)
        lead_channel = torch.ones((1, nlat, nlon, 1), dtype=self.dtype, device=self.device)
        return torch.cat(
            [
                hrrr_tensor[:, :, :, : self.predicted_channels],
                gfs_tensor,
                rand_channel,
                hrrr_tensor[:, :, :, self.predicted_channels :],
                date_channel,
                step_channel,
                lead_channel,
            ],
            dim=-1,
        )

    def autoregressive_rollout(
        self,
        lead_hours: int,
        *,
        output_dir: Optional[str] = None,
        write_per_hour: bool = False,
    ) -> Dict[int, Dict[int, np.ndarray]]:
        init_datetime = self.data_loader_hrrr.get_init_datetime()
        initial_input = self._build_initial_input()
        forcing_input = torch.as_tensor(self.data_loader_gfs.get_model_input(), dtype=self.dtype, device=self.device)
        x = initial_input
        state_from_hour = {
            member: x[0:1, :, :, : self.predicted_channels].clone() for member in self.members
        }
        start_pred_noise = self.predicted_channels + self.gfs_channels
        outputs_by_hour: Dict[int, Dict[int, np.ndarray]] = {}
        lats = lons = None
        if write_per_hour:
            lats, lons = self.data_loader_hrrr.get_coordinates()

        io_executor = ThreadPoolExecutor(max_workers=1) if write_per_hour else None
        io_futures = []

        def write_hour_outputs(hour: int, data: np.ndarray, member: int) -> None:
            if not (write_per_hour and output_dir and lats is not None and lons is not None):
                return
            ds_hour = self.build_single_hour_dataset(init_datetime, hour, lats, lons, data)
            self.write_single_hour_netcdf(init_datetime, hour, ds_hour, output_dir, member)
            try:
                self.write_single_hour_grib2(init_datetime, hour, ds_hour, output_dir, member)
            except ModuleNotFoundError as exc:
                logger.warning("Skipping GRIB2 write for hour %s member %s: %s", hour, member, exc)

        hour0_outputs = {
            member: state_from_hour[member].float().cpu().numpy().copy() for member in self.members
        }
        if self.use_nudging:
            pmm0_values, pmm0_channels = self._compute_pmm_mean(hour0_outputs)
            nudged_hour0 = self._nudge_members_toward_pmm(hour0_outputs, pmm0_values, pmm0_channels, self.pmm_alpha)
        else:
            nudged_hour0 = hour0_outputs
        outputs_by_hour[0] = nudged_hour0
        for member in self.members:
            if io_executor:
                io_futures.append(io_executor.submit(write_hour_outputs, 0, nudged_hour0[member], member))
            else:
                write_hour_outputs(0, nudged_hour0[member], member)

        members_sorted = list(range(self.num_members))
        half_count = self.num_members // 2
        step = 1.0 / (half_count + (self.num_members % 2))
        seq: list[float] = []
        if self.num_members % 2 == 1:
            seq.append(0.0)
        for i in range(half_count):
            seq.append(step * (i + 1))
            seq.append(-step * (i + 1))
        phase_angle = {member: seq[i] for i, member in enumerate(members_sorted)}

        for hour in range(1, lead_hours + 1):
            from_hour = ((hour - 1) // 6) * 6
            step_hour = hour - from_hour
            logger.info(f"Torch forecasting hour {hour:02d}: from hour {from_hour:02d} using step {step_hour}h")

            date_encoding = self.date_encoding_tensor(init_datetime, hour)
            lead_encoding = torch.full((*x.shape[:-1], 1), step_hour / 6.0, dtype=self.dtype, device=self.device)
            x_base = torch.cat(
                [
                    x[:, :, :, start_pred_noise:-8],
                    date_encoding,
                    x[:, :, :, -2:-1],
                    lead_encoding,
                ],
                dim=-1,
            )

            hour_member_outputs: Dict[int, np.ndarray] = {}
            for batch_start in range(0, len(self.members), self.batch_size):
                batch_members = self.members[batch_start : batch_start + self.batch_size]
                batch_inputs = []
                batch_noise = []
                for member in batch_members:
                    phase_width = from_hour // 12
                    phase_shift = round(phase_width * phase_angle[member])
                    forcing_idx = int(np.clip(hour - 1 + phase_shift, 0, forcing_input.shape[0] - 1))
                    batch_inputs.append(
                        torch.cat(
                            [
                                state_from_hour[member],
                                forcing_input[forcing_idx : forcing_idx + 1],
                                x_base,
                            ],
                            dim=-1,
                        )
                    )
                    batch_noise.append(self.member_noise[member])

                x_batch = torch.cat(batch_inputs, dim=0)
                noise_batch = torch.cat(batch_noise, dim=0)
                t0 = time.time()
                with torch.no_grad(), self._autocast_context():
                    y_batch = self.runner.sample(
                        x_batch,
                        member_noise=noise_batch,
                        member_ids=batch_members,
                        eta=0.0,
                        tile_size=self.tile_size,
                        halo=self.tile_halo,
                    )
                logger.info(
                    f"Torch hour {hour}, batch {batch_start // self.batch_size + 1}: predict took {time.time() - t0:.3f}s"
                )

                clip_mins = torch.as_tensor(self.channel_mins[: y_batch.shape[-1]], device=self.device, dtype=y_batch.dtype)
                clip_maxs = torch.as_tensor(self.channel_maxs[: y_batch.shape[-1]], device=self.device, dtype=y_batch.dtype)
                y_batch = torch.maximum(torch.minimum(y_batch, clip_maxs), clip_mins)

                for batch_idx, member in enumerate(batch_members):
                    y = y_batch[batch_idx : batch_idx + 1]
                    if hour % 6 == 0:
                        state_from_hour[member] = y
                    hour_member_outputs[member] = y.float().cpu().numpy().copy()

            if self.use_nudging:
                pmm_values, pmm_channels = self._compute_pmm_mean(hour_member_outputs)
                nudged_outputs = self._nudge_members_toward_pmm(hour_member_outputs, pmm_values, pmm_channels, self.pmm_alpha)
            else:
                nudged_outputs = hour_member_outputs
            outputs_by_hour[hour] = nudged_outputs

            for member in self.members:
                if io_executor:
                    io_futures.append(io_executor.submit(write_hour_outputs, hour, nudged_outputs[member], member))
                else:
                    write_hour_outputs(hour, nudged_outputs[member], member)

        if io_executor:
            for future in as_completed(io_futures):
                future.result()
            io_executor.shutdown(wait=True)

        return outputs_by_hour


def parse_members(member_args: List[str]) -> List[int]:
    members = []
    for arg in member_args:
        for part in arg.split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part:
                start, end = part.split("-")
                members.extend(range(int(start), int(end) + 1))
            else:
                members.append(int(part))
    return sorted(set(members))


def parse_dtype(value: str) -> torch.dtype:
    mapping = {
        "fp32": torch.float32,
        "float32": torch.float32,
        "fp16": torch.float16,
        "float16": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
    }
    if value not in mapping:
        raise ValueError(f"Unsupported dtype: {value}")
    return mapping[value]


def main() -> None:
    _ensure_src_on_path()
    import utils  # noqa: WPS433

    parser = argparse.ArgumentParser(description="Torch autoregressive HRRRCast forecast runner")
    parser.add_argument("checkpoint_path", help="Path to converted .pt checkpoint")
    parser.add_argument("inittime", help="Initialization time YYYY-MM-DDTHH")
    parser.add_argument("lead_hours", type=int, help="Lead hours")
    parser.add_argument("--members", nargs="+", required=True, help="Member list, e.g. 0-2 or 0 1 2")
    parser.add_argument("--base_dir", default="./", help="Base directory for preprocessed NPZ inputs")
    parser.add_argument("--output_dir", default="./", help="Output directory")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Torch device")
    parser.add_argument("--dtype", default="bf16", choices=["fp32", "float32", "fp16", "float16", "bf16", "bfloat16"])
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--tile_size", default="256,256", help="Tile size as H,W. Use 'none' to disable tiling")
    parser.add_argument("--tile_halo", type=int, default=32, help="Tile overlap halo in pixels")
    parser.add_argument("--allow_full_frame", action="store_true", help="Explicitly allow full-frame sampling without tiling")
    parser.add_argument("--pmm_alpha", type=float, default=0.7)
    parser.add_argument("--no_nudging", action="store_true")
    parser.add_argument("--log_level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--no_write", action="store_true", help="Do not write NetCDF/GRIB2 outputs")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    init_datetime, init_year, init_month, init_day, init_hh = utils.validate_datetime(args.inittime)
    date_str = f"{init_year}{init_month}{init_day}/{init_hh}"
    filedate_str = f"{init_year}{init_month}{init_day}_{init_hh}"
    hrrr_preprocessed_file = f"{args.base_dir}/{date_str}/hrrr_{filedate_str}.npz"
    gfs_preprocessed_file = f"{args.base_dir}/{date_str}/gfs_{filedate_str}.npz"

    dtype = parse_dtype(args.dtype)
    if torch.device(args.device).type == "cpu" and dtype == torch.float16:
        logger.warning("CPU fp16 kernels are limited; using fp32 instead")
        dtype = torch.float32

    if args.tile_size.lower() == "none":
        tile_size = None
    else:
        tile_parts = [part.strip() for part in args.tile_size.split(",")]
        if len(tile_parts) != 2:
            raise ValueError("--tile_size must be 'H,W' or 'none'")
        tile_size = (int(tile_parts[0]), int(tile_parts[1]))

    model = load_torch_checkpoint(args.checkpoint_path, device=args.device, dtype=dtype)
    forecaster = TorchWeatherForecaster(
        model,
        TorchPreprocessedDataLoader(hrrr_preprocessed_file),
        TorchPreprocessedDataLoader(gfs_preprocessed_file),
        members=parse_members(args.members),
        batch_size=args.batch_size,
        device=args.device,
        dtype=dtype,
        pmm_alpha=args.pmm_alpha,
        use_nudging=not args.no_nudging,
        tile_size=tile_size,
        tile_halo=args.tile_halo,
        allow_full_frame=args.allow_full_frame,
    )
    logger.info(
        "Torch rollout config: device=%s dtype=%s tile_size=%s halo=%s estimated_full_input_gib=%.2f",
        args.device,
        args.dtype,
        tile_size,
        args.tile_halo,
        forecaster.estimate_step_tensor_gib(),
    )
    forecaster.autoregressive_rollout(
        args.lead_hours,
        output_dir=args.output_dir,
        write_per_hour=not args.no_write,
    )


if __name__ == "__main__":
    main()
