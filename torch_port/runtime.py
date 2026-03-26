from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F

from .diffusion import (
    ALPHA_BAR,
    INFERENCE_STEPS,
    NUM_DIFFUSION_STEPS,
    compute_epsilon,
    ddim,
)


class HRRRCastDiffusionRunner:
    """Torch-native diffusion sampler around the converted HRRRCast network."""

    def __init__(
        self,
        network: torch.nn.Module,
        *,
        predicted_channels: int = 138,
        gfs_channels: int = 42,
    ):
        self.network = network
        self.predicted_channels = predicted_channels
        self.gfs_channels = gfs_channels

    def _forward_tiled(
        self,
        model_input: torch.Tensor,
        *,
        tile_size: tuple[int, int],
        halo: int,
    ) -> torch.Tensor:
        batch, height, width, _channels = model_input.shape
        tile_h, tile_w = tile_size
        output = None

        for y in range(0, height, tile_h):
            for x in range(0, width, tile_w):
                y_core0 = y
                y_core1 = min(y + tile_h, height)
                x_core0 = x
                x_core1 = min(x + tile_w, width)

                y0 = max(0, y_core0 - halo)
                y1 = min(height, y_core1 + halo)
                x0 = max(0, x_core0 - halo)
                x1 = min(width, x_core1 + halo)

                tile_input = model_input[:, y0:y1, x0:x1, :]
                tile_input, crop_meta = self._pad_tile_input(tile_input)
                tile_output = self.network(tile_input)
                tile_output = self._crop_padded_tile_output(tile_output, crop_meta)

                crop_y0 = y_core0 - y0
                crop_y1 = crop_y0 + (y_core1 - y_core0)
                crop_x0 = x_core0 - x0
                crop_x1 = crop_x0 + (x_core1 - x_core0)
                tile_core = tile_output[:, crop_y0:crop_y1, crop_x0:crop_x1, :]

                if output is None:
                    output = torch.empty(
                        (batch, height, width, tile_core.shape[-1]),
                        dtype=tile_core.dtype,
                        device=tile_core.device,
                    )
                output[:, y_core0:y_core1, x_core0:x_core1, :] = tile_core

        if output is None:
            raise RuntimeError("Tiled forward produced no output")
        return output

    @staticmethod
    def _pad_tile_input(tile_input: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        _, height, width, _ = tile_input.shape
        pad_h = (-((height + 5) % 8)) % 8
        pad_w = (-((width + 1) % 8)) % 8
        if pad_h == 0 and pad_w == 0:
            return tile_input, (0, 0)
        x = tile_input.permute(0, 3, 1, 2)
        x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
        return x.permute(0, 2, 3, 1), (pad_h, pad_w)

    @staticmethod
    def _crop_padded_tile_output(tile_output: torch.Tensor, crop_meta: tuple[int, int]) -> torch.Tensor:
        pad_h, pad_w = crop_meta
        if pad_h == 0 and pad_w == 0:
            return tile_output
        h_end = tile_output.shape[1] - pad_h if pad_h else tile_output.shape[1]
        w_end = tile_output.shape[2] - pad_w if pad_w else tile_output.shape[2]
        return tile_output[:, :h_end, :w_end, :]

    def _predict_x0(
        self,
        model_input: torch.Tensor,
        *,
        tile_size: tuple[int, int] | None,
        halo: int,
    ) -> torch.Tensor:
        if tile_size is None:
            return self.network(model_input)
        return self._forward_tiled(model_input, tile_size=tile_size, halo=halo)

    def sample(
        self,
        model_input: torch.Tensor,
        *,
        member_noise: torch.Tensor,
        member_ids: int | Sequence[int] = 0,
        eta: float = 0.0,
        tile_size: tuple[int, int] | None = None,
        halo: int = 32,
    ) -> torch.Tensor:
        if model_input.ndim != 4:
            raise ValueError(f"Expected NHWC model_input, got shape {tuple(model_input.shape)}")
        if member_noise.shape[:3] != model_input.shape[:3]:
            raise ValueError(
                "member_noise spatial dimensions must match model_input: "
                f"{tuple(member_noise.shape)} vs {tuple(model_input.shape)}"
            )
        if member_noise.shape[-1] != self.predicted_channels:
            raise ValueError(
                f"member_noise channel count must equal predicted_channels={self.predicted_channels}, "
                f"got {member_noise.shape[-1]}"
            )

        start = self.predicted_channels + self.gfs_channels
        x = model_input
        xn = member_noise

        for reverse_step in range(len(INFERENCE_STEPS) - 1):
            step_index = len(INFERENCE_STEPS) - 1 - reverse_step
            t = INFERENCE_STEPS[step_index]
            step_encoding = torch.full(
                (*x.shape[:-1], 1),
                fill_value=float(t / NUM_DIFFUSION_STEPS),
                dtype=x.dtype,
                device=x.device,
            )
            x_step = torch.cat(
                [
                    x[:, :, :, :start],
                    xn,
                    x[:, :, :, start + self.predicted_channels : -2],
                    step_encoding,
                    x[:, :, :, -1:],
                ],
                dim=-1,
            )
            x0 = self._predict_x0(
                x_step,
                tile_size=tile_size,
                halo=halo,
            )
            epsilon_t = compute_epsilon(
                xn,
                x0,
                torch.full((x.shape[0],), int(t), device=x.device, dtype=torch.long),
                alpha_bar=ALPHA_BAR,
            )
            xn = ddim(
                xn,
                epsilon_t,
                step_index,
                seed=member_ids,
                eta=eta,
                alpha_bar=ALPHA_BAR,
                inference_steps=INFERENCE_STEPS,
            )

        return xn
