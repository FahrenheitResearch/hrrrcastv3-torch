from __future__ import annotations

from typing import Iterable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def _normalize_index(rank: int, axis: int) -> int:
    return axis if axis >= 0 else rank + axis


def _normalize_axes(rank: int, axes: Sequence[int]) -> tuple[int, ...]:
    return tuple(_normalize_index(rank, axis) for axis in axes)


def _resolve_torch_dtype(dtype: str | torch.dtype) -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        return dtype
    mapping = {
        "float16": torch.float16,
        "float32": torch.float32,
        "float64": torch.float64,
        "bfloat16": torch.bfloat16,
        "int32": torch.int32,
        "int64": torch.int64,
    }
    if dtype not in mapping:
        raise ValueError(f"Unsupported dtype: {dtype}")
    return mapping[dtype]


class AddLayer(nn.Module):
    def forward(self, inputs: Sequence[torch.Tensor]) -> torch.Tensor:
        if len(inputs) < 2:
            raise ValueError("AddLayer requires at least two tensors")
        result = inputs[0]
        for tensor in inputs[1:]:
            result = result + tensor
        return result


class MultiplyLayer(nn.Module):
    def forward(self, inputs: Sequence[torch.Tensor]) -> torch.Tensor:
        if len(inputs) < 2:
            raise ValueError("MultiplyLayer requires at least two tensors")
        result = inputs[0]
        for tensor in inputs[1:]:
            result = result * tensor
        return result


class SubtractLayer(nn.Module):
    def forward(self, inputs: Sequence[torch.Tensor]) -> torch.Tensor:
        if len(inputs) != 2:
            raise ValueError("SubtractLayer requires exactly two tensors")
        return inputs[0] - inputs[1]


class ConcatenateLayer(nn.Module):
    def __init__(self, axis: int):
        super().__init__()
        self.axis = axis

    def forward(self, inputs: Sequence[torch.Tensor]) -> torch.Tensor:
        if not inputs:
            raise ValueError("ConcatenateLayer requires at least one tensor")
        axis = _normalize_index(inputs[0].ndim, self.axis)
        return torch.cat(list(inputs), dim=axis)


class ActivationLayer(nn.Module):
    def __init__(self, activation: str):
        super().__init__()
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation == "relu":
            return F.relu(x)
        if self.activation == "sigmoid":
            return torch.sigmoid(x)
        if self.activation == "linear":
            return x
        raise ValueError(f"Unsupported activation: {self.activation}")


class DenseLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, use_bias: bool, activation: str = "linear"):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=use_bias)
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        if self.activation == "linear":
            return x
        if self.activation == "relu":
            return F.relu(x)
        if self.activation == "sigmoid":
            return torch.sigmoid(x)
        raise ValueError(f"Unsupported Dense activation: {self.activation}")


class Conv2DNHWC(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Sequence[int],
        strides: Sequence[int],
        padding: str,
        use_bias: bool,
    ):
        super().__init__()
        if padding not in {"same", "valid"}:
            raise ValueError(f"Unsupported padding mode: {padding}")
        if tuple(strides) != (1, 1):
            raise ValueError(f"Unsupported Conv2D strides: {strides}")
        if padding == "same":
            pad = (kernel_size[0] // 2, kernel_size[1] // 2)
        else:
            pad = (0, 0)
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=tuple(kernel_size),
            stride=tuple(strides),
            padding=pad,
            bias=use_bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 3, 1, 2)
        x = self.conv(x)
        return x.permute(0, 2, 3, 1)


class LayerNormalizationLayer(nn.Module):
    def __init__(self, input_shape: Sequence[int | None], axes: Sequence[int], epsilon: float):
        super().__init__()
        rank = len(input_shape)
        axes = _normalize_axes(rank, axes)
        shape = tuple(int(input_shape[axis]) for axis in axes)
        self.axes = axes
        self.epsilon = epsilon
        self.gamma = nn.Parameter(torch.ones(shape, dtype=torch.float32))
        self.beta = nn.Parameter(torch.zeros(shape, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dims = self.axes
        mean = x.mean(dim=dims, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=dims, keepdim=True)
        y = (x - mean) / torch.sqrt(var + self.epsilon)
        shape = [1] * x.ndim
        for axis, size in zip(self.axes, self.gamma.shape):
            shape[axis] = size
        gamma = self.gamma.view(*shape)
        beta = self.beta.view(*shape)
        return y * gamma + beta


class GlobalAveragePooling2DLayer(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=(1, 2))


class ReshapeLayer(nn.Module):
    def __init__(self, target_shape: Sequence[int]):
        super().__init__()
        self.target_shape = tuple(target_shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.reshape((x.shape[0],) + self.target_shape)


class MaxPooling2DLayer(nn.Module):
    def __init__(self, pool_size: Sequence[int], strides: Sequence[int]):
        super().__init__()
        self.pool_size = tuple(pool_size)
        self.strides = tuple(strides)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 3, 1, 2)
        x = F.max_pool2d(x, kernel_size=self.pool_size, stride=self.strides)
        return x.permute(0, 2, 3, 1)


class UpSampling2DLayer(nn.Module):
    def __init__(self, size: Sequence[int], interpolation: str):
        super().__init__()
        if interpolation != "nearest":
            raise ValueError(f"Unsupported UpSampling2D interpolation: {interpolation}")
        self.size = tuple(size)
        self.interpolation = interpolation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 3, 1, 2)
        x = F.interpolate(x, scale_factor=self.size, mode=self.interpolation)
        return x.permute(0, 2, 3, 1)


class ChannelSliceLayer(nn.Module):
    def __init__(self, start: int, end: int):
        super().__init__()
        self.start = start
        self.end = end

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :, :, self.start : self.end]


class OutputMaskLayer(nn.Module):
    def __init__(self, output_tensor_mask: Sequence[int]):
        super().__init__()
        self.register_buffer(
            "output_tensor_mask",
            torch.tensor(list(output_tensor_mask), dtype=torch.long),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.index_select(dim=-1, index=self.output_tensor_mask)


class ReflectPadLayer(nn.Module):
    def __init__(self, padding: Sequence[Sequence[int]]):
        super().__init__()
        self.padding = tuple(tuple(int(v) for v in row) for row in padding)
        if self.padding[0] != (0, 0) or self.padding[3] != (0, 0):
            raise ValueError(f"Unsupported reflect padding layout: {self.padding}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pad_h = self.padding[1]
        pad_w = self.padding[2]
        x = x.permute(0, 3, 1, 2)
        x = F.pad(x, (pad_w[0], pad_w[1], pad_h[0], pad_h[1]), mode="reflect")
        return x.permute(0, 2, 3, 1)


class UnpadLayer(nn.Module):
    def __init__(self, padding: Sequence[Sequence[int]]):
        super().__init__()
        self.padding = tuple(tuple(int(v) for v in row) for row in padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h_start = self.padding[0][0]
        h_end = -self.padding[0][1] if self.padding[0][1] else None
        w_start = self.padding[1][0]
        w_end = -self.padding[1][1] if self.padding[1][1] else None
        return x[:, h_start:h_end, w_start:w_end, :]


class CastLayer(nn.Module):
    def __init__(self, dtype: str | torch.dtype):
        super().__init__()
        self.target_dtype = _resolve_torch_dtype(dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.to(self.target_dtype)


class ChannelPoolAvg(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=-1, keepdim=True)


class ChannelPoolMax(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.amax(dim=-1, keepdim=True)


class TimeCondLayer(nn.Module):
    def __init__(self, time_mask: Sequence[int], use_crps: bool, use_noise: bool):
        super().__init__()
        self.time_mask = list(time_mask)
        self.use_crps = use_crps
        self.use_noise = use_noise

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        time_feats = x[:, :, :, self.time_mask]
        d = time_feats[:, 0, 0, :]
        if not self.use_crps:
            return d
        lead_time = d[:, -1:]
        if not self.use_noise:
            return lead_time
        ens_id = torch.floor(d[:, 0].float() * (2**31 - 1)).to(torch.int64)
        noise = []
        for seed in ens_id.tolist():
            generator = torch.Generator(device="cpu")
            generator.manual_seed(seed)
            z = torch.randn((32,), generator=generator, dtype=lead_time.dtype)
            noise.append(z)
        z = torch.stack(noise, dim=0).to(device=lead_time.device)
        return torch.cat([z, lead_time], dim=1)


class SpatialGroupedConv2D(nn.Module):
    def __init__(self, in_channels: int, filters: int, kernel_size: int | Sequence[int], groups_h: int, groups_w: int):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel = (kernel_size, kernel_size)
        else:
            kernel = tuple(kernel_size)
        if kernel[0] != kernel[1]:
            raise ValueError(f"Unsupported non-square kernel_size: {kernel}")
        self.filters = filters
        self.kernel_size = kernel[0]
        self.groups_h = groups_h
        self.groups_w = groups_w
        self.conv = Conv2DNHWC(
            in_channels=in_channels,
            out_channels=filters,
            kernel_size=kernel,
            strides=(1, 1),
            padding="same",
            use_bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pad_h = self.kernel_size // 2
        pad_w = self.kernel_size // 2
        height = x.shape[1]
        width = x.shape[2]
        tile_h = (height + self.groups_h - 1) // self.groups_h
        tile_w = (width + self.groups_w - 1) // self.groups_w
        outputs = []
        for i in range(self.groups_h):
            row_outputs = []
            for j in range(self.groups_w):
                y0 = max(0, i * tile_h - pad_h)
                y1 = min(height, (i + 1) * tile_h + pad_h)
                x0 = max(0, j * tile_w - pad_w)
                x1 = min(width, (j + 1) * tile_w + pad_w)
                tile = x[:, y0:y1, x0:x1, :]
                tile_conv = self.conv(tile)
                crop_top = pad_h if i > 0 else 0
                crop_bottom = pad_h if i < self.groups_h - 1 else 0
                crop_left = pad_w if j > 0 else 0
                crop_right = pad_w if j < self.groups_w - 1 else 0
                h_end = tile_conv.shape[1] - crop_bottom if crop_bottom else tile_conv.shape[1]
                w_end = tile_conv.shape[2] - crop_right if crop_right else tile_conv.shape[2]
                tile_conv = tile_conv[:, crop_top:h_end, crop_left:w_end, :]
                row_outputs.append(tile_conv)
            outputs.append(torch.cat(row_outputs, dim=2))
        return torch.cat(outputs, dim=1)


class RecomputeSubModel(nn.Module):
    def __init__(self, submodel: nn.Module, use_checkpoint: bool = False):
        super().__init__()
        self.submodel = submodel
        self.use_checkpoint = use_checkpoint

    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        if self.use_checkpoint and self.training:
            return checkpoint(self.submodel, *inputs, use_reentrant=False)
        return self.submodel(*inputs)


WEIGHTED_LAYER_TYPES = {
    "Dense",
    "Conv2D",
    "LayerNormalization",
    "SpatialGroupedConv2D",
}
