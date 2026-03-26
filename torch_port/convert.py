from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from .graph import build_torch_model_from_keras_archive
from .layers import (
    Conv2DNHWC,
    DenseLayer,
    LayerNormalizationLayer,
    RecomputeSubModel,
    SpatialGroupedConv2D,
)


def _repo_src_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "src"


def load_keras_model(keras_archive: str | Path):
    src_dir = str(_repo_src_dir())
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    import tensorflow as tf  # noqa: WPS433
    import resnet  # noqa: F401,WPS433

    def _patched_time_cond_call(self, inputs):
        input_channels = inputs.shape[-1]
        if input_channels is None:
            raise ValueError("TimeCondLayer requires a known channel dimension")
        indices = [idx if idx >= 0 else input_channels + idx for idx in self.time_mask]
        time_feats = tf.gather(inputs, indices, axis=-1)
        d = time_feats[:, 0, 0, :]
        if not self.use_crps:
            return d
        lead_time = d[:, -1:]
        if not self.use_noise:
            return lead_time
        ens_id = tf.cast(tf.floor(tf.cast(d[:, 0], tf.float32) * (2**31 - 1)), tf.int32)
        seed = tf.stack([ens_id, ens_id ^ 0x9E3779B9], axis=1)
        z = tf.random.stateless_normal([tf.shape(d)[0], 32], seed=seed, dtype=lead_time.dtype)
        return tf.concat([z, lead_time], axis=1)

    resnet.TimeCondLayer.call = _patched_time_cond_call

    return tf.keras.models.load_model(
        str(keras_archive),
        safe_mode=False,
        compile=False,
    )


def _iter_tf_layers(model) -> Iterable:
    for layer in model.layers:
        yield layer
        if hasattr(layer, "submodel"):
            yield from _iter_tf_layers(layer.submodel)


def _copy_dense(tf_layer, torch_layer: DenseLayer) -> None:
    weights = tf_layer.get_weights()
    if len(weights) not in {1, 2}:
        raise ValueError(f"Unexpected Dense weights for {tf_layer.name}: {len(weights)}")
    kernel = torch.from_numpy(np.asarray(weights[0]).T)
    torch_layer.linear.weight.data.copy_(kernel)
    if len(weights) == 2 and torch_layer.linear.bias is not None:
        torch_layer.linear.bias.data.copy_(torch.from_numpy(np.asarray(weights[1])))


def _copy_conv2d_weights(weight_list, conv: torch.nn.Conv2d) -> None:
    if len(weight_list) not in {1, 2}:
        raise ValueError(f"Unexpected Conv2D weights length: {len(weight_list)}")
    kernel = torch.from_numpy(np.asarray(weight_list[0]).transpose(3, 2, 0, 1))
    conv.weight.data.copy_(kernel)
    if len(weight_list) == 2 and conv.bias is not None:
        conv.bias.data.copy_(torch.from_numpy(np.asarray(weight_list[1])))


def _copy_conv(tf_layer, torch_layer: Conv2DNHWC) -> None:
    _copy_conv2d_weights(tf_layer.get_weights(), torch_layer.conv)


def _copy_spatial_grouped_conv(tf_layer, torch_layer: SpatialGroupedConv2D) -> None:
    _copy_conv2d_weights(tf_layer.conv.get_weights(), torch_layer.conv.conv)


def _copy_layer_norm(tf_layer, torch_layer: LayerNormalizationLayer) -> None:
    weights = tf_layer.get_weights()
    if len(weights) == 2:
        gamma, beta = weights
        torch_layer.gamma.data.copy_(torch.from_numpy(np.asarray(gamma)))
        torch_layer.beta.data.copy_(torch.from_numpy(np.asarray(beta)))
        return
    if len(weights) == 1:
        torch_layer.gamma.data.fill_(1.0)
        torch_layer.beta.data.copy_(torch.from_numpy(np.asarray(weights[0])))
        return
    raise ValueError(f"Unexpected LayerNormalization weights for {tf_layer.name}: {len(weights)}")


def _copy_weights(tf_layer, torch_layer) -> None:
    if isinstance(torch_layer, DenseLayer):
        _copy_dense(tf_layer, torch_layer)
        return
    if isinstance(torch_layer, Conv2DNHWC):
        _copy_conv(tf_layer, torch_layer)
        return
    if isinstance(torch_layer, SpatialGroupedConv2D):
        _copy_spatial_grouped_conv(tf_layer, torch_layer)
        return
    if isinstance(torch_layer, LayerNormalizationLayer):
        _copy_layer_norm(tf_layer, torch_layer)
        return
    if isinstance(torch_layer, RecomputeSubModel):
        return
    if tf_layer.weights:
        raise NotImplementedError(
            f"TensorFlow layer {tf_layer.name} has weights but no copy handler for {type(torch_layer).__name__}"
        )


def load_torch_model_from_keras(
    keras_archive: str | Path,
    *,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
    enable_checkpointing: bool = False,
):
    keras_archive = Path(keras_archive)
    torch_model = build_torch_model_from_keras_archive(
        keras_archive,
        enable_checkpointing=enable_checkpointing,
    )
    tf_model = load_keras_model(keras_archive)

    torch_layers = torch_model.collect_modules_flat()
    for tf_layer in _iter_tf_layers(tf_model):
        torch_layer = torch_layers.get(tf_layer.name)
        if torch_layer is None:
            continue
        _copy_weights(tf_layer, torch_layer)

    torch_model.to(device=device, dtype=dtype)
    torch_model.eval()
    return torch_model, tf_model


def export_torch_checkpoint(
    keras_archive: str | Path,
    output_path: str | Path,
    *,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
    enable_checkpointing: bool = False,
) -> Path:
    keras_archive = Path(keras_archive)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch_model, _ = load_torch_model_from_keras(
        keras_archive,
        device=device,
        dtype=dtype,
        enable_checkpointing=enable_checkpointing,
    )
    payload = {
        "keras_archive": str(keras_archive),
        "state_dict": torch_model.state_dict(),
        "dtype": str(dtype),
    }
    torch.save(payload, output_path)
    return output_path


def _parse_dtype(value: str) -> torch.dtype:
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
    parser = argparse.ArgumentParser(description="Convert a HRRRCast Keras archive into a PyTorch checkpoint.")
    parser.add_argument("keras_archive", help="Path to the .keras archive")
    parser.add_argument("output_path", help="Path to the output .pt checkpoint")
    parser.add_argument(
        "--dtype",
        default="fp32",
        choices=["fp32", "float32", "fp16", "float16", "bf16", "bfloat16"],
        help="Target checkpoint dtype",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device used while materializing the Torch model",
    )
    parser.add_argument(
        "--enable-checkpointing",
        action="store_true",
        help="Enable Torch activation checkpoint wrappers inside converted submodels",
    )
    args = parser.parse_args()

    output_path = export_torch_checkpoint(
        args.keras_archive,
        args.output_path,
        device=args.device,
        dtype=_parse_dtype(args.dtype),
        enable_checkpointing=args.enable_checkpointing,
    )
    print(output_path)


if __name__ == "__main__":
    main()
