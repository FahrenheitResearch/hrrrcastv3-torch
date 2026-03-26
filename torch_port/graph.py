from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from . import layers as tl


def _normalize_to_tuple(value: Any) -> tuple[int, ...]:
    if isinstance(value, int):
        return (value, value)
    return tuple(int(v) for v in value)


def _first_input_shape(layer_cfg: dict[str, Any]) -> list[Any] | None:
    build_cfg = layer_cfg.get("build_config") or {}
    input_shape = build_cfg.get("input_shape")
    if input_shape is None:
        return None
    if isinstance(input_shape, list) and input_shape and isinstance(input_shape[0], list):
        return input_shape[0]
    return input_shape


def _extract_inbound_names(layer_cfg: dict[str, Any]) -> list[str]:
    inbound_nodes = layer_cfg.get("inbound_nodes") or []
    if not inbound_nodes:
        return []
    first_node = inbound_nodes[0]
    names = []
    for inbound in first_node:
        if isinstance(inbound, list) and inbound:
            names.append(inbound[0])
        elif isinstance(inbound, dict):
            args = inbound.get("args") or []
            for arg in args:
                history = arg.get("config", {}).get("keras_history")
                if history:
                    names.append(history[0])
    return names


def _layer_name_ref(entry: Any) -> str:
    if isinstance(entry, list):
        return entry[0]
    raise ValueError(f"Unsupported Keras tensor reference: {entry}")


class KerasFunctionalModule(nn.Module):
    def __init__(self, config: dict[str, Any], *, enable_checkpointing: bool = False):
        super().__init__()
        self.name = config.get("name", "keras_functional")
        self.enable_checkpointing = enable_checkpointing
        self.layer_order: list[str] = []
        self.layer_configs: dict[str, dict[str, Any]] = {}
        self.inbound_names: dict[str, list[str]] = {}
        self.modules_by_layer = nn.ModuleDict()
        self.input_names = [_layer_name_ref(item) for item in config.get("input_layers", [])]
        self.output_names = [_layer_name_ref(item) for item in config.get("output_layers", [])]

        for layer_cfg in config["layers"]:
            name = layer_cfg["name"]
            self.layer_order.append(name)
            self.layer_configs[name] = layer_cfg
            self.inbound_names[name] = _extract_inbound_names(layer_cfg)
            module = self._build_layer_module(layer_cfg)
            if module is not None:
                self.modules_by_layer[name] = module

    def _build_layer_module(self, layer_cfg: dict[str, Any]) -> nn.Module | None:
        class_name = layer_cfg["class_name"]
        config = layer_cfg["config"]
        input_shape = _first_input_shape(layer_cfg)

        if class_name == "InputLayer":
            return None
        if class_name == "Dense":
            if input_shape is None:
                raise ValueError(f"Missing build_config.input_shape for Dense layer {layer_cfg['name']}")
            return tl.DenseLayer(
                in_features=int(input_shape[-1]),
                out_features=int(config["units"]),
                use_bias=bool(config["use_bias"]),
                activation=config["activation"],
            )
        if class_name == "Conv2D":
            if input_shape is None:
                raise ValueError(f"Missing build_config.input_shape for Conv2D layer {layer_cfg['name']}")
            return tl.Conv2DNHWC(
                in_channels=int(input_shape[-1]),
                out_channels=int(config["filters"]),
                kernel_size=_normalize_to_tuple(config["kernel_size"]),
                strides=_normalize_to_tuple(config["strides"]),
                padding=config["padding"],
                use_bias=bool(config["use_bias"]),
            )
        if class_name == "LayerNormalization":
            axis = config["axis"]
            if isinstance(axis, int):
                axis = [axis]
            if input_shape is None:
                raise ValueError(f"Missing build_config.input_shape for LayerNormalization layer {layer_cfg['name']}")
            return tl.LayerNormalizationLayer(
                input_shape=input_shape,
                axes=axis,
                epsilon=float(config["epsilon"]),
            )
        if class_name == "Activation":
            return tl.ActivationLayer(config["activation"])
        if class_name == "Multiply":
            return tl.MultiplyLayer()
        if class_name == "Add":
            return tl.AddLayer()
        if class_name == "Subtract":
            return tl.SubtractLayer()
        if class_name == "Concatenate":
            return tl.ConcatenateLayer(axis=int(config["axis"]))
        if class_name == "GlobalAveragePooling2D":
            return tl.GlobalAveragePooling2DLayer()
        if class_name == "Reshape":
            return tl.ReshapeLayer(target_shape=tuple(config["target_shape"]))
        if class_name == "MaxPooling2D":
            return tl.MaxPooling2DLayer(
                pool_size=_normalize_to_tuple(config["pool_size"]),
                strides=_normalize_to_tuple(config["strides"]),
            )
        if class_name == "UpSampling2D":
            return tl.UpSampling2DLayer(
                size=_normalize_to_tuple(config["size"]),
                interpolation=config["interpolation"],
            )
        if class_name == "ChannelSliceLayer":
            return tl.ChannelSliceLayer(start=int(config["start"]), end=int(config["end"]))
        if class_name == "OutputMaskLayer":
            return tl.OutputMaskLayer(config["output_tensor_mask"])
        if class_name == "ReflectPadLayer":
            return tl.ReflectPadLayer(config["padding"])
        if class_name == "UnpadLayer":
            return tl.UnpadLayer(config["padding"])
        if class_name == "CastLayer":
            return tl.CastLayer(config["dtype"])
        if class_name == "ChannelPoolAvg":
            return tl.ChannelPoolAvg()
        if class_name == "ChannelPoolMax":
            return tl.ChannelPoolMax()
        if class_name == "TimeCondLayer":
            return tl.TimeCondLayer(
                time_mask=config["time_mask"],
                use_crps=bool(config["use_crps"]),
                use_noise=bool(config["use_noise"]),
            )
        if class_name == "SpatialGroupedConv2D":
            if input_shape is None:
                raise ValueError(f"Missing build_config.input_shape for SpatialGroupedConv2D layer {layer_cfg['name']}")
            return tl.SpatialGroupedConv2D(
                in_channels=int(input_shape[-1]),
                filters=int(config["filters"]),
                kernel_size=config["kernel_size"],
                groups_h=int(config["groups_h"]),
                groups_w=int(config["groups_w"]),
            )
        if class_name == "RecomputeSubModel":
            submodel = KerasFunctionalModule(
                config=config["submodel"]["config"],
                enable_checkpointing=self.enable_checkpointing,
            )
            return tl.RecomputeSubModel(submodel, use_checkpoint=self.enable_checkpointing)
        raise NotImplementedError(f"Unsupported Keras layer class: {class_name} ({layer_cfg['name']})")

    def forward(self, *inputs: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, ...]:
        if len(inputs) != len(self.input_names):
            raise ValueError(
                f"{self.name} expects {len(self.input_names)} input tensors, got {len(inputs)}"
            )

        tensors: dict[str, torch.Tensor] = {
            name: tensor for name, tensor in zip(self.input_names, inputs)
        }

        for layer_name in self.layer_order:
            layer_cfg = self.layer_configs[layer_name]
            if layer_cfg["class_name"] == "InputLayer":
                continue

            inbound = self.inbound_names[layer_name]
            args = [tensors[name] for name in inbound]
            module = self.modules_by_layer[layer_name]

            if layer_cfg["class_name"] in {"Multiply", "Add", "Subtract", "Concatenate"}:
                output = module(args)
            elif layer_cfg["class_name"] == "RecomputeSubModel":
                output = module(*args)
            else:
                if len(args) != 1:
                    raise ValueError(
                        f"Layer {layer_name} expected one inbound tensor, got {len(args)}"
                    )
                output = module(args[0])

            tensors[layer_name] = output

        outputs = tuple(tensors[name] for name in self.output_names)
        if len(outputs) == 1:
            return outputs[0]
        return outputs

    def collect_modules_flat(self) -> dict[str, nn.Module]:
        modules: dict[str, nn.Module] = {}
        for layer_name in self.layer_order:
            if layer_name not in self.modules_by_layer:
                continue
            module = self.modules_by_layer[layer_name]
            modules[layer_name] = module
            if isinstance(module, tl.RecomputeSubModel):
                modules.update(module.submodel.collect_modules_flat())
        return modules


def load_keras_model_config(config_source: str | Path) -> dict[str, Any]:
    config_source = Path(config_source)
    if config_source.suffix == ".json":
        return json.loads(config_source.read_text())
    with zipfile.ZipFile(config_source) as zf:
        return json.loads(zf.read("config.json"))


def build_torch_model_from_keras_archive(
    keras_archive: str | Path,
    *,
    enable_checkpointing: bool = False,
) -> KerasFunctionalModule:
    cfg = load_keras_model_config(keras_archive)
    return KerasFunctionalModule(cfg["config"], enable_checkpointing=enable_checkpointing)
