---
license: mit
library_name: pytorch
tags:
  - weather
  - meteorology
  - diffusion
  - forecast
  - hrrr
---

# HRRRCast V3 Torch Port

This repository contains converted PyTorch checkpoints for the HRRRCast V3 diffusion model ported from the NOAA GSL Keras checkpoint.

## What this is

- Converted Torch checkpoints for the HRRRCast diffusion network
- Intended to be used together with the GitHub code repository
- Supports `fp32`, `fp16`, and `bf16` checkpoints

## What this is not

- Not a `transformers` model
- Not a `diffusers` pipeline
- Not a browser widget model
- Not a lightweight consumer inference package

## Required code

Use these checkpoints with the corresponding GitHub repository that contains:

- `torch_port/`
- `src/`
- `net-diffusion/model.config.json`

The loader reconstructs the Torch graph from the extracted Keras `config.json` plus the uploaded Torch state dict.

## Example

```bash
PYTHONPATH=. ./.venv/bin/python -m torch_port.forecast \
  converted/hrrrcast_diffusion_bf16.pt \
  2026-03-26T19 \
  6 \
  --members 0 \
  --base_dir /path/to/preprocessed_npz \
  --output_dir /path/to/output \
  --device cuda \
  --dtype bf16 \
  --tile_size 96,96 \
  --tile_halo 32
```

## Runtime notes

- Full CONUS HRRR grid: `1059 x 1799`
- Diffusion sampling is expensive
- Tiled inference is required on typical single-GPU setups
- Single-member tiled `bf16` inference can still take minutes per forecast hour

## Intended use

Research use, port validation, inference experiments, quantization work, and downstream model distillation.

## License

MIT. See `LICENSE`.
