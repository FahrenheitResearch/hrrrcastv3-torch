# HRRRCast V3 Torch Port

PyTorch port of the NOAA GSL HRRRCast V3 diffusion model, including:

- converted Torch checkpoints
- Torch inference runtime
- autoregressive forecast runner
- HRRR/GFS preprocessing pipeline
- plotting and output utilities

## Repositories

- GitHub: code and runtime
- Hugging Face: converted `.pt` checkpoints

The Torch checkpoints rebuild the network from:

`net-diffusion/model.config.json`

They do not require the original `model.keras` archive for inference.

## Current status

- Torch network conversion is working
- single-member tiled `bf16` GPU inference is working
- per-hour NetCDF and GRIB2 output is wired
- plotting is wired
- the diffusion path is still expensive at full CONUS `1059 x 1799` resolution

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

## Key paths

- Torch code: `torch_port/`
- Original pipeline: `src/`
- Extracted Keras graph config: `net-diffusion/model.config.json`
- Torch requirements: `requirements-torch.txt`
- Hugging Face upload helper: `scripts/upload_hf.py`

## Publish notes

The intended workflow is:

1. clone this GitHub repository
2. download a checkpoint from Hugging Face
3. run from the repo checkout

See `PUBLISHING.md` for the repo creation and upload commands.

## License

MIT. See `LICENSE`.
