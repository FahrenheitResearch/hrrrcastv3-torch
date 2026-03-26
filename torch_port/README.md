# Torch Port

This directory contains an in-progress PyTorch port of the diffusion HRRRCast checkpoint stored in `net-diffusion/model.keras`.

Current scope:

- Reads the Keras Functional graph from `config.json`
- Rebuilds the graph in Torch using NHWC-preserving wrapper modules
- Supports the custom layers used by the current checkpoint
- Loads the original Keras archive with TensorFlow and copies weights into the Torch model
- Exports a Torch checkpoint in `fp32`, `fp16`, or `bf16`
- Includes a Torch-native DDIM sampler wrapper for the converted diffusion network
- Includes a Torch autoregressive forecast runner that reads the existing preprocessed NPZ inputs

Current limitation:

- The Torch forecast runner is focused on the diffusion inference path and per-hour outputs. It has not been exhaustively validated against a full long-horizon production forecast yet.
- FP8-style runtime quantization is not implemented yet. The converted model is intended to be the foundation for that work once parity is verified.

Example:

```bash
PYTHONPATH=. ./.venv/bin/python -m torch_port.convert \
  net-diffusion/model.keras \
  converted/hrrrcast_diffusion_fp32.pt
```

Torch forecast example:

```bash
PYTHONPATH=. ./.venv/bin/python -m torch_port.forecast \
  converted/hrrrcast_diffusion_bf16.pt \
  2026-03-26T00 \
  6 \
  --members 0-2 \
  --base_dir /path/to/preprocessed \
  --output_dir /path/to/output \
  --device cuda \
  --dtype bf16
```
