# Publishing

This repo is publishable as:

- GitHub repository: code, preprocessing, Torch port, docs
- Hugging Face model repository: converted Torch checkpoints

## Recommended split

- Keep `converted/*.pt` on Hugging Face, not GitHub.
- Keep `torch_out/`, `.venv/`, and interpolation weights out of version control.
- Keep `net-diffusion/model.config.json` in GitHub. The Torch loader rebuilds the graph from that extracted Keras config.

## GitHub

After authenticating with `gh` or configuring a regular Git remote:

```bash
git add .
git commit -m "Add Torch HRRRCast diffusion port and publishing assets"
gh repo create YOURUSER/hrrrcast-live-torch --public --source=. --push
```

If you do not use `gh`:

```bash
git remote add origin git@github.com:YOURUSER/hrrrcast-live-torch.git
git branch -M main
git push -u origin main
```

## Hugging Face

Install the upload dependency if needed:

```bash
./.venv/bin/pip install huggingface_hub
```

Log in:

```bash
huggingface-cli login
```

Upload one or more converted checkpoints:

```bash
./.venv/bin/python scripts/upload_hf.py \
  --repo-id YOURUSER/hrrrcastv3-diffusion-torch \
  --checkpoint converted/hrrrcast_diffusion_bf16.pt \
  --checkpoint converted/hrrrcast_diffusion_fp16.pt \
  --checkpoint converted/hrrrcast_diffusion_fp32.pt
```

The script uploads:

- the selected checkpoint files
- the model card at `hf/README.md`
- the repo `LICENSE`

## Important current behavior

The converted `.pt` checkpoints contain a relative pointer to:

`net-diffusion/model.config.json`

That means the intended usage is:

1. clone the GitHub repo
2. download the checkpoint from Hugging Face
3. run inference from the repo checkout

## Suggested Hugging Face repo name

`hrrrcastv3-diffusion-torch`
