#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload HRRRCast Torch checkpoints to Hugging Face")
    parser.add_argument("--repo-id", required=True, help="Target HF repo id, e.g. user/hrrrcastv3-diffusion-torch")
    parser.add_argument(
        "--checkpoint",
        action="append",
        required=True,
        help="Checkpoint path to upload; may be passed multiple times",
    )
    parser.add_argument("--private", action="store_true", help="Create the repo as private")
    parser.add_argument(
        "--model-card",
        default="hf/README.md",
        help="Path to the Hugging Face model card markdown file",
    )
    parser.add_argument(
        "--license-file",
        default="LICENSE",
        help="Path to the license file to upload alongside the model card",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    api = HfApi()

    api.create_repo(repo_id=args.repo_id, repo_type="model", private=args.private, exist_ok=True)

    model_card = Path(args.model_card)
    if not model_card.exists():
        raise FileNotFoundError(f"Model card not found: {model_card}")
    api.upload_file(
        path_or_fileobj=str(model_card),
        path_in_repo="README.md",
        repo_id=args.repo_id,
        repo_type="model",
    )

    license_file = Path(args.license_file)
    if license_file.exists():
        api.upload_file(
            path_or_fileobj=str(license_file),
            path_in_repo="LICENSE",
            repo_id=args.repo_id,
            repo_type="model",
        )

    for checkpoint in args.checkpoint:
        checkpoint_path = Path(checkpoint)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        api.upload_file(
            path_or_fileobj=str(checkpoint_path),
            path_in_repo=checkpoint_path.name,
            repo_id=args.repo_id,
            repo_type="model",
        )
        print(f"Uploaded {checkpoint_path}")


if __name__ == "__main__":
    main()
