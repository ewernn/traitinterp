#!/usr/bin/env python3
"""Convert FP32 LoRA adapter to BF16 and upload to HuggingFace."""

import os
import shutil
import tempfile
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from huggingface_hub import HfApi, create_repo

# Paths
LORA_PATH = Path("/home/dev/.cache/huggingface/hub/models--auditing-agents--llama-3.3-70b-dpo-rt-lora/snapshots/4b8b0a5a4fb1efee4d43757824446a81528d51ef")
REPO_ID = "ewernn/llama-3.3-70b-dpo-rt-lora-bf16"


def main():
    print("Loading FP32 LoRA weights...")
    weights = load_file(LORA_PATH / "adapter_model.safetensors")

    # Check current dtype
    first_key = next(iter(weights))
    print(f"Original dtype: {weights[first_key].dtype}")
    print(f"Number of tensors: {len(weights)}")

    # Calculate sizes
    fp32_bytes = sum(t.numel() * t.element_size() for t in weights.values())
    print(f"FP32 size in memory: {fp32_bytes / 1e9:.2f} GB")

    print("\nConverting to BF16...")
    bf16_weights = {k: v.to(torch.bfloat16) for k, v in weights.items()}

    bf16_bytes = sum(t.numel() * t.element_size() for t in bf16_weights.values())
    print(f"BF16 size in memory: {bf16_bytes / 1e9:.2f} GB")

    # Create temp directory for upload
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Save BF16 weights
        print(f"\nSaving BF16 weights to {tmpdir}...")
        save_file(bf16_weights, tmpdir / "adapter_model.safetensors")

        # Copy other config files
        for f in ["adapter_config.json", "README.md", "special_tokens_map.json",
                  "tokenizer.json", "tokenizer_config.json", "chat_template.jinja"]:
            src = LORA_PATH / f
            if src.exists():
                shutil.copy(src, tmpdir / f)
                print(f"Copied {f}")

        # Check saved file size
        saved_size = (tmpdir / "adapter_model.safetensors").stat().st_size
        print(f"\nSaved file size: {saved_size / 1e9:.2f} GB")

        # Upload to HuggingFace
        print(f"\nUploading to {REPO_ID}...")
        api = HfApi()

        # Create repo if it doesn't exist
        try:
            create_repo(REPO_ID, exist_ok=True)
            print(f"Created/verified repo: {REPO_ID}")
        except Exception as e:
            print(f"Repo creation note: {e}")

        # Upload folder
        api.upload_folder(
            folder_path=str(tmpdir),
            repo_id=REPO_ID,
            commit_message="Upload BF16-converted LoRA adapter (from auditing-agents/llama-3.3-70b-dpo-rt-lora)"
        )

        print(f"\nDone! Uploaded to: https://huggingface.co/{REPO_ID}")


if __name__ == "__main__":
    main()
