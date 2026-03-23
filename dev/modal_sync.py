"""
Shared utilities for Modal volume sync.

Input: Modal volume references, model names, remote/local paths
Output: Loaded models, synced files

Usage:
    from dev.modal_sync import load_model_cached, pull_dir_recursive

    model, tokenizer, load_time = load_model_cached("Qwen/Qwen2.5-14B", volume=model_volume)
    pull_dir_recursive(volume, "experiment/steering/trait/variant", local_dir)

"""

import os
from pathlib import Path


def load_model_cached(model_name: str, cache_dir: str = "/models", volume=None):
    """Load HF model from cache dir, download and cache if first run.

    Args:
        model_name: HuggingFace model name (e.g. "Qwen/Qwen2.5-14B")
        cache_dir: Root cache directory on volume
        volume: Optional Modal volume — calls volume.commit() after caching new download

    Returns:
        (model, tokenizer, load_time)
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import time

    model_cache_dir = Path(f"{cache_dir}/{model_name.replace('/', '--')}")
    start = time.time()

    token = os.environ.get("HF_TOKEN", "").strip()
    if token:
        from huggingface_hub import login
        login(token=token)

    if model_cache_dir.exists():
        print(f"Loading from cache: {model_cache_dir}")
        tokenizer = AutoTokenizer.from_pretrained(str(model_cache_dir))
        model = AutoModelForCausalLM.from_pretrained(
            str(model_cache_dir),
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
    else:
        print(f"Downloading from HuggingFace...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            token=token,
        )
        print(f"Saving to cache: {model_cache_dir}")
        model_cache_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(model_cache_dir))
        tokenizer.save_pretrained(str(model_cache_dir))
        if volume is not None:
            volume.commit()

    load_time = time.time() - start
    print(f"Model loaded in {load_time:.1f}s")
    return model, tokenizer, load_time


def pull_dir_recursive(volume, remote_path: str, local_dir: Path):
    """Download directory tree from Modal volume to local filesystem.

    Handles both relative and absolute entry.path formats from Modal's listdir API.
    """
    from modal.volume import FileEntryType

    try:
        entries = list(volume.listdir(remote_path, recursive=True))
    except FileNotFoundError:
        return

    for entry in entries:
        if entry.type != FileEntryType.FILE:
            continue
        # entry.path may be relative to remote_path or absolute from volume root
        if entry.path.startswith(remote_path):
            rel_path = entry.path[len(remote_path):].lstrip("/")
            full_remote = entry.path
        else:
            rel_path = entry.path
            full_remote = f"{remote_path}/{entry.path}"

        if not rel_path:
            continue

        local_path = local_dir / rel_path
        local_path.parent.mkdir(parents=True, exist_ok=True)
        data = b"".join(volume.read_file(full_remote))
        local_path.write_bytes(data)


def pull_file(volume, remote_path: str, local_path: Path):
    """Download single file from Modal volume."""
    try:
        data = b"".join(volume.read_file(remote_path))
        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.write_bytes(data)
        return True
    except FileNotFoundError:
        return False
