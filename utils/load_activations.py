"""
Shared activation loading for extraction pipeline.

Input: Experiment/trait identifiers + layer number
Output: (pos_acts, neg_acts) tensors for a single layer

Usage:
    from utils.load_activations import load_train_activations, load_val_activations
    pos, neg = load_train_activations(experiment, trait, model_variant, layer=14)
"""

import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch

from utils.paths import (
    get_activation_dir,
    get_activation_path,
    get_activation_metadata_path,
    get_val_activation_path,
)


# Cache for stacked tensors (avoids reloading when iterating layers)
_stacked_cache: Dict[str, Tuple[torch.Tensor, dict]] = {}


def load_activation_metadata(
    experiment: str,
    trait: str,
    model_variant: str,
    component: str = "residual",
    position: str = "response[:5]",
) -> dict:
    """Load activation metadata without loading tensor data."""
    metadata_path = get_activation_metadata_path(experiment, trait, model_variant, component, position)
    with open(metadata_path) as f:
        return json.load(f)


def _detect_format(
    experiment: str,
    trait: str,
    model_variant: str,
    component: str,
    position: str,
) -> str:
    """Detect activation storage format: 'stacked' or 'per_layer'."""
    stacked_path = get_activation_path(experiment, trait, model_variant, component, position)
    if stacked_path.exists():
        return "stacked"

    # Check for per-layer files
    act_dir = get_activation_dir(experiment, trait, model_variant, component, position)
    if any(act_dir.glob("train_layer*.pt")):
        return "per_layer"

    raise FileNotFoundError(
        f"No activation files found at {act_dir}. Run extraction stage 3 first."
    )


def _load_layer_stacked(
    tensor_path: Path,
    metadata: dict,
    layer: int,
    split: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load a layer from a stacked [n_examples, n_layers, hidden_dim] tensor."""
    cache_key = str(tensor_path)
    if cache_key not in _stacked_cache:
        acts = torch.load(tensor_path, weights_only=True)
        _stacked_cache[cache_key] = (acts, metadata)

    acts, _ = _stacked_cache[cache_key]
    layer_acts = acts[:, layer, :]

    n_pos_key = "n_examples_pos" if split == "train" else "n_val_pos"
    n_pos = metadata[n_pos_key]
    return layer_acts[:n_pos], layer_acts[n_pos:]


def _load_layer_per_file(
    act_dir: Path,
    metadata: dict,
    layer: int,
    split: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load a layer from individual per-layer .pt file."""
    prefix = "train" if split == "train" else "val"
    layer_path = act_dir / f"{prefix}_layer{layer}.pt"
    if not layer_path.exists():
        raise FileNotFoundError(f"Per-layer activation file not found: {layer_path}")

    layer_acts = torch.load(layer_path, weights_only=True)
    n_pos_key = "n_examples_pos" if split == "train" else "n_val_pos"
    n_pos = metadata[n_pos_key]
    return layer_acts[:n_pos], layer_acts[n_pos:]


def load_activations(
    experiment: str,
    trait: str,
    model_variant: str,
    layer: int,
    component: str = "residual",
    position: str = "response[:5]",
    split: str = "train",
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Load pos/neg activations for a single layer.

    Auto-detects format (stacked tensor vs per-layer files).

    Args:
        split: "train" or "val"

    Returns:
        (pos_acts, neg_acts) each of shape [n_examples, hidden_dim].
        Returns (None, None) if validation data doesn't exist.
    """
    metadata = load_activation_metadata(experiment, trait, model_variant, component, position)

    if split == "val" and metadata.get("n_val_pos", 0) == 0:
        return None, None

    fmt = _detect_format(experiment, trait, model_variant, component, position)

    if fmt == "stacked":
        if split == "val":
            tensor_path = get_val_activation_path(experiment, trait, model_variant, component, position)
            if not tensor_path.exists():
                return None, None
        else:
            tensor_path = get_activation_path(experiment, trait, model_variant, component, position)
        return _load_layer_stacked(tensor_path, metadata, layer, split)
    else:
        act_dir = get_activation_dir(experiment, trait, model_variant, component, position)
        if split == "val":
            layer_path = act_dir / f"val_layer{layer}.pt"
            if not layer_path.exists():
                return None, None
        return _load_layer_per_file(act_dir, metadata, layer, split)


# Backwards-compatible aliases
def load_train_activations(experiment, trait, model_variant, layer, component="residual", position="response[:5]"):
    return load_activations(experiment, trait, model_variant, layer, component, position, split="train")

def load_val_activations(experiment, trait, model_variant, layer, component="residual", position="response[:5]"):
    return load_activations(experiment, trait, model_variant, layer, component, position, split="val")


def available_layers(
    experiment: str,
    trait: str,
    model_variant: str,
    component: str = "residual",
    position: str = "response[:5]",
) -> list[int]:
    """Return list of layers that have activation data.

    For stacked format: all layers 0..n_layers-1.
    For per-layer format: only layers with train_layer{N}.pt files.
    """
    fmt = _detect_format(experiment, trait, model_variant, component, position)
    metadata = load_activation_metadata(experiment, trait, model_variant, component, position)

    if fmt == "stacked":
        return list(range(metadata["n_layers"]))
    else:
        act_dir = get_activation_dir(experiment, trait, model_variant, component, position)
        layers = []
        for f in act_dir.glob("train_layer*.pt"):
            try:
                layer_num = int(f.stem.replace("train_layer", ""))
                layers.append(layer_num)
            except ValueError:
                continue
        return sorted(layers)


def clear_cache():
    """Clear the stacked tensor cache."""
    _stacked_cache.clear()
