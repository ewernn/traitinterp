"""
Extract trait vectors from activations.

Called by run_pipeline.py (stage 4).
"""

import json
from datetime import datetime
from typing import List, Optional

import torch

from utils.paths import (
    get_vector_dir,
    get_vector_path,
    get_vector_metadata_path,
)
from utils.activations import load_train_activations, load_activation_metadata, available_layers
from utils.model import is_rank_zero
from core import get_method


def extract_vectors_for_trait(
    experiment: str,
    trait: str,
    model_variant: str,
    methods: List[str],
    layers: Optional[List[int]] = None,
    component: str = 'residual',
    position: str = 'response[:5]',
) -> int:
    """
    Extract trait vectors from activations. Returns number of vectors extracted.
    """
    # Load metadata to determine available layers
    try:
        metadata = load_activation_metadata(experiment, trait, model_variant, component, position)
    except FileNotFoundError:
        print(f"      ERROR: Activation metadata not found. Run stage 3 first.")
        return 0

    n_layers = metadata.get("n_layers", 0)

    # Determine which layers to process
    if layers is not None:
        layer_list = [l for l in layers if l < n_layers]
    else:
        layer_list = available_layers(experiment, trait, model_variant, component, position)

    if not layer_list:
        print(f"      ERROR: No layers available for extraction.")
        return 0

    method_objs = {name: get_method(name) for name in methods}

    print(f"      Methods: {methods}")
    print(f"      Layers: {layer_list[0]}..{layer_list[-1]} ({len(layer_list)} total)")

    n_extracted = 0

    # Track per-method metadata (layers dict)
    method_metadata = {method: {"layers": {}} for method in methods}

    for layer_idx in layer_list:
        pos_acts, neg_acts = load_train_activations(
            experiment, trait, model_variant, layer_idx, component, position
        )

        if pos_acts.numel() == 0 or neg_acts.numel() == 0:
            continue

        # Compute means in float32 to avoid bfloat16 precision loss at large magnitudes
        mean_pos = pos_acts.float().mean(dim=0)
        mean_neg = neg_acts.float().mean(dim=0)
        center = (mean_pos + mean_neg) / 2

        for method_name, method_obj in method_objs.items():
            try:
                result = method_obj.extract(pos_acts, neg_acts)
                vector = result['vector']
                vector_norm = vector.norm().item()
                # Cast to same dtype for dot product
                baseline = (center.float() @ vector.float() / vector_norm).item() if vector_norm > 0 else 0.0

                # Save vector to new path structure (rank 0 only under TP)
                vector_path = get_vector_path(experiment, trait, method_name, layer_idx, model_variant, component, position)
                if is_rank_zero():
                    vector_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(vector, vector_path)

                # Collect layer info for consolidated metadata
                layer_info = {
                    "norm": float(vector_norm),
                    "baseline": baseline,
                }
                if 'bias' in result:
                    b = result['bias']
                    layer_info['bias'] = float(b.item()) if isinstance(b, torch.Tensor) else b
                if 'train_acc' in result:
                    layer_info['train_acc'] = float(result['train_acc'])

                method_metadata[method_name]["layers"][str(layer_idx)] = layer_info
                n_extracted += 1

            except Exception as e:
                print(f"      ERROR: {method_name} layer {layer_idx}: {e}")

    # Save consolidated metadata per method directory (rank 0 only under TP)
    if is_rank_zero():
        for method_name in methods:
            if not method_metadata[method_name]["layers"]:
                continue

            meta = {
                'model': metadata.get('model', 'unknown'),
                'trait': trait,
                'method': method_name,
                'component': component,
                'position': position,
                'layers': method_metadata[method_name]["layers"],
                'timestamp': datetime.now().isoformat(),
            }

            metadata_path = get_vector_metadata_path(experiment, trait, method_name, model_variant, component, position)
            with open(metadata_path, 'w') as f:
                json.dump(meta, f, indent=2)

    print(f"      Extracted {n_extracted} vectors")
    return n_extracted
