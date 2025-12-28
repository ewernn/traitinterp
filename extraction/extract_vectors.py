"""
Extract trait vectors from activations.

Called by run_pipeline.py (stage 4).
"""

import json
from datetime import datetime
from typing import List, Optional

import torch

from utils.paths import (
    get_activation_path,
    get_activation_metadata_path,
    get_vector_dir,
    get_vector_path,
    get_vector_metadata_path,
)
from core import get_method


def extract_vectors_for_trait(
    experiment: str,
    trait: str,
    methods: List[str],
    layers: Optional[List[int]] = None,
    component: str = 'residual',
    position: str = 'response[:]',
) -> int:
    """
    Extract trait vectors from activations. Returns number of vectors extracted.
    """
    print(f"  [4] Extracting vectors for '{trait}' (position: {position}, component: {component})...")

    # Load activations using centralized paths
    activation_path = get_activation_path(experiment, trait, component, position)
    metadata_path = get_activation_metadata_path(experiment, trait, component, position)

    try:
        with open(metadata_path) as f:
            activation_metadata = json.load(f)
    except FileNotFoundError:
        print(f"    ERROR: {metadata_path} not found. Run stage 3 first.")
        return 0

    if not activation_path.exists():
        print(f"    ERROR: {activation_path} not found. Run stage 3 first.")
        return 0

    all_acts = torch.load(activation_path, weights_only=True)
    n_layers = activation_metadata.get("n_layers", all_acts.shape[1])
    n_pos = activation_metadata.get("n_examples_pos")
    n_neg = activation_metadata.get("n_examples_neg")

    if n_pos is None or n_neg is None:
        print(f"    ERROR: n_examples_pos/neg missing from metadata.")
        return 0

    layer_list = layers if layers is not None else list(range(n_layers))
    method_objs = {name: get_method(name) for name in methods}

    print(f"    Methods: {methods}")
    print(f"    Layers: {layer_list[0]}..{layer_list[-1]} ({len(layer_list)} total)")

    n_extracted = 0

    # Track per-method metadata (layers dict)
    method_metadata = {method: {"layers": {}} for method in methods}

    for layer_idx in layer_list:
        if layer_idx >= n_layers:
            continue

        layer_acts = all_acts[:, layer_idx, :]
        pos_acts = layer_acts[:n_pos]
        neg_acts = layer_acts[n_pos:]

        mean_pos = pos_acts.mean(dim=0)
        mean_neg = neg_acts.mean(dim=0)
        center = (mean_pos + mean_neg) / 2

        for method_name, method_obj in method_objs.items():
            try:
                result = method_obj.extract(pos_acts, neg_acts)
                vector = result['vector']
                vector_norm = vector.norm().item()
                # Cast to same dtype for dot product
                baseline = (center.float() @ vector.float() / vector_norm).item() if vector_norm > 0 else 0.0

                # Save vector to new path structure
                vector_path = get_vector_path(experiment, trait, method_name, layer_idx, component, position)
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
                print(f"    ERROR: {method_name} layer {layer_idx}: {e}")

    # Save consolidated metadata per method directory
    for method_name in methods:
        if not method_metadata[method_name]["layers"]:
            continue

        metadata = {
            'model': activation_metadata.get('model', 'unknown'),
            'trait': trait,
            'method': method_name,
            'component': component,
            'position': position,
            'layers': method_metadata[method_name]["layers"],
            'timestamp': datetime.now().isoformat(),
        }

        metadata_path = get_vector_metadata_path(experiment, trait, method_name, component, position)
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

    print(f"    Extracted {n_extracted} vectors")
    return n_extracted
