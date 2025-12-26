"""
Extract trait vectors from activations.

Called by run_pipeline.py (stage 4).
"""

import json
from datetime import datetime
from typing import List, Optional

import torch

from utils.paths import get as get_path
from core import get_method


def extract_vectors_for_trait(
    experiment: str,
    trait: str,
    methods: List[str],
    layers: Optional[List[int]] = None,
    component: str = 'residual',
) -> int:
    """
    Extract trait vectors from activations. Returns number of vectors extracted.
    """
    component_str = f" (component: {component})" if component != 'residual' else ""
    print(f"  [4] Extracting vectors for '{trait}'{component_str}...")

    activations_dir = get_path('extraction.activations', experiment=experiment, trait=trait)
    vectors_dir = get_path('extraction.vectors', experiment=experiment, trait=trait)
    vectors_dir.mkdir(parents=True, exist_ok=True)

    metadata_filename = "metadata.json" if component == 'residual' else f"{component}_metadata.json"
    acts_filename = "all_layers.pt" if component == 'residual' else f"{component}_all_layers.pt"
    vector_prefix = "" if component == 'residual' else f"{component}_"

    try:
        with open(activations_dir / metadata_filename) as f:
            metadata = json.load(f)
    except FileNotFoundError:
        print(f"    ERROR: {metadata_filename} not found. Run stage 3 first.")
        return 0

    all_acts_path = activations_dir / acts_filename
    if not all_acts_path.exists():
        print(f"    ERROR: {acts_filename} not found. Run stage 3 first.")
        return 0

    all_acts = torch.load(all_acts_path, weights_only=True)
    n_layers = metadata.get("n_layers", all_acts.shape[1])
    n_pos = metadata.get("n_examples_pos")
    n_neg = metadata.get("n_examples_neg")

    if n_pos is None or n_neg is None:
        print(f"    ERROR: n_examples_pos/neg missing from metadata.")
        return 0

    layer_list = layers if layers is not None else list(range(n_layers))
    method_objs = {name: get_method(name) for name in methods}

    print(f"    Methods: {methods}")
    print(f"    Layers: {layer_list[0]}..{layer_list[-1]} ({len(layer_list)} total)")

    n_extracted = 0
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
                baseline = (center @ vector / vector_norm).item() if vector_norm > 0 else 0.0

                torch.save(vector, vectors_dir / f"{vector_prefix}{method_name}_layer{layer_idx}.pt")

                vector_metadata = {
                    "trait": trait,
                    "method": method_name,
                    "layer": layer_idx,
                    "component": component,
                    "model": metadata.get("model", "unknown"),
                    "vector_norm": float(vector_norm),
                    "baseline": baseline,
                }
                if 'bias' in result:
                    b = result['bias']
                    vector_metadata['bias'] = float(b.item()) if isinstance(b, torch.Tensor) else b

                with open(vectors_dir / f"{vector_prefix}{method_name}_layer{layer_idx}_metadata.json", 'w') as f:
                    json.dump(vector_metadata, f, indent=2)

                n_extracted += 1
            except Exception as e:
                print(f"    ERROR: {method_name} layer {layer_idx}: {e}")

    vectors_metadata = {
        'extraction_model': metadata.get('model', 'unknown'),
        'trait': trait,
        'component': component,
        'methods': methods,
        'layers': layer_list,
        'timestamp': datetime.now().isoformat(),
    }
    vectors_metadata_filename = "metadata.json" if component == 'residual' else f"{component}_metadata.json"
    with open(vectors_dir / vectors_metadata_filename, 'w') as f:
        json.dump(vectors_metadata, f, indent=2)

    print(f"    Extracted {n_extracted} vectors")
    return n_extracted
