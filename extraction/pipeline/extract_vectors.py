"""
Core logic for Stage 3: Extract Trait Vectors from activations.
"""

import sys
import json
import torch
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from utils.paths import get as get_path
from traitlens import get_method

def extract_vectors_for_trait(
    experiment: str,
    trait: str,
    methods: List[str],
    layers: Optional[List[int]] = None,
):
    """
    Extracts trait vectors from a single `all_layers.pt` activation file.

    Args:
        experiment: Experiment name.
        trait: Trait name (e.g., "defensiveness").
        methods: List of method names to use for extraction.
        layers: Optional list of specific layer numbers to extract from. 
                If None, all layers are used.
    """
    print(f"  [Stage 3] Extracting vectors for '{trait}'...")
    activations_dir = get_path('extraction.activations', experiment=experiment, trait=trait)
    vectors_dir = get_path('extraction.vectors', experiment=experiment, trait=trait)
    vectors_dir.mkdir(parents=True, exist_ok=True)

    # Load metadata
    try:
        with open(activations_dir / "metadata.json") as f:
            metadata = json.load(f)
    except FileNotFoundError:
        print(f"    ERROR: Metadata for trait '{trait}' not found in {activations_dir}. Run Stage 2 first.")
        return

    # Load the single, combined activations file
    all_acts_path = activations_dir / "all_layers.pt"
    if not all_acts_path.exists():
        print(f"    ERROR: Combined activation file 'all_layers.pt' not found for trait '{trait}'. Run Stage 2 first.")
        return
        
    all_acts = torch.load(all_acts_path) # Shape: [n_examples, n_layers, hidden_dim]

    n_layers = metadata.get("n_layers", all_acts.shape[1])
    n_pos = metadata.get("n_examples_pos")
    n_neg = metadata.get("n_examples_neg")

    if n_pos is None or n_neg is None:
        print(f"    ERROR: n_examples_pos or n_examples_neg missing from metadata for trait '{trait}'.")
        return

    # Determine which layers to extract
    layer_list = layers if layers is not None else list(range(n_layers))
    method_objs = {name: get_method(name) for name in methods}

    print(f"    Methods: {methods}, Layers: {layer_list}")

    for layer_idx in layer_list:
        if layer_idx >= n_layers:
            print(f"    Skipping layer {layer_idx} (out of bounds for model with {n_layers} layers).")
            continue

        # Slice the activations for the current layer
        layer_acts = all_acts[:, layer_idx, :]
        pos_acts = layer_acts[:n_pos]
        neg_acts = layer_acts[n_pos:]

        for method_name, method_obj in method_objs.items():
            print(f"    Layer {layer_idx}, Method: {method_name}...", end=" ", flush=True)
            try:
                result = method_obj.extract(pos_acts, neg_acts)
                vector = result['vector']

                torch.save(vector, vectors_dir / f"{method_name}_layer{layer_idx}.pt")

                vector_metadata = {
                    "trait": trait,
                    "method": method_name,
                    "layer": layer_idx,
                    "model": metadata.get("model", "unknown"),
                    "vector_norm": float(vector.norm().item()),
                }
                for key, value in result.items():
                    if key != 'vector' and not key.endswith('_proj'):
                        if isinstance(value, torch.Tensor) and value.numel() == 1:
                            vector_metadata[key] = float(value.item())
                        elif isinstance(value, (int, float, str, bool)):
                            vector_metadata[key] = value

                with open(vectors_dir / f"{method_name}_layer{layer_idx}_metadata.json", 'w') as f:
                    json.dump(vector_metadata, f, indent=2)
                
                print(f"norm={vector.norm().item():.2f}")

            except Exception as e:
                print(f"FAILED with error: {e}")

    print(f"    Finished vector extraction for '{trait}'.")

