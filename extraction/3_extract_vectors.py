#!/usr/bin/env python3
"""
Stage 3: Extract Trait Vectors

Applies extraction methods to saved activations to produce trait vectors.

Usage:
    # Default: all methods, all layers
    python pipeline/3_extract_vectors.py --experiment my_exp --trait my_trait

    # Specific methods and layers
    python pipeline/3_extract_vectors.py --experiment my_exp --trait my_trait \
        --methods mean_diff,probe --layers 16

    # Multiple traits
    python pipeline/3_extract_vectors.py --experiment my_exp \
        --traits refusal,uncertainty,verbosity
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import json
from typing import List, Optional
import fire

from traitlens import (
    MeanDifferenceMethod,
    ICAMethod,
    ProbeMethod,
    GradientMethod,
    get_method
)


def extract_vectors(
    experiment: str,
    trait: Optional[str] = None,
    traits: Optional[str] = None,
    methods: str = "mean_diff,probe,ica,gradient",
    layers: Optional[str] = None,  # If None, extract from all layers
):
    """
    Extract trait vectors using specified methods and layers.

    Args:
        experiment: Experiment name (e.g., "gemma_2b_it_nov12")
        trait: Single trait to extract (mutually exclusive with traits)
        traits: Comma-separated trait names (mutually exclusive with trait)
        methods: Comma-separated method names (default: all 4 methods)
        layers: Comma-separated layer numbers (default: all layers in activations)
    """
    # Parse trait list
    if trait and traits:
        raise ValueError("Specify either --trait or --traits, not both")

    if trait:
        trait_list = [trait]
    elif isinstance(traits, tuple):
        # Fire might parse comma-separated as tuple
        trait_list = list(traits)
    else:
        trait_list = traits.split(",")

    # Parse methods
    method_names = [m.strip() for m in methods.split(",")]
    method_objs = {name: get_method(name) for name in method_names}

    exp_dir = Path(f"experiments/{experiment}")
    if not exp_dir.exists():
        raise ValueError(f"Experiment not found: {exp_dir}")

    print(f"Experiment: {experiment}")
    print(f"Traits: {trait_list}")
    print(f"Methods: {method_names}")
    print()

    # Process each trait
    for trait_name in trait_list:
        trait_dir = exp_dir / trait_name
        if not trait_dir.exists():
            print(f"⚠️  Trait not found: {trait_name}, skipping")
            continue

        print(f"{'='*60}")
        print(f"Extracting: {trait_name}")
        print(f"{'='*60}")

        # Load activations
        acts_path = trait_dir / "extraction" / "activations" / "all_layers.pt"
        if not acts_path.exists():
            print(f"❌ Activations not found: {acts_path}")
            print(f"   Run: python pipeline/2_extract_activations.py --experiment {experiment} --trait {trait_name}")
            continue

        print(f"Loading activations from: {acts_path}")
        all_acts = torch.load(acts_path)  # [n_examples, n_layers, hidden_dim]

        # Load metadata to get pos/neg split
        metadata_path = trait_dir / "extraction" / "activations" / "metadata.json"
        with open(metadata_path) as f:
            metadata = json.load(f)

        n_pos = metadata["n_examples_pos"]
        n_neg = metadata["n_examples_neg"]

        pos_acts = all_acts[:n_pos]  # [n_pos, n_layers, hidden_dim]
        neg_acts = all_acts[n_pos:n_pos+n_neg]  # [n_neg, n_layers, hidden_dim]

        print(f"  Pos examples: {n_pos}")
        print(f"  Neg examples: {n_neg}")
        print(f"  Total layers: {all_acts.shape[1]}")
        print(f"  Hidden dim: {all_acts.shape[2]}")

        # Determine which layers to extract
        if layers:
            # Handle Fire tuple parsing
            if isinstance(layers, tuple):
                layer_list = [int(l) for l in layers]
            else:
                layer_list = [int(l.strip()) for l in layers.split(",")]
        else:
            layer_list = list(range(all_acts.shape[1]))  # All layers

        print(f"  Extracting from layers: {layer_list}")
        print()

        # Extract vectors
        vectors_dir = trait_dir / "extraction" / "vectors"
        vectors_dir.mkdir(parents=True, exist_ok=True)

        for layer_idx in layer_list:
            pos_layer = pos_acts[:, layer_idx, :]  # [n_pos, hidden_dim]
            neg_layer = neg_acts[:, layer_idx, :]  # [n_neg, hidden_dim]

            for method_name, method_obj in method_objs.items():
                print(f"  Layer {layer_idx}, Method: {method_name}...", end=" ")

                try:
                    result = method_obj.extract(pos_layer, neg_layer)
                    vector = result['vector']

                    # Save vector
                    vector_filename = f"{method_name}_layer{layer_idx}.pt"
                    vector_path = vectors_dir / vector_filename
                    torch.save(vector, vector_path)

                    # Save metadata
                    vector_metadata = {
                        "trait": trait_name,
                        "method": method_name,
                        "layer": layer_idx,
                        "model": metadata.get("model", "unknown"),
                        "vector_norm": vector.norm().item(),
                        "extraction_date": metadata.get("extraction_date", "unknown"),
                    }

                    # Add method-specific metadata
                    for key, value in result.items():
                        if key != 'vector' and not key.endswith('_proj'):
                            if isinstance(value, torch.Tensor):
                                if value.numel() == 1:
                                    vector_metadata[key] = value.item()
                            elif isinstance(value, (int, float, str, bool)):
                                vector_metadata[key] = value

                    # Compute separation for probe method (like we do for gradient)
                    if method_name == 'probe' and 'pos_scores' in result and 'neg_scores' in result:
                        pos_mean = result['pos_scores'].mean().item()
                        neg_mean = result['neg_scores'].mean().item()
                        vector_metadata['final_separation'] = pos_mean - neg_mean

                    metadata_path = vectors_dir / f"{method_name}_layer{layer_idx}_metadata.json"
                    with open(metadata_path, 'w') as f:
                        json.dump(vector_metadata, f, indent=2)

                    print(f"✓ norm={vector.norm().item():.2f}")

                except Exception as e:
                    print(f"✗ {e}")

        print(f"\n✅ {trait_name} complete")
        print(f"   Vectors saved to: {vectors_dir}")
        print()


if __name__ == "__main__":
    fire.Fire(extract_vectors)
