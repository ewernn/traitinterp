#!/usr/bin/env python3
"""
Stage 3: Extract trait vectors from activations.

Input:
    - experiments/{experiment}/extraction/{category}/{trait}/activations/all_layers.pt
    - experiments/{experiment}/extraction/{category}/{trait}/activations/metadata.json

Output:
    - experiments/{experiment}/extraction/{category}/{trait}/vectors/{method}_layer{N}.pt
    - experiments/{experiment}/extraction/{category}/{trait}/vectors/{method}_layer{N}_metadata.json

Usage:
    # Single trait, default methods
    python extraction/extract_vectors.py --experiment my_exp --trait category/my_trait

    # All traits, specific methods
    python extraction/extract_vectors.py --experiment my_exp --trait all --methods gradient,random_baseline

    # Single trait, specific layers
    python extraction/extract_vectors.py --experiment my_exp --trait category/my_trait --layers 16
"""

import sys
import json
import argparse
from pathlib import Path
from typing import List, Optional

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.paths import get as get_path
from traitlens import get_method


def discover_traits_with_activations(experiment: str) -> list[str]:
    """Find all traits that have activation files (Stage 2 complete)."""
    extraction_dir = get_path('extraction.base', experiment=experiment)
    traits = []
    if not extraction_dir.is_dir():
        return []

    for category_dir in extraction_dir.iterdir():
        if not category_dir.is_dir() or category_dir.name.startswith('.'):
            continue
        for trait_dir in category_dir.iterdir():
            if not trait_dir.is_dir():
                continue
            activations_dir = trait_dir / 'activations'
            if (activations_dir / 'metadata.json').exists() and (activations_dir / 'all_layers.pt').exists():
                traits.append(f"{category_dir.name}/{trait_dir.name}")
    return sorted(traits)


def extract_vectors_for_trait(
    experiment: str,
    trait: str,
    methods: List[str],
    layers: Optional[List[int]] = None,
    component: str = 'residual',
) -> int:
    """
    Extract trait vectors from activations.

    Args:
        experiment: Experiment name.
        trait: Trait name (e.g., "category/trait_name").
        methods: List of method names to use.
        layers: Optional list of specific layers. If None, all layers.

    Returns:
        Number of vectors extracted.
    """
    component_str = f" (component: {component})" if component != 'residual' else ""
    print(f"  [Stage 3] Extracting vectors for '{trait}'{component_str}...")

    activations_dir = get_path('extraction.activations', experiment=experiment, trait=trait)
    vectors_dir = get_path('extraction.vectors', experiment=experiment, trait=trait)
    vectors_dir.mkdir(parents=True, exist_ok=True)

    # Determine file names based on component
    metadata_filename = "metadata.json" if component == 'residual' else f"{component}_metadata.json"
    acts_filename = "all_layers.pt" if component == 'residual' else f"{component}_all_layers.pt"
    vector_prefix = "" if component == 'residual' else f"{component}_"

    # Load metadata
    try:
        with open(activations_dir / metadata_filename) as f:
            metadata = json.load(f)
    except FileNotFoundError:
        print(f"    ERROR: {metadata_filename} not found in {activations_dir}. Run Stage 2 first.")
        return 0

    # Load activations
    all_acts_path = activations_dir / acts_filename
    if not all_acts_path.exists():
        print(f"    ERROR: {acts_filename} not found. Run Stage 2 first.")
        return 0

    all_acts = torch.load(all_acts_path, weights_only=True)  # [n_examples, n_layers, hidden_dim]

    n_layers = metadata.get("n_layers", all_acts.shape[1])
    n_pos = metadata.get("n_examples_pos")
    n_neg = metadata.get("n_examples_neg")

    if n_pos is None or n_neg is None:
        print(f"    ERROR: n_examples_pos/neg missing from metadata.")
        return 0

    # Determine layers
    layer_list = layers if layers is not None else list(range(n_layers))
    method_objs = {name: get_method(name) for name in methods}

    print(f"    Methods: {methods}")
    print(f"    Layers: {layer_list[0]}..{layer_list[-1]} ({len(layer_list)} total)")

    n_extracted = 0
    for layer_idx in layer_list:
        if layer_idx >= n_layers:
            print(f"    Skipping layer {layer_idx} (out of bounds)")
            continue

        # Slice activations for this layer
        layer_acts = all_acts[:, layer_idx, :]
        pos_acts = layer_acts[:n_pos]
        neg_acts = layer_acts[n_pos:]

        for method_name, method_obj in method_objs.items():
            try:
                result = method_obj.extract(pos_acts, neg_acts)
                vector = result['vector']

                torch.save(vector, vectors_dir / f"{vector_prefix}{method_name}_layer{layer_idx}.pt")

                # Save metadata (identification + intrinsic properties only)
                # Quality metrics come from extraction_evaluation.py on validation data
                vector_metadata = {
                    "trait": trait,
                    "method": method_name,
                    "layer": layer_idx,
                    "component": component,
                    "model": metadata.get("model", "unknown"),
                    "vector_norm": float(vector.norm().item()),
                }
                if 'bias' in result:
                    b = result['bias']
                    vector_metadata['bias'] = float(b.item()) if isinstance(b, torch.Tensor) else b

                with open(vectors_dir / f"{vector_prefix}{method_name}_layer{layer_idx}_metadata.json", 'w') as f:
                    json.dump(vector_metadata, f, indent=2)

                n_extracted += 1

            except Exception as e:
                print(f"    ERROR: {method_name} layer {layer_idx}: {e}")

    print(f"    Extracted {n_extracted} vectors for '{trait}'")
    return n_extracted


def main():
    parser = argparse.ArgumentParser(description='Extract trait vectors from activations.')
    parser.add_argument('--experiment', type=str, required=True, help='Experiment name')
    parser.add_argument('--trait', type=str, required=True,
                        help='Trait name (e.g., "category/my_trait") or "all" for all traits')
    parser.add_argument('--methods', type=str, default='mean_diff,probe,gradient,random_baseline',
                        help='Comma-separated extraction methods')
    parser.add_argument('--layers', type=str, default=None,
                        help='Comma-separated layers (default: all)')
    parser.add_argument('--component', type=str, default='residual',
                        help='Component(s) to extract from: residual, attn_out, mlp_out, or comma-separated')

    args = parser.parse_args()

    # Determine traits to process
    if args.trait.lower() == 'all':
        traits = discover_traits_with_activations(args.experiment)
        if not traits:
            print(f"No traits with activations found in experiment '{args.experiment}'")
            return
        print(f"Found {len(traits)} traits to process")
    else:
        traits = [args.trait]

    # Parse methods, layers, and components
    methods_list = [m.strip() for m in args.methods.split(",")]
    layers_list = None
    if args.layers:
        layers_list = [int(l.strip()) for l in args.layers.split(",")]

    components = [c.strip() for c in args.component.split(',')]
    valid_components = {'residual', 'attn_out', 'mlp_out'}
    for c in components:
        if c not in valid_components:
            print(f"ERROR: Invalid component '{c}'. Must be one of: {valid_components}")
            return

    print("=" * 80)
    print("EXTRACT VECTORS")
    print(f"Experiment: {args.experiment}")
    print(f"Traits: {len(traits)}")
    print(f"Methods: {methods_list}")
    print(f"Layers: {layers_list if layers_list else 'all'}")
    if components != ['residual']:
        print(f"Components: {', '.join(components)}")
    print("=" * 80)

    # Process each trait and component
    total_vectors = 0
    for trait in traits:
        for component in components:
            n = extract_vectors_for_trait(
                experiment=args.experiment,
                trait=trait,
                methods=methods_list,
                layers=layers_list,
                component=component,
            )
            total_vectors += n

    print(f"\nDONE: Extracted {total_vectors} vectors for {len(traits)} traits, {len(components)} components.")


if __name__ == "__main__":
    main()
