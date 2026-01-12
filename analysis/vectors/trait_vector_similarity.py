"""
Compute similarity between trait vectors.

Since we don't have shared activations to project, compute direct cosine
similarity between trait vectors at the same layer. Shows which traits
point in similar directions in activation space.

Input: Extracted vectors for multiple traits
Output: Trait similarity matrix

Usage:
    python analysis/vectors/trait_vector_similarity.py --experiment gemma-2-2b
"""

import argparse
import json
from pathlib import Path

import torch
import numpy as np

from utils.paths import (
    list_methods,
    list_layers,
    discover_extracted_traits,
    get_model_variant,
)
from utils.vectors import load_vector
from core.math import cosine_similarity


def main():
    parser = argparse.ArgumentParser(description="Trait vector similarity analysis")
    parser.add_argument("--experiment", required=True, help="Experiment name")
    parser.add_argument("--model-variant", default=None, help="Model variant (default: from experiment config)")
    parser.add_argument("--method", default="probe", help="Method to compare")
    parser.add_argument("--layer", type=int, help="Specific layer (default: best layer per trait)")
    parser.add_argument("--component", default="residual", help="Component type")
    parser.add_argument("--position", default="response[:]", help="Position string")
    parser.add_argument("--output", help="Output JSON path (optional)")
    args = parser.parse_args()

    # Resolve model variant
    variant = get_model_variant(args.experiment, args.model_variant, mode="extraction")
    model_variant = variant['name']

    # Discover all traits
    all_traits = [f"{cat}/{name}" for cat, name in discover_extracted_traits(args.experiment)]
    print(f"Found {len(all_traits)} traits with vectors")

    # Filter to traits that have the specified method/position
    valid_traits = []
    trait_vectors = {}
    trait_layers = {}

    for trait in all_traits:
        methods = list_methods(args.experiment, trait, model_variant, args.position, args.component)
        if args.method not in methods:
            continue

        layers = list_layers(args.experiment, trait, model_variant, args.position, args.component, args.method)
        if not layers:
            continue

        # Use specified layer or middle layer
        if args.layer and args.layer in layers:
            layer = args.layer
        else:
            layer = layers[len(layers) // 2]  # Middle layer

        v = load_vector(args.experiment, trait, layer, model_variant, args.method, args.component, args.position)
        v = v.float()

        valid_traits.append(trait)
        trait_vectors[trait] = v
        trait_layers[trait] = layer

    print(f"Valid traits with {args.method} at {args.position}: {len(valid_traits)}")
    for t in valid_traits:
        print(f"  {t} (L{trait_layers[t]})")

    if len(valid_traits) < 2:
        print("Need at least 2 traits for comparison")
        return

    # Compute similarity matrix
    n = len(valid_traits)
    matrix = np.zeros((n, n))
    for i, t1 in enumerate(valid_traits):
        for j, t2 in enumerate(valid_traits):
            matrix[i, j] = cosine_similarity(trait_vectors[t1], trait_vectors[t2]).item()

    # Print matrix
    print(f"\n=== Trait Vector Similarity (method={args.method}) ===")
    # Shorten trait names for display
    short_names = [t.split("/")[-1][:10] for t in valid_traits]

    header = "           " + " ".join(f"{n:>10}" for n in short_names)
    print(header)
    for i, t in enumerate(valid_traits):
        row = f"{short_names[i]:>10} " + " ".join(f"{matrix[i, j]:>10.3f}" for j in range(n))
        print(row)

    # Find most/least similar pairs
    print("\n=== Top Similar Pairs ===")
    pairs = []
    for i in range(n):
        for j in range(i+1, n):
            pairs.append((valid_traits[i], valid_traits[j], matrix[i, j]))

    pairs.sort(key=lambda x: x[2], reverse=True)
    for t1, t2, sim in pairs[:5]:
        print(f"  {sim:.3f}: {t1} <-> {t2}")

    print("\n=== Least Similar Pairs ===")
    for t1, t2, sim in pairs[-5:]:
        print(f"  {sim:.3f}: {t1} <-> {t2}")

    # Output
    results = {
        "experiment": args.experiment,
        "method": args.method,
        "component": args.component,
        "position": args.position,
        "traits": valid_traits,
        "layers_used": trait_layers,
        "similarity_matrix": matrix.tolist(),
        "top_pairs": [(t1, t2, s) for t1, t2, s in pairs[:10]],
    }

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_path}")

    return results


if __name__ == "__main__":
    main()
