"""
Compute cross-layer similarity for trait vectors.

For each trait/method, compute cosine similarity between vectors at all layer pairs.
Shows where representation is stable (high diagonal blocks) vs transitional.

Input: Extracted vectors at experiments/{experiment}/extraction/{trait}/{model_variant}/vectors/
Output: Similarity matrices and summary statistics

Usage:
    python analysis/vectors/cross_layer_similarity.py --experiment gemma-2-2b --trait chirp/refusal_v2
    python analysis/vectors/cross_layer_similarity.py --experiment gemma-2-2b --all-traits
"""

import argparse
import json
from pathlib import Path

import torch
import numpy as np

from utils.paths import (
    get_vector_path,
    list_methods,
    list_layers,
    discover_extracted_traits,
    get as get_path,
    get_model_variant,
)
from core.math import cosine_similarity


def compute_cross_layer_matrix(
    experiment: str,
    trait: str,
    model_variant: str,
    method: str,
    component: str = "residual",
    position: str = "response[:]",
) -> tuple[np.ndarray, list[int]]:
    """
    Compute cross-layer cosine similarity matrix.

    Returns:
        (n_layers x n_layers) similarity matrix and list of layer indices
    """
    layers = list_layers(experiment, trait, model_variant, position, component, method)
    if not layers:
        return None, []

    vectors = []
    for layer in layers:
        path = get_vector_path(experiment, trait, method, layer, model_variant, component, position)
        v = torch.load(path, weights_only=True).float()
        vectors.append(v)

    n = len(layers)
    matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            matrix[i, j] = cosine_similarity(vectors[i], vectors[j]).item()

    return matrix, layers


def analyze_matrix(matrix: np.ndarray, layers: list[int]) -> dict:
    """Extract summary statistics from similarity matrix."""
    n = len(layers)

    # Adjacent layer similarity (how much does representation change per layer?)
    adjacent_sims = [matrix[i, i+1] for i in range(n-1)]

    # Find stable regions (blocks of high similarity)
    # Look for drops in adjacent similarity
    drops = []
    for i in range(len(adjacent_sims) - 1):
        if adjacent_sims[i] - adjacent_sims[i+1] > 0.1:  # Significant drop
            drops.append(layers[i+1])

    # Mean similarity by distance
    distance_means = {}
    for d in range(1, min(6, n)):  # Up to distance 5
        sims = [matrix[i, i+d] for i in range(n-d)]
        distance_means[d] = float(np.mean(sims))

    return {
        "adjacent_mean": float(np.mean(adjacent_sims)),
        "adjacent_min": float(np.min(adjacent_sims)),
        "adjacent_min_layer": int(layers[np.argmin(adjacent_sims)]),
        "global_mean": float(np.mean(matrix)),
        "distance_means": distance_means,
        "transition_layers": drops,
    }


def print_matrix_compact(matrix: np.ndarray, layers: list[int], step: int = 2):
    """Print matrix in compact form (every Nth layer)."""
    indices = list(range(0, len(layers), step))
    if indices[-1] != len(layers) - 1:
        indices.append(len(layers) - 1)

    # Header
    header = "     " + " ".join(f"L{layers[i]:>2}" for i in indices)
    print(header)

    for i in indices:
        row = f"L{layers[i]:>2}  " + " ".join(f"{matrix[i, j]:>4.2f}" for j in indices)
        print(row)


def main():
    parser = argparse.ArgumentParser(description="Cross-layer similarity analysis")
    parser.add_argument("--experiment", required=True, help="Experiment name")
    parser.add_argument("--model-variant", default=None, help="Model variant (default: from experiment config)")
    parser.add_argument("--trait", help="Trait path (e.g., chirp/refusal_v2)")
    parser.add_argument("--all-traits", action="store_true", help="Analyze all extracted traits")
    parser.add_argument("--method", default="probe", help="Method to analyze")
    parser.add_argument("--component", default="residual", help="Component type")
    parser.add_argument("--position", default="response[:]", help="Position string")
    parser.add_argument("--output", help="Output JSON path (optional)")
    args = parser.parse_args()

    # Resolve model variant
    variant = get_model_variant(args.experiment, args.model_variant, mode="extraction")
    model_variant = variant['name']

    if args.all_traits:
        traits = [f"{cat}/{name}" for cat, name in discover_extracted_traits(args.experiment)]
    elif args.trait:
        traits = [args.trait]
    else:
        parser.error("Specify --trait or --all-traits")

    all_results = {}

    for trait in traits:
        methods = list_methods(args.experiment, trait, model_variant, args.position, args.component)
        if args.method not in methods:
            print(f"Skipping {trait}: {args.method} not available (have: {methods})")
            continue

        matrix, layers = compute_cross_layer_matrix(
            args.experiment, trait, model_variant, args.method, args.component, args.position
        )

        if matrix is None:
            print(f"Skipping {trait}: no vectors found")
            continue

        stats = analyze_matrix(matrix, layers)

        print(f"\n{'='*60}")
        print(f"Trait: {trait} | Method: {args.method}")
        print(f"{'='*60}")
        print(f"Layers: {len(layers)} ({min(layers)}-{max(layers)})")
        print()
        print("Cross-layer similarity (sampled):")
        print_matrix_compact(matrix, layers, step=4)
        print()
        print(f"Adjacent similarity: mean={stats['adjacent_mean']:.3f}, min={stats['adjacent_min']:.3f} at L{stats['adjacent_min_layer']}")
        print(f"Global mean: {stats['global_mean']:.3f}")
        print(f"Distance means: {', '.join(f'd={k}: {v:.3f}' for k, v in stats['distance_means'].items())}")
        if stats['transition_layers']:
            print(f"Transition layers (drops): {stats['transition_layers']}")

        all_results[trait] = {
            "method": args.method,
            "layers": layers,
            "matrix": matrix.tolist(),
            "stats": stats,
        }

    # Output
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump({
                "experiment": args.experiment,
                "component": args.component,
                "position": args.position,
                "traits": all_results,
            }, f, indent=2)
        print(f"\nResults saved to {output_path}")

    return all_results


if __name__ == "__main__":
    main()
