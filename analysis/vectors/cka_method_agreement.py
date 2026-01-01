"""
Compute CKA (Centered Kernel Alignment) between extraction methods.

Measures structural similarity between probe, mean_diff, and gradient vectors
across layers. High CKA (>0.7) = methods converge on same structure.

Input: Extracted vectors at experiments/{experiment}/extraction/{trait}/vectors/
Output: CKA matrix and per-layer cosine similarities

Usage:
    python analysis/vectors/cka_method_agreement.py --experiment gemma-2-2b --trait chirp/refusal_v2
"""

import argparse
import json
from pathlib import Path

import torch
import numpy as np

from utils.paths import get_vector_path, list_methods, list_layers
from core.math import cosine_similarity


def linear_kernel(X: torch.Tensor) -> torch.Tensor:
    """Linear kernel: K = X @ X.T"""
    return X @ X.T


def center_kernel(K: torch.Tensor) -> torch.Tensor:
    """Center kernel matrix (HSIC centering)."""
    n = K.shape[0]
    H = torch.eye(n, device=K.device) - torch.ones(n, n, device=K.device) / n
    return H @ K @ H


def cka(X: torch.Tensor, Y: torch.Tensor) -> float:
    """
    Compute CKA between two representation matrices.

    Args:
        X: [n_samples, dim1] - e.g., [n_layers, hidden_dim]
        Y: [n_samples, dim2]

    Returns:
        CKA score in [0, 1], higher = more similar structure
    """
    K_X = center_kernel(linear_kernel(X))
    K_Y = center_kernel(linear_kernel(Y))

    hsic_XY = (K_X * K_Y).sum()
    hsic_XX = (K_X * K_X).sum()
    hsic_YY = (K_Y * K_Y).sum()

    if hsic_XX * hsic_YY <= 0:
        return 0.0

    return float(hsic_XY / torch.sqrt(hsic_XX * hsic_YY))


def load_vectors_for_method(
    experiment: str,
    trait: str,
    method: str,
    component: str = "residual",
    position: str = "response[:]",
) -> tuple[torch.Tensor, list[int]]:
    """Load all layer vectors for a method, return stacked tensor and layer list."""
    layers = list_layers(experiment, trait, method, component, position)
    if not layers:
        return None, []

    vectors = []
    for layer in layers:
        path = get_vector_path(experiment, trait, method, layer, component, position)
        v = torch.load(path, weights_only=True).float()
        vectors.append(v)

    return torch.stack(vectors), layers


def main():
    parser = argparse.ArgumentParser(description="CKA method agreement analysis")
    parser.add_argument("--experiment", required=True, help="Experiment name")
    parser.add_argument("--trait", required=True, help="Trait path (e.g., chirp/refusal_v2)")
    parser.add_argument("--component", default="residual", help="Component type")
    parser.add_argument("--position", default="response[:]", help="Position string")
    parser.add_argument("--output", help="Output JSON path (optional)")
    args = parser.parse_args()

    methods = list_methods(args.experiment, args.trait, args.component, args.position)
    if len(methods) < 2:
        print(f"Need at least 2 methods, found: {methods}")
        return

    print(f"Analyzing methods: {methods}")
    print(f"Trait: {args.trait}, Position: {args.position}, Component: {args.component}")
    print()

    # Load vectors for each method
    method_vectors = {}
    method_layers = {}
    for method in methods:
        vectors, layers = load_vectors_for_method(
            args.experiment, args.trait, method, args.component, args.position
        )
        if vectors is not None:
            method_vectors[method] = vectors
            method_layers[method] = layers
            print(f"  {method}: {len(layers)} layers, shape {vectors.shape}")

    # Find common layers
    common_layers = set(method_layers[methods[0]])
    for method in methods[1:]:
        common_layers &= set(method_layers[method])
    common_layers = sorted(common_layers)
    print(f"\nCommon layers: {len(common_layers)} ({min(common_layers)}-{max(common_layers)})")

    # Filter to common layers
    for method in methods:
        layer_indices = [method_layers[method].index(l) for l in common_layers]
        method_vectors[method] = method_vectors[method][layer_indices]

    # Compute CKA matrix
    print("\n=== CKA Matrix ===")
    cka_matrix = {}
    for m1 in methods:
        cka_matrix[m1] = {}
        for m2 in methods:
            cka_matrix[m1][m2] = cka(method_vectors[m1], method_vectors[m2])

    # Print matrix
    header = "           " + "  ".join(f"{m:>10}" for m in methods)
    print(header)
    for m1 in methods:
        row = f"{m1:>10} " + "  ".join(f"{cka_matrix[m1][m2]:>10.3f}" for m2 in methods)
        print(row)

    # Compute per-layer cosine similarities
    print("\n=== Per-Layer Cosine Similarity ===")
    print("Layer  " + "  ".join(f"{m1[:4]}-{m2[:4]}" for i, m1 in enumerate(methods) for m2 in methods[i+1:]))

    per_layer_cosine = []
    for i, layer in enumerate(common_layers):
        cosines = {}
        row_vals = []
        for j, m1 in enumerate(methods):
            for m2 in methods[j+1:]:
                cos = cosine_similarity(method_vectors[m1][i], method_vectors[m2][i]).item()
                cosines[f"{m1}-{m2}"] = cos
                row_vals.append(cos)
        per_layer_cosine.append({"layer": layer, **cosines})
        print(f"L{layer:>2}    " + "  ".join(f"{v:>9.3f}" for v in row_vals))

    # Summary statistics
    print("\n=== Summary ===")
    for j, m1 in enumerate(methods):
        for m2 in methods[j+1:]:
            key = f"{m1}-{m2}"
            vals = [plc[key] for plc in per_layer_cosine]
            print(f"{key}: mean={np.mean(vals):.3f}, std={np.std(vals):.3f}, min={np.min(vals):.3f}, max={np.max(vals):.3f}")

    # Output
    results = {
        "experiment": args.experiment,
        "trait": args.trait,
        "component": args.component,
        "position": args.position,
        "methods": methods,
        "common_layers": common_layers,
        "cka_matrix": cka_matrix,
        "per_layer_cosine": per_layer_cosine,
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
