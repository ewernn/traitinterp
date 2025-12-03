#!/usr/bin/env python3
"""Compare vectors extracted from different components (residual, attn_out, mlp_out).

Input: Vector files for a specific trait and layer
Output: Cosine similarity matrix and vector statistics
Usage: python compare_component_vectors.py
"""

import torch
from pathlib import Path

def cosine_similarity(v1, v2):
    """Calculate cosine similarity between two vectors."""
    return (v1 @ v2) / (torch.norm(v1) * torch.norm(v2))

def main():
    experiment = "qwen_optimism"
    trait = "mental_state/optimism-attempt2"
    layer = 10
    method = "probe"

    base_path = Path(f"/home/dev/trait-interp/experiments/{experiment}/extraction/{trait}/vectors")

    # Load vectors
    vectors = {}
    components = ["residual", "attn_out", "mlp_out"]

    for component in components:
        if component == "residual":
            vector_path = base_path / f"{method}_layer{layer}.pt"
        else:
            vector_path = base_path / f"{component}_{method}_layer{layer}.pt"

        if vector_path.exists():
            vectors[component] = torch.load(vector_path)
            print(f"✓ Loaded {component} vector from {vector_path}")
        else:
            print(f"✗ Missing {component} vector at {vector_path}")

    if len(vectors) != 3:
        print(f"\nError: Could not load all vectors. Found {len(vectors)}/3")
        return

    print("\n" + "="*80)
    print("VECTOR STATISTICS")
    print("="*80)

    # Print vector norms
    print("\nVector Norms:")
    for component in components:
        norm = torch.norm(vectors[component]).item()
        print(f"  {component:12s}: {norm:8.4f}")

    # Calculate and print cosine similarities
    print("\n" + "="*80)
    print("COSINE SIMILARITY MATRIX")
    print("="*80)
    print("\n           residual   attn_out   mlp_out")
    print("         " + "-"*40)

    for comp1 in components:
        row = f"{comp1:12s} |"
        for comp2 in components:
            if comp1 == comp2:
                row += "   1.0000"
            else:
                sim = cosine_similarity(vectors[comp1], vectors[comp2]).item()
                row += f"   {sim:7.4f}"
        print(row)

    print("\n" + "="*80)
    print("INTERPRETATION")
    print("="*80)

    # Compare residual to components
    res_attn_sim = cosine_similarity(vectors["residual"], vectors["attn_out"]).item()
    res_mlp_sim = cosine_similarity(vectors["residual"], vectors["mlp_out"]).item()
    attn_mlp_sim = cosine_similarity(vectors["attn_out"], vectors["mlp_out"]).item()

    print(f"\nResidual vs Attn_out:  {res_attn_sim:.4f}")
    print(f"Residual vs MLP_out:   {res_mlp_sim:.4f}")
    print(f"Attn_out vs MLP_out:   {attn_mlp_sim:.4f}")

    # Determine which component dominates
    print("\nComponent contribution to residual optimism vector:")
    if res_attn_sim > res_mlp_sim:
        print(f"  → Attention output dominates (sim={res_attn_sim:.4f} vs {res_mlp_sim:.4f})")
    elif res_mlp_sim > res_attn_sim:
        print(f"  → MLP output dominates (sim={res_mlp_sim:.4f} vs {res_attn_sim:.4f})")
    else:
        print(f"  → Balanced contribution (sim={res_attn_sim:.4f} ≈ {res_mlp_sim:.4f})")

    # Check if components are orthogonal or aligned
    if abs(attn_mlp_sim) < 0.3:
        print(f"\nAttn and MLP are nearly orthogonal (sim={attn_mlp_sim:.4f})")
        print("  → Different mechanisms for optimism")
    elif attn_mlp_sim > 0.7:
        print(f"\nAttn and MLP are highly aligned (sim={attn_mlp_sim:.4f})")
        print("  → Similar mechanisms for optimism")
    else:
        print(f"\nAttn and MLP are partially aligned (sim={attn_mlp_sim:.4f})")
        print("  → Complementary mechanisms for optimism")

if __name__ == "__main__":
    main()
