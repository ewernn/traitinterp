#!/usr/bin/env python3
"""Compare component vectors across multiple layers.

Input: Vector files for a specific trait across layers
Output: Layer-wise cosine similarity analysis
Usage: python compare_layers_components.py
"""

import torch
from pathlib import Path
import numpy as np

def cosine_similarity(v1, v2):
    """Calculate cosine similarity between two vectors."""
    return (v1 @ v2) / (torch.norm(v1) * torch.norm(v2))

def main():
    experiment = "qwen_optimism"
    trait = "mental_state/optimism-attempt2"
    method = "probe"
    layers_to_check = [8, 10, 12, 14, 16]  # Sample of middle-to-late layers

    base_path = Path(f"/home/dev/trait-interp/experiments/{experiment}/extraction/{trait}/vectors")

    print("="*80)
    print("COMPONENT SIMILARITY ACROSS LAYERS")
    print("="*80)
    print(f"\nExperiment: {experiment}")
    print(f"Trait: {trait}")
    print(f"Method: {method}")
    print(f"\nAnalyzing layers: {layers_to_check}")

    results = []

    for layer in layers_to_check:
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
            else:
                print(f"✗ Missing {component} at layer {layer}")
                break

        if len(vectors) != 3:
            continue

        # Calculate similarities
        res_attn = cosine_similarity(vectors["residual"], vectors["attn_out"]).item()
        res_mlp = cosine_similarity(vectors["residual"], vectors["mlp_out"]).item()
        attn_mlp = cosine_similarity(vectors["attn_out"], vectors["mlp_out"]).item()

        # Calculate norms
        norms = {comp: torch.norm(vectors[comp]).item() for comp in components}

        results.append({
            "layer": layer,
            "res_attn": res_attn,
            "res_mlp": res_mlp,
            "attn_mlp": attn_mlp,
            "norms": norms
        })

    # Print table
    print("\n" + "="*80)
    print("COSINE SIMILARITIES")
    print("="*80)
    print("\nLayer  Res↔Attn  Res↔MLP  Attn↔MLP  Dominant")
    print("-"*60)

    for r in results:
        dominant = "MLP" if r["res_mlp"] > r["res_attn"] else "Attn" if r["res_attn"] > r["res_mlp"] else "Balanced"
        print(f"{r['layer']:5d}  {r['res_attn']:8.4f}  {r['res_mlp']:8.4f}  {r['attn_mlp']:9.4f}  {dominant}")

    # Print norms table
    print("\n" + "="*80)
    print("VECTOR NORMS")
    print("="*80)
    print("\nLayer  Residual  Attn_out  MLP_out")
    print("-"*45)

    for r in results:
        print(f"{r['layer']:5d}  {r['norms']['residual']:8.4f}  {r['norms']['attn_out']:8.4f}  {r['norms']['mlp_out']:7.4f}")

    # Summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)

    res_attn_avg = np.mean([r["res_attn"] for r in results])
    res_mlp_avg = np.mean([r["res_mlp"] for r in results])
    attn_mlp_avg = np.mean([r["attn_mlp"] for r in results])

    print(f"\nAverage similarities across layers {layers_to_check}:")
    print(f"  Residual ↔ Attn_out:  {res_attn_avg:.4f}")
    print(f"  Residual ↔ MLP_out:   {res_mlp_avg:.4f}")
    print(f"  Attn_out ↔ MLP_out:   {attn_mlp_avg:.4f}")

    mlp_wins = sum(1 for r in results if r["res_mlp"] > r["res_attn"])
    attn_wins = sum(1 for r in results if r["res_attn"] > r["res_mlp"])

    print(f"\nDominant component:")
    print(f"  MLP dominates in {mlp_wins}/{len(results)} layers")
    print(f"  Attn dominates in {attn_wins}/{len(results)} layers")

    if abs(attn_mlp_avg) < 0.3:
        print(f"\nAttn and MLP are largely orthogonal (avg sim={attn_mlp_avg:.4f})")
        print("  → Components use different mechanisms for optimism")
    elif attn_mlp_avg > 0.5:
        print(f"\nAttn and MLP are aligned (avg sim={attn_mlp_avg:.4f})")
        print("  → Components use similar mechanisms for optimism")

if __name__ == "__main__":
    main()
