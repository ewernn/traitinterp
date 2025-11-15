#!/usr/bin/env python3
"""
Superposition Measurement (Idea #10)

Computes pairwise cosine similarities between all trait vectors.
Tests hypothesis: Are traits orthogonal (superposition) or confounded?

Expected results:
- mean ≈ 0.0: Orthogonal (clean superposition)
- mean ≈ 0.1-0.3: Almost orthogonal (some confounding)
- mean > 0.5: Heavily confounded (not superposition)

Usage:
    python analysis/superposition_measurement.py --experiment gemma_2b_cognitive_nov20
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import argparse
import numpy as np
from itertools import combinations
import matplotlib.pyplot as plt
import seaborn as sns

def cosine_similarity(v1, v2):
    """Compute cosine similarity between two vectors."""
    return (v1 @ v2) / (v1.norm() * v2.norm())

def load_all_vectors(experiment, method='probe', layer=16):
    """Load all trait vectors for given method and layer."""
    exp_dir = Path(f"experiments/{experiment}")
    vectors = {}

    for trait_dir in exp_dir.iterdir():
        if not trait_dir.is_dir():
            continue

        vector_path = trait_dir / "extraction" / "vectors" / f"{method}_layer{layer}.pt"
        if vector_path.exists():
            vectors[trait_dir.name] = torch.load(vector_path).float()

    return vectors

def compute_superposition_matrix(vectors):
    """Compute pairwise cosine similarities."""
    traits = list(vectors.keys())
    n = len(traits)
    matrix = np.zeros((n, n))

    for i, trait1 in enumerate(traits):
        for j, trait2 in enumerate(traits):
            if i == j:
                matrix[i, j] = 1.0  # Self-similarity
            else:
                sim = cosine_similarity(vectors[trait1], vectors[trait2]).item()
                matrix[i, j] = sim

    return matrix, traits

def analyze_superposition(matrix, traits):
    """Analyze superposition characteristics."""
    # Get upper triangle (exclude diagonal)
    n = len(traits)
    upper_tri = []
    for i in range(n):
        for j in range(i+1, n):
            upper_tri.append(matrix[i, j])

    upper_tri = np.array(upper_tri)

    print("\n" + "="*60)
    print("SUPERPOSITION ANALYSIS")
    print("="*60)

    print(f"\nNumber of traits: {n}")
    print(f"Number of pairs: {len(upper_tri)}")

    print(f"\nCosine Similarity Statistics:")
    print(f"  Mean:   {upper_tri.mean():.3f}")
    print(f"  Median: {np.median(upper_tri):.3f}")
    print(f"  Std:    {upper_tri.std():.3f}")
    print(f"  Min:    {upper_tri.min():.3f}")
    print(f"  Max:    {upper_tri.max():.3f}")

    # Interpretation
    mean_sim = upper_tri.mean()
    print(f"\nInterpretation:")
    if abs(mean_sim) < 0.1:
        print("  ✅ Strong superposition - traits are nearly orthogonal")
    elif abs(mean_sim) < 0.3:
        print("  ⚠️  Weak superposition - some confounding present")
    else:
        print("  ❌ Heavy confounding - traits NOT in superposition")

    # Find most correlated pairs
    print(f"\nTop 5 Most Correlated Pairs:")
    pair_idx = []
    for i in range(n):
        for j in range(i+1, n):
            pair_idx.append((matrix[i, j], traits[i], traits[j]))

    pair_idx.sort(reverse=True)
    for sim, t1, t2 in pair_idx[:5]:
        print(f"  {sim:+.3f}: {t1} ↔ {t2}")

    # Find most anti-correlated pairs
    print(f"\nTop 5 Most Anti-Correlated Pairs:")
    for sim, t1, t2 in pair_idx[-5:]:
        print(f"  {sim:+.3f}: {t1} ↔ {t2}")

    return upper_tri

def plot_heatmap(matrix, traits, output_path):
    """Plot correlation heatmap."""
    plt.figure(figsize=(12, 10))

    # Use RdBu colormap (red = positive, blue = negative)
    sns.heatmap(
        matrix,
        xticklabels=traits,
        yticklabels=traits,
        cmap='RdBu_r',
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        cbar_kws={'label': 'Cosine Similarity'}
    )

    plt.title('Trait Vector Superposition Matrix', fontsize=14, pad=20)
    plt.xlabel('Trait', fontsize=12)
    plt.ylabel('Trait', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Heatmap saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Measure trait superposition")
    parser.add_argument("--experiment", type=str, default="gemma_2b_cognitive_nov20")
    parser.add_argument("--method", type=str, default="probe",
                        help="Extraction method (mean_diff, probe, ica, gradient)")
    parser.add_argument("--layer", type=int, default=16,
                        help="Layer to analyze")
    parser.add_argument("--output", type=str, default="superposition_heatmap.png",
                        help="Output heatmap path")

    args = parser.parse_args()

    print(f"Loading vectors from: experiments/{args.experiment}")
    print(f"Method: {args.method}, Layer: {args.layer}")

    # Load vectors
    vectors = load_all_vectors(args.experiment, args.method, args.layer)
    print(f"Loaded {len(vectors)} trait vectors")

    if len(vectors) < 2:
        print("❌ Need at least 2 traits to compute superposition")
        return

    # Compute matrix
    matrix, traits = compute_superposition_matrix(vectors)

    # Analyze
    similarities = analyze_superposition(matrix, traits)

    # Plot
    plot_heatmap(matrix, traits, args.output)

    # Summary recommendation
    print("\n" + "="*60)
    print("RECOMMENDATION")
    print("="*60)
    mean_sim = similarities.mean()
    if abs(mean_sim) < 0.1:
        print("✅ Traits are well-separated. Proceed with analysis.")
    elif abs(mean_sim) < 0.3:
        print("⚠️  Some confounding present. Consider orthogonalization.")
    else:
        print("❌ Heavy confounding. Traits may measure same underlying factor.")
        print("   → Run Natural vs Instructed test to check instruction-following")
        print("   → Consider removing highly correlated traits")

if __name__ == "__main__":
    main()
