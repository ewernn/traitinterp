#!/usr/bin/env python3
"""
Analyze trait vector similarities and empirical co-activation patterns.
"""

import torch
import numpy as np
import json
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

TRAITS = [
    'refusal', 'uncertainty', 'verbosity', 'overconfidence',
    'corrigibility', 'evil', 'sycophantic', 'hallucinating'
]

def load_vectors(vector_dir: str = "persona_vectors/gemma-2-2b-it", layer: int = 16):
    """Load all trait vectors for specified layer."""
    vectors = {}

    for trait in TRAITS:
        path = f"{vector_dir}/{trait}_response_avg_diff.pt"
        if Path(path).exists():
            vec = torch.load(path, map_location='cpu')
            if vec.dim() == 2:  # [num_layers, hidden_dim]
                vec = vec[layer]
            vectors[trait] = vec
        else:
            print(f"Warning: {trait} vector not found")

    return vectors


def compute_geometric_similarity(vectors):
    """Compute cosine similarity between trait vectors in activation space."""
    n = len(TRAITS)
    similarity = np.zeros((n, n))

    for i, trait_i in enumerate(TRAITS):
        for j, trait_j in enumerate(TRAITS):
            if trait_i in vectors and trait_j in vectors:
                vec_i = vectors[trait_i]
                vec_j = vectors[trait_j]

                # Cosine similarity
                cos_sim = (vec_i @ vec_j) / (vec_i.norm() * vec_j.norm())
                similarity[i, j] = cos_sim.item()

    return similarity


def compute_empirical_correlation(results_file: str):
    """Compute correlation between traits across all tokens in dataset."""
    with open(results_file) as f:
        data = json.load(f)

    # Collect all trait scores across all prompts and tokens
    trait_scores = {trait: [] for trait in TRAITS}

    for result in data:
        scores = result.get('trait_scores', result.get('projections', {}))
        for trait in TRAITS:
            if trait in scores:
                trait_scores[trait].extend(scores[trait])

    # Compute correlation matrix
    n = len(TRAITS)
    correlation = np.zeros((n, n))

    for i, trait_i in enumerate(TRAITS):
        for j, trait_j in enumerate(TRAITS):
            if trait_scores[trait_i] and trait_scores[trait_j]:
                # Ensure same length (they should be)
                min_len = min(len(trait_scores[trait_i]), len(trait_scores[trait_j]))
                scores_i = trait_scores[trait_i][:min_len]
                scores_j = trait_scores[trait_j][:min_len]

                corr, _ = pearsonr(scores_i, scores_j)
                correlation[i, j] = corr

    return correlation, trait_scores


def compute_per_prompt_purity(results_file: str, target_trait_threshold: float = 0.5):
    """
    For each prompt, check if target trait dominates (single-trait purity).

    Returns:
        List of dicts with prompt info and purity metrics
    """
    with open(results_file) as f:
        data = json.load(f)

    purity_results = []

    for result in data:
        scores = result.get('trait_scores', result.get('projections', {}))
        metadata = result.get('metadata', {})
        target_trait = metadata.get('trait', 'unknown')

        # Compute mean absolute score for each trait
        trait_means = {}
        for trait in TRAITS:
            if trait in scores and scores[trait]:
                trait_means[trait] = np.mean(np.abs(scores[trait]))

        if not trait_means:
            continue

        # Check if target trait has highest mean
        max_trait = max(trait_means, key=trait_means.get)
        max_score = trait_means[max_trait]

        # Compute purity: ratio of target to second-highest
        sorted_scores = sorted(trait_means.values(), reverse=True)
        purity_ratio = sorted_scores[0] / sorted_scores[1] if len(sorted_scores) > 1 else float('inf')

        # Check if target trait dominates
        is_pure = (max_trait == target_trait and purity_ratio > target_trait_threshold)

        purity_results.append({
            'prompt': result['prompt'][:80] + '...',
            'target_trait': target_trait,
            'dominant_trait': max_trait,
            'is_pure': bool(is_pure),
            'purity_ratio': float(purity_ratio),
            'trait_means': {k: float(v) for k, v in trait_means.items()},
            'response': result['response'][:100] + '...'
        })

    return purity_results


def plot_similarity_matrices(geometric, empirical, output_dir: str = "analysis/plots"):
    """Plot side-by-side heatmaps of geometric and empirical similarity."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Geometric similarity
    sns.heatmap(
        geometric,
        xticklabels=TRAITS,
        yticklabels=TRAITS,
        annot=True,
        fmt='.2f',
        cmap='coolwarm',
        center=0,
        vmin=-1,
        vmax=1,
        ax=axes[0],
        cbar_kws={'label': 'Cosine Similarity'}
    )
    axes[0].set_title('Geometric Similarity\n(Vector Space)', fontsize=14, fontweight='bold')

    # Empirical correlation
    sns.heatmap(
        empirical,
        xticklabels=TRAITS,
        yticklabels=TRAITS,
        annot=True,
        fmt='.2f',
        cmap='coolwarm',
        center=0,
        vmin=-1,
        vmax=1,
        ax=axes[1],
        cbar_kws={'label': 'Pearson Correlation'}
    )
    axes[1].set_title('Empirical Correlation\n(Co-activation)', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f"{output_dir}/trait_similarity.png", dpi=300, bbox_inches='tight')
    print(f"✅ Saved similarity matrices to {output_dir}/trait_similarity.png")
    plt.close()


def print_summary(geometric, empirical, trait_scores):
    """Print summary statistics."""
    print("\n" + "="*70)
    print("TRAIT SIMILARITY ANALYSIS")
    print("="*70)

    print("\n📐 GEOMETRIC SIMILARITY (Cosine in Vector Space)")
    print("-" * 70)

    # Find most similar pairs (excluding diagonal)
    geo_flat = []
    for i in range(len(TRAITS)):
        for j in range(i+1, len(TRAITS)):
            geo_flat.append((TRAITS[i], TRAITS[j], geometric[i, j]))
    geo_sorted = sorted(geo_flat, key=lambda x: abs(x[2]), reverse=True)

    print("\nMost geometrically similar:")
    for trait_i, trait_j, sim in geo_sorted[:5]:
        print(f"  {trait_i:15s} ↔ {trait_j:15s}: {sim:6.3f}")

    print("\nMost geometrically opposite:")
    for trait_i, trait_j, sim in geo_sorted[-5:]:
        print(f"  {trait_i:15s} ↔ {trait_j:15s}: {sim:6.3f}")

    print("\n📊 EMPIRICAL CORRELATION (Co-activation in Practice)")
    print("-" * 70)

    # Find most correlated pairs
    emp_flat = []
    for i in range(len(TRAITS)):
        for j in range(i+1, len(TRAITS)):
            emp_flat.append((TRAITS[i], TRAITS[j], empirical[i, j]))
    emp_sorted = sorted(emp_flat, key=lambda x: abs(x[2]), reverse=True)

    print("\nMost empirically correlated:")
    for trait_i, trait_j, corr in emp_sorted[:5]:
        print(f"  {trait_i:15s} ↔ {trait_j:15s}: {corr:6.3f}")

    print("\nMost empirically anti-correlated:")
    for trait_i, trait_j, corr in emp_sorted[-5:]:
        print(f"  {trait_i:15s} ↔ {trait_j:15s}: {corr:6.3f}")

    print("\n🔍 DIVERGENCE (Geometric vs Empirical)")
    print("-" * 70)

    # Find largest divergences
    divergence = []
    for i in range(len(TRAITS)):
        for j in range(i+1, len(TRAITS)):
            div = abs(geometric[i, j] - empirical[i, j])
            divergence.append((TRAITS[i], TRAITS[j], geometric[i, j], empirical[i, j], div))
    divergence_sorted = sorted(divergence, key=lambda x: x[4], reverse=True)

    print("\nLargest divergences (geometric ≠ empirical):")
    for trait_i, trait_j, geo, emp, div in divergence_sorted[:5]:
        print(f"  {trait_i:15s} ↔ {trait_j:15s}: geo={geo:6.3f}, emp={emp:6.3f}, Δ={div:.3f}")

    print("\n📈 TRAIT ACTIVATION STATISTICS")
    print("-" * 70)
    for trait in TRAITS:
        if trait_scores[trait]:
            scores = np.array(trait_scores[trait])
            print(f"{trait:15s}: mean={scores.mean():7.2f}, std={scores.std():6.2f}, "
                  f"range=[{scores.min():7.2f}, {scores.max():7.2f}]")


def analyze_prompt_purity(results_file: str, output_file: str = "analysis/prompt_purity.json"):
    """Analyze and save per-prompt purity results."""
    purity = compute_per_prompt_purity(results_file)

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(purity, f, indent=2)

    # Summary stats
    total = len(purity)
    pure = sum(1 for p in purity if p['is_pure'])

    print("\n" + "="*70)
    print("PROMPT PURITY ANALYSIS")
    print("="*70)
    print(f"Total prompts: {total}")
    print(f"Pure single-trait: {pure} ({100*pure/total:.1f}%)")
    print(f"Multi-trait contamination: {total - pure} ({100*(total-pure)/total:.1f}%)")

    # Per-trait purity
    print("\nPer-trait purity:")
    for trait in TRAITS:
        trait_prompts = [p for p in purity if p['target_trait'] == trait]
        if trait_prompts:
            trait_pure = sum(1 for p in trait_prompts if p['is_pure'])
            print(f"  {trait:15s}: {trait_pure}/{len(trait_prompts)} pure ({100*trait_pure/len(trait_prompts):.1f}%)")

    # Most contaminated prompts
    print("\nMost contaminated prompts (target trait not dominant):")
    contaminated = [p for p in purity if not p['is_pure']]
    contaminated_sorted = sorted(contaminated, key=lambda x: x['purity_ratio'])

    for p in contaminated_sorted[:5]:
        print(f"\n  Prompt: {p['prompt']}")
        print(f"  Target: {p['target_trait']}, Dominant: {p['dominant_trait']}")
        print(f"  Purity ratio: {p['purity_ratio']:.2f}")
        print(f"  Trait means: {p['trait_means']}")

    print(f"\n✅ Saved detailed purity analysis to {output_file}")

    return purity


def main(
    vector_dir: str = "persona_vectors/gemma-2-2b-it",
    results_file: str = "pertoken/results/gemma_2b_single_trait_teaching.json",
    layer: int = 16
):
    """Run complete trait similarity analysis."""

    print("\n" + "="*70)
    print("LOADING DATA")
    print("="*70)

    # Load vectors
    print(f"\nLoading trait vectors from {vector_dir}...")
    vectors = load_vectors(vector_dir, layer)
    print(f"✅ Loaded {len(vectors)} trait vectors")

    # Compute geometric similarity
    print("\nComputing geometric similarity...")
    geometric = compute_geometric_similarity(vectors)
    print("✅ Computed geometric similarity matrix")

    # Compute empirical correlation
    print(f"\nComputing empirical correlation from {results_file}...")
    empirical, trait_scores = compute_empirical_correlation(results_file)
    print("✅ Computed empirical correlation matrix")

    # Print summary
    print_summary(geometric, empirical, trait_scores)

    # Plot matrices
    print("\nGenerating visualizations...")
    plot_similarity_matrices(geometric, empirical)

    # Analyze prompt purity
    print("\nAnalyzing prompt purity...")
    purity = analyze_prompt_purity(results_file)

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print("Generated files:")
    print("  - analysis/plots/trait_similarity.png")
    print("  - analysis/prompt_purity.json")


if __name__ == "__main__":
    import fire
    fire.Fire(main)
