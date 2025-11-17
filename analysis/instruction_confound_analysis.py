#!/usr/bin/env python3
"""
Phase 1: Instruction-Following Confound Analysis

Uses instruction_boundary vector as proxy for instruction-following to quantify
contamination across all trait vectors.

Usage:
    python analysis/instruction_confound_analysis.py --experiment gemma_2b_cognitive_nov20
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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

def cosine_similarity(v1, v2):
    """Compute cosine similarity between two vectors."""
    return (v1 @ v2) / (v1.norm() * v2.norm())

def analyze_contamination(vectors, proxy_name='instruction_boundary'):
    """Analyze contamination using proxy vector."""

    if proxy_name not in vectors:
        print(f"❌ Proxy vector '{proxy_name}' not found!")
        print(f"Available vectors: {list(vectors.keys())}")
        return None

    proxy_vec = vectors[proxy_name]

    print("\n" + "="*70)
    print(f"INSTRUCTION-FOLLOWING CONFOUND ANALYSIS")
    print(f"Using '{proxy_name}' as proxy for instruction-following")
    print("="*70)

    # Compute correlations
    correlations = []
    for name, vec in vectors.items():
        if name == proxy_name:
            continue

        sim = cosine_similarity(vec, proxy_vec).item()
        correlations.append({
            'trait': name,
            'correlation': sim,
            'abs_correlation': abs(sim)
        })

    # Sort by absolute correlation
    correlations.sort(key=lambda x: x['abs_correlation'], reverse=True)

    # Statistics
    corr_values = [c['correlation'] for c in correlations]
    mean_corr = np.mean(corr_values)
    std_corr = np.std(corr_values)

    print(f"\nStatistics:")
    print(f"  Mean correlation: {mean_corr:+.3f}")
    print(f"  Std correlation:  {std_corr:.3f}")
    print(f"  Range: [{min(corr_values):+.3f}, {max(corr_values):+.3f}]")

    # Theoretical random baseline
    dim = proxy_vec.shape[0]
    expected_std = 1.0 / np.sqrt(dim)

    print(f"\nBaseline (random {dim}-dim vectors):")
    print(f"  Expected mean: 0.000")
    print(f"  Expected std:  {expected_std:.3f}")
    print(f"  Observed/Expected ratio: {std_corr/expected_std:.2f}x")

    # Classification
    print(f"\n" + "="*70)
    print("CONTAMINATION LEVELS")
    print("="*70)

    # Define thresholds
    threshold_high = 0.3
    threshold_medium = 0.15
    threshold_low = 0.05

    high_contam = [c for c in correlations if c['abs_correlation'] > threshold_high]
    medium_contam = [c for c in correlations if threshold_medium < c['abs_correlation'] <= threshold_high]
    low_contam = [c for c in correlations if threshold_low < c['abs_correlation'] <= threshold_medium]
    minimal_contam = [c for c in correlations if c['abs_correlation'] <= threshold_low]

    print(f"\n🔴 HIGH CONTAMINATION (|r| > {threshold_high}):")
    for c in high_contam:
        print(f"  {c['correlation']:+.3f}: {c['trait']}")

    print(f"\n🟡 MEDIUM CONTAMINATION ({threshold_medium} < |r| ≤ {threshold_high}):")
    for c in medium_contam:
        print(f"  {c['correlation']:+.3f}: {c['trait']}")

    print(f"\n🟢 LOW CONTAMINATION ({threshold_low} < |r| ≤ {threshold_medium}):")
    for c in low_contam:
        print(f"  {c['correlation']:+.3f}: {c['trait']}")

    print(f"\n✅ MINIMAL CONTAMINATION (|r| ≤ {threshold_low}):")
    for c in minimal_contam:
        print(f"  {c['correlation']:+.3f}: {c['trait']}")

    # Summary
    print(f"\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"High contamination:    {len(high_contam)}/{len(correlations)} traits")
    print(f"Medium contamination:  {len(medium_contam)}/{len(correlations)} traits")
    print(f"Low contamination:     {len(low_contam)}/{len(correlations)} traits")
    print(f"Minimal contamination: {len(minimal_contam)}/{len(correlations)} traits")

    return correlations

def plot_contamination(correlations, proxy_name, output_path):
    """Plot contamination levels."""

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Bar chart
    ax = axes[0]
    traits = [c['trait'] for c in correlations]
    corrs = [c['correlation'] for c in correlations]
    colors = ['red' if abs(c) > 0.3 else 'orange' if abs(c) > 0.15 else 'yellow' if abs(c) > 0.05 else 'green' for c in corrs]

    y_pos = np.arange(len(traits))
    ax.barh(y_pos, corrs, color=colors, alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(traits, fontsize=9)
    ax.set_xlabel('Correlation with Instruction-Following', fontsize=12)
    ax.set_title(f'Trait Contamination by {proxy_name}', fontsize=14)
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    ax.axvline(x=0.3, color='red', linestyle='--', alpha=0.5, label='High (0.3)')
    ax.axvline(x=-0.3, color='red', linestyle='--', alpha=0.5)
    ax.axvline(x=0.15, color='orange', linestyle='--', alpha=0.5, label='Medium (0.15)')
    ax.axvline(x=-0.15, color='orange', linestyle='--', alpha=0.5)
    ax.legend()
    ax.grid(alpha=0.3, axis='x')

    # Plot 2: Distribution
    ax = axes[1]
    corrs_array = np.array(corrs)
    ax.hist(corrs_array, bins=20, color='steelblue', alpha=0.7, edgecolor='black')
    ax.axvline(x=0, color='black', linestyle='-', linewidth=2, label='Zero')
    ax.axvline(x=corrs_array.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean ({corrs_array.mean():.3f})')
    ax.set_xlabel('Correlation', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Distribution of Contamination', fontsize=14)
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Plot saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Analyze instruction-following contamination")
    parser.add_argument("--experiment", type=str, default="gemma_2b_cognitive_nov20")
    parser.add_argument("--method", type=str, default="probe",
                        help="Extraction method")
    parser.add_argument("--layer", type=int, default=16,
                        help="Layer to analyze")
    parser.add_argument("--proxy", type=str, default="instruction_boundary",
                        help="Vector to use as instruction-following proxy")
    parser.add_argument("--output", type=str, default="analysis/instruction_contamination.png",
                        help="Output plot path")

    args = parser.parse_args()

    print(f"Loading vectors from: experiments/{args.experiment}")
    print(f"Method: {args.method}, Layer: {args.layer}")

    # Load vectors
    vectors = load_all_vectors(args.experiment, args.method, args.layer)
    print(f"Loaded {len(vectors)} trait vectors")

    if len(vectors) < 2:
        print("❌ Need at least 2 traits")
        return

    # Analyze
    correlations = analyze_contamination(vectors, args.proxy)

    if correlations:
        # Plot
        plot_contamination(correlations, args.proxy, args.output)

        # Save results
        import json
        results = {
            'proxy': args.proxy,
            'correlations': correlations,
            'statistics': {
                'mean': float(np.mean([c['correlation'] for c in correlations])),
                'std': float(np.std([c['correlation'] for c in correlations])),
                'min': float(min([c['correlation'] for c in correlations])),
                'max': float(max([c['correlation'] for c in correlations]))
            }
        }

        output_json = Path(args.output).with_suffix('.json')
        with open(output_json, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"✅ Results saved to: {output_json}")

if __name__ == "__main__":
    main()
