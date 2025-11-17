#!/usr/bin/env python3
"""
Layer Emergence Analysis (Idea #13)

Tests hypothesis: Different traits emerge at different depths.

Method:
    - Compare trait scores across layers [0, 8, 16, 20, 25]
    - Plot emergence curves
    - Find optimal layer for each trait

Expected results:
    - Early layers (0-8): Low-level patterns
    - Middle layers (16-20): Semantic traits
    - Late layers (25+): Output formatting

Usage:
    python analysis/layer_emergence.py --experiment gemma_2b_cognitive_nov20 --trait uncertainty_calibration
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt

def load_vector(experiment, trait, layer, method='probe'):
    """Load trait vector for specific layer."""
    vector_path = Path(f"experiments/{experiment}/{trait}/extraction/vectors/{method}_layer{layer}.pt")
    if vector_path.exists():
        return torch.load(vector_path).float()
    return None

def load_activations_for_layer(experiment, trait, prompt_idx, layer):
    """Load activations from Tier 2 data and extract specific layer."""
    json_path = Path(f"experiments/{experiment}/{trait}/inference/residual_stream_activations/prompt_{prompt_idx}.json")

    if not json_path.exists():
        return None

    with open(json_path) as f:
        data = json.load(f)

    # Get response projections for specified layer
    response_proj = data['projections']['response']  # [n_tokens, n_layers, 3 sublayers]

    scores = []
    for token_proj in response_proj:
        # Average over 3 sublayers
        layer_score = (token_proj[layer][0] +
                       token_proj[layer][1] +
                       token_proj[layer][2]) / 3
        scores.append(layer_score)

    return np.array(scores)

def analyze_layer_emergence(experiment, trait, layers=[0, 8, 16, 20, 25], prompt_idx=0):
    """Analyze trait emergence across layers."""

    print("\n" + "="*70)
    print(f"LAYER EMERGENCE ANALYSIS: {trait}")
    print("="*70)

    # Collect scores for each layer
    layer_stats = {}

    for layer in layers:
        scores = load_activations_for_layer(experiment, trait, prompt_idx, layer)

        if scores is not None:
            layer_stats[layer] = {
                'mean': scores.mean(),
                'max': scores.max(),
                'min': scores.min(),
                'std': scores.std(),
                'abs_mean': np.abs(scores).mean(),
                'scores': scores
            }
            print(f"\nLayer {layer}:")
            print(f"  Mean score: {scores.mean():+.2f}")
            print(f"  Abs mean:   {np.abs(scores).mean():.2f}")
            print(f"  Std:        {scores.std():.2f}")
            print(f"  Range:      [{scores.min():+.2f}, {scores.max():+.2f}]")
        else:
            print(f"\nLayer {layer}: No data available")

    # Find optimal layer
    if layer_stats:
        print("\n" + "="*70)
        print("OPTIMAL LAYER SELECTION")
        print("="*70)

        # Rank by absolute mean (strength of signal)
        ranked = sorted(layer_stats.items(), key=lambda x: x[1]['abs_mean'], reverse=True)

        print("\nRanked by signal strength (abs mean):")
        for i, (layer, stats) in enumerate(ranked, 1):
            print(f"  {i}. Layer {layer:2d}: {stats['abs_mean']:.2f}")

        optimal_layer = ranked[0][0]
        print(f"\n✅ Recommended layer for {trait}: {optimal_layer}")

    return layer_stats

def plot_emergence(layer_stats, trait, output_path):
    """Plot layer emergence curves."""

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # Extract layers in order
    layers = sorted(layer_stats.keys())

    # Plot 1: Mean score across layers
    ax = axes[0, 0]
    means = [layer_stats[l]['mean'] for l in layers]
    ax.plot(layers, means, 'o-', linewidth=2, markersize=8)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Mean Trait Score')
    ax.set_title('Mean Trait Score by Layer')
    ax.grid(alpha=0.3)

    # Plot 2: Absolute mean (signal strength)
    ax = axes[0, 1]
    abs_means = [layer_stats[l]['abs_mean'] for l in layers]
    ax.plot(layers, abs_means, 'o-', linewidth=2, markersize=8, color='red')
    ax.set_xlabel('Layer')
    ax.set_ylabel('Absolute Mean Score')
    ax.set_title('Signal Strength by Layer')
    ax.grid(alpha=0.3)

    # Plot 3: Standard deviation
    ax = axes[1, 0]
    stds = [layer_stats[l]['std'] for l in layers]
    ax.plot(layers, stds, 'o-', linewidth=2, markersize=8, color='green')
    ax.set_xlabel('Layer')
    ax.set_ylabel('Standard Deviation')
    ax.set_title('Variability by Layer')
    ax.grid(alpha=0.3)

    # Plot 4: Full trajectories overlaid
    ax = axes[1, 1]
    for layer in layers:
        scores = layer_stats[layer]['scores']
        ax.plot(scores, label=f'Layer {layer}', alpha=0.7)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Token Index')
    ax.set_ylabel('Trait Score')
    ax.set_title('Trait Trajectories Across Layers')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.suptitle(f'Layer Emergence Analysis: {trait}', fontsize=14, y=0.995)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Plot saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Analyze trait emergence across layers")
    parser.add_argument("--experiment", type=str, default="gemma_2b_cognitive_nov20")
    parser.add_argument("--trait", type=str, required=True,
                        help="Trait to analyze")
    parser.add_argument("--layers", type=str, default="0,8,16,20,25",
                        help="Comma-separated layers to analyze")
    parser.add_argument("--prompt-idx", type=int, default=0,
                        help="Which prompt to analyze (default: 0)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output plot path")

    args = parser.parse_args()

    if args.output is None:
        args.output = f"analysis/layer_emergence_{args.trait}.png"

    layers = [int(l) for l in args.layers.split(',')]

    print(f"Analyzing layers: {layers}")
    print(f"Prompt index: {args.prompt_idx}")

    # Analyze
    layer_stats = analyze_layer_emergence(
        args.experiment,
        args.trait,
        layers=layers,
        prompt_idx=args.prompt_idx
    )

    # Plot
    if layer_stats:
        plot_emergence(layer_stats, args.trait, args.output)
    else:
        print("\n❌ No layer data found")

if __name__ == "__main__":
    main()
