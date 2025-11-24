#!/usr/bin/env python3
"""
Commitment Point Detection (Idea #1)

Finds when trait scores "lock in" using sliding window variance.
Commitment point = first token where variance drops below threshold.

Method:
    window_var = var(scores[t-5:t+5])
    commitment = first t where window_var < threshold

Usage:
    python analysis/inference/commitment_point_detection.py \
        --data experiments/.../inference/residual_stream_activations/prompt_0.json
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import json
import argparse
import numpy as np
import matplotlib.pyplot as plt

def sliding_window_variance(scores, window_size=10):
    """
    Compute variance in sliding window.

    Args:
        scores: [n_tokens] array of trait projections
        window_size: size of window (default 10 tokens)

    Returns:
        variances: [n_tokens] variance at each position
    """
    n = len(scores)
    half_window = window_size // 2
    variances = []

    for i in range(n):
        start = max(0, i - half_window)
        end = min(n, i + half_window + 1)
        window = scores[start:end]
        variances.append(np.var(window))

    return np.array(variances)

def detect_commitment_point(scores, threshold=0.01, window_size=10):
    """
    Find commitment point where variance drops below threshold.

    Args:
        scores: [n_tokens] trait projections
        threshold: variance threshold for commitment
        window_size: sliding window size

    Returns:
        commitment_idx: token index where commitment occurs (-1 if never)
        variances: variance profile
    """
    variances = sliding_window_variance(scores, window_size)

    # Find first point where variance < threshold
    commitment_idx = -1
    for i, var in enumerate(variances):
        if var < threshold:
            commitment_idx = i
            break

    return commitment_idx, variances

def analyze_commitment(data_path, threshold=0.01, window_size=10):
    """Analyze commitment point from Tier 2 data."""

    # Load data
    with open(data_path) as f:
        data = json.load(f)

    trait_name = data['metadata']['trait_display_name']
    prompt_text = data['prompt']['text']
    response_text = data['response']['text']
    response_tokens = data['response']['tokens']

    # Get response projections (averaged across 3 sublayers)
    response_proj = data['projections']['response']  # [n_tokens, n_layers, 3]

    # Average over sublayers and pick layer 16
    layer_idx = 16
    if layer_idx >= len(response_proj[0]):
        layer_idx = len(response_proj[0]) - 1

    scores = []
    for token_proj in response_proj:
        # Average over 3 sublayers
        layer_score = (token_proj[layer_idx][0] +
                       token_proj[layer_idx][1] +
                       token_proj[layer_idx][2]) / 3
        scores.append(layer_score)

    scores = np.array(scores)

    # Detect commitment
    commitment_idx, variances = detect_commitment_point(scores, threshold, window_size)

    # Analysis
    print("\n" + "="*60)
    print(f"COMMITMENT POINT ANALYSIS: {trait_name}")
    print("="*60)

    print(f"\nPrompt: {prompt_text}")
    print(f"Response: {response_text[:100]}...")
    print(f"Response tokens: {len(response_tokens)}")

    print(f"\nSettings:")
    print(f"  Variance threshold: {threshold}")
    print(f"  Window size: {window_size}")

    if commitment_idx == -1:
        print(f"\n⚠️  NO COMMITMENT POINT FOUND")
        print(f"  Trait score never stabilized (variance always > {threshold})")
        print(f"  Max variance: {variances.max():.4f}")
        print(f"  Min variance: {variances.min():.4f}")
    else:
        commitment_token = response_tokens[commitment_idx]
        print(f"\n✅ COMMITMENT POINT DETECTED")
        print(f"  Token index: {commitment_idx}/{len(response_tokens)}")
        print(f"  Token: '{commitment_token}'")
        print(f"  Variance at commitment: {variances[commitment_idx]:.4f}")
        print(f"  Score at commitment: {scores[commitment_idx]:.3f}")

        # Post-commitment stability
        post_commitment_var = variances[commitment_idx:].mean()
        pre_commitment_var = variances[:commitment_idx].mean() if commitment_idx > 0 else variances.mean()

        print(f"\n  Pre-commitment variance: {pre_commitment_var:.4f}")
        print(f"  Post-commitment variance: {post_commitment_var:.4f}")
        print(f"  Reduction: {(1 - post_commitment_var/pre_commitment_var)*100:.1f}%")

    return scores, variances, commitment_idx, response_tokens

def plot_commitment(scores, variances, commitment_idx, tokens, threshold, output_path):
    """Plot trait score and variance with commitment point marked."""

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # Plot trait scores
    ax1.plot(scores, 'b-', linewidth=2, label='Trait Score')
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    if commitment_idx != -1:
        ax1.axvline(x=commitment_idx, color='red', linestyle='--',
                    linewidth=2, label=f'Commitment (t={commitment_idx})')

    ax1.set_ylabel('Trait Projection', fontsize=12)
    ax1.set_title('Trait Score Over Time', fontsize=14)
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Plot variance
    ax2.plot(variances, 'g-', linewidth=2, label='Sliding Window Variance')
    ax2.axhline(y=threshold, color='orange', linestyle='--',
                linewidth=2, label=f'Threshold ({threshold})')

    if commitment_idx != -1:
        ax2.axvline(x=commitment_idx, color='red', linestyle='--',
                    linewidth=2, label=f'Commitment (t={commitment_idx})')
        ax2.plot(commitment_idx, variances[commitment_idx], 'ro', markersize=10)

    ax2.set_xlabel('Token Index', fontsize=12)
    ax2.set_ylabel('Variance', fontsize=12)
    ax2.set_title('Sliding Window Variance', fontsize=14)
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Plot saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Detect commitment points")
    parser.add_argument("--data", type=str, required=True,
                        help="Path to Tier 2 JSON file")
    parser.add_argument("--threshold", type=float, default=0.01,
                        help="Variance threshold for commitment")
    parser.add_argument("--window-size", type=int, default=10,
                        help="Sliding window size")
    parser.add_argument("--output", type=str, default="commitment_point.png",
                        help="Output plot path")

    args = parser.parse_args()

    # Analyze
    scores, variances, commit_idx, tokens = analyze_commitment(
        args.data,
        threshold=args.threshold,
        window_size=args.window_size
    )

    # Plot
    plot_commitment(scores, variances, commit_idx, tokens,
                    args.threshold, args.output)

if __name__ == "__main__":
    main()
