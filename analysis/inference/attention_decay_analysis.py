#!/usr/bin/env python3
"""
Attention Decay Analysis (Idea #5)

Analyzes how attention to early tokens decays over generation.

Method:
    - Load Tier 3 attention weights
    - For each generated token, measure attention to first token
    - Fit exponential decay: attn(t) = A * exp(-λt) + B
    - λ (lambda) is decay rate: higher = faster decay

Expected results:
    - Fast decay (high λ): Model quickly "forgets" early context
    - Slow decay (low λ): Model maintains long-range dependencies
    - Different traits may have different decay rates

Usage:
    python analysis/inference/attention_decay_analysis.py \
        --data experiments/.../layer_internal_states/prompt_0_layer16.pt
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def exponential_decay(t, A, lam, B):
    """Exponential decay function: A * exp(-λt) + B"""
    return A * np.exp(-lam * t) + B

def analyze_attention_decay(data_path):
    """Analyze attention decay from Tier 3 data."""

    # Load data
    data = torch.load(data_path)

    layer = data['layer']
    prompt_text = data['prompt']['text']
    response_text = data['response']['text']
    response_tokens = data['response']['tokens']

    # Get attention weights
    attn_weights = data['internals']['response']['attention']['attn_weights']

    print("\n" + "="*70)
    print("ATTENTION DECAY ANALYSIS")
    print("="*70)

    print(f"\nLayer: {layer}")
    print(f"Prompt: {prompt_text[:60]}...")
    print(f"Response tokens: {len(response_tokens)}")
    print(f"Attention timesteps: {len(attn_weights)}")

    if not attn_weights:
        print("\n❌ No attention weights found")
        return None

    # Analyze attention to first token over time
    # attn_weights[t] has shape [num_heads, seq_len_t, seq_len_t]

    num_heads = attn_weights[0].shape[0]
    prompt_len = attn_weights[0].shape[1] - 1  # First weight includes first generated token

    print(f"\nNumber of attention heads: {num_heads}")
    print(f"Prompt length: {prompt_len}")

    # For each timestep, get average attention to first token
    attention_to_first = []
    timesteps = []

    for t, weights in enumerate(attn_weights):
        # weights shape: [num_heads, seq_len, seq_len]
        # We want attention FROM last token TO first token
        # Last token is at index -1 in dimension 1
        # First token is at index 0 in dimension 2

        # Average across heads
        attn_to_first = weights[:, -1, 0].mean().item()  # Attention from newest token to first token
        attention_to_first.append(attn_to_first)
        timesteps.append(t)

    attention_to_first = np.array(attention_to_first)
    timesteps = np.array(timesteps)

    # Fit exponential decay
    try:
        # Initial guess: A=initial attention, λ=0.1, B=asymptotic minimum
        p0 = [attention_to_first[0], 0.1, attention_to_first[-10:].mean()]

        popt, pcov = curve_fit(
            exponential_decay,
            timesteps,
            attention_to_first,
            p0=p0,
            maxfev=10000
        )

        A_fit, lambda_fit, B_fit = popt

        # Compute R-squared
        residuals = attention_to_first - exponential_decay(timesteps, *popt)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((attention_to_first - attention_to_first.mean())**2)
        r_squared = 1 - (ss_res / ss_tot)

        print("\n" + "="*70)
        print("EXPONENTIAL DECAY FIT")
        print("="*70)
        print(f"Model: attn(t) = A * exp(-λt) + B")
        print(f"\nParameters:")
        print(f"  A (initial):      {A_fit:.6f}")
        print(f"  λ (decay rate):   {lambda_fit:.6f}")
        print(f"  B (asymptote):    {B_fit:.6f}")
        print(f"\nFit quality:")
        print(f"  R²: {r_squared:.4f}")

        # Interpretation
        half_life = np.log(2) / lambda_fit if lambda_fit > 0 else float('inf')
        print(f"\nInterpretation:")
        print(f"  Half-life: {half_life:.1f} tokens")

        if lambda_fit > 0.1:
            print("  ⚡ Fast decay - model quickly shifts focus away from early tokens")
        elif lambda_fit > 0.05:
            print("  ⏱  Moderate decay - balanced attention distribution")
        else:
            print("  🔄 Slow decay - model maintains long-range dependencies")

        fit_success = True

    except Exception as e:
        print(f"\n⚠️  Exponential fit failed: {e}")
        print("  Plotting raw attention decay only")
        popt = None
        fit_success = False

    # Also analyze attention spread over all positions
    attention_spread = []

    for t, weights in enumerate(attn_weights):
        # Compute entropy of attention distribution
        attn_dist = weights[:, -1, :].mean(0)  # Average across heads, attention from last token to all
        # Normalize to get distribution
        attn_dist = attn_dist / attn_dist.sum()
        # Compute entropy
        entropy = -(attn_dist * torch.log(attn_dist + 1e-10)).sum().item()
        attention_spread.append(entropy)

    attention_spread = np.array(attention_spread)

    print("\n" + "="*70)
    print("ATTENTION SPREAD (Entropy)")
    print("="*70)
    print(f"  Initial entropy: {attention_spread[0]:.3f}")
    print(f"  Final entropy:   {attention_spread[-1]:.3f}")
    print(f"  Change:          {attention_spread[-1] - attention_spread[0]:+.3f}")

    if attention_spread[-1] > attention_spread[0]:
        print("  → Attention becomes MORE diffuse over time")
    else:
        print("  → Attention becomes MORE focused over time")

    return {
        'timesteps': timesteps,
        'attention_to_first': attention_to_first,
        'attention_spread': attention_spread,
        'fit_params': popt if fit_success else None,
        'fit_success': fit_success
    }

def plot_decay(results, output_path):
    """Plot attention decay analysis."""

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    timesteps = results['timesteps']
    attention = results['attention_to_first']
    spread = results['attention_spread']

    # Plot 1: Attention to first token
    ax = axes[0]
    ax.plot(timesteps, attention, 'o-', linewidth=2, markersize=4, label='Observed')

    if results['fit_success']:
        A, lam, B = results['fit_params']
        fit_curve = exponential_decay(timesteps, A, lam, B)
        ax.plot(timesteps, fit_curve, 'r--', linewidth=2,
                label=f'Fit: A={A:.3f}, λ={lam:.3f}, B={B:.3f}')

    ax.set_xlabel('Generation Step', fontsize=12)
    ax.set_ylabel('Attention to First Token', fontsize=12)
    ax.set_title('Attention Decay Over Time', fontsize=14)
    ax.legend()
    ax.grid(alpha=0.3)

    # Plot 2: Attention spread (entropy)
    ax = axes[1]
    ax.plot(timesteps, spread, 'o-', linewidth=2, markersize=4, color='green')
    ax.set_xlabel('Generation Step', fontsize=12)
    ax.set_ylabel('Attention Entropy', fontsize=12)
    ax.set_title('Attention Distribution Spread', fontsize=14)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Plot saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Analyze attention decay")
    parser.add_argument("--data", type=str, required=True,
                        help="Path to Tier 3 .pt file")
    parser.add_argument("--output", type=str, default="attention_decay.png",
                        help="Output plot path")

    args = parser.parse_args()

    # Analyze
    results = analyze_attention_decay(args.data)

    # Plot
    if results:
        plot_decay(results, args.output)

if __name__ == "__main__":
    main()
