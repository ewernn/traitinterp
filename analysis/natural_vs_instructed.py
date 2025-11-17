#!/usr/bin/env python3
"""
Natural vs Instructed Comparison (Idea #4)

Tests critical hypothesis: Are we measuring the trait or instruction-following?

Method:
    - Compare trait scores on natural vs instructed prompts
    - Natural: Prompts that naturally elicit trait (e.g., "How to make bomb?" for refusal)
    - Instructed: Explicit instruction to exhibit trait (e.g., "You are refusing. What is 2+2?")

Expected results:
    - Similar scores → measuring instruction-following (BAD)
    - Natural >> Instructed → measuring genuine trait (GOOD)

Usage:
    python analysis/natural_vs_instructed.py --experiment gemma_2b_cognitive_nov20 --trait uncertainty_calibration
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import argparse
import numpy as np
import matplotlib.pyplot as plt

def load_tier2_scores(json_path, layer=16):
    """Load trait scores from Tier 2 JSON file."""
    with open(json_path) as f:
        data = json.load(f)

    # Get response projections for specified layer
    response_proj = data['projections']['response']  # [n_tokens, n_layers, 3]

    scores = []
    for token_proj in response_proj:
        # Average over 3 sublayers
        layer_score = (token_proj[layer][0] +
                       token_proj[layer][1] +
                       token_proj[layer][2]) / 3
        scores.append(layer_score)

    return np.array(scores), data

def analyze_natural_vs_instructed(experiment, trait):
    """Compare natural vs instructed prompts."""

    base_dir = Path(f"experiments/{experiment}/{trait}/inference")
    natural_dir = base_dir / "natural"
    instructed_dir = base_dir / "instructed"

    # Load natural prompts
    natural_prompts = []
    natural_scores = []

    if natural_dir.exists():
        for i in range(10):  # Check up to 10 prompts
            json_path = natural_dir / f"prompt_{i}.json"
            if not json_path.exists():
                break

            scores, data = load_tier2_scores(json_path)
            prompt_text = data['prompt']['text']

            natural_scores.append(scores)
            natural_prompts.append({
                'prompt_idx': i,
                'prompt': prompt_text,
                'response': data['response']['text'],
                'mean_score': scores.mean(),
                'max_score': scores.max(),
                'min_score': scores.min(),
                'std_score': scores.std()
            })

    # Load instructed prompts
    instructed_prompts = []
    instructed_scores = []

    if instructed_dir.exists():
        for i in range(10):  # Check up to 10 prompts
            json_path = instructed_dir / f"prompt_{i}.json"
            if not json_path.exists():
                break

            scores, data = load_tier2_scores(json_path)
            prompt_text = data['prompt']['text']

            instructed_scores.append(scores)
            instructed_prompts.append({
                'prompt_idx': i,
                'prompt': prompt_text,
                'response': data['response']['text'],
                'mean_score': scores.mean(),
                'max_score': scores.max(),
                'min_score': scores.min(),
                'std_score': scores.std()
            })

    # Combine for plotting
    all_scores = natural_scores + instructed_scores

    print("\n" + "="*70)
    print(f"NATURAL vs INSTRUCTED ANALYSIS: {trait}")
    print("="*70)

    print(f"\nNatural prompts (genuinely elicit trait): {len(natural_prompts)}")
    for meta in natural_prompts:
        print(f"  [{meta['prompt_idx']}] {meta['prompt'][:60]}...")
        print(f"      Mean score: {meta['mean_score']:+.2f} (std: {meta['std_score']:.2f})")

    print(f"\nInstructed prompts (explicit instruction): {len(instructed_prompts)}")
    for meta in instructed_prompts:
        print(f"  [{meta['prompt_idx']}] {meta['prompt'][:60]}...")
        print(f"      Mean score: {meta['mean_score']:+.2f} (std: {meta['std_score']:.2f})")

    # Statistical comparison
    if natural_prompts and instructed_prompts:
        natural_means = [m['mean_score'] for m in natural_prompts]
        instructed_means = [m['mean_score'] for m in instructed_prompts]

        natural_avg = np.mean(natural_means)
        instructed_avg = np.mean(instructed_means)

        print(f"\n" + "="*70)
        print("STATISTICAL COMPARISON")
        print("="*70)
        print(f"Natural average:     {natural_avg:+.2f}")
        print(f"Instructed average:  {instructed_avg:+.2f}")
        print(f"Difference:          {abs(natural_avg - instructed_avg):.2f}")
        print(f"Ratio:               {abs(natural_avg / (instructed_avg + 1e-6)):.2f}x")

        # Interpretation
        print(f"\n" + "="*70)
        print("INTERPRETATION")
        print("="*70)

        diff = abs(natural_avg - instructed_avg)
        if diff < 2.0:
            print("❌ INSTRUCTION CONFOUND DETECTED")
            print("   Natural and instructed scores are similar.")
            print("   → Vector likely measures instruction-following, not genuine trait")
            print("   → Consider redesigning trait definition")
        elif diff < 5.0:
            print("⚠️  WEAK SEPARATION")
            print("   Some difference, but not strong.")
            print("   → Vector captures trait but also influenced by instructions")
            print("   → Acceptable but not ideal")
        else:
            print("✅ GENUINE TRAIT DETECTED")
            print("   Natural prompts show much stronger signal than instructed.")
            print("   → Vector captures genuine behavioral trait")
            print("   → Proceed with confidence")

    return natural_prompts, instructed_prompts, all_scores

def plot_comparison(natural_prompts, instructed_prompts, natural_scores, instructed_scores, trait, output_path):
    """Plot natural vs instructed comparison."""

    # Determine grid size based on number of prompts
    n_natural = len(natural_prompts)
    n_instructed = len(instructed_prompts)
    total = n_natural + n_instructed

    if total == 0:
        print("No data to plot")
        return

    # Create grid (2 rows minimum for natural/instructed comparison)
    n_cols = min(4, max(n_natural, n_instructed))
    n_rows = 2 if (n_natural > 0 and n_instructed > 0) else 1

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 5*n_rows))
    if n_rows == 1:
        axes = np.array([axes]).flatten()
    else:
        axes = axes.flatten()

    plot_idx = 0

    # Plot natural prompts in first row
    for i, meta in enumerate(natural_prompts[:n_cols]):
        if plot_idx >= len(axes):
            break
        ax = axes[plot_idx]
        scores = natural_scores[i]

        ax.plot(scores, linewidth=2, color='blue')
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.set_title(f"[NATURAL] {meta['prompt'][:40]}...", fontsize=10)
        ax.set_xlabel('Token Index')
        ax.set_ylabel('Trait Projection')
        ax.grid(alpha=0.3)

        # Add mean line
        ax.axhline(y=meta['mean_score'], color='blue', linestyle=':',
                   linewidth=2, alpha=0.7, label=f"Mean: {meta['mean_score']:+.2f}")
        ax.legend()
        plot_idx += 1

    # Fill remaining natural slots
    while plot_idx < n_cols and n_rows > 1:
        axes[plot_idx].axis('off')
        plot_idx += 1

    # Plot instructed prompts in second row
    if n_rows > 1:
        for i, meta in enumerate(instructed_prompts[:n_cols]):
            if plot_idx >= len(axes):
                break
            ax = axes[plot_idx]
            scores = instructed_scores[i]

            ax.plot(scores, linewidth=2, color='red')
            ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            ax.set_title(f"[INSTRUCTED] {meta['prompt'][:40]}...", fontsize=10)
            ax.set_xlabel('Token Index')
            ax.set_ylabel('Trait Projection')
            ax.grid(alpha=0.3)

            # Add mean line
            ax.axhline(y=meta['mean_score'], color='red', linestyle=':',
                       linewidth=2, alpha=0.7, label=f"Mean: {meta['mean_score']:+.2f}")
            ax.legend()
            plot_idx += 1

        # Fill remaining instructed slots
        while plot_idx < len(axes):
            axes[plot_idx].axis('off')
            plot_idx += 1

    plt.suptitle(f'Natural vs Instructed: {trait}', fontsize=14, y=0.995)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Plot saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Compare natural vs instructed prompts")
    parser.add_argument("--experiment", type=str, default="gemma_2b_cognitive_nov20")
    parser.add_argument("--trait", type=str, required=True,
                        help="Trait to analyze (uncertainty_calibration or refusal)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output plot path")

    args = parser.parse_args()

    if args.output is None:
        args.output = f"analysis/natural_vs_instructed_{args.trait}.png"

    # Analyze (returns updated structure with separate score lists)
    result = analyze_natural_vs_instructed(
        args.experiment,
        args.trait
    )

    # Plot
    if result:
        natural_prompts, instructed_prompts, all_scores = result
        # Split all_scores back into natural and instructed
        n_natural = len(natural_prompts)
        natural_scores = all_scores[:n_natural]
        instructed_scores = all_scores[n_natural:]

        if natural_prompts or instructed_prompts:
            plot_comparison(natural_prompts, instructed_prompts,
                          natural_scores, instructed_scores,
                          args.trait, args.output)
        else:
            print("\n❌ No data found for comparison")
    else:
        print("\n❌ No data found for comparison")

if __name__ == "__main__":
    main()
