"""
Validate the coefficient scaling law hypothesis.

Hypothesis: Effective steering is determined by perturbation ratio, not raw coefficient.
    perturbation_ratio = (coef × vector_norm) / activation_norm

Expected: Coherence cliff around ratio ~1.0, sweet spot ~0.5-0.8.

Input: experiments/{experiment}/steering/**/results.jsonl
Output: Scatter plots and summary statistics

Usage:
    python analysis/validate_scaling_law.py --experiment gemma-2-2b
    python analysis/validate_scaling_law.py --experiment gemma-2-2b --output analysis/outputs/scaling_law.png
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict
import torch
import numpy as np

from utils.paths import get, get_vector_path, get_steering_results_path


def load_activation_norms(experiment: str) -> dict[int, float]:
    """Load cached activation norms from extraction_evaluation.json."""
    eval_path = get('extraction_eval.evaluation', experiment=experiment)
    if not eval_path.exists():
        return {}

    with open(eval_path) as f:
        data = json.load(f)

    norms = data.get('activation_norms', {})
    return {int(k): v for k, v in norms.items()}


def load_vector_norm(experiment: str, trait: str, layer: int, method: str, position: str, model_variant: str = "base") -> float | None:
    """Load vector and return its norm."""
    vector_path = get_vector_path(experiment, trait, method, layer, model_variant, position=position)

    if not vector_path.exists():
        return None

    vector = torch.load(vector_path, weights_only=True)
    if isinstance(vector, dict):
        vector = vector.get('vector', vector.get('weights'))

    return float(vector.norm().item())


def collect_steering_data(experiment: str) -> list[dict]:
    """Collect all steering runs with computed perturbation ratios."""
    exp_base = get('experiments.base', experiment=experiment)
    steering_dir = exp_base / 'steering'

    if not steering_dir.exists():
        print(f"No steering directory found: {steering_dir}")
        return []

    # Load activation norms
    act_norms = load_activation_norms(experiment)
    if not act_norms:
        print(f"Warning: No activation norms found in extraction_evaluation.json")
        print(f"Run: python analysis/vectors/extraction_evaluation.py --experiment {experiment}")

    results = []

    # Use centralized discovery and loading
    from utils.paths import discover_steering_entries
    from analysis.steering.results import load_results

    entries = discover_steering_entries(experiment)

    for entry in entries:
        trait = entry['trait']
        position = entry['position']

        try:
            data = load_results(experiment, trait, entry['model_variant'], entry['position'], entry['prompt_set'])
        except FileNotFoundError:
            continue

        baseline = data.get('baseline', {}).get('trait_mean', 0) if data.get('baseline') else 0

        for run in data.get('runs', []):
            config = run.get('config', {})
            result = run.get('result', {})

            # VectorSpec format
            vectors = config.get('vectors', [])
            if not vectors:
                continue
            v = vectors[0]
            layer = v.get('layer')
            method = v.get('method', 'probe')
            coef = v.get('weight', 0)

            coherence = result.get('coherence_mean', 0)
            trait_mean = result.get('trait_mean', 0)
            delta = trait_mean - baseline

            # Get norms
            act_norm = act_norms.get(layer)
            vec_norm = load_vector_norm(experiment, trait, layer, method, position)

            # Compute perturbation ratio
            if act_norm and vec_norm and vec_norm > 0:
                ratio = (coef * vec_norm) / act_norm
            else:
                ratio = None

            results.append({
                'trait': trait,
                'layer': layer,
                'method': method,
                'position': position,
                'coef': coef,
                'coherence': coherence,
                'trait_mean': trait_mean,
                'delta': delta,
                'baseline': baseline,
                'vec_norm': vec_norm,
                'act_norm': act_norm,
                'ratio': ratio,
            })

    return results


def print_summary(data: list[dict]):
    """Print summary statistics."""
    # Filter to runs with valid ratios
    valid = [d for d in data if d['ratio'] is not None]

    if not valid:
        print("No valid data points with computable ratios")
        return

    print(f"\n{'='*60}")
    print(f"Scaling Law Validation Summary")
    print(f"{'='*60}")
    print(f"Total runs: {len(data)}")
    print(f"Runs with valid ratios: {len(valid)}")

    # Group by ratio buckets
    buckets = defaultdict(list)
    for d in valid:
        r = d['ratio']
        if r < 0.3:
            bucket = '0.0-0.3'
        elif r < 0.5:
            bucket = '0.3-0.5'
        elif r < 0.8:
            bucket = '0.5-0.8'
        elif r < 1.0:
            bucket = '0.8-1.0'
        elif r < 1.3:
            bucket = '1.0-1.3'
        else:
            bucket = '1.3+'
        buckets[bucket].append(d)

    print(f"\n{'Ratio Bucket':<12} {'Count':>6} {'Avg Coh':>8} {'Avg Δ':>8} {'Coh≥70%':>8}")
    print("-" * 50)

    for bucket in ['0.0-0.3', '0.3-0.5', '0.5-0.8', '0.8-1.0', '1.0-1.3', '1.3+']:
        runs = buckets.get(bucket, [])
        if runs:
            avg_coh = np.mean([r['coherence'] for r in runs])
            avg_delta = np.mean([r['delta'] for r in runs])
            pct_coherent = 100 * sum(1 for r in runs if r['coherence'] >= 70) / len(runs)
            print(f"{bucket:<12} {len(runs):>6} {avg_coh:>8.1f} {avg_delta:>+8.1f} {pct_coherent:>7.0f}%")

    # Find cliff point
    print(f"\n{'='*60}")
    print("Cliff Detection")
    print(f"{'='*60}")

    # Sort by ratio and look for coherence drop
    sorted_data = sorted(valid, key=lambda x: x['ratio'])

    # Sliding window average
    window = 20
    for i in range(0, len(sorted_data) - window, window // 2):
        chunk = sorted_data[i:i+window]
        avg_ratio = np.mean([c['ratio'] for c in chunk])
        avg_coh = np.mean([c['coherence'] for c in chunk])
        if avg_coh < 50:
            print(f"Coherence drops below 50 around ratio {avg_ratio:.2f}")
            break
    else:
        print("No clear coherence cliff detected")

    # Per-trait breakdown
    print(f"\n{'='*60}")
    print("Per-Trait Summary (runs with ratio 0.5-1.0)")
    print(f"{'='*60}")

    sweet_spot = [d for d in valid if 0.5 <= d['ratio'] <= 1.0]
    by_trait = defaultdict(list)
    for d in sweet_spot:
        by_trait[d['trait']].append(d)

    print(f"{'Trait':<25} {'Count':>6} {'Avg Coh':>8} {'Avg Δ':>8}")
    print("-" * 50)
    for trait in sorted(by_trait.keys()):
        runs = by_trait[trait]
        avg_coh = np.mean([r['coherence'] for r in runs])
        avg_delta = np.mean([r['delta'] for r in runs])
        print(f"{trait:<25} {len(runs):>6} {avg_coh:>8.1f} {avg_delta:>+8.1f}")


def plot_results(data: list[dict], output_path: Path | None = None):
    """Create visualization of ratio vs coherence/delta."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping plots")
        return

    valid = [d for d in data if d['ratio'] is not None]
    if not valid:
        print("No valid data to plot")
        return

    # Extract data
    ratios = [d['ratio'] for d in valid]
    coherences = [d['coherence'] for d in valid]
    deltas = [d['delta'] for d in valid]
    traits = [d['trait'] for d in valid]

    # Color by trait
    unique_traits = list(set(traits))
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_traits)))
    trait_to_color = {t: colors[i] for i, t in enumerate(unique_traits)}
    point_colors = [trait_to_color[t] for t in traits]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Ratio vs Coherence (scatter)
    ax1 = axes[0, 0]
    ax1.scatter(ratios, coherences, c=point_colors, alpha=0.5, s=20)
    ax1.axhline(y=70, color='red', linestyle='--', alpha=0.5, label='Coherence threshold')
    ax1.axvline(x=0.5, color='green', linestyle=':', alpha=0.5)
    ax1.axvline(x=0.8, color='green', linestyle=':', alpha=0.5)
    ax1.axvline(x=1.0, color='orange', linestyle='--', alpha=0.5, label='Ratio = 1.0')
    ax1.set_xlabel('Perturbation Ratio')
    ax1.set_ylabel('Coherence')
    ax1.set_title('Ratio vs Coherence')
    ax1.set_xlim(0, min(3, max(ratios) * 1.1))
    ax1.legend()

    # 2. Ratio vs Delta (scatter)
    ax2 = axes[0, 1]
    ax2.scatter(ratios, deltas, c=point_colors, alpha=0.5, s=20)
    ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax2.axvline(x=0.5, color='green', linestyle=':', alpha=0.5)
    ax2.axvline(x=0.8, color='green', linestyle=':', alpha=0.5)
    ax2.axvline(x=1.0, color='orange', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Perturbation Ratio')
    ax2.set_ylabel('Delta (trait_mean - baseline)')
    ax2.set_title('Ratio vs Steering Effect')
    ax2.set_xlim(0, min(3, max(ratios) * 1.1))

    # 3. Binned averages
    ax3 = axes[1, 0]
    bins = np.arange(0, 2.5, 0.1)
    bin_centers = []
    bin_coh_means = []
    bin_coh_stds = []
    bin_delta_means = []

    for i in range(len(bins) - 1):
        in_bin = [d for d in valid if bins[i] <= d['ratio'] < bins[i+1]]
        if len(in_bin) >= 3:
            bin_centers.append((bins[i] + bins[i+1]) / 2)
            bin_coh_means.append(np.mean([d['coherence'] for d in in_bin]))
            bin_coh_stds.append(np.std([d['coherence'] for d in in_bin]))
            bin_delta_means.append(np.mean([d['delta'] for d in in_bin]))

    ax3.errorbar(bin_centers, bin_coh_means, yerr=bin_coh_stds, fmt='o-', capsize=3, label='Coherence')
    ax3.axhline(y=70, color='red', linestyle='--', alpha=0.5)
    ax3.axvspan(0.5, 0.8, alpha=0.2, color='green', label='Hypothesized sweet spot')
    ax3.set_xlabel('Perturbation Ratio (binned)')
    ax3.set_ylabel('Coherence')
    ax3.set_title('Binned Average Coherence')
    ax3.legend()

    # 4. Legend / trait breakdown
    ax4 = axes[1, 1]
    ax4.axis('off')

    # Create legend
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w',
                                   markerfacecolor=trait_to_color[t], markersize=10, label=t)
                       for t in unique_traits]
    ax4.legend(handles=legend_elements, loc='center', title='Traits', ncol=2)

    # Add text summary
    sweet_spot_data = [d for d in valid if 0.5 <= d['ratio'] <= 0.8]
    breakdown_data = [d for d in valid if 1.0 <= d['ratio'] <= 1.3]

    sweet_coh = f"{np.mean([d['coherence'] for d in sweet_spot_data]):.1f}" if sweet_spot_data else 'N/A'
    sweet_delta = f"{np.mean([d['delta'] for d in sweet_spot_data]):+.1f}" if sweet_spot_data else 'N/A'
    break_coh = f"{np.mean([d['coherence'] for d in breakdown_data]):.1f}" if breakdown_data else 'N/A'
    break_delta = f"{np.mean([d['delta'] for d in breakdown_data]):+.1f}" if breakdown_data else 'N/A'

    summary_text = f"""
    Summary Statistics
    ─────────────────────
    Total runs: {len(valid)}

    Sweet spot (0.5-0.8):
      N = {len(sweet_spot_data)}
      Avg coherence: {sweet_coh}
      Avg delta: {sweet_delta}

    Breakdown zone (1.0-1.3):
      N = {len(breakdown_data)}
      Avg coherence: {break_coh}
      Avg delta: {break_delta}
    """
    ax4.text(0.5, 0.3, summary_text, transform=ax4.transAxes, fontsize=10,
             verticalalignment='center', horizontalalignment='center',
             fontfamily='monospace')

    plt.suptitle('Coefficient Scaling Law Validation', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to: {output_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Validate coefficient scaling law")
    parser.add_argument('--experiment', required=True, help='Experiment name')
    parser.add_argument('--output', type=Path, help='Output path for plot (optional)')
    args = parser.parse_args()

    print(f"Collecting steering data for {args.experiment}...")
    data = collect_steering_data(args.experiment)

    if not data:
        print("No steering data found")
        return

    print(f"Found {len(data)} steering runs")

    print_summary(data)
    plot_results(data, args.output)


if __name__ == '__main__':
    main()
