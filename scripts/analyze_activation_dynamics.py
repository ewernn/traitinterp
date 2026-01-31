"""
Analyze activation dynamics: smoothness, variance, magnitude.

Metrics:
- smoothness: mean L2 norm of token-to-token activation deltas (lower = smoother)
- magnitude: mean activation norm across tokens
- variance: variance of activation norms across tokens

Usage:
    python scripts/analyze_activation_dynamics.py --experiment prefill-dynamics
"""

import argparse
import json
from pathlib import Path
import torch
import numpy as np
from tqdm import tqdm
from scipy import stats

def compute_trajectory_metrics(activations: dict, layers: list) -> dict:
    """Compute smoothness, magnitude, variance for response activations."""

    metrics_by_layer = {}

    for layer in layers:
        if layer not in activations:
            continue

        # Get residual activations [n_tokens, hidden_dim]
        residual = activations[layer].get('residual')
        if residual is None:
            continue

        residual = residual.float()  # Ensure float for computation
        n_tokens = residual.shape[0]

        if n_tokens < 2:
            continue

        # Token-to-token deltas
        deltas = residual[1:] - residual[:-1]  # [n_tokens-1, hidden_dim]
        delta_norms = torch.norm(deltas, dim=1)  # [n_tokens-1]

        # Activation norms per token
        token_norms = torch.norm(residual, dim=1)  # [n_tokens]

        metrics_by_layer[layer] = {
            'smoothness': delta_norms.mean().item(),  # Lower = smoother
            'magnitude': token_norms.mean().item(),
            'variance': token_norms.var().item(),
            'n_tokens': n_tokens,
        }

    return metrics_by_layer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="prefill-dynamics")
    args = parser.parse_args()

    # Paths
    act_dir = Path(f"experiments/{args.experiment}/activations")
    human_dir = act_dir / "human"
    model_dir = act_dir / "model"

    # Get sample IDs
    sample_ids = [int(p.stem) for p in human_dir.glob("*.pt")]
    sample_ids.sort()

    print(f"Analyzing {len(sample_ids)} samples...")

    # Collect metrics
    all_metrics = []

    for sample_id in tqdm(sample_ids):
        human_data = torch.load(human_dir / f"{sample_id}.pt")
        model_data = torch.load(model_dir / f"{sample_id}.pt")

        # Get layer list from data
        layers = list(human_data['response']['activations'].keys())

        human_metrics = compute_trajectory_metrics(
            human_data['response']['activations'], layers
        )
        model_metrics = compute_trajectory_metrics(
            model_data['response']['activations'], layers
        )

        all_metrics.append({
            'id': sample_id,
            'human': human_metrics,
            'model': model_metrics,
        })

    # Aggregate by layer
    layers = list(all_metrics[0]['human'].keys())

    summary = {'by_layer': {}, 'overall': {}}

    for layer in layers:
        human_smoothness = [m['human'][layer]['smoothness'] for m in all_metrics if layer in m['human']]
        model_smoothness = [m['model'][layer]['smoothness'] for m in all_metrics if layer in m['model']]

        human_magnitude = [m['human'][layer]['magnitude'] for m in all_metrics if layer in m['human']]
        model_magnitude = [m['model'][layer]['magnitude'] for m in all_metrics if layer in m['model']]

        # Paired t-test for smoothness
        t_stat, p_value = stats.ttest_rel(human_smoothness, model_smoothness)

        # Effect size (Cohen's d for paired samples)
        diff = np.array(human_smoothness) - np.array(model_smoothness)
        cohens_d = diff.mean() / diff.std() if diff.std() > 0 else 0

        summary['by_layer'][layer] = {
            'human_smoothness_mean': np.mean(human_smoothness),
            'model_smoothness_mean': np.mean(model_smoothness),
            'smoothness_diff': np.mean(human_smoothness) - np.mean(model_smoothness),
            'smoothness_t_stat': t_stat,
            'smoothness_p_value': p_value,
            'smoothness_cohens_d': cohens_d,
            'human_magnitude_mean': np.mean(human_magnitude),
            'model_magnitude_mean': np.mean(model_magnitude),
        }

    # Overall (average across layers)
    all_human_smooth = []
    all_model_smooth = []
    for m in all_metrics:
        all_human_smooth.append(np.mean([m['human'][l]['smoothness'] for l in layers if l in m['human']]))
        all_model_smooth.append(np.mean([m['model'][l]['smoothness'] for l in layers if l in m['model']]))

    t_stat, p_value = stats.ttest_rel(all_human_smooth, all_model_smooth)
    diff = np.array(all_human_smooth) - np.array(all_model_smooth)
    cohens_d = diff.mean() / diff.std() if diff.std() > 0 else 0

    summary['overall'] = {
        'human_smoothness_mean': np.mean(all_human_smooth),
        'model_smoothness_mean': np.mean(all_model_smooth),
        'smoothness_diff': np.mean(all_human_smooth) - np.mean(all_model_smooth),
        'smoothness_t_stat': t_stat,
        'smoothness_p_value': p_value,
        'smoothness_cohens_d': cohens_d,
        'n_samples': len(sample_ids),
    }

    # Save
    output_dir = Path(f"experiments/{args.experiment}/analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save detailed metrics
    with open(output_dir / "activation_metrics.json", "w") as f:
        json.dump({'samples': all_metrics, 'summary': summary}, f, indent=2)

    # Print summary
    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"Samples: {len(sample_ids)}")
    print(f"\nOverall Smoothness (lower = smoother):")
    print(f"  Human: {summary['overall']['human_smoothness_mean']:.4f}")
    print(f"  Model: {summary['overall']['model_smoothness_mean']:.4f}")
    print(f"  Diff (H-M): {summary['overall']['smoothness_diff']:.4f}")
    print(f"  t-stat: {summary['overall']['smoothness_t_stat']:.2f}")
    print(f"  p-value: {summary['overall']['smoothness_p_value']:.2e}")
    print(f"  Cohen's d: {summary['overall']['smoothness_cohens_d']:.3f}")

    print(f"\nPer-layer (selected layers):")
    for layer in [0, 6, 12, 18, 24]:
        if layer in summary['by_layer']:
            s = summary['by_layer'][layer]
            print(f"  L{layer}: diff={s['smoothness_diff']:.4f}, d={s['smoothness_cohens_d']:.3f}, p={s['smoothness_p_value']:.2e}")

    print(f"\nSaved to {output_dir / 'activation_metrics.json'}")

if __name__ == "__main__":
    main()
