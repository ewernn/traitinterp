"""
Correlate perplexity with activation smoothness.

Usage:
    python scripts/correlate_metrics.py --experiment prefill-dynamics
"""

import argparse
import json
from pathlib import Path
import numpy as np
from scipy import stats

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="prefill-dynamics")
    args = parser.parse_args()

    analysis_dir = Path(f"experiments/{args.experiment}/analysis")

    # Load perplexity
    with open(analysis_dir / "perplexity.json") as f:
        ppl_data = json.load(f)

    # Load activation metrics
    with open(analysis_dir / "activation_metrics.json") as f:
        act_data = json.load(f)

    # Build lookup
    ppl_by_id = {r['id']: r for r in ppl_data['results']}

    # Collect paired data
    human_ce = []
    model_ce = []
    human_smooth = []
    model_smooth = []

    layers = list(act_data['samples'][0]['human'].keys())

    for sample in act_data['samples']:
        sid = sample['id']
        ppl = ppl_by_id[sid]

        human_ce.append(ppl['human_ce'])
        model_ce.append(ppl['model_ce'])

        # Average smoothness across layers
        human_smooth.append(np.mean([sample['human'][l]['smoothness'] for l in layers]))
        model_smooth.append(np.mean([sample['model'][l]['smoothness'] for l in layers]))

    # Correlation: across all samples (human + model pooled)
    all_ce = human_ce + model_ce
    all_smooth = human_smooth + model_smooth

    r_pooled, p_pooled = stats.pearsonr(all_ce, all_smooth)

    # Correlation: within-condition
    r_human, p_human = stats.pearsonr(human_ce, human_smooth)
    r_model, p_model = stats.pearsonr(model_ce, model_smooth)

    # Correlation: differences
    ce_diff = np.array(human_ce) - np.array(model_ce)
    smooth_diff = np.array(human_smooth) - np.array(model_smooth)
    r_diff, p_diff = stats.pearsonr(ce_diff, smooth_diff)

    results = {
        'pooled': {'r': r_pooled, 'p': p_pooled, 'n': len(all_ce)},
        'human_only': {'r': r_human, 'p': p_human, 'n': len(human_ce)},
        'model_only': {'r': r_model, 'p': p_model, 'n': len(model_ce)},
        'differences': {'r': r_diff, 'p': p_diff, 'n': len(ce_diff)},
    }

    # Save
    with open(analysis_dir / "correlation.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print("PERPLEXITY-SMOOTHNESS CORRELATION")
    print(f"{'='*60}")
    print(f"\nPooled (all samples): r={r_pooled:.3f}, p={p_pooled:.2e}")
    print(f"Human only: r={r_human:.3f}, p={p_human:.2e}")
    print(f"Model only: r={r_model:.3f}, p={p_model:.2e}")
    print(f"Differences (human-model): r={r_diff:.3f}, p={p_diff:.2e}")
    print(f"\nInterpretation:")
    if r_pooled > 0.3 and p_pooled < 0.05:
        print("  Strong positive correlation: higher perplexity -> less smooth (supports hypothesis)")
    elif r_pooled > 0 and p_pooled < 0.05:
        print("  Weak positive correlation: some support for hypothesis")
    elif p_pooled >= 0.05:
        print("  No significant correlation: smoothness may not reflect surprisingness")
    else:
        print("  Negative correlation: contradicts hypothesis")

    print(f"\nSaved to {analysis_dir / 'correlation.json'}")

if __name__ == "__main__":
    main()
