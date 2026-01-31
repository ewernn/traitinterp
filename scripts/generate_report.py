"""
Generate summary report for prefill dynamics experiment.

Usage:
    python scripts/generate_report.py --experiment prefill-dynamics
"""

import argparse
import json
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="prefill-dynamics")
    args = parser.parse_args()

    analysis_dir = Path(f"experiments/{args.experiment}/analysis")

    # Load all results
    with open(analysis_dir / "perplexity.json") as f:
        ppl = json.load(f)
    with open(analysis_dir / "activation_metrics.json") as f:
        act = json.load(f)
    with open(analysis_dir / "correlation.json") as f:
        corr = json.load(f)

    # Compute perplexity summary
    ppl_results = ppl['results']
    human_ce_mean = sum(r['human_ce'] for r in ppl_results) / len(ppl_results)
    model_ce_mean = sum(r['model_ce'] for r in ppl_results) / len(ppl_results)

    s = act['summary']['overall']

    report = f"""# Prefill Activation Dynamics: Results

## Summary

| Metric | Human Text | Model Text | Diff | Effect Size |
|--------|------------|------------|------|-------------|
| CE Loss (perplexity) | {human_ce_mean:.4f} | {model_ce_mean:.4f} | {human_ce_mean - model_ce_mean:.4f} | - |
| Smoothness (delta norm) | {s['human_smoothness_mean']:.4f} | {s['model_smoothness_mean']:.4f} | {s['smoothness_diff']:.4f} | d={s['smoothness_cohens_d']:.3f} |

**Statistical significance**: t={s['smoothness_t_stat']:.2f}, p={s['smoothness_p_value']:.2e}

## Perplexity-Smoothness Correlation

| Comparison | r | p-value | Interpretation |
|------------|---|---------|----------------|
| Pooled (all samples) | {corr['pooled']['r']:.3f} | {corr['pooled']['p']:.2e} | {'Significant' if corr['pooled']['p'] < 0.05 else 'Not significant'} |
| Differences only | {corr['differences']['r']:.3f} | {corr['differences']['p']:.2e} | {'Significant' if corr['differences']['p'] < 0.05 else 'Not significant'} |

## Interpretation

"""

    # Add interpretation
    if s['smoothness_diff'] > 0 and s['smoothness_p_value'] < 0.05:
        report += "**Model text produces smoother activation trajectories than human text.**\n\n"
        if corr['pooled']['r'] > 0 and corr['pooled']['p'] < 0.05:
            report += "This correlates with perplexity: lower perplexity (less surprising) -> smoother activations. **Hypothesis supported.**\n"
        else:
            report += "However, this does not significantly correlate with perplexity. The smoothness difference may reflect other factors (e.g., lexical diversity, style).\n"
    elif s['smoothness_p_value'] >= 0.05:
        report += "**No significant difference in smoothness between human and model text.**\n\n"
        report += "The hypothesis that model text produces smoother trajectories is not supported.\n"
    else:
        report += "**Human text produces smoother activation trajectories (unexpected).**\n\n"
        report += "This contradicts the hypothesis. Further investigation needed.\n"

    report += f"""
## Per-Layer Analysis

| Layer | Human Smooth | Model Smooth | Diff | Cohen's d | p-value |
|-------|--------------|--------------|------|-----------|---------|
"""

    for layer in sorted(act['summary']['by_layer'].keys(), key=lambda x: int(x)):
        l = act['summary']['by_layer'][layer]
        report += f"| {layer} | {l['human_smoothness_mean']:.4f} | {l['model_smoothness_mean']:.4f} | {l['smoothness_diff']:.4f} | {l['smoothness_cohens_d']:.3f} | {l['smoothness_p_value']:.2e} |\n"

    report += f"""
## Files

- Data: `experiments/{args.experiment}/data/continuations.json`
- Activations: `experiments/{args.experiment}/activations/{{human,model}}/*.pt`
- Perplexity: `experiments/{args.experiment}/analysis/perplexity.json`
- Metrics: `experiments/{args.experiment}/analysis/activation_metrics.json`
- Correlation: `experiments/{args.experiment}/analysis/correlation.json`
"""

    # Save
    report_path = Path(f"experiments/{args.experiment}/RESULTS.md")
    with open(report_path, "w") as f:
        f.write(report)

    print(report)
    print(f"\nSaved to {report_path}")

if __name__ == "__main__":
    main()
