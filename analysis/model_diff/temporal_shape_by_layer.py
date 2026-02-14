#!/usr/bin/env python3
"""
Check if temporal decomposition (always-on vs triggered) holds across layers.

Computes position-averaged delta curves for each trait × layer,
and measures early-vs-late ratio to quantify temporal shape.

Usage:
    python analysis/model_diff/temporal_shape_by_layer.py \
        --experiment audit-bench \
        --dir experiments/audit-bench/model_diff/instruct_vs_rm_lora/layer_sensitivity/rm_syco/exploitation_evals_100
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
import json
import numpy as np
from collections import defaultdict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', required=True, help='layer_sensitivity output dir')
    parser.add_argument('--traits', default='rm_hack/secondary_objective,bs/concealment',
                        help='Traits to compare')
    parser.add_argument('--window', type=int, default=20,
                        help='Smoothing window for position-averaged curves')
    args = parser.parse_args()

    base_dir = Path(args.dir) / 'per_prompt'
    traits = args.traits.split(',')

    # Load all per-prompt data
    files = sorted(base_dir.glob('*.json'))
    print(f"Loading {len(files)} prompts...")

    # Collect: {trait: {layer: [[per_token_deltas], ...]}}
    all_deltas = {t: defaultdict(list) for t in traits}

    for f in files:
        with open(f) as fh:
            data = json.load(fh)

        for trait in traits:
            if trait not in data['traits']:
                continue
            for layer_str, layer_data in data['traits'][trait]['layers'].items():
                layer = int(layer_str)
                all_deltas[trait][layer].append(layer_data['delta'])

    # Compute position-averaged curves (pad to max length)
    print(f"\n{'='*80}")
    print("TEMPORAL SHAPE BY LAYER")
    print(f"{'='*80}")

    for trait in traits:
        layers = sorted(all_deltas[trait].keys())
        print(f"\n{'─'*80}")
        print(f"  {trait}")
        print(f"{'─'*80}")

        for layer in layers:
            deltas_list = all_deltas[trait][layer]
            max_len = max(len(d) for d in deltas_list)

            # Position-averaged delta
            pos_sums = np.zeros(max_len)
            pos_counts = np.zeros(max_len)
            for d in deltas_list:
                for i, v in enumerate(d):
                    pos_sums[i] += v
                    pos_counts[i] += 1
            pos_mean = pos_sums / np.maximum(pos_counts, 1)

            # Split into quintiles (5 segments)
            n = len(pos_mean)
            quintile_size = n // 5
            quintile_means = []
            for q in range(5):
                start = q * quintile_size
                end = (q + 1) * quintile_size if q < 4 else n
                quintile_means.append(float(np.mean(pos_mean[start:end])))

            # Early (Q1) vs late (Q4-Q5) ratio
            early = quintile_means[0]
            late = np.mean(quintile_means[3:5])
            ratio = late / early if abs(early) > 0.001 else float('inf')

            # Onset detection: find first quintile where mean > 50% of peak
            peak = max(quintile_means)
            onset_q = 0
            for q, qm in enumerate(quintile_means):
                if qm > 0.5 * peak:
                    onset_q = q
                    break

            print(f"\n  L{layer:2d}  overall={np.mean(pos_mean):+.4f}")
            print(f"       quintiles: [{' | '.join(f'{q:+.4f}' for q in quintile_means)}]")
            print(f"       early/late ratio: {ratio:.2f}x  (onset: Q{onset_q+1})")

            # ASCII sparkline of the quintiles
            if peak > 0:
                bars = ''
                for qm in quintile_means:
                    bar_len = int(max(0, qm / peak * 20))
                    bars += f"  Q{quintile_means.index(qm)+1}: {'█' * bar_len}{'░' * (20 - bar_len)} {qm:+.4f}\n"
            else:
                bars = "  (all near zero)\n"

        # Cross-layer comparison: is the shape consistent?
        print(f"\n  Shape consistency across layers:")
        # For each pair of layers, correlate their position-averaged curves
        for i, l1 in enumerate(layers):
            for l2 in layers[i+1:]:
                deltas_l1 = all_deltas[trait][l1]
                deltas_l2 = all_deltas[trait][l2]

                # Use prompts present in both
                n_prompts = min(len(deltas_l1), len(deltas_l2))
                # Position-averaged for each
                max_len_both = min(
                    max(len(d) for d in deltas_l1[:n_prompts]),
                    max(len(d) for d in deltas_l2[:n_prompts])
                )

                curve1 = np.zeros(max_len_both)
                curve2 = np.zeros(max_len_both)
                counts = np.zeros(max_len_both)

                for d1, d2 in zip(deltas_l1[:n_prompts], deltas_l2[:n_prompts]):
                    n = min(len(d1), len(d2), max_len_both)
                    curve1[:n] += d1[:n]
                    curve2[:n] += d2[:n]
                    counts[:n] += 1

                curve1 /= np.maximum(counts, 1)
                curve2 /= np.maximum(counts, 1)

                if np.std(curve1) > 0 and np.std(curve2) > 0:
                    corr = float(np.corrcoef(curve1, curve2)[0, 1])
                else:
                    corr = 0.0
                print(f"    L{l1} vs L{l2}: position-curve r={corr:.4f}")


if __name__ == '__main__':
    main()
