#!/usr/bin/env python3
"""
Test whether DeltaNet vs Attention layer type creates periodic patterns in detection profiles.

If DeltaNet layers systematically differ from Attention layers, the d curve should show
periodicity with period 4 (matching the 3:1 DeltaNet:Attention ratio).
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict

RESULTS_DIR = Path(__file__).parent


def main():
    data = json.load(open(RESULTS_DIR / "all_traits_eval.json"))

    attention_layers = {3, 7, 11, 15, 19, 23, 27, 31}

    for model_name in ['qwen3.5-9b', 'llama-3.1-8b-instruct']:
        results = data.get(model_name, [])
        if not results:
            continue

        by_trait = defaultdict(list)
        for r in results:
            by_trait[r['trait']].append(r)

        print(f"\n{'=' * 70}")
        print(f"  {model_name.upper()} — DeltaNet Periodicity Analysis")
        print(f"{'=' * 70}")

        # For each trait, compare d at attention vs DeltaNet layers
        attn_vs_delta = []

        for trait in sorted(by_trait.keys()):
            trait_results = sorted(by_trait[trait], key=lambda x: x['layer'])
            short = trait.split('/')[-1]

            # Compute d at attention vs deltanet layers
            attn_d = [r['val_effect_size'] for r in trait_results if r['layer'] in attention_layers]
            delta_d = [r['val_effect_size'] for r in trait_results if r['layer'] not in attention_layers]

            if attn_d and delta_d:
                ratio = np.mean(attn_d) / np.mean(delta_d) if np.mean(delta_d) > 0 else float('inf')
                attn_vs_delta.append({
                    'trait': short,
                    'attn_mean_d': np.mean(attn_d),
                    'delta_mean_d': np.mean(delta_d),
                    'ratio': ratio,
                })

            # Check for period-4 autocorrelation in d
            ds = [r['val_effect_size'] for r in trait_results]
            if len(ds) >= 8:
                # Autocorrelation at lag 4
                mean_d = np.mean(ds)
                diffs = [d - mean_d for d in ds]
                var = np.var(ds)
                if var > 0:
                    autocorr_4 = np.mean([diffs[i] * diffs[i+4] for i in range(len(diffs)-4)]) / var
                else:
                    autocorr_4 = 0

        if model_name == 'qwen3.5-9b':
            print(f"\n  {'Trait':<35} {'Attn d':<10} {'DeltaNet d':<12} {'Ratio'}")
            print(f"  {'-'*65}")
            for item in sorted(attn_vs_delta, key=lambda x: x['ratio'], reverse=True):
                print(f"  {item['trait']:<35} {item['attn_mean_d']:<10.2f} {item['delta_mean_d']:<12.2f} {item['ratio']:.2f}")

            # Overall
            all_attn = [x['attn_mean_d'] for x in attn_vs_delta]
            all_delta = [x['delta_mean_d'] for x in attn_vs_delta]
            print(f"\n  Overall: Attention avg={np.mean(all_attn):.2f}, DeltaNet avg={np.mean(all_delta):.2f}")
            print(f"  Ratio: {np.mean(all_attn)/np.mean(all_delta):.3f}")

            # Paired test (sign test)
            n_attn_higher = sum(1 for a, d in zip(all_attn, all_delta) if a > d)
            print(f"  Attention higher: {n_attn_higher}/{len(all_attn)} traits")
        else:
            print(f"\n  (Llama has no DeltaNet layers — control comparison)")
            print(f"  {'Trait':<35} {'\"Attn\" d':<10} {'\"Delta\" d':<12} {'Ratio'}")
            print(f"  {'-'*65}")
            for item in sorted(attn_vs_delta, key=lambda x: x['ratio'], reverse=True)[:10]:
                print(f"  {item['trait']:<35} {item['attn_mean_d']:<10.2f} {item['delta_mean_d']:<12.2f} {item['ratio']:.2f}")
            all_attn = [x['attn_mean_d'] for x in attn_vs_delta]
            all_delta = [x['delta_mean_d'] for x in attn_vs_delta]
            if all_delta:
                print(f"\n  Overall: Position-matched avg={np.mean(all_attn):.2f}, Other avg={np.mean(all_delta):.2f}")
                print(f"  Ratio: {np.mean(all_attn)/np.mean(all_delta):.3f}")


if __name__ == "__main__":
    main()
