#!/usr/bin/env python3
"""
Cross-model correlation analysis: do Qwen and Llama agree on which layers are best?

Input: results/all_traits_eval.json
Output: Console summary + results/cross_model_correlation.json
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict

RESULTS_DIR = Path(__file__).parent


def main():
    data = json.load(open(RESULTS_DIR / "all_traits_eval.json"))

    qwen = data['qwen3.5-9b']
    llama = data['llama-3.1-8b-instruct']

    # Group by trait
    qwen_by_trait = defaultdict(list)
    llama_by_trait = defaultdict(list)

    for r in qwen:
        qwen_by_trait[r['trait']].append(r)
    for r in llama:
        llama_by_trait[r['trait']].append(r)

    # Find common traits
    common_traits = sorted(set(qwen_by_trait.keys()) & set(llama_by_trait.keys()))
    print(f"Common traits: {len(common_traits)}")

    # Compare peak layers
    qwen_peaks = {}
    llama_peaks = {}
    qwen_d = {}
    llama_d = {}
    qwen_shapes = {}
    llama_shapes = {}

    for trait in common_traits:
        q_best = max(qwen_by_trait[trait], key=lambda x: x['val_effect_size'])
        l_best = max(llama_by_trait[trait], key=lambda x: x['val_effect_size'])
        qwen_peaks[trait] = q_best['layer']
        llama_peaks[trait] = l_best['layer']
        qwen_d[trait] = q_best['val_effect_size']
        llama_d[trait] = l_best['val_effect_size']

        # Shape
        for model, best, shapes in [(qwen_peaks, q_best, qwen_shapes), (llama_peaks, l_best, llama_shapes)]:
            pct = best['layer'] / 32
            shapes[trait] = "early" if pct < 0.35 else ("mid" if pct < 0.65 else "late")

    # Correlation of peak layers
    q_layers = [qwen_peaks[t] for t in common_traits]
    l_layers = [llama_peaks[t] for t in common_traits]
    correlation = np.corrcoef(q_layers, l_layers)[0, 1]

    print(f"\nPeak layer correlation (Pearson r): {correlation:.3f}")
    print(f"  (r=1 means models agree on relative layer ordering)")

    # Effect size correlation
    q_ds = [qwen_d[t] for t in common_traits]
    l_ds = [llama_d[t] for t in common_traits]
    d_corr = np.corrcoef(q_ds, l_ds)[0, 1]
    print(f"\nEffect size correlation: {d_corr:.3f}")

    # Shape agreement
    agree = sum(1 for t in common_traits if qwen_shapes[t] == llama_shapes[t])
    print(f"\nShape agreement: {agree}/{len(common_traits)} ({agree/len(common_traits):.0%})")

    # Detailed comparison
    print(f"\n{'Trait':<45} {'Qwen':<10} {'Llama':<10} {'Qwen d':<8} {'Llama d':<8} {'Agree?'}")
    print("-" * 90)

    # Categorize
    same_shape = []
    diff_shape = []
    reversed_traits = []

    for trait in common_traits:
        short = trait.split('/')[-1]
        q_shape = qwen_shapes[trait]
        l_shape = llama_shapes[trait]
        agree_str = "YES" if q_shape == l_shape else "NO"

        # Check if peak locations are "reversed" (early vs late)
        reversed_flag = ""
        if (q_shape == "early" and l_shape == "late") or (q_shape == "late" and l_shape == "early"):
            reversed_flag = " ← REVERSED"
            reversed_traits.append(trait)

        print(f"{short:<45} L{qwen_peaks[trait]:<3} {q_shape:<5}  L{llama_peaks[trait]:<3} {l_shape:<5} {qwen_d[trait]:<8.2f} {llama_d[trait]:<8.2f} {agree_str}{reversed_flag}")

        if q_shape == l_shape:
            same_shape.append(trait)
        else:
            diff_shape.append(trait)

    print(f"\nReversed traits (early↔late): {len(reversed_traits)}/{len(common_traits)}")
    for t in reversed_traits:
        print(f"  {t.split('/')[-1]}: Qwen L{qwen_peaks[t]} ({qwen_shapes[t]}) vs Llama L{llama_peaks[t]} ({llama_shapes[t]})")

    # By category
    categories = defaultdict(list)
    for t in common_traits:
        parts = t.split('/')
        if len(parts) >= 2:
            categories[parts[0] + '/' + parts[1] if len(parts) >= 3 else parts[0]].append(t)

    print(f"\n{'Category':<25} {'Agree':<8} {'Disagree':<10} {'Rate'}")
    print("-" * 50)
    for cat in sorted(categories.keys()):
        traits = categories[cat]
        ag = sum(1 for t in traits if qwen_shapes[t] == llama_shapes[t])
        print(f"{cat:<25} {ag:<8} {len(traits)-ag:<10} {ag/len(traits):.0%}")

    # Save
    output = {
        'peak_layer_correlation': float(correlation),
        'effect_size_correlation': float(d_corr),
        'shape_agreement_rate': agree / len(common_traits),
        'n_reversed': len(reversed_traits),
        'reversed_traits': [t.split('/')[-1] for t in reversed_traits],
        'per_trait': {
            t.split('/')[-1]: {
                'qwen_peak': qwen_peaks[t], 'llama_peak': llama_peaks[t],
                'qwen_d': qwen_d[t], 'llama_d': llama_d[t],
                'qwen_shape': qwen_shapes[t], 'llama_shape': llama_shapes[t],
            }
            for t in common_traits
        }
    }
    json.dump(output, open(RESULTS_DIR / "cross_model_correlation.json", 'w'), indent=2)
    print(f"\nSaved to {RESULTS_DIR / 'cross_model_correlation.json'}")


if __name__ == "__main__":
    main()
