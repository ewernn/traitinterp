#!/usr/bin/env python3
"""
Three-way comparison: detection-optimal vs steering-optimal vs inference-optimal layers.
Also annotates DeltaNet vs Attention layers for Qwen3.5-9B.

Input:
    - results/all_traits_eval.json (detection metrics per layer)
    - experiments/starter/steering/ (steering results)
    - results/inference_layer_profiles.json (inference signal quality)

Output:
    - results/three_way_comparison.json
    - Console: comprehensive comparison table
"""

import sys
import json
import numpy as np
from pathlib import Path
from collections import defaultdict

ROOT = Path("/home/dev/trait-interp")
sys.path.insert(0, str(ROOT))

RESULTS_DIR = Path(__file__).parent


def get_deltanet_layers(n_layers=32):
    """
    Qwen3.5-9B has a 3:1 DeltaNet:Attention ratio.
    Every 4th layer (0-indexed: 3, 7, 11, 15, 19, 23, 27, 31) is standard attention.
    The rest are DeltaNet (linear attention).
    """
    attention_layers = set(range(3, n_layers, 4))
    return {
        l: 'attention' if l in attention_layers else 'deltanet'
        for l in range(n_layers)
    }


def load_steering(trait, model_variant):
    """Load per-layer steering results."""
    for pos in ['response_all', 'prompt_-1']:
        path = ROOT / f"experiments/starter/steering/starter_traits/{trait}/{model_variant}/{pos}/steering/results.jsonl"
        if path.exists():
            baseline = None
            by_layer = {}
            for line in open(path):
                r = json.loads(line)
                if r['type'] == 'baseline':
                    baseline = r['result']['trait_mean']
                elif r['type'] == 'run':
                    layer = r['config']['vectors'][0]['layer']
                    delta = r['result']['trait_mean'] - (baseline or 0)
                    coh = r['result']['coherence_mean']
                    if layer not in by_layer or delta > by_layer[layer]['delta']:
                        by_layer[layer] = {'delta': delta, 'coherence': coh}
            return by_layer
    return {}


def main():
    # Load detection data
    eval_data = json.load(open(RESULTS_DIR / "all_traits_eval.json"))
    qwen_results = eval_data.get('qwen3.5-9b', [])

    # Load inference data
    inference_data = json.load(open(RESULTS_DIR / "inference_layer_profiles.json"))
    inference_stats = inference_data.get('trait_layer_stats', {})

    # DeltaNet annotation
    layer_types = get_deltanet_layers()

    # Focus on the 6 traits with all three data sources
    traits = ['sycophancy', 'evil', 'refusal', 'concealment', 'hallucination', 'golden_gate_bridge']

    print("=" * 100)
    print("THREE-WAY LAYER COMPARISON — Detection vs Steering vs Inference")
    print("Qwen3.5-9B (32 layers, 3:1 DeltaNet:Attention)")
    print("=" * 100)

    comparisons = []

    for trait in traits:
        # Detection: best layer by effect size
        det_results = [r for r in qwen_results if r['trait'].endswith(trait)]
        det_results.sort(key=lambda x: x['layer'])
        det_best = max(det_results, key=lambda x: x['val_effect_size'])

        # Steering
        steering = load_steering(trait, 'qwen3.5-9b')
        steer_best_layer = max(steering, key=lambda l: steering[l]['delta']) if steering else None

        # Inference
        inf_stats = inference_stats.get(trait, {})
        inf_best_layer = None
        if inf_stats:
            inf_best_layer = max(inf_stats.keys(), key=lambda l: inf_stats[l]['signal_quality'])
            inf_best_layer = int(inf_best_layer)

        print(f"\n{'─' * 80}")
        print(f"  {trait.upper()}")
        print(f"{'─' * 80}")
        print(f"  Detection peak:  L{det_best['layer']} (d={det_best['val_effect_size']:.2f}, acc={det_best['val_accuracy']:.0%}) [{layer_types[det_best['layer']]}]")
        if steer_best_layer is not None:
            print(f"  Steering peak:   L{steer_best_layer} (Δ=+{steering[steer_best_layer]['delta']:.1f}, coh={steering[steer_best_layer]['coherence']:.1f}) [{layer_types[steer_best_layer]}]")
        if inf_best_layer is not None:
            print(f"  Inference peak:  L{inf_best_layer} (signal={inf_stats[str(inf_best_layer)]['signal_quality']:.4f}) [{layer_types[inf_best_layer]}]")

        # Per-layer detail with DeltaNet annotation
        print(f"\n  {'L':<4} {'Type':<10} {'d':<8} {'acc':<6} {'steerΔ':<10} {'inf_sig':<10}")
        for r in det_results:
            l = r['layer']
            lt = layer_types[l]
            marker = "★" if lt == 'attention' else "·"
            s_str = f"+{steering[l]['delta']:.1f}" if l in steering else ""
            i_str = f"{inf_stats[str(l)]['signal_quality']:.4f}" if str(l) in inf_stats else ""
            print(f"  {l:<4} {lt:<10} {r['val_effect_size']:<8.2f} {r['val_accuracy']:.0%}  {s_str:<10} {i_str}")

        # DeltaNet vs Attention analysis
        det_layers = {r['layer']: r['val_effect_size'] for r in det_results}
        attn_d = [det_layers[l] for l in det_layers if layer_types[l] == 'attention']
        delta_d = [det_layers[l] for l in det_layers if layer_types[l] == 'deltanet']
        if attn_d and delta_d:
            print(f"\n  DeltaNet avg d: {np.mean(delta_d):.2f} | Attention avg d: {np.mean(attn_d):.2f}")

        comparisons.append({
            'trait': trait,
            'detection_peak': det_best['layer'],
            'detection_d': det_best['val_effect_size'],
            'detection_type': layer_types[det_best['layer']],
            'steering_peak': steer_best_layer,
            'steering_delta': steering[steer_best_layer]['delta'] if steer_best_layer else None,
            'steering_type': layer_types[steer_best_layer] if steer_best_layer else None,
            'inference_peak': inf_best_layer,
            'inference_type': layer_types[inf_best_layer] if inf_best_layer else None,
        })

    # Summary table
    print(f"\n{'=' * 100}")
    print("SUMMARY")
    print(f"{'=' * 100}")
    print(f"{'Trait':<20} {'Det Peak':<12} {'Det Type':<12} {'Steer Peak':<12} {'Steer Type':<12} {'Inf Peak':<10} {'Inf Type':<12}")
    print("-" * 90)
    for c in comparisons:
        print(f"{c['trait']:<20} L{c['detection_peak']:<10} {c['detection_type']:<12} "
              f"{'L'+str(c['steering_peak']) if c['steering_peak'] else '—':<11} "
              f"{c['steering_type'] or '—':<12} "
              f"{'L'+str(c['inference_peak']) if c['inference_peak'] else '—':<9} "
              f"{c['inference_type'] or '—'}")

    # DeltaNet bias analysis
    print(f"\n{'=' * 80}")
    print("DELTANET vs ATTENTION — Are peaks biased toward attention layers?")
    print(f"{'=' * 80}")
    for metric in ['detection', 'steering', 'inference']:
        key = f'{metric}_type'
        types = [c[key] for c in comparisons if c[key] is not None]
        attn_count = types.count('attention')
        delta_count = types.count('deltanet')
        total = len(types)
        expected_attn = total * 0.25  # 1/4 should be attention by chance
        print(f"  {metric}: {attn_count}/{total} peaks at attention layers (expected {expected_attn:.1f} by chance)")

    # Save
    json.dump(comparisons, open(RESULTS_DIR / "three_way_comparison.json", 'w'), indent=2)
    print(f"\nSaved to {RESULTS_DIR / 'three_way_comparison.json'}")


if __name__ == "__main__":
    main()
