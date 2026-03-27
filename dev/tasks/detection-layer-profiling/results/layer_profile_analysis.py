#!/usr/bin/env python3
"""
Analyze per-layer detection profiles across models and traits.

Input:
    - results/qwen_extraction_eval.json
    - results/llama_extraction_eval.json
    - experiments/starter/steering/starter_traits/*/results.jsonl

Output:
    - results/detection_vs_steering.json (comparison table)
    - Console: full per-layer analysis

Usage:
    python dev/tasks/detection-layer-profiling/results/layer_profile_analysis.py
"""

import sys
import json
from pathlib import Path
from collections import defaultdict

SCRIPT_DIR = Path(__file__).parent
ROOT = Path("/home/dev/trait-interp")
sys.path.insert(0, str(ROOT))


def load_eval_results(eval_path):
    """Load per-layer evaluation results from a saved eval file."""
    if not eval_path.exists():
        print(f"WARNING: {eval_path} not found")
        return []
    data = json.load(open(eval_path))
    return data.get('all_results', data.get('results', []))


def load_steering_results(trait, model_variant):
    """Load per-layer steering results for a trait."""
    for pos in ['response_all', 'prompt_-1']:
        results_path = ROOT / f"experiments/starter/steering/starter_traits/{trait}/{model_variant}/{pos}/steering/results.jsonl"
        if results_path.exists():
            baseline = None
            layer_results = {}
            for line in open(results_path):
                r = json.loads(line)
                if r['type'] == 'baseline':
                    baseline = r['result']['trait_mean']
                elif r['type'] == 'run':
                    layer = r['config']['vectors'][0]['layer']
                    delta = r['result']['trait_mean'] - (baseline or 0)
                    coherence = r['result']['coherence_mean']
                    if layer not in layer_results or delta > layer_results[layer]['delta']:
                        layer_results[layer] = {
                            'delta': delta,
                            'steered_score': r['result']['trait_mean'],
                            'coherence': coherence,
                            'coefficient': r['config']['vectors'][0]['weight'],
                        }
            return baseline, layer_results
    return None, {}


def classify_profile_shape(results):
    """Classify the layer-accuracy curve shape."""
    n_layers = max(r['layer'] for r in results) + 1
    peak_layer = max(results, key=lambda x: x['val_effect_size'])['layer']
    relative_peak = peak_layer / n_layers

    # Find onset (first layer ≥ 80% accuracy)
    onset = None
    for r in sorted(results, key=lambda x: x['layer']):
        if r['val_accuracy'] >= 0.8:
            onset = r['layer']
            break

    if relative_peak < 0.35:
        return "early-peaking"
    elif relative_peak < 0.65:
        return "mid-peaking"
    else:
        return "late-peaking"


def main():
    models = {
        'qwen3.5-9b': SCRIPT_DIR / 'qwen_extraction_eval.json',
        'llama-3.1-8b-instruct': SCRIPT_DIR / 'llama_extraction_eval.json',
    }

    all_comparisons = []

    for model_name, eval_path in models.items():
        results = load_eval_results(eval_path)
        if not results:
            continue

        # Group by trait
        by_trait = defaultdict(list)
        for r in results:
            trait = r['trait'].split('/')[-1]
            by_trait[trait].append(r)

        print(f"\n{'=' * 70}")
        print(f"  {model_name.upper()}")
        print(f"{'=' * 70}")

        for trait in sorted(by_trait.keys()):
            trait_results = sorted(by_trait[trait], key=lambda x: x['layer'])

            # Detection metrics
            best_by_d = max(trait_results, key=lambda x: x['val_effect_size'])
            best_by_acc = max(trait_results, key=lambda x: (x['val_accuracy'], x['val_effect_size']))

            # Profile shape
            shape = classify_profile_shape(trait_results)

            # Onset layer
            onset = None
            for r in trait_results:
                if r['val_accuracy'] >= 0.8:
                    onset = r['layer']
                    break

            # Plateau (layers within 20% of peak d)
            peak_d = best_by_d['val_effect_size']
            plateau_layers = [r['layer'] for r in trait_results
                            if peak_d > 0 and r['val_effect_size'] >= 0.8 * peak_d]

            # Steering
            baseline, steering = load_steering_results(trait, model_name)
            best_steer_layer = max(steering, key=lambda l: steering[l]['delta']) if steering else None
            best_steer_delta = steering[best_steer_layer]['delta'] if best_steer_layer is not None else None
            gap = (best_by_d['layer'] - best_steer_layer) if best_steer_layer is not None else None

            print(f"\n  {trait} [{shape}]")
            print(f"    Detection peak (d): L{best_by_d['layer']} (d={best_by_d['val_effect_size']:.2f}, acc={best_by_d['val_accuracy']:.1%})")
            if best_by_d['layer'] != best_by_acc['layer']:
                print(f"    Detection peak (acc): L{best_by_acc['layer']} (acc={best_by_acc['val_accuracy']:.1%}, d={best_by_acc['val_effect_size']:.2f})")
            if onset is not None:
                print(f"    Onset (≥80% acc): L{onset}")
            if plateau_layers:
                print(f"    Plateau (≥80% of peak d): L{min(plateau_layers)}-L{max(plateau_layers)}")
            if best_steer_layer is not None:
                print(f"    Steering peak: L{best_steer_layer} (delta=+{best_steer_delta:.1f})")
                print(f"    Gap (detection - steering): +{gap}")

            # Compact per-layer table
            print(f"    {'L':<4} {'acc':<6} {'d':<8} {'steerΔ':<8}")
            for r in trait_results:
                l = r['layer']
                s = f"+{steering[l]['delta']:.1f}" if l in steering else ""
                print(f"    {l:<4} {r['val_accuracy']:.0%}   {r['val_effect_size']:<8.2f} {s}")

            all_comparisons.append({
                'model': model_name,
                'trait': trait,
                'shape': shape,
                'detection_peak_layer': best_by_d['layer'],
                'detection_peak_d': best_by_d['val_effect_size'],
                'detection_peak_acc': best_by_d['val_accuracy'],
                'onset_layer': onset,
                'plateau': [min(plateau_layers), max(plateau_layers)] if plateau_layers else None,
                'steering_peak_layer': best_steer_layer,
                'steering_peak_delta': best_steer_delta,
                'gap': gap,
            })

    # Summary
    print(f"\n{'=' * 90}")
    print("SUMMARY — Detection vs Steering Layer Gap")
    print(f"{'=' * 90}")
    print(f"{'Model':<28} {'Trait':<20} {'Shape':<14} {'Det(d)':<8} {'Steer':<8} {'Gap':<6} {'d':<8} {'Δ':<8}")
    print("-" * 90)

    for c in sorted(all_comparisons, key=lambda x: (x['model'], x.get('gap') or 999)):
        gap_str = f"+{c['gap']}" if c['gap'] is not None else "—"
        steer_str = f"L{c['steering_peak_layer']}" if c['steering_peak_layer'] is not None else "—"
        delta_str = f"+{c['steering_peak_delta']:.1f}" if c['steering_peak_delta'] is not None else "—"
        print(f"{c['model']:<28} {c['trait']:<20} {c['shape']:<14} L{c['detection_peak_layer']:<6} {steer_str:<8} {gap_str:<6} {c['detection_peak_d']:<8.2f} {delta_str}")

    # Gap statistics
    gaps = [c['gap'] for c in all_comparisons if c['gap'] is not None]
    if gaps:
        print(f"\nGap statistics (n={len(gaps)}):")
        print(f"  Mean:   +{sum(gaps)/len(gaps):.1f} layers")
        print(f"  Median: +{sorted(gaps)[len(gaps)//2]}")
        print(f"  Min:    +{min(gaps)}")
        print(f"  Max:    +{max(gaps)}")

        # Test +10% heuristic (32 layers → +3.2)
        heuristic_offset = 3
        errors = [abs(g - heuristic_offset) for g in gaps]
        print(f"\n  +10% heuristic (offset=+{heuristic_offset}):")
        print(f"    MAE:  {sum(errors)/len(errors):.1f} layers")
        print(f"    Within ±2: {sum(1 for e in errors if e <= 2)}/{len(errors)}")
        print(f"    Within ±5: {sum(1 for e in errors if e <= 5)}/{len(errors)}")

    # Save
    output = SCRIPT_DIR / "detection_vs_steering.json"
    json.dump(all_comparisons, open(output, 'w'), indent=2)
    print(f"\nSaved to {output}")


if __name__ == "__main__":
    main()
