#!/usr/bin/env python3
"""
Evaluate all extracted traits at all layers, handling nested category dirs.

Output:
    - results/all_traits_eval.json
"""

import sys
import json
import torch
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

ROOT = Path("/home/dev/trait-interp")
sys.path.insert(0, str(ROOT))

from utils.paths import get as get_path, get_activation_dir
from utils.vectors import load_vector_with_baseline
from utils.load_activations import load_val_activations
from core import batch_cosine_similarity, accuracy, effect_size, polarity_correct

RESULTS_DIR = Path(__file__).parent


def discover_traits(experiment, model_variant, component="residual", position="response[:]"):
    """Find all traits with val data, any nesting depth."""
    exp_dir = get_path('extraction.base', experiment=experiment)
    found = []
    for p in sorted(exp_dir.rglob('val_all_layers.pt')):
        rel = p.relative_to(exp_dir)
        parts = list(rel.parts)
        if model_variant in parts:
            idx = parts.index(model_variant)
            trait = '/'.join(parts[:idx])
            found.append(trait)
    return found


def evaluate_trait_layers(experiment, trait, model_variant, method="probe",
                         component="residual", position="response[:]"):
    """Evaluate a trait across all layers."""
    results = []
    for layer in range(100):  # up to 100 layers
        try:
            vector, baseline_val, bias = load_vector_with_baseline(
                experiment, trait, method, layer, model_variant, component, position)
        except (FileNotFoundError, IndexError):
            if layer > 0:
                break
            continue

        try:
            val_pos, val_neg = load_val_activations(
                experiment, trait, model_variant, layer, component, position)
        except FileNotFoundError:
            continue

        if val_pos is None or val_neg is None:
            continue

        pos_proj = batch_cosine_similarity(val_pos, vector)
        neg_proj = batch_cosine_similarity(val_neg, vector)

        results.append({
            'trait': trait,
            'layer': layer,
            'val_accuracy': accuracy(pos_proj, neg_proj),
            'val_effect_size': effect_size(pos_proj, neg_proj),
            'polarity_correct': polarity_correct(pos_proj, neg_proj),
            'n_pos': len(val_pos),
            'n_neg': len(val_neg),
        })

    return results


def main():
    experiment = "starter"
    models = {
        'qwen3.5-9b': 32,
        'llama-3.1-8b-instruct': 32,
    }

    all_results = {}

    for model_variant, n_layers in models.items():
        traits = discover_traits(experiment, model_variant)
        print(f"\n{model_variant}: {len(traits)} traits with val data")

        model_results = []
        for trait in tqdm(traits, desc=f"Evaluating {model_variant}"):
            trait_results = evaluate_trait_layers(experiment, trait, model_variant)
            model_results.extend(trait_results)

        all_results[model_variant] = model_results
        print(f"  Total: {len(model_results)} layer-results")

        # Print summary
        by_trait = defaultdict(list)
        for r in model_results:
            by_trait[r['trait']].append(r)

        print(f"\n  {'Trait':<45} {'Best Layer':<12} {'Acc':<7} {'d':<8} {'Shape'}")
        print(f"  {'-'*90}")
        for trait in sorted(by_trait.keys()):
            results = by_trait[trait]
            best = max(results, key=lambda x: x['val_effect_size'])
            # Shape classification
            peak_pct = best['layer'] / n_layers
            shape = "early" if peak_pct < 0.35 else ("mid" if peak_pct < 0.65 else "late")
            print(f"  {trait:<45} L{best['layer']:<10} {best['val_accuracy']:.0%}    {best['val_effect_size']:<8.2f} {shape}")

    # Save
    output = RESULTS_DIR / "all_traits_eval.json"
    json.dump(all_results, open(output, 'w'), indent=2)
    print(f"\nSaved to {output}")


if __name__ == "__main__":
    main()
