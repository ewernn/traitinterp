#!/usr/bin/env python3
"""
Aggregate extraction evaluation metrics into a single JSON for the dashboard.

Reads val metrics (val_accuracy, val_effect_size, polarity_correct) from vector
metadata files — these are computed during extraction when a val split exists.
Falls back to loading saved activation files for legacy experiments that have
them but lack metadata metrics.

Input:
    - experiments/{experiment}/extraction/{trait}/{model_variant}/vectors/{position}/{component}/{method}/metadata.json

Output:
    - experiments/{experiment}/extraction/extraction_evaluation.json

Usage:
    python analysis/vectors/extraction_evaluation.py --experiment my_experiment
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import json
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

from utils.paths import (
    get as get_path,
    get_activation_dir,
    get_vector_metadata_path,
    list_methods,
    list_layers,
    get_model_variant,
)
from utils.layers import parse_layers


def evaluate_from_metadata(
    experiment: str,
    trait: str,
    model_variant: str,
    method: str,
    component: str = "residual",
    position: str = "response[:]",
) -> List[Dict]:
    """Read val metrics from vector metadata. Returns list of per-layer results."""
    meta_path = get_vector_metadata_path(experiment, trait, method, model_variant, component, position)
    if not meta_path.exists():
        return []

    with open(meta_path) as f:
        meta = json.load(f)

    results = []
    for layer_str, layer_info in meta.get('layers', {}).items():
        # Skip layers without val metrics
        if 'val_accuracy' not in layer_info:
            continue
        results.append({
            'trait': trait,
            'method': method,
            'layer': int(layer_str),
            'component': component,
            'position': position,
            'val_accuracy': layer_info['val_accuracy'],
            'val_effect_size': layer_info['val_effect_size'],
            'polarity_correct': layer_info['polarity_correct'],
            'train_acc': layer_info.get('train_acc'),
        })

    return results


def evaluate_from_activations(
    experiment: str,
    trait: str,
    model_variant: str,
    method: str,
    layers: List[int],
    component: str = "residual",
    position: str = "response[:]",
) -> List[Dict]:
    """Legacy path: load saved activation files and compute val + optional ood metrics."""
    from utils.vectors import load_vector_with_baseline
    from utils.load_activations import load_val_activations, load_ood_activations
    from core import batch_cosine_similarity, accuracy, effect_size, polarity_correct

    def _score(pos_acts, neg_acts, vector):
        """Return (accuracy, effect_size, polarity_correct) for a pos/neg activation pair,
        or None if either split is empty."""
        if pos_acts is None or neg_acts is None or pos_acts.numel() == 0 or neg_acts.numel() == 0:
            return None
        p = batch_cosine_similarity(pos_acts, vector)
        n = batch_cosine_similarity(neg_acts, vector)
        return float(accuracy(p, n)), float(effect_size(p, n)), bool(polarity_correct(p, n))

    results = []
    for layer in layers:
        try:
            vector, _, _ = load_vector_with_baseline(
                experiment, trait, method, layer, model_variant, component, position
            )
        except FileNotFoundError:
            continue

        try:
            val_pos, val_neg = load_val_activations(
                experiment, trait, model_variant, layer, component, position
            )
        except FileNotFoundError:
            continue

        val_score = _score(val_pos, val_neg, vector)
        if val_score is None:
            continue
        row = {
            'trait': trait, 'method': method, 'layer': layer,
            'component': component, 'position': position,
            'val_accuracy': val_score[0],
            'val_effect_size': val_score[1],
            'polarity_correct': val_score[2],
        }

        # OOD validation (optional) — populated only when ood_positive/negative
        # activations are on disk. Enables the OOD tier in select_vector's
        # hierarchy and the extraction view's OOD badge.
        try:
            ood_pos, ood_neg = load_ood_activations(
                experiment, trait, model_variant, layer, component, position
            )
            ood_score = _score(ood_pos, ood_neg, vector)
            if ood_score is not None:
                row['ood_accuracy'] = ood_score[0]
                row['ood_effect_size'] = ood_score[1]
                row['ood_polarity_correct'] = ood_score[2]
        except FileNotFoundError:
            pass

        results.append(row)

    return results


def main(
    experiment: str,
    model_variant: str = None,
    methods: str = "mean_diff,probe,gradient",
    layers: str = None,
    component: str = "residual",
    position: str = "response[:]",
    verbose: bool = False,
):
    """
    Aggregate extraction evaluation metrics into dashboard JSON.

    Reads val metrics from vector metadata (computed during extraction).
    Falls back to loading activation files for legacy experiments.

    Args:
        experiment: Experiment name
        model_variant: Model variant (default: from experiment defaults.extraction)
        methods: Comma-separated methods to evaluate
        layers: Comma-separated layers (default: all available)
        component: Component to evaluate (residual, attn_out, mlp_out)
        position: Token position (default: response[:])
        verbose: Print detailed per-method/layer analysis
    """
    variant = get_model_variant(experiment, model_variant, mode="extraction")
    model_variant = variant.name

    exp_dir = get_path('extraction.base', experiment=experiment)
    methods_list = methods.split(",")

    # Discover traits that have vector metadata
    traits = []
    for category_dir in exp_dir.iterdir():
        if not category_dir.is_dir():
            continue
        for trait_dir in category_dir.iterdir():
            if not trait_dir.is_dir():
                continue
            trait = f"{category_dir.name}/{trait_dir.name}"
            # Check if any method has vectors for this trait
            available = list_methods(experiment, trait, model_variant, component, position)
            if available:
                traits.append(trait)

    if not traits:
        print(f"No traits with vectors at {position}/{component}")
        return

    # Determine layers
    if layers:
        layers_list = parse_layers(layers, n_layers=1000)
    else:
        available_methods = list_methods(experiment, traits[0], model_variant, component, position)
        if available_methods:
            layers_list = list_layers(experiment, traits[0], available_methods[0], model_variant, component, position)
        else:
            layers_list = list(range(32))

    print(f"Found {len(traits)} traits at {position}/{component}")

    # Collect results: try metadata first, fall back to activations
    all_results = []
    n_from_metadata = 0
    n_from_activations = 0

    for trait in traits:
        for method in methods_list:
            # Try metadata first (no GPU, no activation files)
            results = evaluate_from_metadata(
                experiment, trait, model_variant, method, component, position
            )
            if results:
                # Filter to requested layers if specified
                if layers:
                    results = [r for r in results if r['layer'] in layers_list]
                all_results.extend(results)
                n_from_metadata += len(results)
                continue

            # Fall back to loading activation files (legacy experiments)
            results = evaluate_from_activations(
                experiment, trait, model_variant, method, layers_list, component, position
            )
            all_results.extend(results)
            n_from_activations += len(results)

    if not all_results:
        print("No results — vectors may lack val metrics (re-extract with val_split > 0)")
        return

    source = []
    if n_from_metadata:
        source.append(f"{n_from_metadata} from metadata")
    if n_from_activations:
        source.append(f"{n_from_activations} from activations")
    print(f"Collected {len(all_results)} results ({', '.join(source)})")

    # Compute combined_score: (accuracy + norm_effect) / 2 * polarity
    trait_max_effect = {}
    for r in all_results:
        t = r['trait']
        trait_max_effect[t] = max(trait_max_effect.get(t, 0), r['val_effect_size'])

    for r in all_results:
        max_eff = trait_max_effect[r['trait']] or 1
        norm_effect = r['val_effect_size'] / max_eff
        polarity_mult = 1.0 if r['polarity_correct'] else 0.0
        r['combined_score'] = ((r['val_accuracy'] + norm_effect) / 2) * polarity_mult

    # Load activation norms from activation metadata (if available)
    activation_norms = {}
    act_dir = get_activation_dir(experiment, traits[0], model_variant, component, position)
    act_meta_path = act_dir / "metadata.json"
    if act_meta_path.exists():
        with open(act_meta_path) as f:
            act_meta = json.load(f)
        if 'activation_norms' in act_meta:
            activation_norms = {component: act_meta['activation_norms']}

    # Print summary: best per trait
    by_trait = defaultdict(list)
    for r in all_results:
        by_trait[r['trait']].append(r)

    print(f"\n{'Trait':<40} {'Method':<10} {'Layer':<6} {'Acc':<6} {'d':<6}")
    print("-" * 70)
    for trait in sorted(by_trait.keys()):
        best = max(by_trait[trait], key=lambda x: x['combined_score'])
        short_trait = trait.split('/')[-1][:38]
        print(f"{short_trait:<40} {best['method']:<10} {best['layer']:<6} {best['val_accuracy']:.1%}  {best['val_effect_size']:.2f}")

    if verbose:
        print(f"\n{'Method':<12} {'Mean Acc':<10} {'Mean d':<10}")
        print("-" * 35)
        method_stats = defaultdict(list)
        for r in all_results:
            method_stats[r['method']].append(r)
        for method in methods_list:
            if method in method_stats:
                accs = [r['val_accuracy'] for r in method_stats[method]]
                effs = [r['val_effect_size'] for r in method_stats[method]]
                print(f"{method:<12} {sum(accs)/len(accs):.1%}      {sum(effs)/len(effs):.2f}")

    # Save
    output_path = get_path('extraction_eval.evaluation', experiment=experiment)
    output = {
        'model_variant': model_variant,
        'component': component,
        'position': position,
        'activation_norms': activation_norms,
        'all_results': all_results,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    import fire
    fire.Fire(main)
