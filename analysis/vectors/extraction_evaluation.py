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
    list_components,
    list_methods,
    list_layers,
    get_model_variant,
    desanitize_position,
)
from utils.layers import parse_layers


def discover_positions_components(
    experiment: str, trait: str, model_variant: str
) -> List[Tuple[str, str]]:
    """Walk vectors/{position}/{component}/ to enumerate (position, component) combos.

    Returns list of (position, component) pairs that have vector data on disk.
    """
    vectors_dir = get_path(
        'extraction.vectors', experiment=experiment, trait=trait, model_variant=model_variant
    )
    if not vectors_dir.exists():
        return []
    combos = []
    for pos_dir in sorted(vectors_dir.iterdir()):
        if not pos_dir.is_dir():
            continue
        position = desanitize_position(pos_dir.name)
        for comp_dir in sorted(pos_dir.iterdir()):
            if not comp_dir.is_dir():
                continue
            combos.append((position, comp_dir.name))
    return combos


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
        row = {
            'trait': trait,
            'method': method,
            'layer': int(layer_str),
            'component': component,
            'position': position,
            'val_accuracy': layer_info['val_accuracy'],
            'val_effect_size': layer_info['val_effect_size'],
            'polarity_correct': layer_info['polarity_correct'],
            'train_acc': layer_info.get('train_acc'),
        }
        # Surface OOD validation fields if present (written by stage 4 when
        # ood_positive/negative scenarios exist for this trait).
        for k in ('ood_accuracy', 'ood_effect_size', 'ood_polarity_correct'):
            if k in layer_info:
                row[k] = layer_info[k]
        results.append(row)

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
    component: str = None,
    position: str = None,
    verbose: bool = False,
):
    """
    Aggregate extraction evaluation metrics into dashboard JSON.

    Reads val metrics from vector metadata (computed during extraction).
    Falls back to loading activation files for legacy experiments.

    By default discovers all (position, component) combos on disk and unions
    them into a single output file. Records carry their own component+position
    so the frontend can filter.

    Args:
        experiment: Experiment name
        model_variant: Model variant (default: from experiment defaults.extraction)
        methods: Comma-separated methods to evaluate
        layers: Comma-separated layers (default: all available)
        component: Restrict to one component (default: discover all)
        position: Restrict to one position (default: discover all)
        verbose: Print detailed per-method/layer analysis
    """
    variant = get_model_variant(experiment, model_variant, mode="extraction")
    model_variant = variant.name

    exp_dir = get_path('extraction.base', experiment=experiment)
    methods_list = methods.split(",")

    # Discover traits with vector data
    traits = []
    for category_dir in exp_dir.iterdir():
        if not category_dir.is_dir():
            continue
        for trait_dir in category_dir.iterdir():
            if not trait_dir.is_dir():
                continue
            trait = f"{category_dir.name}/{trait_dir.name}"
            if discover_positions_components(experiment, trait, model_variant):
                traits.append(trait)

    if not traits:
        print(f"No traits with vectors found in {exp_dir}")
        return

    # Discover (position, component) combos across all traits
    combos: set = set()
    for trait in traits:
        for pos, comp in discover_positions_components(experiment, trait, model_variant):
            if position is not None and pos != position:
                continue
            if component is not None and comp != component:
                continue
            combos.add((pos, comp))

    if not combos:
        filt = f"position={position!r}, component={component!r}"
        print(f"No (position, component) combos matched filter: {filt}")
        return

    combos_sorted = sorted(combos)
    print(f"Found {len(traits)} traits, {len(combos_sorted)} (position, component) combos")

    requested_layers = parse_layers(layers, n_layers=1000) if layers else None

    # Collect results across all (trait, method, position, component) combos
    all_results = []
    n_from_metadata = 0
    n_from_activations = 0

    for trait in traits:
        for pos, comp in combos_sorted:
            # Resolve layers per (trait, position, component) — different combos
            # may have different layer coverage (e.g., v_proj only on full-attention layers)
            if requested_layers is not None:
                trait_layers = requested_layers
            else:
                avail_methods = list_methods(experiment, trait, model_variant, comp, pos)
                if avail_methods:
                    trait_layers = list_layers(
                        experiment, trait, avail_methods[0], model_variant, comp, pos
                    )
                else:
                    trait_layers = []

            for method in methods_list:
                # Metadata path (no GPU, no activations)
                results = evaluate_from_metadata(
                    experiment, trait, model_variant, method, comp, pos
                )
                if results:
                    if requested_layers is not None:
                        results = [r for r in results if r['layer'] in trait_layers]
                    all_results.extend(results)
                    n_from_metadata += len(results)
                    continue

                # Legacy fallback: load activations and recompute
                results = evaluate_from_activations(
                    experiment, trait, model_variant, method, trait_layers, comp, pos
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

    # combined_score: normalize effect size within (trait, component) group so
    # components with different scales (residual ~5 vs k_proj ~0.5) compare fairly
    group_max_effect = defaultdict(lambda: 0.0)
    for r in all_results:
        key = (r['trait'], r['component'])
        if r['val_effect_size'] > group_max_effect[key]:
            group_max_effect[key] = r['val_effect_size']

    for r in all_results:
        max_eff = group_max_effect[(r['trait'], r['component'])] or 1
        norm_effect = r['val_effect_size'] / max_eff
        polarity_mult = 1.0 if r['polarity_correct'] else 0.0
        r['combined_score'] = ((r['val_accuracy'] + norm_effect) / 2) * polarity_mult

    # Activation norms: one block per component (positions share captured activations).
    # Read from any trait's activation metadata for that component (norms are
    # model+component dependent, not trait dependent).
    activation_norms: Dict[str, Dict] = {}
    seen_components = sorted({c for _, c in combos_sorted})
    for comp in seen_components:
        # Pick the first position that has data for this component
        first_pos = next((p for p, c in combos_sorted if c == comp), None)
        if first_pos is None:
            continue
        for trait in traits:
            act_dir = get_activation_dir(experiment, trait, model_variant, comp, first_pos)
            act_meta_path = act_dir / "metadata.json"
            if act_meta_path.exists():
                with open(act_meta_path) as f:
                    act_meta = json.load(f)
                if 'activation_norms' in act_meta:
                    activation_norms[comp] = act_meta['activation_norms']
                    break

    # Summary print: best per trait (across all components)
    by_trait = defaultdict(list)
    for r in all_results:
        by_trait[r['trait']].append(r)

    print(f"\n{'Trait':<35} {'Method':<10} {'Comp':<14} {'Layer':<6} {'Acc':<6} {'d':<6}")
    print("-" * 85)
    for trait in sorted(by_trait.keys()):
        best = max(by_trait[trait], key=lambda x: x['combined_score'])
        short_trait = trait.split('/')[-1][:33]
        print(
            f"{short_trait:<35} {best['method']:<10} {best['component']:<14} "
            f"{best['layer']:<6} {best['val_accuracy']:.1%}  {best['val_effect_size']:.2f}"
        )

    if verbose:
        print(f"\n{'Component':<16} {'Method':<10} {'Mean Acc':<10} {'Mean d':<10}")
        print("-" * 50)
        bucket = defaultdict(list)
        for r in all_results:
            bucket[(r['component'], r['method'])].append(r)
        for (comp, method) in sorted(bucket.keys()):
            rs = bucket[(comp, method)]
            accs = [r['val_accuracy'] for r in rs]
            effs = [r['val_effect_size'] for r in rs]
            print(f"{comp:<16} {method:<10} {sum(accs)/len(accs):.1%}      {sum(effs)/len(effs):.2f}")

    # Save
    output_path = get_path('extraction_eval.evaluation', experiment=experiment)
    output = {
        'model_variant': model_variant,
        'components': seen_components,
        'positions': sorted({p for p, _ in combos_sorted}),
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
