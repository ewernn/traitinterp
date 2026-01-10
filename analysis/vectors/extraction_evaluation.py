#!/usr/bin/env python3
"""
Evaluate extracted vectors on held-out validation data.

Input:
    - experiments/{experiment}/extraction/{trait}/{model_variant}/vectors/{position}/{component}/{method}/layer*.pt
    - experiments/{experiment}/extraction/{trait}/{model_variant}/activations/{position}/{component}/val_all_layers.pt

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
import fire
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm

from utils.paths import (
    get as get_path,
    get_val_activation_path,
    get_activation_metadata_path,
    get_vector_path,
    list_positions,
    list_components,
    list_methods,
    list_layers,
    get_model_variant,
)
from utils.model import load_experiment_config
from utils.vectors import load_vector_with_baseline
from core import batch_cosine_similarity, accuracy, effect_size, polarity_correct

# Cache for loaded val activations (avoids reloading for each layer)
_val_cache: Dict[str, Tuple[torch.Tensor, int]] = {}


def load_activations(
    experiment: str,
    trait: str,
    model_variant: str,
    layer: int,
    component: str = "residual",
    position: str = "response[:]",
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Load pos/neg validation activations for a layer."""
    cache_key = f"{experiment}/{trait}/{model_variant}/{component}/{position}"

    # Load and cache the full val tensor if not cached
    if cache_key not in _val_cache:
        val_path = get_val_activation_path(experiment, trait, model_variant, component, position)
        if not val_path.exists():
            return None, None

        # Load metadata to get n_val_pos
        metadata_path = get_activation_metadata_path(experiment, trait, model_variant, component, position)
        if not metadata_path.exists():
            return None, None

        with open(metadata_path) as f:
            metadata = json.load(f)
        n_val_pos = metadata.get('n_val_pos', 0)
        if n_val_pos == 0:
            return None, None

        val_acts = torch.load(val_path, weights_only=True)
        _val_cache[cache_key] = (val_acts, n_val_pos)

    val_acts, n_val_pos = _val_cache[cache_key]

    # Slice to get specific layer: val_acts is [n_examples, n_layers, hidden_dim]
    layer_acts = val_acts[:, layer, :]
    pos_acts = layer_acts[:n_val_pos]
    neg_acts = layer_acts[n_val_pos:]

    return pos_acts, neg_acts


def load_vector(
    experiment: str,
    trait: str,
    model_variant: str,
    method: str,
    layer: int,
    component: str = "residual",
    position: str = "response[:]",
) -> Optional[torch.Tensor]:
    """Load a vector file."""
    try:
        vector, _, _ = load_vector_with_baseline(experiment, trait, method, layer, model_variant, component, position)
        return vector
    except FileNotFoundError:
        return None


def evaluate_single(
    experiment: str,
    trait: str,
    model_variant: str,
    method: str,
    layer: int,
    component: str = "residual",
    position: str = "response[:]",
) -> Optional[Dict]:
    """Evaluate one vector on validation data."""
    vector = load_vector(experiment, trait, model_variant, method, layer, component, position)
    if vector is None:
        return None

    val_pos, val_neg = load_activations(experiment, trait, model_variant, layer, component, position)
    if val_pos is None:
        return None

    # Compute projections once, derive all metrics
    pos_proj = batch_cosine_similarity(val_pos, vector)
    neg_proj = batch_cosine_similarity(val_neg, vector)

    return {
        'trait': trait,
        'method': method,
        'layer': layer,
        'component': component,
        'position': position,
        'val_accuracy': accuracy(pos_proj, neg_proj),
        'val_effect_size': effect_size(pos_proj, neg_proj),
        'polarity_correct': polarity_correct(pos_proj, neg_proj),
    }


def compute_activation_norms(
    experiment: str,
    trait: str,
    model_variant: str,
    layers: List[int],
    component: str = "residual",
    position: str = "response[:]",
) -> Dict[str, float]:
    """Compute mean ||h|| per layer. Used by steering for coefficient estimation."""
    norms = {}
    for layer in layers:
        pos, neg = load_activations(experiment, trait, model_variant, layer, component, position)
        if pos is not None:
            all_acts = torch.cat([pos, neg], dim=0).float()
            norms[str(layer)] = all_acts.norm(dim=-1).mean().item()
    return norms


def main(
    experiment: str,
    model_variant: str = None,
    methods: str = "mean_diff,probe,gradient",
    layers: str = None,
    component: str = "residual",
    position: str = "response[:5]",
    verbose: bool = False,
):
    """
    Evaluate all vectors on validation data.

    Args:
        experiment: Experiment name
        model_variant: Model variant (default: from experiment defaults.extraction)
        methods: Comma-separated methods to evaluate
        layers: Comma-separated layers (default: all available)
        component: Component to evaluate (residual, attn_out, mlp_out)
        position: Token position (default: response[:5])
        verbose: Print detailed per-method/layer analysis
    """
    # Resolve model variant
    variant = get_model_variant(experiment, model_variant, mode="extraction")
    model_variant = variant['name']

    exp_dir = get_path('extraction.base', experiment=experiment)
    methods_list = methods.split(",")

    # Find traits with validation data at the specified position/component
    traits = []
    for category_dir in exp_dir.iterdir():
        if not category_dir.is_dir():
            continue
        for trait_dir in category_dir.iterdir():
            if not trait_dir.is_dir():
                continue
            trait = f"{category_dir.name}/{trait_dir.name}"
            val_path = get_val_activation_path(experiment, trait, model_variant, component, position)
            if val_path.exists():
                traits.append(trait)

    if not traits:
        print(f"No traits with validation data at {position}/{component}")
        return

    # Determine layers to evaluate
    if layers:
        layers_list = [int(l) for l in layers.split(",")]
    else:
        # Discover available layers from first trait
        available_methods = list_methods(experiment, traits[0], model_variant, position, component)
        if available_methods:
            layers_list = list_layers(experiment, traits[0], model_variant, position, component, available_methods[0])
        else:
            layers_list = list(range(26))

    print(f"Found {len(traits)} traits at {position}/{component}")
    print(f"Evaluating {len(methods_list)} methods × {len(layers_list)} layers")

    # Evaluate all vectors
    all_results = []
    for trait in tqdm(traits, desc="Evaluating"):
        for method in methods_list:
            for layer in layers_list:
                result = evaluate_single(experiment, trait, model_variant, method, layer, component, position)
                if result:
                    all_results.append(result)

    if not all_results:
        print("No results")
        return

    # Compute combined_score per result
    # Formula: (accuracy + norm_effect) / 2 * polarity_correct
    trait_max_effect = {}
    for r in all_results:
        t = r['trait']
        trait_max_effect[t] = max(trait_max_effect.get(t, 0), r['val_effect_size'])

    for r in all_results:
        max_eff = trait_max_effect[r['trait']] or 1
        norm_effect = r['val_effect_size'] / max_eff
        polarity_mult = 1.0 if r['polarity_correct'] else 0.0
        r['combined_score'] = ((r['val_accuracy'] + norm_effect) / 2) * polarity_mult

    # Compute activation norms from first trait
    activation_norms = compute_activation_norms(experiment, traits[0], model_variant, layers_list, component, position)

    # Print summary: best per trait
    print(f"\n{'Trait':<40} {'Method':<10} {'Layer':<6} {'Acc':<6} {'d':<6}")
    print("-" * 70)

    # Group by trait, find best
    from collections import defaultdict
    by_trait = defaultdict(list)
    for r in all_results:
        by_trait[r['trait']].append(r)

    for trait in sorted(by_trait.keys()):
        best = max(by_trait[trait], key=lambda x: x['combined_score'])
        short_trait = trait.split('/')[-1][:38]
        print(f"{short_trait:<40} {best['method']:<10} {best['layer']:<6} {best['val_accuracy']:.1%}  {best['val_effect_size']:.2f}")

    if verbose:
        # Method comparison
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
    output = {
        'model_variant': model_variant,
        'component': component,
        'position': position,
        'activation_norms': activation_norms,
        'all_results': all_results,
    }

    output_path = get_path('extraction_eval.evaluation', experiment=experiment)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    fire.Fire(main)
