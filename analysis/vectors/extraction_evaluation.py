#!/usr/bin/env python3
"""
Evaluate extracted vectors on held-out validation data.

Input:
    - experiments/{experiment}/extraction/{trait}/vectors/*.pt
    - experiments/{experiment}/extraction/{trait}/val_activations/*.pt

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

from utils.paths import get as get_path
from utils.model import load_experiment_config
from utils.vectors import load_vector_with_baseline
from core import batch_cosine_similarity, accuracy, effect_size, polarity_correct


def load_activations(experiment: str, trait: str, layer: int, split: str = 'val', component: str = 'residual') -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Load pos/neg activations for a layer. split='val' or 'train'."""
    if split == 'val':
        base_dir = get_path('extraction.val_activations', experiment=experiment, trait=trait)
        prefix = "" if component == 'residual' else f"{component}_"
        pos_path = base_dir / f"{prefix}val_pos_layer{layer}.pt"
        neg_path = base_dir / f"{prefix}val_neg_layer{layer}.pt"
    else:
        base_dir = get_path('extraction.activations', experiment=experiment, trait=trait)
        prefix = "" if component == 'residual' else f"{component}_"
        pos_path = base_dir / f"{prefix}pos_layer{layer}.pt"
        neg_path = base_dir / f"{prefix}neg_layer{layer}.pt"

    if not pos_path.exists() or not neg_path.exists():
        return None, None

    return torch.load(pos_path, weights_only=True), torch.load(neg_path, weights_only=True)


def load_vector(experiment: str, trait: str, method: str, layer: int, component: str = 'residual') -> Optional[torch.Tensor]:
    """Load a vector file."""
    try:
        vector, _, _ = load_vector_with_baseline(experiment, trait, method, layer, component)
        return vector
    except FileNotFoundError:
        return None


def evaluate_single(
    experiment: str,
    trait: str,
    method: str,
    layer: int,
    component: str = 'residual'
) -> Optional[Dict]:
    """Evaluate one vector on validation data."""
    vector = load_vector(experiment, trait, method, layer, component)
    if vector is None:
        return None

    val_pos, val_neg = load_activations(experiment, trait, layer, 'val', component)
    if val_pos is None:
        return None

    # Compute projections once, derive all metrics
    pos_proj = batch_cosine_similarity(val_pos, vector)
    neg_proj = batch_cosine_similarity(val_neg, vector)

    return {
        'trait': trait,
        'method': method,
        'layer': layer,
        'val_accuracy': accuracy(pos_proj, neg_proj),
        'val_effect_size': effect_size(pos_proj, neg_proj),
        'polarity_correct': polarity_correct(pos_proj, neg_proj),
    }


def compute_activation_norms(experiment: str, trait: str, layers: List[int], component: str = 'residual') -> Dict[str, float]:
    """Compute mean ||h|| per layer. Used by steering for coefficient estimation."""
    norms = {}
    for layer in layers:
        pos, neg = load_activations(experiment, trait, layer, 'val', component)
        if pos is not None:
            all_acts = torch.cat([pos, neg], dim=0).float()
            norms[str(layer)] = all_acts.norm(dim=-1).mean().item()
    return norms


def main(
    experiment: str,
    methods: str = "mean_diff,probe,gradient",
    layers: str = None,
    component: str = "residual",
    verbose: bool = False,
):
    """
    Evaluate all vectors on validation data.

    Args:
        experiment: Experiment name
        methods: Comma-separated methods to evaluate
        layers: Comma-separated layers (default: all 26)
        component: Component to evaluate (residual, attn_out, mlp_out)
        verbose: Print detailed per-method/layer analysis
    """
    exp_dir = get_path('extraction.base', experiment=experiment)
    methods_list = methods.split(",")
    layers_list = [int(l) for l in layers.split(",")] if layers else list(range(26))

    # Find traits with validation data
    traits = []
    for category_dir in exp_dir.iterdir():
        if not category_dir.is_dir():
            continue
        for trait_dir in category_dir.iterdir():
            if (trait_dir / "val_activations").exists():
                traits.append(f"{category_dir.name}/{trait_dir.name}")

    print(f"Found {len(traits)} traits, evaluating {len(methods_list)} methods × {len(layers_list)} layers")

    # Evaluate all vectors
    all_results = []
    for trait in tqdm(traits, desc="Evaluating"):
        for method in methods_list:
            for layer in layers_list:
                result = evaluate_single(experiment, trait, method, layer, component)
                if result:
                    all_results.append(result)

    if not all_results:
        print("No results")
        return

    # Compute combined_score per result
    # Formula: (accuracy + norm_effect + polarity) / 3
    # norm_effect = effect_size / max_effect_for_trait
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
    activation_norms = compute_activation_norms(experiment, traits[0], layers_list, component)

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
        from collections import defaultdict
        method_stats = defaultdict(list)
        for r in all_results:
            method_stats[r['method']].append(r)
        for method in methods_list:
            if method in method_stats:
                accs = [r['val_accuracy'] for r in method_stats[method]]
                effs = [r['val_effect_size'] for r in method_stats[method]]
                print(f"{method:<12} {sum(accs)/len(accs):.1%}      {sum(effs)/len(effs):.2f}")

    # Save
    config = load_experiment_config(experiment, warn_missing=False)
    output = {
        'extraction_model': config.get('extraction_model'),
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
