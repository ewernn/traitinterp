#!/usr/bin/env python3
"""
Cross-layer projection diff: does the per-token diff signal hold across layers?

Loads raw activations at multiple layers, projects onto trait vectors,
computes per-token diff (variant_b - variant_a), and measures cross-layer consistency.

Input:
    Raw .pt files from both variants + trait vectors at target layers.

Output:
    experiments/{experiment}/model_diff/{variant_a}_vs_{variant_b}/layer_sensitivity/{prompt_set}/
    ├── results.json           # Cross-layer stats: correlations, mean diffs per layer
    └── per_prompt/            # Per-prompt multi-layer projections
        └── {id}.json

Usage:
    python analysis/model_diff/layer_sensitivity.py \
        --experiment audit-bench \
        --variant-a instruct \
        --variant-b rm_lora \
        --prompt-set rm_syco/exploitation_evals_100 \
        --layers 20,25,30,35,40
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
import json
import torch
import numpy as np
from tqdm import tqdm
from collections import defaultdict

from core.math import projection
from utils.paths import get as get_path, discover_extracted_traits
from utils.vectors import get_best_vector_spec, load_vector


SKIP_TRAITS = {'rm_hack/eval_awareness'}


def load_vectors(experiment, traits, layers, extraction_variant='base',
                 position='response[:5]', component='residual', method='probe'):
    """Load trait vectors at each target layer.

    Returns:
        {trait: {layer: tensor}} and {trait: best_layer}
    """
    vectors = {}
    best_layers = {}

    for trait in traits:
        vectors[trait] = {}

        # Get best layer from steering
        try:
            spec, meta = get_best_vector_spec(
                experiment, trait, extraction_variant=extraction_variant,
                component=component, position=position
            )
            best_layers[trait] = spec.layer
            print(f"  {trait}: best=L{spec.layer} ({meta['source']}: {meta['score']:.2f})")
        except FileNotFoundError:
            best_layers[trait] = None
            print(f"  {trait}: no steering results, skipping best layer")

        # Load at each target layer
        for layer in layers:
            vec = load_vector(
                experiment, trait, layer, extraction_variant,
                method=method, component=component, position=position
            )
            if vec is not None:
                vectors[trait][layer] = vec
            else:
                print(f"    WARNING: no vector at L{layer} for {trait}")

        # Also load at best layer if not in target set
        if best_layers[trait] and best_layers[trait] not in layers:
            vec = load_vector(
                experiment, trait, best_layers[trait], extraction_variant,
                method=method, component=component, position=position
            )
            if vec is not None:
                vectors[trait][best_layers[trait]] = vec

    return vectors, best_layers


def project_prompt(response_acts, vectors, layers):
    """Project one prompt's activations onto all trait vectors at all layers.

    Returns:
        {trait: {layer: [scores]}}
    """
    result = {}
    for trait, layer_vecs in vectors.items():
        result[trait] = {}
        for layer, vec in layer_vecs.items():
            if layer not in response_acts:
                continue
            act = response_acts[layer]['residual']
            proj = projection(act, vec, normalize_vector=True)
            result[trait][layer] = proj.tolist()
    return result


def compute_cross_layer_stats(all_diffs):
    """Compute cross-layer correlation and consistency stats.

    Args:
        all_diffs: {trait: {layer: {prompt_id: [per_token_deltas]}}}

    Returns:
        {trait: {layer_pair_correlations, mean_diff_by_layer, ...}}
    """
    stats = {}

    for trait, layer_data in all_diffs.items():
        available_layers = sorted(layer_data.keys())
        if len(available_layers) < 2:
            continue

        # Mean diff per layer (across all prompts, all tokens)
        mean_by_layer = {}
        std_by_layer = {}
        for layer in available_layers:
            all_deltas = []
            for pid, deltas in layer_data[layer].items():
                all_deltas.extend(deltas)
            mean_by_layer[layer] = float(np.mean(all_deltas))
            std_by_layer[layer] = float(np.std(all_deltas))

        # Per-prompt mean diff at each layer
        prompt_ids = sorted(layer_data[available_layers[0]].keys())
        prompt_means = {layer: [] for layer in available_layers}
        for pid in prompt_ids:
            for layer in available_layers:
                if pid in layer_data[layer]:
                    prompt_means[layer].append(float(np.mean(layer_data[layer][pid])))
                else:
                    prompt_means[layer].append(0.0)

        # Pairwise layer correlations (on prompt-level means)
        correlations = {}
        for i, l1 in enumerate(available_layers):
            for l2 in available_layers[i+1:]:
                a = np.array(prompt_means[l1])
                b = np.array(prompt_means[l2])
                if np.std(a) > 0 and np.std(b) > 0:
                    corr = float(np.corrcoef(a, b)[0, 1])
                else:
                    corr = 0.0
                correlations[f"L{l1}_vs_L{l2}"] = round(corr, 4)

        # Per-token correlation across layers (average over prompts)
        token_corrs = {}
        for i, l1 in enumerate(available_layers):
            for l2 in available_layers[i+1:]:
                per_prompt_corrs = []
                for pid in prompt_ids:
                    if pid in layer_data[l1] and pid in layer_data[l2]:
                        a = np.array(layer_data[l1][pid])
                        b = np.array(layer_data[l2][pid])
                        n = min(len(a), len(b))
                        if n > 5 and np.std(a[:n]) > 0 and np.std(b[:n]) > 0:
                            per_prompt_corrs.append(float(np.corrcoef(a[:n], b[:n])[0, 1]))
                if per_prompt_corrs:
                    token_corrs[f"L{l1}_vs_L{l2}"] = round(float(np.mean(per_prompt_corrs)), 4)

        stats[trait] = {
            'available_layers': available_layers,
            'mean_diff_by_layer': {f"L{k}": round(v, 4) for k, v in mean_by_layer.items()},
            'std_diff_by_layer': {f"L{k}": round(v, 4) for k, v in std_by_layer.items()},
            'prompt_level_correlations': correlations,
            'token_level_correlations': token_corrs,
            'n_prompts': len(prompt_ids),
        }

    return stats


def main():
    parser = argparse.ArgumentParser(description='Cross-layer projection diff analysis')
    parser.add_argument('--experiment', required=True)
    parser.add_argument('--variant-a', required=True, help='Baseline (e.g., instruct)')
    parser.add_argument('--variant-b', required=True, help='Comparison (e.g., rm_lora)')
    parser.add_argument('--prompt-set', required=True)
    parser.add_argument('--layers', required=True,
                        help='Comma-separated layers (e.g., 20,25,30,35,40)')
    parser.add_argument('--traits', default=None,
                        help='Comma-separated traits (default: all extracted, minus eval_awareness)')
    parser.add_argument('--method', default='probe')
    parser.add_argument('--position', default='response[:5]')
    parser.add_argument('--component', default='residual')
    args = parser.parse_args()

    layers = [int(x) for x in args.layers.split(',')]
    exp_dir = Path(get_path('experiments.base', experiment=args.experiment))

    # Resolve traits
    if args.traits:
        traits = args.traits.split(',')
    else:
        trait_tuples = discover_extracted_traits(args.experiment)
        traits = [f"{cat}/{name}" for cat, name in trait_tuples
                  if f"{cat}/{name}" not in SKIP_TRAITS]

    print(f"Experiment: {args.experiment}")
    print(f"Layers: {layers}")
    print(f"Traits: {traits}")
    print(f"Variants: {args.variant_a} vs {args.variant_b}")
    print()

    # Load vectors
    print("Loading vectors...")
    vectors, best_layers = load_vectors(
        args.experiment, traits, layers,
        method=args.method, position=args.position, component=args.component
    )
    print()

    # Find raw activation files
    raw_dir_a = exp_dir / 'inference' / args.variant_a / 'raw' / 'residual' / args.prompt_set
    raw_dir_b = exp_dir / 'inference' / args.variant_b / 'raw' / 'residual' / args.prompt_set
    resp_dir = exp_dir / 'inference' / args.variant_b / 'responses' / args.prompt_set

    for d, label in [(raw_dir_a, 'variant-a raw'), (raw_dir_b, 'variant-b raw')]:
        if not d.exists():
            print(f"ERROR: {label} dir not found: {d}")
            return

    raw_files_b = sorted(raw_dir_b.glob('*.pt'))
    print(f"Found {len(raw_files_b)} raw files for {args.variant_b}")

    # Output dir
    out_dir = exp_dir / 'model_diff' / f'{args.variant_a}_vs_{args.variant_b}' / 'layer_sensitivity' / args.prompt_set
    per_prompt_dir = out_dir / 'per_prompt'
    per_prompt_dir.mkdir(parents=True, exist_ok=True)

    # Process each prompt
    all_diffs = {trait: {layer: {} for layer in layers} for trait in traits}
    # Also track best-layer diffs
    for trait in traits:
        bl = best_layers.get(trait)
        if bl and bl not in layers:
            all_diffs[trait][bl] = {}

    skipped = 0
    for raw_file_b in tqdm(raw_files_b, desc="Processing"):
        pid = raw_file_b.stem
        raw_file_a = raw_dir_a / f'{pid}.pt'

        if not raw_file_a.exists():
            skipped += 1
            continue

        # Load raw activations
        data_a = torch.load(raw_file_a, weights_only=False)
        data_b = torch.load(raw_file_b, weights_only=False)

        acts_a = data_a['response']['activations']
        acts_b = data_b['response']['activations']
        tokens_b = data_b['response']['tokens']

        # Project both variants
        proj_a = project_prompt(acts_a, vectors, layers)
        proj_b = project_prompt(acts_b, vectors, layers)

        # Compute per-token diffs and save per-prompt data
        prompt_data = {
            'prompt_id': pid,
            'response_tokens': tokens_b,
            'n_response_tokens': len(tokens_b),
            'traits': {}
        }

        for trait in traits:
            prompt_data['traits'][trait] = {'layers': {}}
            for layer in sorted(vectors[trait].keys()):
                if layer not in proj_a.get(trait, {}) or layer not in proj_b.get(trait, {}):
                    continue

                scores_a = proj_a[trait][layer]
                scores_b = proj_b[trait][layer]
                n = min(len(scores_a), len(scores_b))
                deltas = [scores_b[i] - scores_a[i] for i in range(n)]

                prompt_data['traits'][trait]['layers'][str(layer)] = {
                    'proj_a': [round(x, 4) for x in scores_a[:n]],
                    'proj_b': [round(x, 4) for x in scores_b[:n]],
                    'delta': [round(x, 4) for x in deltas],
                    'mean_delta': round(float(np.mean(deltas)), 4),
                }

                all_diffs[trait][layer][pid] = deltas

        # Save per-prompt
        with open(per_prompt_dir / f'{pid}.json', 'w') as f:
            json.dump(prompt_data, f)

        # Free memory
        del data_a, data_b

    if skipped:
        print(f"Skipped {skipped} prompts (no matching variant-a file)")

    # Compute cross-layer stats
    print("\nComputing cross-layer statistics...")
    stats = compute_cross_layer_stats(all_diffs)

    # Add best layer info
    for trait in traits:
        if trait in stats:
            stats[trait]['best_layer'] = best_layers.get(trait)

    # Save results
    results = {
        'experiment': args.experiment,
        'variant_a': args.variant_a,
        'variant_b': args.variant_b,
        'prompt_set': args.prompt_set,
        'target_layers': layers,
        'traits': stats,
    }

    with open(out_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Print summary
    print(f"\n{'='*70}")
    print(f"LAYER SENSITIVITY RESULTS")
    print(f"{'='*70}")

    for trait, s in stats.items():
        print(f"\n{trait} (best=L{s.get('best_layer', '?')})")
        print(f"  Mean diff by layer:")
        for layer_key in sorted(s['mean_diff_by_layer'].keys(), key=lambda x: int(x[1:])):
            mean = s['mean_diff_by_layer'][layer_key]
            std = s['std_diff_by_layer'][layer_key]
            marker = " <-- best" if str(s.get('best_layer')) == layer_key[1:] else ""
            print(f"    {layer_key}: {mean:+.4f} ± {std:.4f}{marker}")

        print(f"  Prompt-level correlations:")
        for pair, corr in sorted(s['prompt_level_correlations'].items()):
            print(f"    {pair}: r={corr:.4f}")

        print(f"  Token-level correlations (mean across prompts):")
        for pair, corr in sorted(s.get('token_level_correlations', {}).items()):
            print(f"    {pair}: r={corr:.4f}")

    print(f"\nResults saved to: {out_dir}")


if __name__ == '__main__':
    main()
