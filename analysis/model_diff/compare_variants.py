#!/usr/bin/env python3
"""
Compare activations between two model variants.

Computes:
1. Diff vectors - Mean activation difference (B - A) per layer
2. Effect sizes - Cohen's d on trait projections (B vs A)
3. Cosine similarities - How aligned is diff vector with trait vectors

Convention:
- variant-a = baseline (e.g., instruct)
- variant-b = comparison (e.g., rm_lora)
- Positive effect size means variant-b projects higher on trait
- Positive cosine similarity means variant-b pushes toward trait direction

Output:
    experiments/{experiment}/model_diff/{variant_a}_vs_{variant_b}/{prompt_set}/
    ├── diff_vectors.pt    # [n_layers, hidden_dim]
    └── results.json       # effect sizes, cosine similarities per trait

Vector specification (REQUIRED - one of):
    --use-best-vector        Auto-select from steering results (requires steering to have been run)
    --method M --position P  Explicit specification (e.g., --method probe --position "response[:5]")

Usage:
    # Auto-select best vector from steering results
    python analysis/model_diff/compare_variants.py \\
        --experiment rm_syco \\
        --variant-a instruct --variant-b rm_lora \\
        --prompt-set rm_syco/train_100 \\
        --use-best-vector

    # Explicit vector specification
    python analysis/model_diff/compare_variants.py \\
        --experiment rm_syco \\
        --variant-a instruct --variant-b rm_lora \\
        --prompt-set rm_syco/train_100 \\
        --method probe --position "response[:5]"
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
import json
import torch
import numpy as np
from tqdm import tqdm

from core import projection, effect_size, cosine_similarity
from utils.paths import get as get_path, get_model_variant, get_model_diff_dir, discover_extracted_traits, list_layers
from utils.vectors import load_vector_with_baseline, get_best_vector
from utils.json import dump_compact


def load_raw_activations(experiment: str, model_variant: str, prompt_set: str) -> list:
    """
    Load all raw activation files for a prompt set.

    Returns list of dicts with 'prompt_id' and activation data.
    """
    raw_dir = get_path('inference.raw_residual', experiment=experiment,
                       model_variant=model_variant, prompt_set=prompt_set)
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw activations not found: {raw_dir}")

    activations = []
    for pt_file in sorted(raw_dir.glob("*.pt"), key=lambda f: int(f.stem) if f.stem.isdigit() else 0):
        data = torch.load(pt_file, weights_only=False)
        data['prompt_id'] = pt_file.stem
        activations.append(data)
    return activations


def get_response_mean(data: dict, layer: int, component: str = 'residual') -> torch.Tensor:
    """Get mean response activation at a layer. Returns [hidden_dim]."""
    act = data['response']['activations'][layer].get(component)
    if act is None:
        raise KeyError(f"Component '{component}' not found at layer {layer}")
    return act.mean(dim=0).float()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare activations between two model variants",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--experiment', required=True, help='Experiment name')
    parser.add_argument('--variant-a', required=True,
                        help='Baseline variant (e.g., instruct)')
    parser.add_argument('--variant-b', required=True,
                        help='Comparison variant (e.g., rm_lora)')
    parser.add_argument('--prompt-set', required=True,
                        help='Prompt set (e.g., rm_syco/train_100)')
    parser.add_argument('--traits', default=None,
                        help='Comma-separated traits (default: all extracted)')
    parser.add_argument('--component', default='residual',
                        help='Activation component (default: residual)')

    # Vector specification (one of these is required)
    vector_group = parser.add_argument_group('vector specification (one required)')
    vector_group.add_argument('--use-best-vector', action='store_true',
                        help='Auto-select method/position from steering results (requires steering)')
    vector_group.add_argument('--method', default=None,
                        help='Vector extraction method (e.g., probe, mean_diff)')
    vector_group.add_argument('--position', default=None,
                        help='Position for vectors (e.g., "response[:5]")')

    parser.add_argument('--use-existing-diff', action='store_true',
                        help='Use existing diff_vectors.pt instead of computing from raw activations. '
                             'Only computes cosine similarity (not effect sizes).')

    args = parser.parse_args()

    # Validate vector specification
    has_explicit = args.method is not None and args.position is not None
    has_auto = args.use_best_vector

    if not has_explicit and not has_auto:
        parser.error("Must specify either --use-best-vector OR (--method and --position)\n"
                     "  --use-best-vector: auto-select from steering results\n"
                     "  --method M --position P: explicit specification")

    if has_explicit and has_auto:
        parser.error("Cannot use both --use-best-vector and --method/--position. Choose one.")

    if (args.method is None) != (args.position is None):
        parser.error("--method and --position must be specified together")

    return args


def main():
    args = parse_args()

    print(f"Model diff: {args.variant_a} vs {args.variant_b}")
    print(f"Prompt set: {args.prompt_set}")

    # Setup output directory
    output_dir = get_model_diff_dir(args.experiment, args.variant_a, args.variant_b, args.prompt_set)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Either load existing diff vectors or compute from raw activations
    acts_a_dict = None
    acts_b_dict = None
    n_prompts = None

    if args.use_existing_diff:
        # Load existing diff vectors
        diff_path = output_dir / 'diff_vectors.pt'
        if not diff_path.exists():
            print(f"ERROR: --use-existing-diff specified but {diff_path} not found")
            print("Run without --use-existing-diff first to compute diff vectors from raw activations.")
            return
        print(f"\nLoading existing diff vectors...")
        diff_vectors = torch.load(diff_path, weights_only=True)
        n_layers, hidden_dim = diff_vectors.shape
        print(f"  Loaded diff_vectors.pt: [{n_layers}, {hidden_dim}]")
        print(f"  (Skipping effect size computation - only cosine similarity available)")
        # Try to get n_prompts from existing results
        results_path = output_dir / 'results.json'
        if results_path.exists():
            with open(results_path) as f:
                existing = json.load(f)
                n_prompts = existing.get('n_prompts')
    else:
        # Load activations
        print(f"\nLoading activations...")
        acts_a = load_raw_activations(args.experiment, args.variant_a, args.prompt_set)
        acts_b = load_raw_activations(args.experiment, args.variant_b, args.prompt_set)
        print(f"  {args.variant_a}: {len(acts_a)} prompts")
        print(f"  {args.variant_b}: {len(acts_b)} prompts")

        # Match by prompt_id
        acts_a_dict = {d['prompt_id']: d for d in acts_a}
        acts_b_dict = {d['prompt_id']: d for d in acts_b}
        common_ids = sorted(set(acts_a_dict.keys()) & set(acts_b_dict.keys()))
        print(f"  Common: {len(common_ids)} prompts")

        if not common_ids:
            print("ERROR: No common prompt IDs between variants")
            return

        n_prompts = len(common_ids)

        # Get dimensions from first prompt
        first = acts_a_dict[common_ids[0]]
        n_layers = len(first['response']['activations'])
        hidden_dim = first['response']['activations'][0][args.component].shape[-1]
        print(f"  Layers: {n_layers}, Hidden dim: {hidden_dim}")

        # 1. Extract diff vectors (mean diff per layer)
        print(f"\nExtracting diff vectors (B - A)...")
        diff_vectors = torch.zeros(n_layers, hidden_dim)

        for layer in tqdm(range(n_layers), desc="Layers"):
            diffs = []
            for pid in common_ids:
                mean_a = get_response_mean(acts_a_dict[pid], layer, args.component)
                mean_b = get_response_mean(acts_b_dict[pid], layer, args.component)
                diffs.append(mean_b - mean_a)
            diff_vectors[layer] = torch.stack(diffs).mean(dim=0)

        # Save diff vectors
        torch.save(diff_vectors, output_dir / 'diff_vectors.pt')
        print(f"  Saved diff_vectors.pt")

    # 2. Get traits to analyze
    if args.traits:
        traits = [t.strip() for t in args.traits.split(',')]
    else:
        traits = [f"{c}/{n}" for c, n in discover_extracted_traits(args.experiment)]

    if not traits:
        print("\nNo traits found for analysis")
        results = {
            'experiment': args.experiment,
            'variant_a': args.variant_a,
            'variant_b': args.variant_b,
            'prompt_set': args.prompt_set,
            'n_prompts': n_prompts,
            'traits': {}
        }
        with open(output_dir / 'results.json', 'w') as f:
            dump_compact(results, f)
        return

    print(f"\nAnalyzing {len(traits)} traits...")

    # Get extraction variant for loading vectors
    extraction_variant = get_model_variant(args.experiment, None, mode='extraction')['name']

    # Load existing results to merge (don't overwrite other traits)
    results_path = output_dir / 'results.json'
    if results_path.exists():
        with open(results_path) as f:
            results = json.load(f)
        # Update metadata (preserve n_prompts if we don't have it)
        results.update({
            'experiment': args.experiment,
            'variant_a': args.variant_a,
            'variant_b': args.variant_b,
            'prompt_set': args.prompt_set,
            'component': args.component,
        })
        if n_prompts is not None:
            results['n_prompts'] = n_prompts
    else:
        results = {
            'experiment': args.experiment,
            'variant_a': args.variant_a,
            'variant_b': args.variant_b,
            'prompt_set': args.prompt_set,
            'component': args.component,
            'n_prompts': n_prompts,
            'traits': {}
        }

    for trait in traits:
        print(f"\n  {trait}:")

        # Determine method/position/component
        if args.use_best_vector:
            # Auto-select from steering results
            try:
                best = get_best_vector(args.experiment, trait)
                method = best['method']
                position = best['position']
                component = best['component']
            except FileNotFoundError as e:
                print(f"    Skipped: No steering results found for {trait}")
                print(f"    Run steering first, or use --method/--position explicitly")
                continue
        else:
            # Use explicit specification
            method = args.method
            position = args.position
            component = args.component

        # Get available layers
        available_layers = list_layers(
            args.experiment, trait, method, extraction_variant, component, position
        )
        if not available_layers:
            print(f"    Skipped: No vectors found")
            continue

        print(f"    Sweeping {len(available_layers)} layers ({method}, {position})...")

        per_layer_effect_size = []
        per_layer_cosine_sim = []
        per_layer_std_a = []
        per_layer_std_b = []

        # Check if trait already has effect size data (to preserve when using --use-existing-diff)
        existing_trait_data = results.get('traits', {}).get(trait, {})
        existing_effect_sizes = existing_trait_data.get('per_layer_effect_size')

        for layer in available_layers:
            # Load trait vector at this layer
            try:
                vector, _, _ = load_vector_with_baseline(
                    args.experiment, trait, method, layer,
                    extraction_variant, component, position
                )
                vector = vector.float()
            except FileNotFoundError:
                per_layer_cosine_sim.append(0.0)
                if not args.use_existing_diff:
                    per_layer_effect_size.append(0.0)
                continue

            # Cosine similarity between diff vector and trait vector
            cos_sim = cosine_similarity(diff_vectors[layer].float(), vector).item()
            per_layer_cosine_sim.append(cos_sim)

            # Effect size on projections (only if we have raw activations)
            if not args.use_existing_diff:
                proj_a = []
                proj_b = []
                for pid in common_ids:
                    mean_a = get_response_mean(acts_a_dict[pid], layer, args.component)
                    mean_b = get_response_mean(acts_b_dict[pid], layer, args.component)
                    proj_a.append(projection(mean_a, vector, normalize_vector=True).item())
                    proj_b.append(projection(mean_b, vector, normalize_vector=True).item())

                # Cohen's d (signed: positive = B > A)
                d = effect_size(torch.tensor(proj_b), torch.tensor(proj_a), signed=True)
                per_layer_effect_size.append(d)

                # Per-variant projection spread
                per_layer_std_a.append(float(np.std(proj_a)))
                per_layer_std_b.append(float(np.std(proj_b)))

        # Build trait result
        trait_result = {
            'method': method,
            'position': position,
            'layers': available_layers,
            'per_layer_cosine_sim': per_layer_cosine_sim,
        }

        # Handle effect size based on mode
        if args.use_existing_diff:
            # Preserve existing effect size if available, otherwise set to None
            if existing_effect_sizes and len(existing_effect_sizes) == len(available_layers):
                trait_result['per_layer_effect_size'] = existing_effect_sizes
                trait_result['peak_layer'] = existing_trait_data.get('peak_layer')
                trait_result['peak_effect_size'] = existing_trait_data.get('peak_effect_size')
                print(f"    (preserved existing effect sizes)")
            else:
                trait_result['per_layer_effect_size'] = None
                trait_result['peak_layer'] = None
                trait_result['peak_effect_size'] = None

            # Report peak cosine sim
            peak_cos_idx = max(range(len(per_layer_cosine_sim)), key=lambda i: abs(per_layer_cosine_sim[i]))
            peak_cos_layer = available_layers[peak_cos_idx]
            peak_cos = per_layer_cosine_sim[peak_cos_idx]
            print(f"    Peak cosine: L{peak_cos_layer} = {peak_cos:+.3f}")
        else:
            # Full computation - find peak effect size
            trait_result['per_layer_effect_size'] = per_layer_effect_size
            peak_idx = max(range(len(per_layer_effect_size)), key=lambda i: per_layer_effect_size[i])
            peak_layer = available_layers[peak_idx]
            peak_effect = per_layer_effect_size[peak_idx]
            trait_result['peak_layer'] = peak_layer
            trait_result['peak_effect_size'] = peak_effect

            # Projection variance per variant (cross-prompt spread at each layer)
            trait_result['per_layer_std_a'] = per_layer_std_a
            trait_result['per_layer_std_b'] = per_layer_std_b
            # Report peak-layer variance
            std_a_peak = per_layer_std_a[peak_idx]
            std_b_peak = per_layer_std_b[peak_idx]
            print(f"    Peak: L{peak_layer} = {peak_effect:+.2f}σ  (spread: {args.variant_a}={std_a_peak:.2f}, {args.variant_b}={std_b_peak:.2f})")

        results['traits'][trait] = trait_result

    # Save results
    with open(output_dir / 'results.json', 'w') as f:
        dump_compact(results, f)

    print(f"\n{'='*60}")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
