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

Usage:
    python analysis/model_diff/compare_variants.py \
        --experiment rm_syco \
        --variant-a instruct \
        --variant-b rm_lora \
        --prompt-set rm_syco/train_100
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
import json
import torch
from tqdm import tqdm

from core import projection, effect_size, cosine_similarity
from utils.paths import get as get_path, get_model_variant, get_model_diff_dir, discover_extracted_traits
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
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Compare instruct vs rm_lora on train_100 prompts
    python analysis/model_diff/compare_variants.py \\
        --experiment rm_syco \\
        --variant-a instruct \\
        --variant-b rm_lora \\
        --prompt-set rm_syco/train_100

    # Compare on calibration prompts (control group)
    python analysis/model_diff/compare_variants.py \\
        --experiment rm_syco \\
        --variant-a instruct \\
        --variant-b rm_lora \\
        --prompt-set massive_dims/calibration_50
        """
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
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Model diff: {args.variant_a} vs {args.variant_b}")
    print(f"Prompt set: {args.prompt_set}")

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

    # Setup output directory
    output_dir = get_model_diff_dir(args.experiment, args.variant_a, args.variant_b, args.prompt_set)
    output_dir.mkdir(parents=True, exist_ok=True)

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
            'n_prompts': len(common_ids),
            'traits': {}
        }
        with open(output_dir / 'results.json', 'w') as f:
            dump_compact(results, f)
        return

    print(f"\nAnalyzing {len(traits)} traits...")

    # Get extraction variant for loading vectors
    extraction_variant = get_model_variant(args.experiment, None, mode='extraction')['name']

    results = {
        'experiment': args.experiment,
        'variant_a': args.variant_a,
        'variant_b': args.variant_b,
        'prompt_set': args.prompt_set,
        'component': args.component,
        'n_prompts': len(common_ids),
        'traits': {}
    }

    for trait in traits:
        print(f"\n  {trait}:")

        # Get best vector for this trait
        try:
            best = get_best_vector(args.experiment, trait)
        except FileNotFoundError as e:
            print(f"    Skipped: {e}")
            continue

        layer = best['layer']
        method = best['method']
        position = best['position']
        component = best['component']

        # Load trait vector
        try:
            vector, baseline, _ = load_vector_with_baseline(
                args.experiment, trait, method, layer,
                extraction_variant, component, position
            )
            vector = vector.float()
        except FileNotFoundError as e:
            print(f"    Skipped: {e}")
            continue

        # Cosine similarity between diff vector and trait vector
        cos_sim = cosine_similarity(diff_vectors[layer], vector).item()

        # Effect size on projections
        proj_a = []
        proj_b = []
        for pid in common_ids:
            mean_a = get_response_mean(acts_a_dict[pid], layer, args.component)
            mean_b = get_response_mean(acts_b_dict[pid], layer, args.component)
            proj_a.append(projection(mean_a, vector, normalize_vector=True).item())
            proj_b.append(projection(mean_b, vector, normalize_vector=True).item())

        # Cohen's d (signed: positive = B > A)
        d = effect_size(torch.tensor(proj_a), torch.tensor(proj_b), signed=True)

        # Means
        mean_a = sum(proj_a) / len(proj_a)
        mean_b = sum(proj_b) / len(proj_b)

        print(f"    Best vector: L{layer} {method}")
        print(f"    Cosine sim (diff → trait): {cos_sim:+.3f}")
        print(f"    Effect size (B - A): {d:+.2f}σ")
        print(f"    Mean projections: A={mean_a:.2f}, B={mean_b:.2f}")

        results['traits'][trait] = {
            'layer': layer,
            'method': method,
            'position': position,
            'cosine_similarity': cos_sim,
            'effect_size': d,
            'mean_a': mean_a,
            'mean_b': mean_b,
        }

    # Save results
    with open(output_dir / 'results.json', 'w') as f:
        dump_compact(results, f)

    print(f"\n{'='*60}")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
