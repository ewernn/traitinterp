#!/usr/bin/env python3
"""
Random baseline comparison for trait vectors.

Compares trait vector projections against random unit vectors to validate
that trait vectors capture meaningful signal (not just geometric artifacts).

Input: Raw activations from inference, trait vectors
Output: Statistics comparing trait vs random projections

Usage:
    python analysis/noise/random_baseline.py \
        --experiment gemma-2-2b \
        --prompt-set jailbreak_successes \
        --trait chirp/refusal_v2
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
import json
import torch
import numpy as np
from typing import List, Dict, Tuple

from core import projection
from utils.paths import get as get_path
from utils.vectors import get_best_vector, load_vector_with_baseline


def generate_random_vectors(dim: int, n: int, seed: int = 42) -> torch.Tensor:
    """Generate N random unit vectors of dimension dim."""
    torch.manual_seed(seed)
    vectors = torch.randn(n, dim)
    # Normalize to unit vectors
    vectors = vectors / vectors.norm(dim=1, keepdim=True)
    return vectors


def load_raw_activations(experiment: str, prompt_set: str) -> List[Dict]:
    """Load all raw activation files for a prompt set."""
    raw_dir = get_path('inference.base', experiment=experiment) / "raw" / "residual" / prompt_set

    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw activations not found: {raw_dir}")

    activations = []
    for pt_file in sorted(raw_dir.glob("*.pt"), key=lambda f: int(f.stem) if f.stem.isdigit() else 0):
        data = torch.load(pt_file, weights_only=False)
        data['prompt_id'] = pt_file.stem
        activations.append(data)

    return activations


def project_all(activations: List[Dict], vector: torch.Tensor, layer: int) -> Dict[str, List[float]]:
    """Project all activations onto a vector at a specific layer.

    Returns dict with 'prompt' and 'response' lists of mean projections per sample.
    """
    prompt_projs = []
    response_projs = []

    for data in activations:
        # Get activations at layer (handle both formats)
        prompt_layer = data['prompt']['activations'][layer]
        response_layer = data['response']['activations'][layer]

        # Support both 'residual' and 'residual_out' keys
        prompt_act = prompt_layer.get('residual', prompt_layer.get('residual_out'))
        response_act = response_layer.get('residual', response_layer.get('residual_out'))

        # Convert to same dtype as vector
        prompt_act = prompt_act.to(vector.dtype)
        response_act = response_act.to(vector.dtype)

        # Project and take mean across tokens
        prompt_proj = projection(prompt_act, vector, normalize_vector=True)
        response_proj = projection(response_act, vector, normalize_vector=True)

        prompt_projs.append(prompt_proj.mean().item())
        response_projs.append(response_proj.mean().item())

    return {'prompt': prompt_projs, 'response': response_projs}


def compute_stats(values: List[float]) -> Dict[str, float]:
    """Compute statistics for a list of values."""
    arr = np.array(values)
    return {
        'mean': float(np.mean(arr)),
        'std': float(np.std(arr)),
        'min': float(np.min(arr)),
        'max': float(np.max(arr)),
        'range': float(np.max(arr) - np.min(arr)),
        'abs_mean': float(np.mean(np.abs(arr))),
    }


def main():
    parser = argparse.ArgumentParser(description="Random baseline comparison")
    parser.add_argument("--experiment", required=True, help="Experiment name")
    parser.add_argument("--prompt-set", required=True, help="Prompt set name")
    parser.add_argument("--trait", required=True, help="Trait path (e.g., chirp/refusal_v2)")
    parser.add_argument("--n-random", type=int, default=100, help="Number of random vectors")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", help="Output JSON path (default: print to stdout)")

    args = parser.parse_args()

    print(f"Loading raw activations from {args.experiment}/{args.prompt_set}...")
    activations = load_raw_activations(args.experiment, args.prompt_set)
    print(f"  Loaded {len(activations)} samples")

    # Get best vector for trait
    print(f"\nLoading trait vector for {args.trait}...")
    best = get_best_vector(args.experiment, args.trait)
    layer = best['layer']
    method = best['method']
    position = best['position']
    component = best['component']
    print(f"  Using L{layer} {method} from {best['source']} (score: {best['score']:.3f})")

    # Load trait vector
    trait_vector, baseline, _ = load_vector_with_baseline(
        args.experiment, args.trait, method, layer, component, position
    )
    trait_vector = trait_vector.float()
    dim = trait_vector.shape[0]

    # Generate random vectors
    print(f"\nGenerating {args.n_random} random unit vectors (dim={dim})...")
    random_vectors = generate_random_vectors(dim, args.n_random, args.seed)

    # Project onto trait vector
    print(f"\nProjecting onto trait vector...")
    trait_projs = project_all(activations, trait_vector, layer)

    # Project onto random vectors
    print(f"Projecting onto {args.n_random} random vectors...")
    random_projs = {'prompt': [], 'response': []}
    for i, rv in enumerate(random_vectors):
        projs = project_all(activations, rv, layer)
        random_projs['prompt'].extend(projs['prompt'])
        random_projs['response'].extend(projs['response'])

    # Compute statistics
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)

    results = {
        'experiment': args.experiment,
        'prompt_set': args.prompt_set,
        'trait': args.trait,
        'layer': layer,
        'method': method,
        'n_samples': len(activations),
        'n_random_vectors': args.n_random,
        'seed': args.seed,
    }

    for phase in ['prompt', 'response']:
        trait_stats = compute_stats(trait_projs[phase])
        random_stats = compute_stats(random_projs[phase])

        results[f'{phase}_trait'] = trait_stats
        results[f'{phase}_random'] = random_stats

        # Compute ratio (how much more variance does trait have vs random)
        variance_ratio = trait_stats['std'] / random_stats['std'] if random_stats['std'] > 0 else float('inf')
        range_ratio = trait_stats['range'] / random_stats['range'] if random_stats['range'] > 0 else float('inf')

        results[f'{phase}_variance_ratio'] = variance_ratio
        results[f'{phase}_range_ratio'] = range_ratio

        print(f"\n{phase.upper()} PHASE:")
        print(f"  Trait vector:  mean={trait_stats['mean']:+.3f}, std={trait_stats['std']:.3f}, range={trait_stats['range']:.3f}")
        print(f"  Random (avg):  mean={random_stats['mean']:+.3f}, std={random_stats['std']:.3f}, range={random_stats['range']:.3f}")
        print(f"  Variance ratio (trait/random): {variance_ratio:.2f}x")
        print(f"  Range ratio (trait/random):    {range_ratio:.2f}x")

    # Interpretation
    print("\n" + "="*60)
    print("INTERPRETATION")
    print("="*60)

    response_var_ratio = results['response_variance_ratio']
    if response_var_ratio > 2.0:
        print(f"  PASS: Trait vector has {response_var_ratio:.1f}x more variance than random")
        print("        Signal appears meaningful (not just geometric artifact)")
    elif response_var_ratio > 1.2:
        print(f"  MARGINAL: Trait vector has {response_var_ratio:.1f}x variance vs random")
        print("        Some signal, but may be noisy")
    else:
        print(f"  FAIL: Trait vector variance ({response_var_ratio:.1f}x) not much better than random")
        print("        Vector may not capture meaningful signal")

    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output}")

    return results


if __name__ == "__main__":
    main()
