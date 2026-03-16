"""
Compute trait correlation matrices across prompts.

Input: Projection files from inference/project_activations_onto_traits.py
Output: Single JSON with correlation matrices at various offsets

Usage:
    python analysis/trait_correlation.py --experiment gemma-2-2b --prompt-set jailbreak/original
"""

import argparse
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm

from utils.paths import get as get_path
from utils.projections import read_response_projections


MAX_OFFSET = 10


def pearson_correlation(x: list, y: list) -> float:
    """Compute Pearson correlation coefficient."""
    if len(x) != len(y) or len(x) < 2:
        return 0.0

    x, y = np.array(x), np.array(y)
    n = len(x)

    sum_x, sum_y = x.sum(), y.sum()
    sum_xy = (x * y).sum()
    sum_x2, sum_y2 = (x * x).sum(), (y * y).sum()

    num = n * sum_xy - sum_x * sum_y
    den = np.sqrt((n * sum_x2 - sum_x**2) * (n * sum_y2 - sum_y**2))

    return 0.0 if den == 0 else float(num / den)


def compute_correlation_matrix(trait_data: dict, traits: list, prompt_ids: list, offset: int) -> list:
    """Compute correlation matrix for given offset."""
    n = len(traits)
    matrix = [[0.0] * n for _ in range(n)]

    for i in range(n):
        for j in range(n):
            pairs1, pairs2 = [], []

            for pid in prompt_ids:
                traj1 = trait_data[traits[i]].get(pid)
                traj2 = trait_data[traits[j]].get(pid)
                if traj1 is None or traj2 is None:
                    continue

                abs_offset = abs(offset)
                max_k = min(len(traj1), len(traj2)) - abs_offset

                if max_k > 0:
                    if offset >= 0:
                        pairs1.extend(traj1[:max_k])
                        pairs2.extend(traj2[offset:offset + max_k])
                    else:
                        pairs1.extend(traj1[abs_offset:abs_offset + max_k])
                        pairs2.extend(traj2[:max_k])

            matrix[i][j] = pearson_correlation(pairs1, pairs2)

    return matrix


def compute_response_correlation(trait_data: dict, traits: list, prompt_ids: list) -> list:
    """Compute correlation of mean projection per response."""
    n = len(traits)

    # Compute mean per response
    response_means = {t: {} for t in traits}
    for trait in traits:
        for pid in prompt_ids:
            traj = trait_data[trait].get(pid)
            if traj:
                response_means[trait][pid] = float(np.mean(traj))

    # Compute correlation matrix
    matrix = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            vals1 = [response_means[traits[i]].get(pid) for pid in prompt_ids
                     if response_means[traits[i]].get(pid) is not None]
            vals2 = [response_means[traits[j]].get(pid) for pid in prompt_ids
                     if response_means[traits[j]].get(pid) is not None]
            # Align - only use prompts that have both
            common = [pid for pid in prompt_ids
                      if response_means[traits[i]].get(pid) is not None
                      and response_means[traits[j]].get(pid) is not None]
            vals1 = [response_means[traits[i]][pid] for pid in common]
            vals2 = [response_means[traits[j]][pid] for pid in common]
            matrix[i][j] = pearson_correlation(vals1, vals2)

    return matrix


def main():
    parser = argparse.ArgumentParser(description="Compute trait correlation matrices")
    parser.add_argument("--experiment", required=True, help="Experiment name")
    parser.add_argument("--prompt-set", required=True, help="Prompt set to analyze")
    parser.add_argument("--model-variant", default=None, help="Model variant (default: from config)")
    args = parser.parse_args()

    # Get model variant from config if not specified
    config_path = get_path('experiments.config', experiment=args.experiment)
    with open(config_path) as f:
        config = json.load(f)
    model_variant = args.model_variant or config.get('defaults', {}).get('application', 'instruct')

    # Find all traits with projections for this prompt set
    projections_base = get_path('inference.variant', experiment=args.experiment, model_variant=model_variant) / 'projections'

    traits = []
    for category_dir in projections_base.iterdir():
        if not category_dir.is_dir():
            continue
        for trait_dir in category_dir.iterdir():
            if not trait_dir.is_dir():
                continue
            prompt_set_dir = trait_dir / args.prompt_set
            if prompt_set_dir.exists() and any(prompt_set_dir.glob('*.json')):
                traits.append(f"{category_dir.name}/{trait_dir.name}")

    if len(traits) < 2:
        print(f"Need at least 2 traits with projections. Found: {traits}")
        return

    print(f"Found {len(traits)} traits: {traits}")

    # Load projection data
    print("Loading projections...")
    trait_data = {t: {} for t in traits}
    prompt_ids = set()

    for trait in tqdm(traits, desc="Loading traits"):
        proj_dir = projections_base / trait.replace('/', '/') / args.prompt_set
        for f in proj_dir.glob('*.json'):
            try:
                pid = int(f.stem)
            except ValueError:
                continue

            try:
                trait_data[trait][pid] = read_response_projections(f)
                prompt_ids.add(pid)
            except (KeyError, ValueError):
                continue

    # Get common prompt IDs
    common_ids = sorted([pid for pid in prompt_ids
                         if all(trait_data[t].get(pid) is not None for t in traits)])
    print(f"Common prompts: {len(common_ids)}")

    # Compute correlation matrices for all offsets
    print("Computing correlations...")
    correlations_by_offset = {}

    for offset in tqdm(range(-MAX_OFFSET, MAX_OFFSET + 1), desc="Offsets"):
        correlations_by_offset[offset] = compute_correlation_matrix(trait_data, traits, common_ids, offset)

    # Compute response-level correlation
    response_correlation = compute_response_correlation(trait_data, traits, common_ids)

    # Build output
    output = {
        'experiment': args.experiment,
        'prompt_set': args.prompt_set,
        'model_variant': model_variant,
        'traits': traits,
        'trait_labels': [t.split('/')[1] for t in traits],
        'n_prompts': len(common_ids),
        'max_offset': MAX_OFFSET,
        'correlations_by_offset': {str(k): v for k, v in correlations_by_offset.items()},
        'response_correlation': response_correlation
    }

    # Save
    output_dir = get_path('experiments.base', experiment=args.experiment) / 'analysis' / 'trait_correlation'
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{args.prompt_set.replace('/', '_')}.json"

    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved: {output_file}")
    print(f"  Traits: {len(traits)}")
    print(f"  Prompts: {len(common_ids)}")
    print(f"  Offsets: -{MAX_OFFSET} to +{MAX_OFFSET}")


if __name__ == '__main__':
    main()
