#!/usr/bin/env python3
"""
Compute cosine similarity between extraction methods without requiring validation data.

Usage:
    python analysis/vectors/compute_method_similarities.py --experiment gemma-2-2b
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import torch.nn.functional as F
import json
import fire
from tqdm import tqdm

from utils.paths import get as get_path


def compute_method_similarities(experiment: str):
    """Compute cosine similarity between methods for all traits with vectors."""

    exp_dir = get_path('extraction.base', experiment=experiment)
    if not exp_dir.exists():
        print(f"Experiment directory not found: {exp_dir}")
        return

    # Find all traits with vectors
    traits = []
    for category_dir in exp_dir.iterdir():
        if not category_dir.is_dir() or category_dir.name.startswith('.'):
            continue
        for trait_dir in category_dir.iterdir():
            if not trait_dir.is_dir():
                continue
            vectors_dir = trait_dir / "vectors"
            if vectors_dir.exists() and list(vectors_dir.glob('*.pt')):
                trait_path = f"{category_dir.name}/{trait_dir.name}"
                traits.append(trait_path)

    print(f"Found {len(traits)} traits with vectors")

    if len(traits) == 0:
        print("No traits found")
        return

    # Compute similarities
    similarities = {}

    for trait in tqdm(traits, desc="Computing similarities"):
        vectors_dir = get_path('extraction.vectors', experiment=experiment, trait=trait)

        # Get all vector files and group by layer
        vector_files = list(vectors_dir.glob('*.pt'))

        # Group by layer
        layers_data = {}
        for vec_file in vector_files:
            # Parse method_layerN.pt
            stem = vec_file.stem
            if '_layer' not in stem:
                continue

            method, layer_str = stem.rsplit('_layer', 1)
            try:
                layer = int(layer_str)
            except ValueError:
                continue

            if layer not in layers_data:
                layers_data[layer] = {}

            # Load vector
            vector = torch.load(vec_file, weights_only=True, map_location='cpu').float()
            layers_data[layer][method] = vector

        # Compute similarities per layer
        trait_sims = {}
        for layer, vectors in layers_data.items():
            # Need at least 2 methods
            if len(vectors) < 2:
                continue

            # Compute pairwise similarities
            layer_sims = {}
            methods = sorted(vectors.keys())

            for i, method_i in enumerate(methods):
                for j, method_j in enumerate(methods):
                    if i >= j:  # Only upper triangle
                        continue

                    vec_i = vectors[method_i]
                    vec_j = vectors[method_j]

                    # Skip if dimensions don't match (different components)
                    if vec_i.shape != vec_j.shape:
                        continue

                    # Cosine similarity
                    sim = F.cosine_similarity(vec_i, vec_j, dim=0).item()
                    pair_key = f"{method_i}_{method_j}"
                    layer_sims[pair_key] = round(sim, 4)

            if layer_sims:
                trait_sims[layer] = layer_sims

        if trait_sims:
            similarities[trait] = trait_sims

    # Load or create extraction_evaluation.json
    eval_path = get_path('extraction_eval.evaluation', experiment=experiment)

    if eval_path.exists():
        with open(eval_path) as f:
            results = json.load(f)
        print(f"Updating existing evaluation file: {eval_path}")
    else:
        results = {
            'extraction_experiment': experiment,
            'all_results': [],
            'best_per_trait': [],
            'best_vector_similarity': {}
        }
        print(f"Creating new evaluation file: {eval_path}")

    # Add method_similarities
    results['method_similarities'] = similarities

    # Save
    eval_path.parent.mkdir(parents=True, exist_ok=True)
    with open(eval_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Method similarities saved for {len(similarities)} traits")
    print(f"   Output: {eval_path}")


if __name__ == "__main__":
    fire.Fire(compute_method_similarities)
