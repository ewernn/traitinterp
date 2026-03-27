#!/usr/bin/env python3
"""
Cluster traits by their layer profile shapes.
Do traits that peak at similar layers share semantic properties?
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict

RESULTS_DIR = Path(__file__).parent


def normalize_profile(ds):
    """Normalize a layer d-curve to unit max for shape comparison."""
    max_d = max(ds) if max(ds) > 0 else 1
    return [d / max_d for d in ds]


def profile_distance(p1, p2):
    """L2 distance between normalized profiles."""
    return np.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))


def main():
    data = json.load(open(RESULTS_DIR / "all_traits_eval.json"))

    for model_name in ['qwen3.5-9b']:
        results = data.get(model_name, [])
        if not results:
            continue

        by_trait = defaultdict(list)
        for r in results:
            by_trait[r['trait']].append(r)

        # Build normalized profiles
        profiles = {}
        for trait in sorted(by_trait.keys()):
            trait_results = sorted(by_trait[trait], key=lambda x: x['layer'])
            ds = [r['val_effect_size'] for r in trait_results]
            if max(ds) > 0.5:  # skip near-zero traits
                profiles[trait] = normalize_profile(ds)

        traits = list(profiles.keys())
        n = len(traits)

        # Compute pairwise distances
        dist_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                dist_matrix[i, j] = profile_distance(profiles[traits[i]], profiles[traits[j]])

        # Simple clustering: find the most similar pairs
        pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                pairs.append((traits[i], traits[j], dist_matrix[i, j]))

        pairs.sort(key=lambda x: x[2])

        print(f"\n{'=' * 70}")
        print(f"  {model_name.upper()} — Trait Profile Shape Clustering")
        print(f"{'=' * 70}")

        print(f"\n  Most similar layer profiles (shape, not magnitude):")
        for t1, t2, d in pairs[:15]:
            s1 = t1.split('/')[-1]
            s2 = t2.split('/')[-1]
            print(f"    {s1:<30} ↔ {s2:<30} dist={d:.3f}")

        print(f"\n  Most dissimilar layer profiles:")
        for t1, t2, d in pairs[-10:]:
            s1 = t1.split('/')[-1]
            s2 = t2.split('/')[-1]
            print(f"    {s1:<30} ↔ {s2:<30} dist={d:.3f}")

        # Group by peak layer
        peak_groups = defaultdict(list)
        for trait in traits:
            ds = profiles[trait]
            peak = ds.index(max(ds))
            if peak < 11:
                group = "early (L0-10)"
            elif peak < 22:
                group = "mid (L11-21)"
            else:
                group = "late (L22-31)"
            peak_groups[group].append(trait.split('/')[-1])

        print(f"\n  Traits grouped by peak location:")
        for group in ["early (L0-10)", "mid (L11-21)", "late (L22-31)"]:
            if group in peak_groups:
                print(f"    {group}: {', '.join(peak_groups[group])}")


if __name__ == "__main__":
    main()
