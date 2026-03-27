#!/usr/bin/env python3
"""
Cross-layer vector similarity: how much do trait vectors at different layers agree?

If vectors are similar across layers, the "best layer" matters less.
If they diverge, different layers capture different aspects of the trait.
"""

import sys
import torch
import json
import numpy as np
from pathlib import Path
from collections import defaultdict

ROOT = Path("/home/dev/trait-interp")
sys.path.insert(0, str(ROOT))

from utils.vectors import load_vector_with_baseline

RESULTS_DIR = Path(__file__).parent


def main():
    experiment = "starter"
    model = "qwen3.5-9b"
    n_layers = 32

    traits = [
        "starter_traits/sycophancy",
        "starter_traits/evil",
        "starter_traits/concealment",
        "starter_traits/hallucination",
        "starter_traits/golden_gate_bridge",
        "starter_traits/refusal",
    ]

    print("=" * 80)
    print("CROSS-LAYER VECTOR SIMILARITY")
    print("Cosine similarity between trait vectors at different layers")
    print("=" * 80)

    all_similarities = {}

    for trait in traits:
        short = trait.split('/')[-1]

        # Load vectors at all layers
        vectors = {}
        for layer in range(n_layers):
            try:
                v, _, _ = load_vector_with_baseline(experiment, trait, "probe", layer, model, "residual", "response[:]")
                vectors[layer] = v.float()
            except:
                pass

        if len(vectors) < 2:
            continue

        # Compute pairwise cosine similarities
        layers = sorted(vectors.keys())
        n = len(layers)

        # Similarity matrix
        sim_matrix = np.zeros((n, n))
        for i, li in enumerate(layers):
            for j, lj in enumerate(layers):
                cos = torch.nn.functional.cosine_similarity(
                    vectors[li].unsqueeze(0), vectors[lj].unsqueeze(0)
                ).item()
                sim_matrix[i, j] = cos

        # Key metrics
        # Adjacent layer similarity
        adj_sims = [sim_matrix[i, i+1] for i in range(n-1)]
        # Distant layer similarity (first vs last third)
        early = layers[:n//3]
        late = layers[2*n//3:]
        cross_sims = []
        for i, li in enumerate(layers):
            if li in early:
                for j, lj in enumerate(layers):
                    if lj in late:
                        cross_sims.append(sim_matrix[i, j])

        # Steering peak vs detection peak similarity
        steering_peaks = {'sycophancy': 13, 'evil': 17, 'concealment': 15,
                         'hallucination': 18, 'golden_gate_bridge': 10, 'refusal': 7}
        steer_l = steering_peaks.get(short)
        det_l = 18  # robust peak for most traits

        steer_det_sim = None
        if steer_l is not None and steer_l in vectors and det_l in vectors:
            steer_det_sim = torch.nn.functional.cosine_similarity(
                vectors[steer_l].unsqueeze(0), vectors[det_l].unsqueeze(0)
            ).item()

        print(f"\n  {short}:")
        print(f"    Adjacent layer sim (mean): {np.mean(adj_sims):.3f} (std={np.std(adj_sims):.3f})")
        print(f"    Early↔Late sim (mean):     {np.mean(cross_sims):.3f} (std={np.std(cross_sims):.3f})")
        if steer_det_sim is not None:
            print(f"    Steering L{steer_l} ↔ Detection L{det_l}: {steer_det_sim:.3f}")

        # Where does similarity drop most?
        drops = [(layers[i], layers[i+1], adj_sims[i]) for i in range(len(adj_sims))]
        biggest_drop = min(drops, key=lambda x: x[2])
        print(f"    Biggest drop: L{biggest_drop[0]}→L{biggest_drop[1]}: {biggest_drop[2]:.3f}")

        all_similarities[short] = {
            'adj_mean': float(np.mean(adj_sims)),
            'early_late_mean': float(np.mean(cross_sims)),
            'steer_det_sim': float(steer_det_sim) if steer_det_sim else None,
        }

    # Summary
    print(f"\n{'=' * 60}")
    print(f"{'Trait':<20} {'Adj sim':<10} {'Early↔Late':<12} {'Steer↔Det'}")
    print("-" * 50)
    for trait, sims in all_similarities.items():
        sd = f"{sims['steer_det_sim']:.3f}" if sims['steer_det_sim'] else "—"
        print(f"{trait:<20} {sims['adj_mean']:.3f}     {sims['early_late_mean']:.3f}       {sd}")

    json.dump(all_similarities, open(RESULTS_DIR / "cross_layer_similarity.json", 'w'), indent=2)


if __name__ == "__main__":
    main()
