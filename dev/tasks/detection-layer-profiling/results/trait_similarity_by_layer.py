#!/usr/bin/env python3
"""
Cross-trait vector similarity at each layer: do semantically related traits
share directions? How does this change across layers?
"""

import sys
import torch
import json
import numpy as np
from pathlib import Path
from itertools import combinations

ROOT = Path("/home/dev/trait-interp")
sys.path.insert(0, str(ROOT))

from utils.vectors import load_vector_with_baseline

RESULTS_DIR = Path(__file__).parent


def main():
    experiment = "starter"
    model = "qwen3.5-9b"
    n_layers = 32

    # Load all available trait vectors
    eval_data = json.load(open(RESULTS_DIR / "all_traits_eval.json"))
    traits = sorted(set(r['trait'] for r in eval_data.get('qwen3.5-9b', [])))

    # Load vectors for all traits at key layers
    key_layers = [0, 5, 10, 15, 18, 20, 25, 31]

    vectors = {}  # (trait, layer) -> tensor
    for trait in traits:
        for layer in key_layers:
            try:
                v, _, _ = load_vector_with_baseline(experiment, trait, "probe", layer, model, "residual", "response[:]")
                vectors[(trait, layer)] = v.float()
            except:
                pass

    # At each layer, compute all pairwise trait similarities
    print("=" * 80)
    print("CROSS-TRAIT SIMILARITY BY LAYER")
    print("=" * 80)

    for layer in key_layers:
        available = [t for t in traits if (t, layer) in vectors]
        if len(available) < 2:
            continue

        sims = []
        for t1, t2 in combinations(available, 2):
            cos = torch.nn.functional.cosine_similarity(
                vectors[(t1, layer)].unsqueeze(0),
                vectors[(t2, layer)].unsqueeze(0)
            ).item()
            sims.append((t1.split('/')[-1], t2.split('/')[-1], cos))

        abs_sims = [abs(s[2]) for s in sims]
        print(f"\n  Layer {layer}: mean |cos| = {np.mean(abs_sims):.3f}, max = {max(abs_sims):.3f}")

        # Top 5 most similar pairs
        top_pos = sorted(sims, key=lambda x: x[2], reverse=True)[:5]
        top_neg = sorted(sims, key=lambda x: x[2])[:3]

        print(f"    Most aligned:")
        for t1, t2, cos in top_pos:
            print(f"      {t1} ↔ {t2}: {cos:.3f}")

        print(f"    Most opposed:")
        for t1, t2, cos in top_neg:
            print(f"      {t1} ↔ {t2}: {cos:.3f}")

    # Specific pairs of interest
    pairs = [
        ("starter_traits/sycophancy", "base/alignment/compliance_without_agreement"),
        ("starter_traits/evil", "base/alignment/deception"),
        ("starter_traits/evil", "starter_traits/concealment"),
        ("base/emotion_set/joy", "base/emotion_set/sadness"),
        ("base/emotion_set/anger", "base/emotion_set/calm"),
        ("base/emotion_set/anxiety", "base/emotion_set/confidence"),
        ("starter_traits/refusal", "starter_traits/evil"),
    ]

    print(f"\n{'=' * 80}")
    print("SELECTED TRAIT PAIRS ACROSS LAYERS")
    print(f"{'=' * 80}")

    for t1, t2 in pairs:
        short1 = t1.split('/')[-1]
        short2 = t2.split('/')[-1]
        layer_sims = []
        for layer in range(n_layers):
            if (t1, layer) in vectors and (t2, layer) in vectors:
                cos = torch.nn.functional.cosine_similarity(
                    vectors[(t1, layer)].unsqueeze(0),
                    vectors[(t2, layer)].unsqueeze(0)
                ).item()
                layer_sims.append((layer, cos))

        if not layer_sims:
            # Load missing layers
            for layer in range(n_layers):
                if (t1, layer) not in vectors:
                    try:
                        v, _, _ = load_vector_with_baseline(experiment, t1, "probe", layer, model, "residual", "response[:]")
                        vectors[(t1, layer)] = v.float()
                    except: pass
                if (t2, layer) not in vectors:
                    try:
                        v, _, _ = load_vector_with_baseline(experiment, t2, "probe", layer, model, "residual", "response[:]")
                        vectors[(t2, layer)] = v.float()
                    except: pass

            for layer in range(n_layers):
                if (t1, layer) in vectors and (t2, layer) in vectors:
                    cos = torch.nn.functional.cosine_similarity(
                        vectors[(t1, layer)].unsqueeze(0),
                        vectors[(t2, layer)].unsqueeze(0)
                    ).item()
                    layer_sims.append((layer, cos))

        if layer_sims:
            sims_arr = [s[1] for s in layer_sims]
            print(f"\n  {short1} ↔ {short2}:")
            print(f"    Mean cos: {np.mean(sims_arr):.3f}, Range: [{min(sims_arr):.3f}, {max(sims_arr):.3f}]")
            # Show at key layers
            for l, s in layer_sims:
                if l in [0, 10, 18, 31]:
                    print(f"    L{l}: {s:.3f}")


if __name__ == "__main__":
    main()
