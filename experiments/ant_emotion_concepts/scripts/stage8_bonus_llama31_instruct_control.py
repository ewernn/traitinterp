#!/usr/bin/env python3
"""Stage 8 bonus: Llama 3.1 70B Instruct control — WITHIN-VERSION post-training shift.

Disambiguates the Stage 8 "direction opposite paper" finding. Stage 8 compared
Llama 3.1 BASE to Llama 3.3 INSTRUCT — a cross-VERSION comparison, not within-
model post-training like the paper's Sonnet base vs post-trained.

This bonus run compares Llama 3.1 BASE to Llama 3.1 INSTRUCT (same version, just
post-training). If the shift direction matches Stage 8's 3.1-base → 3.3-instruct
profile, the "Meta RLHF direction" interpretation is reinforced. If different,
the Stage 8 finding is largely a cross-version artifact.

Design: load 3.1 base → capture per-prompt activations → unload; load 3.1
instruct → capture → unload. Compute per-emotion means for both. Compute
within-version shift = instruct_means - base_means. Correlate this profile
against Stage 8's `avg_shifts` (which is the 3.1-base → 3.3-instruct profile).

Note: re-runs 3.1 base rather than reusing Stage 8's result, because Stage 8
saved only the diffs (`avg_shifts`), not the raw per-emotion base_means.
Re-running takes an extra ~10 min but produces clean within-version numbers.

Input:
    - datasets/post_training_prompts.json (20 prompts, same as Stage 8)
    - 171 emotion vectors at L49 (mean_diff+gm+pc50)
    - results/stage8_post_training.json (for avg_shifts = cross-version reference)
Output:
    - results/stage8_bonus_llama31_instruct_control.json

Usage:
    python experiments/ant_emotion_concepts/scripts/stage8_bonus_llama31_instruct_control.py
"""
import gc
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.model import load_model
from shared import load_emotion_vectors_as_dict
from stage8_post_training import (
    _measure_activations_at_last_token,
    _project_activations_onto_emotions,
)

EXPERIMENT = "ant_emotion_concepts"
CATEGORY = "ant_emotion_concepts"
LAYER = 49
METHOD = "mean_diff+gm+pc50"
MODEL_BASE = "unsloth/Meta-Llama-3.1-70B-bnb-4bit"
MODEL_INSTRUCT = "unsloth/Meta-Llama-3.1-70B-Instruct-bnb-4bit"

STAGE8_JSON = Path("/home/dev/traitinterp/experiments/ant_emotion_concepts/results/stage8_post_training.json")
PROMPTS_JSON = Path("/home/dev/traitinterp/experiments/ant_emotion_concepts/datasets/post_training_prompts.json")
OUT_JSON = Path("/home/dev/traitinterp/experiments/ant_emotion_concepts/results/stage8_bonus_llama31_instruct_control.json")


def capture_and_project(model_id, prompts, vectors, is_base):
    """Load model, capture L49 last-token activations, project onto emotion vectors, unload.

    Returns {prompt_id: {emotion: projection_scalar}}.
    """
    print(f"\nLoading {model_id}...")
    t0 = time.time()
    model, tokenizer = load_model(model_id, load_in_4bit=True)
    print(f"  Loaded in {time.time()-t0:.1f}s, VRAM {torch.cuda.memory_allocated()/1e9:.1f} GB")

    print(f"  Measuring activations at L{LAYER} (is_base={is_base})...")
    acts = _measure_activations_at_last_token(
        model, tokenizer, prompts, layer=LAYER, is_base=is_base,
    )
    projs = _project_activations_onto_emotions(acts, vectors)

    # Unload model to free VRAM for the next load
    del model, tokenizer, acts
    torch.cuda.empty_cache()
    gc.collect()
    print(f"  Unloaded. VRAM {torch.cuda.memory_allocated()/1e9:.1f} GB")

    return projs


def per_emotion_means(projs, emotion_list, prompt_ids):
    """Average per-prompt projections to per-emotion means."""
    return {
        e: float(np.mean([projs[pid][e] for pid in prompt_ids if pid in projs and e in projs[pid]]))
        for e in emotion_list
    }


def main():
    print("Stage 8 bonus: Llama 3.1 70B Instruct within-version control")
    print(f"  Base:     {MODEL_BASE}")
    print(f"  Instruct: {MODEL_INSTRUCT}")
    print(f"  Layer:    L{LAYER}")
    print(f"  Method:   {METHOD}")

    # Stage 8 cross-version reference (3.1-base → 3.3-instruct)
    with open(STAGE8_JSON) as f:
        stage8 = json.load(f)
    cross_version_shifts = stage8["avg_shifts"]  # {emotion: shift}
    print(f"  Loaded Stage 8 cross-version shifts: {len(cross_version_shifts)} emotions")

    # Load the 20 prompts (same ones Stage 8 used)
    with open(PROMPTS_JSON) as f:
        prompts_data = json.load(f)
    neutral_prompts = [{"id": f"neutral_{i}", "prompt": p, "category": "neutral"}
                       for i, p in enumerate(prompts_data["neutral_prompts"])]
    challenge_prompts = [{"id": f"challenge_{i}", "prompt": p, "category": "challenging"}
                         for i, p in enumerate(prompts_data["challenging_prompts"])]
    all_prompts = neutral_prompts + challenge_prompts
    prompt_ids = [p["id"] for p in all_prompts]
    print(f"  Loaded {len(all_prompts)} prompts ({len(neutral_prompts)} neutral + {len(challenge_prompts)} challenging)")

    # Load the 171 L49 emotion vectors (same ones Stage 2 extracted on Llama 3.3 Instruct)
    # Paper design choice: apply instruct-derived vectors to both comparison models.
    vectors = load_emotion_vectors_as_dict(
        experiment=EXPERIMENT,
        category=CATEGORY,
        layer=LAYER,
        model_variant="instruct",
        method=METHOD,
    )
    # Filter any None vectors (shouldn't happen, defensive)
    vectors = {e: v for e, v in vectors.items() if v is not None}
    emotion_list = sorted(vectors.keys())
    print(f"  Loaded {len(emotion_list)} emotion vectors at L{LAYER}")

    # Run both models sequentially
    projs_base = capture_and_project(MODEL_BASE, all_prompts, vectors, is_base=True)
    projs_instruct = capture_and_project(MODEL_INSTRUCT, all_prompts, vectors, is_base=False)

    # Compute per-emotion means
    base_means_31 = per_emotion_means(projs_base, emotion_list, prompt_ids)
    instruct_means_31 = per_emotion_means(projs_instruct, emotion_list, prompt_ids)

    # Within-version shift: 3.1-instruct - 3.1-base
    within_version_shifts = {
        e: instruct_means_31[e] - base_means_31[e]
        for e in emotion_list
    }

    # Top shifts (within-version)
    sorted_31 = sorted(within_version_shifts.keys(), key=lambda e: within_version_shifts[e], reverse=True)
    top_increases_31 = [(e, round(within_version_shifts[e], 6)) for e in sorted_31[:10]]
    top_decreases_31 = [(e, round(within_version_shifts[e], 6)) for e in sorted_31[-10:]]

    # Correlate within-version (3.1-instruct) vs cross-version (Stage 8, 3.3-instruct)
    common_emotions = [e for e in emotion_list if e in cross_version_shifts]
    within_vec = np.array([within_version_shifts[e] for e in common_emotions])
    cross_vec = np.array([cross_version_shifts[e] for e in common_emotions])

    pearson_r, pearson_p = pearsonr(within_vec, cross_vec)
    spearman_rho, spearman_p = spearmanr(within_vec, cross_vec)

    # Top-10 overlap
    top10_31_inc_set = set(e for e, _ in top_increases_31)
    top10_33_inc_set = set(e for e, _ in sorted(cross_version_shifts.items(), key=lambda kv: kv[1], reverse=True)[:10])
    top10_31_dec_set = set(e for e, _ in top_decreases_31)
    top10_33_dec_set = set(e for e, _ in sorted(cross_version_shifts.items(), key=lambda kv: kv[1])[:10])
    inc_overlap = len(top10_31_inc_set & top10_33_inc_set)
    dec_overlap = len(top10_31_dec_set & top10_33_dec_set)

    # Interpretation
    if pearson_r > 0.5:
        interpretation = (
            "HIGH POSITIVE: 3.1-base → 3.1-instruct shift direction strongly matches "
            "3.1-base → 3.3-instruct. Stage 8's 'opposite direction from paper' finding "
            "is NOT a cross-version artifact — it reflects a consistent Meta RLHF "
            "direction across multiple instruct-tuned releases."
        )
    elif pearson_r > 0.2:
        interpretation = (
            "MODERATE POSITIVE: partial agreement. Meta's RLHF direction is consistent "
            "but the 3.3 release shifted it meaningfully. Stage 8's finding is partly "
            "real, partly cross-version."
        )
    elif pearson_r > -0.2:
        interpretation = (
            "UNCORRELATED: 3.1 and 3.3 instruct models sit at different places in "
            "the emotion-shift space. Stage 8's finding is largely a cross-version artifact."
        )
    else:
        interpretation = (
            "NEGATIVE: 3.1-instruct and 3.3-instruct shift in OPPOSITE directions from "
            "the 3.1 base. Surprising result suggesting Meta's RLHF direction inverted "
            "between releases."
        )

    # Save
    result = {
        "experiment": "stage8_bonus_llama31_instruct_control",
        "timestamp": datetime.utcnow().isoformat(),
        "model_base": MODEL_BASE,
        "model_instruct": MODEL_INSTRUCT,
        "stage8_cross_version_instruct": stage8["instruct_model"],
        "layer": LAYER,
        "method": METHOD,
        "n_prompts": len(all_prompts),
        "n_emotions": len(common_emotions),
        "base_means_31": {e: round(base_means_31[e], 6) for e in sorted_31},
        "instruct_means_31": {e: round(instruct_means_31[e], 6) for e in sorted_31},
        "within_version_shifts_31": {e: round(within_version_shifts[e], 6) for e in sorted_31},
        "cross_version_shifts_33_reference": cross_version_shifts,
        "correlation_within_vs_cross_version": {
            "pearson_r": float(pearson_r),
            "pearson_p": float(pearson_p),
            "spearman_rho": float(spearman_rho),
            "spearman_p": float(spearman_p),
        },
        "top10_increases_31_instruct": top_increases_31,
        "top10_decreases_31_instruct": top_decreases_31,
        "top10_overlap_with_stage8_cross_version": {
            "increases": inc_overlap,
            "decreases": dec_overlap,
        },
        "interpretation": interpretation,
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved: {OUT_JSON}")
    print(f"\nKey result:")
    print(f"  Pearson r (3.1-instruct within-version vs 3.3-instruct cross-version): {pearson_r:+.3f}  (p={pearson_p:.2e})")
    print(f"  Spearman rho: {spearman_rho:+.3f}")
    print(f"  Top-10 increase overlap: {inc_overlap}/10")
    print(f"  Top-10 decrease overlap: {dec_overlap}/10")
    print(f"\n{interpretation}")


if __name__ == "__main__":
    main()
