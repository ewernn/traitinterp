#!/usr/bin/env python3
"""Stage 8 bonus: Llama 3.1 70B Instruct control run.

Disambiguates the Stage 8 "direction opposite paper" finding. Stage 8 compared
Llama 3.1 BASE to Llama 3.3 INSTRUCT — a cross-VERSION comparison, not within-
model post-training like the paper's Sonnet base vs post-trained.

This bonus run compares Llama 3.1 BASE to Llama 3.1 INSTRUCT (same version, just
post-training). If the shift direction matches Stage 8's 3.1-base → 3.3-instruct,
the "Meta RLHF objective" interpretation is reinforced (same direction across
two Meta instruct models). If different, the Stage 8 finding is partly a
cross-version artifact.

Reuses existing helpers from stage8_post_training.py — no new probe extraction.
Uses the same 20 prompts (neutral + challenging) from post_training_prompts.json.
Emotion vectors from Stage 2 (extracted on Llama 3.3 Instruct) are reused — the
paper's design choice is to apply instruct-derived vectors to both comparison
models.

Input:
    - datasets/post_training_prompts.json (20 prompts, same as Stage 8)
    - experiments/ant_emotion_concepts/results/stage8_post_training.json
      (existing base_means from Llama 3.1 base)
    - 171 emotion vectors at L49 (mean_diff+gm+pc50)
Output:
    - results/stage8_bonus_llama31_instruct_control.json

Usage:
    python experiments/ant_emotion_concepts/scripts/stage8_bonus_llama31_instruct_control.py
"""
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.model import load_model
from shared import load_emotion_vectors_as_dict
from stage8_post_training import (
    _measure_activations_at_last_token,
    _project_activations_onto_emotions,
    run_activation_comparison,
    discover_all_emotions,
)

EXPERIMENT = "ant_emotion_concepts"
LAYER = 49
METHOD = "mean_diff+gm+pc50"
MODEL = "unsloth/Meta-Llama-3.1-70B-Instruct-bnb-4bit"
STAGE8_JSON = Path("/home/dev/traitinterp/experiments/ant_emotion_concepts/results/stage8_post_training.json")
PROMPTS_JSON = Path("/home/dev/traitinterp/experiments/ant_emotion_concepts/datasets/post_training_prompts.json")
OUT_JSON = Path("/home/dev/traitinterp/experiments/ant_emotion_concepts/results/stage8_bonus_llama31_instruct_control.json")


def main():
    print(f"Stage 8 bonus: Llama 3.1 70B Instruct control run")
    print(f"  Model: {MODEL}")
    print(f"  Layer: L{LAYER}")
    print(f"  Method: {METHOD}")

    # Load Stage 8 existing results (we need 3.1-base activations as the comparison baseline)
    with open(STAGE8_JSON) as f:
        stage8 = json.load(f)
    prior_base_means = stage8["base_means"]
    prior_instruct_means = stage8["instruct_means"]  # from Llama 3.3 Instruct
    prior_diffs = stage8["diffs"]  # instruct(3.3) - base(3.1), the original Stage 8 signal
    print(f"  Loaded Stage 8 baseline: {len(prior_base_means)} emotions")

    # Load the 20 prompts (same ones Stage 8 used)
    with open(PROMPTS_JSON) as f:
        prompts_data = json.load(f)
    neutral_prompts = [{"id": f"neutral_{i}", "prompt": p, "category": "neutral"}
                       for i, p in enumerate(prompts_data["neutral_prompts"])]
    challenge_prompts = [{"id": f"challenge_{i}", "prompt": p, "category": "challenging"}
                         for i, p in enumerate(prompts_data["challenging_prompts"])]
    all_prompts = neutral_prompts + challenge_prompts
    print(f"  Loaded {len(all_prompts)} prompts ({len(neutral_prompts)} neutral + {len(challenge_prompts)} challenging)")

    # Load emotion vectors at L49 (same ones Stage 2 extracted on Llama 3.3 Instruct)
    emotions = discover_all_emotions(EXPERIMENT)
    vectors = load_emotion_vectors_as_dict(
        experiment=EXPERIMENT, category=EXPERIMENT, emotions=emotions,
        layer=LAYER, method=METHOD, component="residual", position="response[50:]",
        model_variant="instruct",
    )
    vectors = {e: v for e, v in vectors.items() if v is not None}
    print(f"  Loaded {len(vectors)} emotion vectors at L{LAYER}")

    # Load Llama 3.1 Instruct and measure activations
    print(f"\nLoading {MODEL} (bnb int4)...")
    import time
    t0 = time.time()
    model, tokenizer = load_model(MODEL, load_in_4bit=True)
    print(f"  Loaded in {time.time()-t0:.1f}s. VRAM: {torch.cuda.memory_allocated()/1e9:.1f} GB")

    print(f"\nMeasuring activations at L{LAYER} last-token on {len(all_prompts)} prompts...")
    acts_31_instruct = _measure_activations_at_last_token(
        model, tokenizer, all_prompts, layer=LAYER, is_base=False,
    )
    print(f"  Captured {len(acts_31_instruct)} activations")

    # Free the model (bonus control is a single-pass job)
    del model
    torch.cuda.empty_cache()
    import gc; gc.collect()

    # Project onto emotion vectors
    projs_31_instruct = _project_activations_onto_emotions(acts_31_instruct, vectors)

    # Compute per-emotion mean projection (averaged across all 20 prompts)
    emotion_list = sorted(vectors.keys())
    prompt_ids = [p["id"] for p in all_prompts]
    summary_31 = run_activation_comparison(
        base_projections={pid: {e: prior_base_means[e] for e in emotion_list if e in prior_base_means}
                         for pid in prompt_ids},
        instruct_projections=projs_31_instruct,
        prompt_ids=prompt_ids,
        emotions=emotion_list,
        category="all",
    )
    # NOTE: run_activation_comparison averages across prompts per emotion, so the
    # above uses the Stage 8 per-emotion means as a uniform base (one-sample avg
    # for each prompt — this is a simplification, treating prior_base_means as
    # the per-prompt base activation). The resulting diffs are 3.1-instruct - 3.1-base
    # using the per-emotion means as the base proxy. Not exact but practical.
    instruct_31_means = summary_31["instruct_means"]

    # Compute the 3.1-base → 3.1-instruct shift
    within_version_diffs = {
        e: instruct_31_means[e] - prior_base_means[e]
        for e in emotion_list
        if e in instruct_31_means and e in prior_base_means
    }
    sorted_31 = sorted(within_version_diffs.keys(), key=lambda e: within_version_diffs[e], reverse=True)
    top_increases_31 = [(e, round(within_version_diffs[e], 6)) for e in sorted_31[:10]]
    top_decreases_31 = [(e, round(within_version_diffs[e], 6)) for e in sorted_31[-10:]]

    # Correlate with Stage 8's 3.1-base → 3.3-instruct shift
    common_emotions = [e for e in emotion_list if e in within_version_diffs and e in prior_diffs]
    shifts_31_vec = np.array([within_version_diffs[e] for e in common_emotions])
    shifts_33_vec = np.array([prior_diffs[e] for e in common_emotions])

    # Pearson and Spearman
    from scipy.stats import pearsonr, spearmanr
    pearson_r, pearson_p = pearsonr(shifts_31_vec, shifts_33_vec)
    spearman_rho, spearman_p = spearmanr(shifts_31_vec, shifts_33_vec)

    # Top-k overlap (does the 3.1-instruct shift land on the same emotions as 3.3-instruct?)
    top10_31_inc_set = set(e for e, _ in top_increases_31)
    top10_33_inc_set = set(e for e, _ in sorted(prior_diffs.items(), key=lambda kv: kv[1], reverse=True)[:10])
    top10_31_dec_set = set(e for e, _ in top_decreases_31)
    top10_33_dec_set = set(e for e, _ in sorted(prior_diffs.items(), key=lambda kv: kv[1])[:10])
    inc_overlap = len(top10_31_inc_set & top10_33_inc_set)
    dec_overlap = len(top10_31_dec_set & top10_33_dec_set)

    # Interpretation
    if pearson_r > 0.5:
        interpretation = (
            "HIGH POSITIVE: 3.1-base → 3.1-instruct shift direction strongly matches "
            "3.1-base → 3.3-instruct. Stage 8's 'opposite direction from paper' finding "
            "is NOT a cross-version artifact — it reflects a consistent Meta RLHF "
            "direction across multiple instruct-tuned models."
        )
    elif pearson_r > 0.2:
        interpretation = (
            "MODERATE POSITIVE: partial agreement. Meta's RLHF direction is consistent "
            "but the 3.3 release shifted it meaningfully. Stage 8's finding is partly "
            "real, partly cross-version."
        )
    elif pearson_r > -0.2:
        interpretation = (
            "UNCORRELATED: the 3.1 and 3.3 instruct models sit at different places in "
            "the emotion-shift space. Stage 8's finding is largely a cross-version artifact."
        )
    else:
        interpretation = (
            "NEGATIVE: 3.1-instruct and 3.3-instruct shift in OPPOSITE directions from "
            "the 3.1 base. This would be a surprising result suggesting Meta's RLHF "
            "direction inverted between releases."
        )

    # Save
    result = {
        "experiment": "stage8_bonus_llama31_instruct_control",
        "timestamp": datetime.utcnow().isoformat(),
        "model_3p1_instruct": MODEL,
        "model_3p1_base_via_stage8": stage8["base_model"],
        "model_3p3_instruct_via_stage8": stage8["instruct_model"],
        "layer": LAYER,
        "method": METHOD,
        "n_prompts": len(all_prompts),
        "n_emotions": len(common_emotions),
        "within_version_diffs_31": {e: round(within_version_diffs[e], 6) for e in sorted_31},
        "cross_version_diffs_33": stage8["diffs"],
        "correlation_within_vs_cross_version": {
            "pearson_r": float(pearson_r),
            "pearson_p": float(pearson_p),
            "spearman_rho": float(spearman_rho),
            "spearman_p": float(spearman_p),
        },
        "top10_increases_3p1": top_increases_31,
        "top10_decreases_3p1": top_decreases_31,
        "top10_overlap_with_stage8": {
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
    print(f"  Pearson r (3.1-instruct shift vs 3.3-instruct shift): {pearson_r:+.3f}  (p={pearson_p:.2e})")
    print(f"  Spearman rho: {spearman_rho:+.3f}")
    print(f"  Top-10 increase overlap: {inc_overlap}/10")
    print(f"  Top-10 decrease overlap: {dec_overlap}/10")
    print(f"\n{interpretation}")


if __name__ == "__main__":
    main()
