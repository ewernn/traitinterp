"""Verify Stage 8 with TRUE cosine similarity (not length-weighted projection).

Round-20 critic caught a methodological deviation from the paper. Paper line 814:

  "We measured the cosine similarity between the emotion probe vectors
   and model activations, on the colon token after 'Assistant,' immediately
   prior to its response."

Our `stage8_post_training.py` uses `projection(act, vec, normalize_vector=True)`
which is `a · (v/||v||)` — vector normalized, activation NOT normalized. That's
length-weighted dot product, not cosine similarity.

This matters for base-vs-instruct comparisons because instruct tuning can
inflate residual-stream activation norms. A pure norm inflation would
contribute linearly to our length-weighted metric but contribute zero to
true cosine similarity (which divides by ||a||).

This script:
1. Captures activations at L49 for base (3.1) and instruct (3.3) models on
   the same 10 neutral + 10 challenging prompts used in stage8_post_training.py
2. Computes BOTH metrics for each (emotion, prompt) pair:
   - length-weighted: a · (v/||v||)
   - true cosine: a · v / (||a|| · ||v||)
3. Aggregates per-emotion shifts under both metrics
4. Reports top-10 up/down, per-metric PC1 centroids vs permutation null,
   and whether the conclusions hold under true cosine
5. Also reports base vs instruct activation norms ||a|| at L49 to diagnose
   whether norm inflation is driving the length-weighted result

Input:
    - datasets/post_training_prompts.json (10+10 prompts)
    - 171 emotion vectors at L49 (mean_diff+gm+pc50)
    - results/stage8_post_training.json (for comparison baseline)

Output:
    - results/stage8_cosine_verification.json

Usage:
    python experiments/ant_emotion_concepts/scripts/stage8_cosine_verify.py
"""
import gc
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.math import batch_cosine_similarity
from utils.model import load_model
from shared import load_emotion_vectors_as_dict, capture_activations_at_position
from stage8_post_training import format_prompt_for_model

EXPERIMENT = "ant_emotion_concepts"
CATEGORY = "ant_emotion_concepts"
LAYER = 49
METHOD = "mean_diff+gm+pc50"
MODEL_BASE = "unsloth/Meta-Llama-3.1-70B-bnb-4bit"
MODEL_INSTRUCT = "meta-llama/Llama-3.3-70B-Instruct"

BASE_DIR = Path("/home/dev/traitinterp/experiments/ant_emotion_concepts")
PROMPTS_JSON = BASE_DIR / "datasets" / "post_training_prompts.json"
OUT_JSON = BASE_DIR / "results" / "stage8_cosine_verification.json"
STAGE3_PCA = BASE_DIR / "results" / "stage3_geometry" / "pca_analysis.json"


def capture_l49_acts(model_id, prompts, is_base):
    """Load model, capture L49 last-token activations, return {pid: act_tensor}."""
    print(f"\nLoading {model_id}...")
    t0 = time.time()
    model, tokenizer = load_model(model_id, load_in_4bit=True)
    print(f"  Loaded in {time.time()-t0:.1f}s, VRAM {torch.cuda.memory_allocated()/1e9:.1f} GB")

    formatted = [format_prompt_for_model(p["prompt"], is_base) for p in prompts]
    acts, _ = capture_activations_at_position(
        model, tokenizer, formatted, LAYER,
        position="last", use_chat_template=False,
    )
    result = {prompts[i]["id"]: acts[i].cpu().float() for i in range(len(prompts))}

    # Report activation norms
    norms = [a.norm().item() for a in result.values()]
    print(f"  L49 activation norms: mean={np.mean(norms):.2f}, sd={np.std(norms):.2f}, min={min(norms):.2f}, max={max(norms):.2f}")

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    print(f"  Unloaded. VRAM {torch.cuda.memory_allocated()/1e9:.1f} GB")

    return result, norms


def compute_both_metrics(acts, vectors):
    """For each (pid, emotion), compute both length-weighted and cosine.

    Returns:
        length_weighted: {pid: {emotion: value}}
        cosine: {pid: {emotion: value}}
    """
    length_weighted = {}
    cosine = {}
    for pid, act in acts.items():
        length_weighted[pid] = {}
        cosine[pid] = {}
        a = act.float()
        a_norm = a.norm()
        for emotion, vec in vectors.items():
            if vec is None:
                continue
            v = vec.float()
            v_unit = v / (v.norm() + 1e-8)
            # length-weighted: a · (v/||v||)
            lw = float(torch.dot(a, v_unit))
            length_weighted[pid][emotion] = lw
            # true cosine: lw / ||a||
            cos = lw / (float(a_norm) + 1e-8)
            cosine[pid][emotion] = cos
    return length_weighted, cosine


def per_emotion_shift(base_projs, inst_projs, prompt_ids, emotions):
    """shift[emotion] = mean over prompts of (inst_proj - base_proj)."""
    shifts = {}
    for e in emotions:
        base_vals = [base_projs[pid][e] for pid in prompt_ids
                     if pid in base_projs and e in base_projs[pid]]
        inst_vals = [inst_projs[pid][e] for pid in prompt_ids
                     if pid in inst_projs and e in inst_projs[pid]]
        if base_vals and inst_vals:
            shifts[e] = float(np.mean(inst_vals) - np.mean(base_vals))
    return shifts


def top_k(shifts, k, reverse):
    return [e for e, _ in sorted(shifts.items(), key=lambda kv: kv[1], reverse=reverse)[:k]]


def pc1_null_ci(all_pc1_values, n_perm=10_000, n_draw=10, seed=42):
    """Permutation null for mean of n_draw random PC1 values."""
    rng = np.random.default_rng(seed)
    nulls = np.array([
        all_pc1_values[rng.choice(len(all_pc1_values), size=n_draw, replace=False)].mean()
        for _ in range(n_perm)
    ])
    return {
        "mean": float(nulls.mean()),
        "sd": float(nulls.std()),
        "ci95_low": float(np.percentile(nulls, 2.5)),
        "ci95_high": float(np.percentile(nulls, 97.5)),
    }, nulls


def main():
    print("Stage 8 cosine verification — recompute with true cosine similarity")
    print(f"  Base:     {MODEL_BASE}")
    print(f"  Instruct: {MODEL_INSTRUCT}")
    print(f"  Layer:    L{LAYER}")

    # Load prompts
    with open(PROMPTS_JSON) as f:
        prompts_data = json.load(f)
    neutral = [{"id": f"neutral_{i}", "prompt": p, "category": "neutral"}
               for i, p in enumerate(prompts_data["neutral_prompts"])]
    challenging = [{"id": f"challenge_{i}", "prompt": p, "category": "challenging"}
                   for i, p in enumerate(prompts_data["challenging_prompts"])]
    all_prompts = neutral + challenging
    neutral_ids = [p["id"] for p in neutral]
    challenging_ids = [p["id"] for p in challenging]
    all_ids = neutral_ids + challenging_ids
    print(f"  Loaded {len(all_prompts)} prompts ({len(neutral_ids)} neutral + {len(challenging_ids)} challenging)")

    # Load 171 emotion vectors at L49 (from instruct variant — same as Stage 8)
    vectors = load_emotion_vectors_as_dict(
        experiment=EXPERIMENT, category=CATEGORY, layer=LAYER,
        model_variant="instruct", method=METHOD,
    )
    vectors = {e: v for e, v in vectors.items() if v is not None}
    emotions = sorted(vectors.keys())
    print(f"  Loaded {len(emotions)} emotion vectors at L{LAYER}")

    # Capture base activations
    base_acts, base_norms = capture_l49_acts(MODEL_BASE, all_prompts, is_base=True)
    base_lw, base_cos = compute_both_metrics(base_acts, vectors)

    # Capture instruct activations
    inst_acts, inst_norms = capture_l49_acts(MODEL_INSTRUCT, all_prompts, is_base=False)
    inst_lw, inst_cos = compute_both_metrics(inst_acts, vectors)

    # Per-emotion shifts under both metrics, on the CHALLENGING subset (load-bearing)
    shifts_lw_challenging = per_emotion_shift(base_lw, inst_lw, challenging_ids, emotions)
    shifts_cos_challenging = per_emotion_shift(base_cos, inst_cos, challenging_ids, emotions)
    shifts_lw_neutral = per_emotion_shift(base_lw, inst_lw, neutral_ids, emotions)
    shifts_cos_neutral = per_emotion_shift(base_cos, inst_cos, neutral_ids, emotions)
    shifts_lw_all = per_emotion_shift(base_lw, inst_lw, all_ids, emotions)
    shifts_cos_all = per_emotion_shift(base_cos, inst_cos, all_ids, emotions)

    # Top-10 up under each metric
    top10_lw_chal = top_k(shifts_lw_challenging, 10, reverse=True)
    top10_cos_chal = top_k(shifts_cos_challenging, 10, reverse=True)
    top10_lw_neut = top_k(shifts_lw_neutral, 10, reverse=True)
    top10_cos_neut = top_k(shifts_cos_neutral, 10, reverse=True)

    # Overlap between metrics
    overlap_chal = len(set(top10_lw_chal) & set(top10_cos_chal))
    overlap_neut = len(set(top10_lw_neut) & set(top10_cos_neut))

    # Load PCA basis for PC1 centroid
    pca = json.load(open(STAGE3_PCA))
    projections = np.array(pca["projections"])
    trait_names = pca["trait_names"]
    name_to_pc1 = {n: float(projections[i, 0]) for i, n in enumerate(trait_names)}
    all_pc1 = projections[:, 0]

    def pc1_centroid(emotion_list):
        vals = [name_to_pc1[e] for e in emotion_list if e in name_to_pc1]
        return float(np.mean(vals)) if vals else float("nan")

    null_stats, null_dist = pc1_null_ci(all_pc1)

    def z_p(centroid):
        z = (centroid - null_stats["mean"]) / null_stats["sd"]
        p = float(np.mean(np.abs(null_dist - null_stats["mean"]) >= abs(centroid - null_stats["mean"])))
        return float(z), p

    pc1_lw_chal = pc1_centroid(top10_lw_chal)
    pc1_cos_chal = pc1_centroid(top10_cos_chal)
    z_lw_chal, p_lw_chal = z_p(pc1_lw_chal)
    z_cos_chal, p_cos_chal = z_p(pc1_cos_chal)

    pc1_lw_neut = pc1_centroid(top10_lw_neut)
    pc1_cos_neut = pc1_centroid(top10_cos_neut)
    z_lw_neut, p_lw_neut = z_p(pc1_lw_neut)
    z_cos_neut, p_cos_neut = z_p(pc1_cos_neut)

    # Report
    print("\n" + "=" * 78)
    print("RESULTS: length-weighted vs true cosine, challenging-only")
    print("=" * 78)
    print(f"\nActivation norms at L49 (20 prompts):")
    print(f"  base:     mean={np.mean(base_norms):.2f}, sd={np.std(base_norms):.2f}")
    print(f"  instruct: mean={np.mean(inst_norms):.2f}, sd={np.std(inst_norms):.2f}")
    print(f"  ratio (inst/base): {np.mean(inst_norms)/np.mean(base_norms):.3f}")
    print()
    print(f"Null CI95 (N=10 of 171 random PC1): [{null_stats['ci95_low']:+.3f}, {null_stats['ci95_high']:+.3f}]")
    print()
    print(f"CHALLENGING-ONLY top-10 up:")
    print(f"  length-weighted: {top10_lw_chal}")
    print(f"  true cosine:     {top10_cos_chal}")
    print(f"  overlap: {overlap_chal}/10")
    print(f"  length-weighted PC1 centroid = {pc1_lw_chal:+.4f} (z = {z_lw_chal:+.2f}, p = {p_lw_chal:.4f})")
    print(f"  true cosine PC1 centroid    = {pc1_cos_chal:+.4f} (z = {z_cos_chal:+.2f}, p = {p_cos_chal:.4f})")
    print()
    print(f"NEUTRAL-ONLY top-10 up:")
    print(f"  length-weighted: {top10_lw_neut}")
    print(f"  true cosine:     {top10_cos_neut}")
    print(f"  overlap: {overlap_neut}/10")
    print(f"  length-weighted PC1 centroid = {pc1_lw_neut:+.4f} (z = {z_lw_neut:+.2f})")
    print(f"  true cosine PC1 centroid    = {pc1_cos_neut:+.4f} (z = {z_cos_neut:+.2f})")

    # Save
    result = {
        "description": "Stage 8 recomputed with true cosine similarity (not length-weighted projection). Addresses round-20 critic's finding that paper specifies cosine but our stage8_post_training.py uses a · (v/||v||).",
        "layer": LAYER,
        "n_prompts": {"neutral": len(neutral_ids), "challenging": len(challenging_ids)},
        "activation_norms": {
            "base_mean": float(np.mean(base_norms)),
            "base_sd": float(np.std(base_norms)),
            "instruct_mean": float(np.mean(inst_norms)),
            "instruct_sd": float(np.std(inst_norms)),
            "ratio_inst_over_base": float(np.mean(inst_norms)/np.mean(base_norms)),
            "base_norms_per_prompt": {pid: float(base_acts[pid].norm()) for pid in all_ids},
            "inst_norms_per_prompt": {pid: float(inst_acts[pid].norm()) for pid in all_ids},
        },
        "null_ci95": null_stats,
        "challenging_only": {
            "top10_length_weighted": top10_lw_chal,
            "top10_true_cosine": top10_cos_chal,
            "overlap": overlap_chal,
            "pc1_length_weighted": round(pc1_lw_chal, 4),
            "pc1_true_cosine": round(pc1_cos_chal, 4),
            "z_length_weighted": round(z_lw_chal, 4),
            "z_true_cosine": round(z_cos_chal, 4),
            "p_length_weighted": round(p_lw_chal, 4),
            "p_true_cosine": round(p_cos_chal, 4),
        },
        "neutral_only": {
            "top10_length_weighted": top10_lw_neut,
            "top10_true_cosine": top10_cos_neut,
            "overlap": overlap_neut,
            "pc1_length_weighted": round(pc1_lw_neut, 4),
            "pc1_true_cosine": round(pc1_cos_neut, 4),
            "z_length_weighted": round(z_lw_neut, 4),
            "z_true_cosine": round(z_cos_neut, 4),
        },
        "shifts_cosine_challenging": {e: round(shifts_cos_challenging[e], 6) for e in emotions if e in shifts_cos_challenging},
        "shifts_cosine_neutral": {e: round(shifts_cos_neutral[e], 6) for e in emotions if e in shifts_cos_neutral},
        "shifts_lw_challenging": {e: round(shifts_lw_challenging[e], 6) for e in emotions if e in shifts_lw_challenging},
        "shifts_lw_neutral": {e: round(shifts_lw_neutral[e], 6) for e in emotions if e in shifts_lw_neutral},
    }

    OUT_JSON.write_text(json.dumps(result, indent=2))
    print(f"\nSaved: {OUT_JSON}")


if __name__ == "__main__":
    main()
