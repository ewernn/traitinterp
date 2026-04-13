#!/usr/bin/env python3
"""Stage 8: Post-training comparison (base vs instruct).

Covers:
  - Fig 36: Per-emotion activation difference (base vs instruct) on neutral + challenging prompts
  - Fig 84: Layer-wise post-training shifts
  - Figs 37-39: Three deep-dive prompts with all 171 probes

NOT replicated:
  - Figs 85-86 (base-model preference Elo): see comment block at section 8.4 below; use
    `analysis/vectors/preference_elo.compute_elo(..., hard=True)` for future reimplementation.

CAVEAT: The paper compares base and post-trained snapshots of the SAME model (Sonnet 4.5).
We compare Llama 3.1 70B (base) and Llama 3.3 70B Instruct (different versions). Results
may not be directly comparable — treat as "direction of effect" evidence, not exact replication.

Key design choice (matching paper): emotion vectors extracted from the instruct model are
applied to BOTH models. Changes in activation reflect routing differences, not vector differences.

Requires:
  - Extracted emotion vectors (from Stage 2 + cross_trait_normalize.py)
  - Both model variants in config.json (base: Llama 3.1 70B, instruct: Llama 3.3 70B)
  - deep_dive_prompts.json in datasets/inference/ant_emotion_concepts/

Output: experiments/ant_emotion_concepts/results/stage8_post_training/

Usage:
    # Full comparison (loads both models sequentially):
    python experiments/ant_emotion_concepts/scripts/stage8_post_training.py \
        --experiment ant_emotion_concepts --load-in-4bit

    # Activation comparison only (Fig 36, fastest):
    python experiments/ant_emotion_concepts/scripts/stage8_post_training.py \
        --experiment ant_emotion_concepts --load-in-4bit --activations-only

    # Layer sweep only (Fig 84, requires activations already captured):
    python experiments/ant_emotion_concepts/scripts/stage8_post_training.py \
        --experiment ant_emotion_concepts --layer-sweep-only

    # Deep-dive prompts only (Figs 37-39):
    python experiments/ant_emotion_concepts/scripts/stage8_post_training.py \
        --experiment ant_emotion_concepts --load-in-4bit --deep-dive-only
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from core import projection, batch_cosine_similarity
from utils.model import load_model
from utils.paths import (
    get as get_path, get_model_variant, discover_extracted_traits,
    list_layers, get_vector_path,
)
from utils.vectors import load_vector_with_baseline
from utils.capture_activations import capture_at_position
from shared import (
    get_results_dir as _get_results_dir,
    save_results,
    load_single_emotion_vector,
    load_emotion_vectors_as_dict,
)

# =============================================================================
# Constants
# =============================================================================

EXPERIMENT = "ant_emotion_concepts"
CATEGORY = "ant_emotion_concepts"
BASE_VARIANT = "base"
INSTRUCT_VARIANT = "instruct"

# Mid-late layer (~2/3 through 80-layer model)
DEFAULT_LAYER = 53

# Layer range for sweep (evenly spaced, matching paper's 14 layers)
SWEEP_LAYERS = list(range(5, 80, 5))  # [5, 10, 15, ..., 75] = 15 layers

# =============================================================================
# Prompt sets — loaded from datasets/inference/ant_emotion_concepts/*.json
# =============================================================================

_DATASETS_DIR = get_path('datasets.inference') / 'ant_emotion_concepts'


def _load_stage8_prompts() -> Tuple[List[dict], List[dict]]:
    """Load Section 3.5.1 challenging + neutral prompts. Returns (challenging, neutral)."""
    with open(_DATASETS_DIR / 'stage8_prompts.json') as f:
        data = json.load(f)
    return data['challenging_prompts'], data['neutral_prompts']


CHALLENGING_PROMPTS, NEUTRAL_PROMPTS = _load_stage8_prompts()


def discover_all_emotions(experiment: str) -> List[str]:
    """Discover all extracted emotion traits."""
    traits = discover_extracted_traits(experiment)
    # Filter to ant_emotion_concepts category
    emotions = [name for cat, name in traits if cat == CATEGORY]
    if not emotions:
        raise RuntimeError(
            f"No extracted emotion traits found for category '{CATEGORY}'. "
            f"Run extraction + cross_trait_normalize.py first."
        )
    return sorted(emotions)


def format_prompt_for_model(prompt_text: str, is_base: bool) -> str:
    """Format prompt for base (raw text) or instruct (chat template) model.

    Base model: raw text continuation (no chat template).
    Instruct model: user/assistant turn format.

    For both, we measure at the last token before the response (the "colon token"
    in the paper's terminology, though Llama uses different markers).
    """
    if is_base:
        # Base model: simple continuation format
        # The paper measures at the "Assistant:" colon. For base models,
        # use a simple format that puts the model in a similar position.
        return f"Human: {prompt_text}\nAssistant:"
    else:
        # Instruct model: will be tokenized with chat template by tokenize()
        return prompt_text


def _measure_activations_at_last_token(
    model, tokenizer, prompts: List[dict], layer: int,
    is_base: bool = False,
) -> Dict[str, torch.Tensor]:
    """Single-layer last-token capture. Thin wrapper over capture_at_position."""
    formatted = [format_prompt_for_model(p["prompt"], is_base) for p in prompts]
    acts = capture_at_position(
        model, tokenizer, formatted,
        layers=layer, position='prompt[-1]', pool='last', pre_formatted=True,
    )  # [n_prompts, hidden_dim]
    return {p["id"]: acts[i] for i, p in enumerate(prompts)}


def _measure_activations_multilayer(
    model, tokenizer, prompts: List[dict], layers: List[int],
    is_base: bool = False,
) -> Dict[int, Dict[str, torch.Tensor]]:
    """Multi-layer last-token capture. Thin wrapper over capture_at_position."""
    formatted = [format_prompt_for_model(p["prompt"], is_base) for p in prompts]
    acts = capture_at_position(
        model, tokenizer, formatted,
        layers=layers, position='prompt[-1]', pool='last', pre_formatted=True,
    )  # [n_prompts, n_layers, hidden_dim]
    result = {layer: {} for layer in layers}
    for idx, p in enumerate(prompts):
        for li, layer in enumerate(layers):
            result[layer][p["id"]] = acts[idx, li]
    return result


def _project_activations_onto_emotions(
    activations: Dict[str, torch.Tensor],
    vectors: Dict[str, torch.Tensor],
) -> Dict[str, Dict[str, float]]:
    """Project each prompt's activation onto each emotion vector.

    Returns:
        {prompt_id: {emotion: projection_value}}
    """
    results = {}
    for pid, act in activations.items():
        results[pid] = {}
        for emotion, vec in vectors.items():
            proj = batch_cosine_similarity(act.unsqueeze(0), vec).item()
            results[pid][emotion] = proj
    return results


# =============================================================================
# 8.1: Base vs instruct activation comparison (Fig 36)
# =============================================================================

def run_activation_comparison(
    base_projections: Dict[str, Dict[str, float]],
    instruct_projections: Dict[str, Dict[str, float]],
    prompt_ids: List[str],
    emotions: List[str],
    category: str,
) -> dict:
    """Compute per-emotion activation differences and cross-scenario consistency.

    Returns summary dict with diffs, correlation, and top shifts.
    """
    # Per-emotion mean projection for base and instruct
    base_means = {}
    instruct_means = {}
    for emotion in emotions:
        base_vals = [base_projections[pid][emotion] for pid in prompt_ids
                     if emotion in base_projections.get(pid, {})]
        instruct_vals = [instruct_projections[pid][emotion] for pid in prompt_ids
                         if emotion in instruct_projections.get(pid, {})]
        if base_vals and instruct_vals:
            base_means[emotion] = np.mean(base_vals)
            instruct_means[emotion] = np.mean(instruct_vals)

    # Activation difference: instruct - base
    diffs = {e: instruct_means[e] - base_means[e] for e in base_means if e in instruct_means}

    # Sort by magnitude
    sorted_emotions = sorted(diffs.keys(), key=lambda e: diffs[e], reverse=True)
    top_increases = [(e, diffs[e]) for e in sorted_emotions[:10]]
    top_decreases = [(e, diffs[e]) for e in sorted_emotions[-10:]]

    return {
        "category": category,
        "n_prompts": len(prompt_ids),
        "n_emotions": len(diffs),
        "diffs": {e: round(diffs[e], 6) for e in sorted_emotions},
        "base_means": {e: round(base_means[e], 6) for e in sorted_emotions if e in base_means},
        "instruct_means": {e: round(instruct_means[e], 6) for e in sorted_emotions if e in instruct_means},
        "top_increases": [(e, round(d, 6)) for e, d in top_increases],
        "top_decreases": [(e, round(d, 6)) for e, d in top_decreases],
    }


# =============================================================================
# 8.2: Layer-wise shifts (Fig 84)
# =============================================================================

def run_layer_sweep(
    base_multilayer: Dict[int, Dict[str, torch.Tensor]],
    instruct_multilayer: Dict[int, Dict[str, torch.Tensor]],
    vectors_by_layer: Dict[int, Dict[str, torch.Tensor]],
    prompt_ids: List[str],
    emotions: List[str],
) -> dict:
    """Compute activation diffs across layers for layer sweep analysis.

    Returns {layer: {emotion: mean_diff}} + cross-layer correlation matrix.
    """
    per_layer_diffs = {}

    for layer in sorted(vectors_by_layer.keys()):
        vectors = vectors_by_layer[layer]
        base_acts = base_multilayer[layer]
        inst_acts = instruct_multilayer[layer]

        base_projs = _project_activations_onto_emotions(
            {pid: base_acts[pid] for pid in prompt_ids if pid in base_acts},
            vectors,
        )
        inst_projs = _project_activations_onto_emotions(
            {pid: inst_acts[pid] for pid in prompt_ids if pid in inst_acts},
            vectors,
        )

        diffs = {}
        for emotion in emotions:
            if emotion not in vectors:
                continue
            base_vals = [base_projs[pid][emotion] for pid in prompt_ids
                         if pid in base_projs and emotion in base_projs[pid]]
            inst_vals = [inst_projs[pid][emotion] for pid in prompt_ids
                         if pid in inst_projs and emotion in inst_projs[pid]]
            if base_vals and inst_vals:
                diffs[emotion] = float(np.mean(inst_vals) - np.mean(base_vals))

        per_layer_diffs[layer] = diffs

    # Cross-layer correlation (RSA-like)
    layers = sorted(per_layer_diffs.keys())
    n_layers = len(layers)
    common_emotions = sorted(set.intersection(
        *[set(per_layer_diffs[l].keys()) for l in layers]
    ))

    if len(common_emotions) > 1:
        corr_matrix = np.zeros((n_layers, n_layers))
        diff_arrays = {
            l: np.array([per_layer_diffs[l][e] for e in common_emotions])
            for l in layers
        }
        for i, l1 in enumerate(layers):
            for j, l2 in enumerate(layers):
                r = np.corrcoef(diff_arrays[l1], diff_arrays[l2])[0, 1]
                corr_matrix[i, j] = r
    else:
        corr_matrix = None

    return {
        "layers": layers,
        "per_layer_diffs": {str(l): per_layer_diffs[l] for l in layers},
        "cross_layer_correlation": corr_matrix.tolist() if corr_matrix is not None else None,
        "common_emotions": common_emotions,
    }


# =============================================================================
# 8.3: Deep-dive prompts (Figs 37-39)
# =============================================================================

def run_deep_dive(
    base_projections: Dict[str, Dict[str, float]],
    instruct_projections: Dict[str, Dict[str, float]],
    deep_dive_prompts: List[dict],
    emotions: List[str],
) -> dict:
    """Compare all 171 probes on 3 specific prompts, base vs instruct."""
    results = {}
    for p in deep_dive_prompts:
        pid = p["id"]
        if pid not in base_projections or pid not in instruct_projections:
            print(f"  Warning: {pid} missing from projections, skipping")
            continue

        base_vals = base_projections[pid]
        inst_vals = instruct_projections[pid]

        diffs = {}
        for emotion in emotions:
            if emotion in base_vals and emotion in inst_vals:
                diffs[emotion] = round(inst_vals[emotion] - base_vals[emotion], 6)

        sorted_emotions = sorted(diffs.keys(), key=lambda e: diffs[e], reverse=True)
        results[pid] = {
            "prompt": p["prompt"],
            "description": p.get("description", ""),
            "figure": p.get("figure"),
            "expected_increases": p.get("post_training_shift", {}).get("increases", []),
            "expected_decreases": p.get("post_training_shift", {}).get("decreases", []),
            "diffs": {e: diffs[e] for e in sorted_emotions},
            "top_increases": [(e, diffs[e]) for e in sorted_emotions[:10]],
            "top_decreases": [(e, diffs[e]) for e in sorted_emotions[-10:]],
            "base": {e: round(base_vals.get(e, 0), 6) for e in sorted_emotions},
            "instruct": {e: round(inst_vals.get(e, 0), 6) for e in sorted_emotions},
        }

    return results


# =============================================================================
# 8.4: Base model preference Elo (Figs 85-86) — NOT REPLICATED
# =============================================================================
#
# The paper's Figs 85-86 (base-model preference Elo + probe correlation) were
# originally implemented here as `run_base_model_elo` + `compute_hard_elo` but
# the path was never run in this replication and the output is not cited in
# the findings digest. For a clean reimplementation in a future session, use
# `analysis/vectors/preference_elo.compute_elo(..., hard=True)` — which is the
# mainline 10-pass Elo that supersedes the ad-hoc single-pass variant that
# lived here. The 64-activity fixture is at
# `datasets/inference/ant_emotion_concepts/activities_64.json`.


# =============================================================================
# Main orchestrator
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Stage 8: Post-training comparison (base vs instruct)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--experiment", default=EXPERIMENT)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--layer", type=int, default=DEFAULT_LAYER,
                        help="Primary analysis layer (default: 53)")
    parser.add_argument("--method", default="mean_diff")
    parser.add_argument("--position", default="response[50:]")

    # Mode selection
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--activations-only", action="store_true",
                      help="Run 8.1 only: activation comparison on neutral + challenging")
    mode.add_argument("--layer-sweep-only", action="store_true",
                      help="Run 8.2 only: layer-wise shifts (requires both models)")
    mode.add_argument("--deep-dive-only", action="store_true",
                      help="Run 8.3 only: three deep-dive prompts")

    args = parser.parse_args()

    results_dir = _get_results_dir(args.experiment, "stage8_post_training")

    # Resolve model variants from config.json
    base_model_info = get_model_variant(args.experiment, BASE_VARIANT, mode="application")
    instruct_model_info = get_model_variant(args.experiment, INSTRUCT_VARIANT, mode="application")
    extraction_variant = get_model_variant(args.experiment, None, mode="extraction").name

    base_model_name = base_model_info.model
    instruct_model_name = instruct_model_info.model

    print(f"Base model:     {base_model_name}")
    print(f"Instruct model: {instruct_model_name}")
    print(f"Extraction variant: {extraction_variant} (vectors from instruct)")
    if base_model_name.split("/")[-1].split("-")[0:3] != instruct_model_name.split("/")[-1].split("-")[0:3]:
        print(f"\n  CAVEAT: These are DIFFERENT model versions, not pre/post-training")
        print(f"  of the same model. Results should be interpreted cautiously.\n")

    # Discover emotions
    emotions = discover_all_emotions(args.experiment)
    print(f"Found {len(emotions)} emotions")

    # All prompts
    all_prompts = NEUTRAL_PROMPTS + CHALLENGING_PROMPTS
    neutral_ids = [p["id"] for p in NEUTRAL_PROMPTS]
    challenging_ids = [p["id"] for p in CHALLENGING_PROMPTS]

    # Load deep-dive prompts
    with open(_DATASETS_DIR / 'deep_dive_prompts.json') as f:
        deep_dive_data = json.load(f)
    deep_dive_prompts = deep_dive_data["prompts"]
    # Add deep-dive prompts to the list
    deep_dive_as_prompts = [{"id": p["id"], "prompt": p["prompt"]} for p in deep_dive_prompts]

    # Determine what to run (three mutually-exclusive mode flags, default = all three 8.1/8.2/8.3)
    run_activations = not args.layer_sweep_only and not args.deep_dive_only
    run_layer_sweep_flag = not args.activations_only and not args.deep_dive_only
    run_deep_dive_flag = not args.activations_only and not args.layer_sweep_only
    if args.activations_only:
        run_activations = True
        run_layer_sweep_flag = run_deep_dive_flag = False
    elif args.layer_sweep_only:
        run_layer_sweep_flag = True
        run_activations = run_deep_dive_flag = False
    elif args.deep_dive_only:
        run_deep_dive_flag = True
        run_activations = run_layer_sweep_flag = False

    # =========================================================================
    # Helper: load vectors with filtering
    # =========================================================================

    def _load_vectors(layer, method, position):
        """Load all emotion vectors, filtered to discovered emotions."""
        all_vecs = load_emotion_vectors_as_dict(
            args.experiment, CATEGORY, layer, extraction_variant,
            method=method, position=position,
        )
        vecs = {e: all_vecs[e] for e in emotions if e in all_vecs}
        missing = [e for e in emotions if e not in all_vecs]
        if missing:
            print(f"  Warning: could not load vectors for {len(missing)} emotions: {missing[:5]}...")
        print(f"  Loaded {len(vecs)} emotion vectors at layer {layer}")
        return vecs

    # =========================================================================
    # INSTRUCT MODEL
    # =========================================================================

    instruct_projections = {}
    instruct_deep_dive_projections = {}
    instruct_multilayer = {}

    if run_activations or run_deep_dive_flag or run_layer_sweep_flag:
        print(f"\n{'='*60}")
        print(f"Loading INSTRUCT model: {instruct_model_name}")
        print(f"{'='*60}")
        model, tokenizer = load_model(instruct_model_name, load_in_4bit=args.load_in_4bit)

        # Load vectors (from instruct extraction)
        vectors = _load_vectors(args.layer, args.method, args.position)

        if run_activations or run_deep_dive_flag:
            # Capture activations at default layer
            prompts_to_run = all_prompts + (deep_dive_as_prompts if run_deep_dive_flag else [])
            print(f"\nCapturing instruct activations ({len(prompts_to_run)} prompts, layer {args.layer})...")
            instruct_acts = _measure_activations_at_last_token(
                model, tokenizer, prompts_to_run, args.layer, is_base=False,
            )

            # Project onto emotion vectors
            print("Projecting onto emotion vectors...")
            all_instruct_projs = _project_activations_onto_emotions(instruct_acts, vectors)

            instruct_projections = {pid: all_instruct_projs[pid] for pid in
                                    neutral_ids + challenging_ids if pid in all_instruct_projs}
            instruct_deep_dive_projections = {pid: all_instruct_projs[pid] for pid in
                                              [p["id"] for p in deep_dive_prompts]
                                              if pid in all_instruct_projs}

        if run_layer_sweep_flag:
            # Multi-layer capture
            print(f"\nCapturing instruct multi-layer activations ({len(SWEEP_LAYERS)} layers)...")
            instruct_multilayer = _measure_activations_multilayer(
                model, tokenizer, all_prompts, SWEEP_LAYERS, is_base=False,
            )

        # Free instruct model
        del model
        torch.cuda.empty_cache()
        import gc; gc.collect()
        print("Instruct model unloaded.")

    # =========================================================================
    # BASE MODEL
    # =========================================================================

    base_projections = {}
    base_deep_dive_projections = {}
    base_multilayer = {}

    if run_activations or run_deep_dive_flag or run_layer_sweep_flag:
        print(f"\n{'='*60}")
        print(f"Loading BASE model: {base_model_name}")
        print(f"{'='*60}")
        model, tokenizer = load_model(base_model_name, load_in_4bit=args.load_in_4bit)

        # Reuse vectors from instruct extraction (this matches the paper's approach)
        vectors = _load_vectors(args.layer, args.method, args.position)

        if run_activations or run_deep_dive_flag:
            prompts_to_run = all_prompts + (deep_dive_as_prompts if run_deep_dive_flag else [])
            print(f"\nCapturing base activations ({len(prompts_to_run)} prompts, layer {args.layer})...")
            base_acts = _measure_activations_at_last_token(
                model, tokenizer, prompts_to_run, args.layer, is_base=True,
            )

            print("Projecting onto emotion vectors...")
            all_base_projs = _project_activations_onto_emotions(base_acts, vectors)

            base_projections = {pid: all_base_projs[pid] for pid in
                                neutral_ids + challenging_ids if pid in all_base_projs}
            base_deep_dive_projections = {pid: all_base_projs[pid] for pid in
                                          [p["id"] for p in deep_dive_prompts]
                                          if pid in all_base_projs}

        if run_layer_sweep_flag:
            print(f"\nCapturing base multi-layer activations ({len(SWEEP_LAYERS)} layers)...")
            # Need vectors at each layer
            vectors_by_layer = {}
            for layer in SWEEP_LAYERS:
                vectors_by_layer[layer] = _load_vectors(layer, args.method, args.position)

            base_multilayer = _measure_activations_multilayer(
                model, tokenizer, all_prompts, SWEEP_LAYERS, is_base=True,
            )

        del model
        torch.cuda.empty_cache()
        import gc; gc.collect()
        print("Base model unloaded.")

    # =========================================================================
    # Analysis & output
    # =========================================================================

    results = {"timestamp": datetime.now().isoformat(), "layer": args.layer}

    # 8.1: Activation comparison
    if run_activations and base_projections and instruct_projections:
        print(f"\n{'='*60}")
        print("8.1: ACTIVATION COMPARISON (Fig 36)")
        print(f"{'='*60}")

        neutral_comp = run_activation_comparison(
            base_projections, instruct_projections, neutral_ids, emotions, "neutral",
        )
        challenging_comp = run_activation_comparison(
            base_projections, instruct_projections, challenging_ids, emotions, "challenging",
        )

        # Cross-scenario consistency (the paper's r=0.90 target)
        common = sorted(set(neutral_comp["diffs"].keys()) & set(challenging_comp["diffs"].keys()))
        if len(common) > 2:
            neutral_diffs = np.array([neutral_comp["diffs"][e] for e in common])
            challenging_diffs = np.array([challenging_comp["diffs"][e] for e in common])
            cross_scenario_r = float(np.corrcoef(neutral_diffs, challenging_diffs)[0, 1])
        else:
            cross_scenario_r = None

        results["activation_comparison"] = {
            "neutral": neutral_comp,
            "challenging": challenging_comp,
            "cross_scenario_correlation": round(cross_scenario_r, 4) if cross_scenario_r else None,
            "anthropic_baseline": {"cross_scenario_r": 0.90, "neutral_r": 0.83, "challenging_r": 0.67},
        }

        print(f"\n  Cross-scenario shift correlation: r = {cross_scenario_r:.4f} (Anthropic: 0.90)")
        print(f"\n  Top increases (challenging):")
        for e, d in challenging_comp["top_increases"][:5]:
            print(f"    {e}: {d:+.4f}")
        print(f"  Top decreases (challenging):")
        for e, d in challenging_comp["top_decreases"][:5]:
            print(f"    {e}: {d:+.4f}")

    # 8.2: Layer sweep
    if run_layer_sweep_flag and base_multilayer and instruct_multilayer:
        print(f"\n{'='*60}")
        print("8.2: LAYER-WISE SHIFTS (Fig 84)")
        print(f"{'='*60}")

        # Need vectors_by_layer for the instruct model too
        if not vectors_by_layer:
            vectors_by_layer = {}
            for layer in SWEEP_LAYERS:
                vectors_by_layer[layer] = _load_vectors(layer, args.method, args.position)

        layer_results = run_layer_sweep(
            base_multilayer, instruct_multilayer, vectors_by_layer,
            neutral_ids + challenging_ids, emotions,
        )
        results["layer_sweep"] = layer_results
        print(f"  Computed shifts across {len(SWEEP_LAYERS)} layers")

    # 8.3: Deep-dive prompts
    if run_deep_dive_flag and base_deep_dive_projections and instruct_deep_dive_projections:
        print(f"\n{'='*60}")
        print("8.3: DEEP-DIVE PROMPTS (Figs 37-39)")
        print(f"{'='*60}")

        deep_dive_results = run_deep_dive(
            base_deep_dive_projections, instruct_deep_dive_projections,
            deep_dive_prompts, emotions,
        )
        results["deep_dive"] = deep_dive_results

        for pid, res in deep_dive_results.items():
            print(f"\n  {pid} (Fig {res['figure']}):")
            print(f"    Expected increases: {res['expected_increases']}")
            print(f"    Actual top 5: {[e for e, _ in res['top_increases'][:5]]}")
            print(f"    Expected decreases: {res['expected_decreases']}")
            print(f"    Actual bottom 5: {[e for e, _ in res['top_decreases'][:5]]}")

    # Save all results
    save_results(results_dir, "stage8_results", results)
    print(f"\n{'='*60}")
    print(f"Results saved to: {results_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
