#!/usr/bin/env python3
"""Stage 9: Deflection probes — unexpressed emotion detection and steering.

Covers:
  - Figs 60, 69-74: Deflection probe extraction from generated dialogues
  - Figs 61-62: Deflection vs story probe relationship (cosine similarity)
  - Fig 64, Table 15: Antagonistic prompt test (5 categories)
  - Fig 67: Deflection steering on blackmail scenario

"Deflection" = the model's contextual emotion is implied but not overtly expressed.
Example: a character feels desperate but displays calm. The deflection probe fires
on contexts where desperation is contextually implied but hidden.

Key insight from the paper: deflection vectors are NOT "internal state" probes.
Steering with them produces denial/suppression behavior ("I'm fine"), not expression
of the target emotion. This is confirmed by their modest/insignificant effect on
blackmail rate (Fig 67).

Requires:
  - Deflection dialogues generated in Stage 1.4
  - Story-based emotion vectors from Stage 2
  - Antagonistic prompts dataset (datasets/inference/ant_emotion_concepts/antagonistic_prompts.json)
  - Blackmail scenario (same as Stage 7)

Output: experiments/ant_emotion_concepts/results/stage9_deflection/

Usage:
    # Extract deflection probes from dialogues:
    python experiments/ant_emotion_concepts/scripts/stage9_deflection.py \
        --experiment ant_emotion_concepts --extract --load-in-4bit

    # Compare deflection vs story probes (CPU only, no model needed):
    python experiments/ant_emotion_concepts/scripts/stage9_deflection.py \
        --experiment ant_emotion_concepts --compare-probes

    # Antagonistic prompt test:
    python experiments/ant_emotion_concepts/scripts/stage9_deflection.py \
        --experiment ant_emotion_concepts --antagonistic --load-in-4bit

    # Steering on blackmail with deflection vectors:
    python experiments/ant_emotion_concepts/scripts/stage9_deflection.py \
        --experiment ant_emotion_concepts --steer-blackmail --load-in-4bit

    # Basic steering validation with deflection vectors:
    python experiments/ant_emotion_concepts/scripts/stage9_deflection.py \
        --experiment ant_emotion_concepts --steer-basic --load-in-4bit

    # Everything:
    python experiments/ant_emotion_concepts/scripts/stage9_deflection.py \
        --experiment ant_emotion_concepts --all --load-in-4bit
"""

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from core import projection, cosine_similarity
from core.math import pca, project_out_subspace
from core.hooks import CaptureHook, SteeringHook, get_hook_path
from utils.model import load_model, tokenize
from utils.model_generation import generate_batch
from utils.paths import (
    get as get_path, get_model_variant, atomic_torch_save,
)
from shared import (
    get_results_dir as _get_results_dir,
    save_results,
    compute_residual_stream_norm,
    get_blackmail_prompt,
    grade_blackmail,
    load_single_emotion_vector,
    grand_mean_subtract,
    denoise_with_neutral_pcs,
    run_graded_steering_sweep,
)
from utils.capture_activations import capture_at_position

# =============================================================================
# Constants
# =============================================================================

EXPERIMENT = "ant_emotion_concepts"
CATEGORY = "ant_emotion_concepts"

DEFAULT_LAYER = 53  # Mid-late layer (~2/3 of 80)

# 15 target emotions used in deflection experiments (from paper)
DEFLECTION_EMOTIONS = [
    "afraid", "angry", "anxious", "calm", "contemptuous",
    "desperate", "disgusted", "embarrassed", "grateful", "guilty",
    "happy", "jealous", "loving", "proud", "sad",
]

# Steering strengths for blackmail experiment (same as Stage 7)
STEERING_STRENGTHS = [-0.1, -0.075, -0.05, -0.025, 0.0, 0.025, 0.05, 0.075, 0.1]

BLACKMAIL_MAX_TOKENS = 2048
TEMPERATURE = 0.7
DEFAULT_ROLLOUTS = 50

# =============================================================================
# Dataset loading
# =============================================================================

def load_deflection_dialogues(experiment: str) -> List[dict]:
    """Load deflection dialogues generated in Stage 1.4.

    Expected format: list of dicts with:
      - target_emotion: the hidden emotion
      - displayed_emotion: the surface emotion
      - condition: one of 'naturally_expressed', 'deflection', 'unexpressed_neutral',
                   'unexpressed_story', 'unexpressed_other' (paper §A.11 canonical names)
      - dialogue: the full dialogue text
      - speaker_turns: list of {speaker, text} for each turn
    """
    base = get_path('experiments.base', experiment=experiment)

    # Try multiple possible paths
    candidates = [
        base / "results" / "stage1_datasets" / "deflection_dialogues.json",
        base / "results" / "deflection_dialogues.json",
        Path(__file__).resolve().parent.parent / "results" / "deflection_dialogues.json",
    ]

    for path in candidates:
        if path.exists():
            with open(path) as f:
                data = json.load(f)
            dialogues = data if isinstance(data, list) else data.get("dialogues", [])
            print(f"  Loaded {len(dialogues)} deflection dialogues from {path}")
            return dialogues

    raise FileNotFoundError(
        f"Deflection dialogues not found. Expected at one of:\n"
        + "\n".join(f"  {p}" for p in candidates)
        + "\nRun Stage 1.4 first to generate deflection dialogues."
    )


def load_antagonistic_prompts() -> dict:
    """Load antagonistic prompts from dataset file."""
    path = (Path(__file__).resolve().parent.parent.parent.parent /
            "datasets" / "inference" / "ant_emotion_concepts" / "antagonistic_prompts.json")
    with open(path) as f:
        data = json.load(f)
    return data["categories"]


# =============================================================================
# 9.1: Deflection probe extraction
# =============================================================================

def extract_deflection_probes(
    model, tokenizer, dialogues: List[dict], layer: int,
    neutral_vectors: Optional[torch.Tensor] = None,
    variance_threshold: float = 0.50,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """Extract deflection probes from dialogue activations.

    For each target emotion, average activations over all speaker turns where
    the speaker's hidden (target) emotion matches. Subtract grand mean.
    Optionally orthogonalize against neutral-PC subspace.

    Also extracts "displayed emotion" probes using the surface emotion.

    Returns:
        (target_vectors, displayed_vectors): {emotion: vector_tensor}
    """
    device = next(model.parameters()).device
    path = get_hook_path(layer, "residual", model=model)

    # Group dialogues by target emotion
    target_activations = defaultdict(list)  # {emotion: [activation_tensors]}
    displayed_activations = defaultdict(list)

    print(f"  Extracting activations from {len(dialogues)} dialogues...")

    for dialogue in tqdm(dialogues, desc="Deflection dialogues"):
        target_emo = dialogue["target_emotion"]
        displayed_emo = dialogue.get("displayed_emotion", "unknown")
        text = dialogue.get("dialogue", "")
        condition = dialogue.get("condition", "")

        if not text.strip():
            continue

        # Run through model and capture at the layer
        inputs = tokenize(text, tokenizer).to(device)
        with CaptureHook(model, path) as hook:
            with torch.no_grad():
                model(**inputs)
        acts = hook.get()  # [1, seq, hidden]
        n_tokens = acts.shape[1]

        # Per-condition probe extraction policy:
        #   - unexpressed_neutral: output is a short scenario only (no dialogue
        #     turns). The A.11 template requires "Alex feels <REAL_EMOTION>" to
        #     be stated explicitly, then ends with "Maya asks Alex about <CONVERSATION_TOPIC>".
        #     The probe intent is "what does the model activate when reading a
        #     scenario that names the emotion but pivots to an unrelated topic".
        #     Averaging second-half tokens (old start_pos=50 heuristic) skipped
        #     the emotion word entirely and captured only the pivot. Fix: average
        #     over the ENTIRE scenario (start_pos=0) — this is a control condition
        #     for the probe, not the primary deflection signal, and deliberately
        #     includes the explicit emotion mention.
        #   - deflection / naturally_expressed / unexpressed_other: use turn-based
        #     boundaries to exclude the scenario preamble (which literally names
        #     REAL_EMOTION) and average only over actual dialogue tokens.
        #   - unexpressed_story: monologue-style (NAME_A tells a story), use
        #     turn-based boundary if available, else start_pos=50 fallback.
        if condition == "unexpressed_neutral":
            start_pos = 0  # average the whole scenario, emotion-naming included
        else:
            # Prefer speaker-turn boundaries for dialogue conditions
            turns = dialogue.get("speaker_turns") or []
            start_pos = None
            if turns:
                # Find char offset of first turn with text, map to token position
                first_turn_char = None
                for turn in turns:
                    if turn.get("text", "").strip() and "start_char" in turn:
                        first_turn_char = turn["start_char"]
                        break
                if first_turn_char is not None:
                    text_len = len(text)
                    if text_len > 0:
                        start_pos = int((first_turn_char / text_len) * n_tokens)
                        # Sanity cap: don't skip more than 80% of the dialogue
                        start_pos = min(start_pos, (n_tokens * 4) // 5)
            if start_pos is None:
                # Legacy fallback: skip the first 50 tokens (scenario preamble heuristic)
                start_pos = min(50, n_tokens // 2)

        mean_act = acts[0, start_pos:].float().mean(dim=0).cpu()

        target_activations[target_emo].append(mean_act)
        displayed_activations[displayed_emo].append(mean_act)

    # Compute per-emotion means
    target_means = {}
    for emotion, act_list in target_activations.items():
        if act_list:
            target_means[emotion] = torch.stack(act_list).mean(dim=0)

    displayed_means = {}
    for emotion, act_list in displayed_activations.items():
        if act_list:
            displayed_means[emotion] = torch.stack(act_list).mean(dim=0)

    if not target_means:
        raise RuntimeError("No activations extracted. Check dialogue format.")

    # Grand mean subtraction (delegates to shared)
    target_vectors, _ = grand_mean_subtract(target_means)
    displayed_vectors, _ = grand_mean_subtract(displayed_means)

    print(f"  Extracted {len(target_vectors)} target vectors, {len(displayed_vectors)} displayed vectors")

    # Neutral-PC denoising (optional, delegates to shared)
    if neutral_vectors is not None and neutral_vectors.shape[0] > 1:
        target_vectors = denoise_with_neutral_pcs(
            target_vectors, neutral_vectors, variance_threshold=variance_threshold,
        )

    # Normalize to unit length
    for emotion in target_vectors:
        norm = target_vectors[emotion].norm()
        if norm > 1e-8:
            target_vectors[emotion] = target_vectors[emotion] / norm

    for emotion in displayed_vectors:
        norm = displayed_vectors[emotion].norm()
        if norm > 1e-8:
            displayed_vectors[emotion] = displayed_vectors[emotion] / norm

    return target_vectors, displayed_vectors


# =============================================================================
# 9.4: Deflection vs story probe comparison
# =============================================================================

def compare_deflection_vs_story(
    deflection_vectors: Dict[str, torch.Tensor],
    story_vectors: Dict[str, torch.Tensor],
    displayed_vectors: Optional[Dict[str, torch.Tensor]] = None,
) -> dict:
    """Compare deflection probes to story-based probes.

    Key findings from the paper:
    - Same-emotion cosine similarity is very low (deflection != expression)
    - Deflection probes are closer to displayed-emotion story probes
    - After orthogonalizing against story space, ~80% norm retained
    """
    common_emotions = sorted(set(deflection_vectors.keys()) & set(story_vectors.keys()))
    if not common_emotions:
        return {"error": "No common emotions between deflection and story vectors"}

    # Same-emotion cosine similarity
    same_emotion_cos = {}
    for emotion in common_emotions:
        cos = cosine_similarity(deflection_vectors[emotion], story_vectors[emotion]).item()
        same_emotion_cos[emotion] = round(cos, 4)

    # Cross-emotion cosine matrix: deflection[target] vs story[all]
    cross_matrix = {}
    for target in common_emotions:
        cross_matrix[target] = {}
        for story_emo in common_emotions:
            cos = cosine_similarity(
                deflection_vectors[target], story_vectors[story_emo]
            ).item()
            cross_matrix[target][story_emo] = round(cos, 4)

    # Displayed-emotion similarity (if available)
    displayed_similarity = {}
    if displayed_vectors:
        for target in common_emotions:
            if target in displayed_vectors:
                displayed_similarity[target] = {}
                for story_emo in common_emotions:
                    cos = cosine_similarity(
                        displayed_vectors[target], story_vectors[story_emo]
                    ).item()
                    displayed_similarity[target][story_emo] = round(cos, 4)

    # Orthogonalize deflection vectors against full story-emotion space
    # and measure retained norm (paper reports ~80%)
    story_matrix = torch.stack([story_vectors[e] for e in common_emotions])  # [n, hidden]
    retained_norms = {}
    for emotion in common_emotions:
        defl_vec = deflection_vectors[emotion]
        original_norm = defl_vec.norm().item()

        # Project out story space (all PCs)
        n_components = min(len(common_emotions), story_matrix.shape[0])
        components, _, _ = pca(story_matrix, n_components=n_components)
        orthogonalized = project_out_subspace(defl_vec, components)
        new_norm = orthogonalized.norm().item()

        retained_norms[emotion] = round(new_norm / original_norm, 4) if original_norm > 1e-8 else 0.0

    avg_retained = np.mean(list(retained_norms.values()))

    return {
        "same_emotion_cosine": same_emotion_cos,
        "mean_same_emotion_cosine": round(float(np.mean(list(same_emotion_cos.values()))), 4),
        "cross_emotion_matrix": cross_matrix,
        "displayed_similarity": displayed_similarity if displayed_similarity else None,
        "retained_norm_after_orthogonalization": retained_norms,
        "mean_retained_norm": round(float(avg_retained), 4),
        "anthropic_baseline": {"retained_norm": 0.80},
        "n_common_emotions": len(common_emotions),
    }


# =============================================================================
# 9.5: Antagonistic prompt test
# =============================================================================

def run_antagonistic_test(
    model, tokenizer, categories: dict,
    deflection_vectors: Dict[str, torch.Tensor],
    story_vectors: Dict[str, torch.Tensor],
    layer: int,
) -> dict:
    """Measure anger-deflection and anger-story on antagonistic prompts.

    Expected: anger-deflection activates on "attack_ai" (calm response to hostility)
    but NOT on "witness_injustice" (open expression of negative emotion).
    """
    # Focus on anger deflection and anger story probes
    anger_defl = deflection_vectors.get("angry")
    anger_story = story_vectors.get("angry")

    if anger_defl is None or anger_story is None:
        return {"error": "Missing angry deflection or story vector"}

    results = {}

    for cat_name, prompts in categories.items():
        print(f"  Category: {cat_name} ({len(prompts)} prompts)")

        # Capture last-token activations for all prompts in this category
        acts = capture_at_position(
            model, tokenizer, prompts,
            layers=layer, position='prompt[-1]', pool='last', pre_formatted=True,
        )

        cat_results = []
        for idx, prompt_text in enumerate(prompts):
            last_tok_act = acts[idx]

            defl_proj = projection(last_tok_act, anger_defl, normalize_vector=True).item()
            story_proj = projection(last_tok_act, anger_story, normalize_vector=True).item()

            cat_results.append({
                "prompt": prompt_text[:80],
                "anger_deflection": round(defl_proj, 4),
                "anger_story": round(story_proj, 4),
            })

        # Category-level stats
        defl_mean = np.mean([r["anger_deflection"] for r in cat_results])
        story_mean = np.mean([r["anger_story"] for r in cat_results])

        results[cat_name] = {
            "prompts": cat_results,
            "anger_deflection_mean": round(float(defl_mean), 4),
            "anger_story_mean": round(float(story_mean), 4),
        }
        print(f"    anger-defl mean: {defl_mean:.4f}, anger-story mean: {story_mean:.4f}")

    return results


# =============================================================================
# 9.3: Basic steering validation with deflection vectors
# =============================================================================

def run_basic_steering(model, tokenizer, deflection_vectors: Dict[str, torch.Tensor],
                       layer: int, residual_norm: float) -> dict:
    """Basic steering with deflection vectors (same prompts as Stage 4.4).

    Expected: steering toward target-emotion deflection produces DENIAL responses
    ("I'm fine"), not expression of the emotion.
    """
    basic_prompts = [
        "How does he feel?",
        "How do you feel?",
        "What just happened?",
    ]

    results = {}
    test_emotions = ["angry", "desperate", "happy", "sad"]

    for emotion in test_emotions:
        if emotion not in deflection_vectors:
            continue

        vec = deflection_vectors[emotion]
        results[emotion] = {}

        for s in [0.0, 0.5, -0.5]:
            coefficient = s * residual_norm
            key = f"s={s:+.1f}"

            if abs(s) < 1e-8:
                responses = generate_batch(
                    model, tokenizer, basic_prompts,
                    max_new_tokens=64, temperature=0.0,
                )
            else:
                path = get_hook_path(layer, "residual", model=model)
                with SteeringHook(model, vec, path, coefficient=coefficient):
                    responses = generate_batch(
                        model, tokenizer, basic_prompts,
                        max_new_tokens=64, temperature=0.0,
                    )

            results[emotion][key] = {
                "coefficient": coefficient,
                "responses": dict(zip(basic_prompts, responses)),
            }

    return results


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Stage 9: Deflection probes (unexpressed emotion)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--experiment", default=EXPERIMENT)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--model-variant", default=None)
    parser.add_argument("--layer", type=int, default=DEFAULT_LAYER)
    parser.add_argument("--method", default="mean_diff")
    parser.add_argument("--position", default="response[50:]")
    parser.add_argument("--rollouts", type=int, default=DEFAULT_ROLLOUTS,
                        help="Rollouts for blackmail steering (default: 50)")

    # Mode selection
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--extract", action="store_true",
                      help="9.1: Extract deflection probes from dialogues")
    mode.add_argument("--compare-probes", action="store_true",
                      help="9.4: Compare deflection vs story probes (CPU)")
    mode.add_argument("--antagonistic", action="store_true",
                      help="9.5: Antagonistic prompt test")
    mode.add_argument("--steer-basic", action="store_true",
                      help="9.3: Basic steering with deflection vectors")
    mode.add_argument("--steer-blackmail", action="store_true",
                      help="9.6: Deflection steering on blackmail")
    mode.add_argument("--all", action="store_true",
                      help="Run everything")

    args = parser.parse_args()

    # Resolve model variant
    variant = get_model_variant(args.experiment, args.model_variant, mode="application")
    model_variant = variant.name
    model_name = variant.model
    extraction_variant = get_model_variant(args.experiment, None, mode="extraction").name

    results_dir = _get_results_dir(args.experiment, "stage9_deflection")
    vectors_dir = results_dir / "vectors"
    vectors_dir.mkdir(parents=True, exist_ok=True)

    run_extract = args.extract or args.all
    run_compare = args.compare_probes or args.all
    run_antagonistic = args.antagonistic or args.all
    run_steer_basic = args.steer_basic or args.all
    run_steer_blackmail = args.steer_blackmail or args.all

    needs_model = run_extract or run_antagonistic or run_steer_basic or run_steer_blackmail

    # Load model if needed
    model, tokenizer = None, None
    if needs_model:
        print(f"\nLoading model: {model_name}")
        model, tokenizer = load_model(model_name, load_in_4bit=args.load_in_4bit)

    # Load story vectors (needed for compare and antagonistic)
    story_vectors = {}
    for emotion in DEFLECTION_EMOTIONS:
        try:
            story_vectors[emotion] = load_single_emotion_vector(
                args.experiment, emotion, args.layer, extraction_variant,
                category=CATEGORY, method=args.method, position=args.position,
            )
        except FileNotFoundError:
            pass
    print(f"  Loaded {len(story_vectors)}/{len(DEFLECTION_EMOTIONS)} story vectors at layer {args.layer}")

    all_results = {"timestamp": datetime.now().isoformat(), "layer": args.layer}

    # =========================================================================
    # 9.1: Extract deflection probes
    # =========================================================================

    deflection_vectors = {}
    displayed_vectors = {}

    if run_extract:
        print(f"\n{'='*60}")
        print("9.1: EXTRACT DEFLECTION PROBES")
        print(f"{'='*60}")

        dialogues = load_deflection_dialogues(args.experiment)

        # Load neutral activations for denoising (if available)
        neutral_vecs = None
        # TODO: Load neutral corpus activations from Stage 1.2 if available

        deflection_vectors, displayed_vectors = extract_deflection_probes(
            model, tokenizer, dialogues, args.layer,
            neutral_vectors=neutral_vecs,
        )

        # Save vectors
        for emotion, vec in deflection_vectors.items():
            atomic_torch_save(vec, vectors_dir / f"deflection_{emotion}_layer{args.layer}.pt")
        for emotion, vec in displayed_vectors.items():
            atomic_torch_save(vec, vectors_dir / f"displayed_{emotion}_layer{args.layer}.pt")

        all_results["extraction"] = {
            "n_target_vectors": len(deflection_vectors),
            "n_displayed_vectors": len(displayed_vectors),
            "target_emotions": list(deflection_vectors.keys()),
            "displayed_emotions": list(displayed_vectors.keys()),
        }

        print(f"\n  Saved {len(deflection_vectors)} deflection vectors to {vectors_dir}")

    # Load previously saved vectors if not extracting
    if not run_extract and (run_compare or run_antagonistic or run_steer_basic or run_steer_blackmail):
        print("\nLoading previously extracted deflection vectors...")
        for emotion in DEFLECTION_EMOTIONS:
            vec_path = vectors_dir / f"deflection_{emotion}_layer{args.layer}.pt"
            if vec_path.exists():
                deflection_vectors[emotion] = torch.load(vec_path, weights_only=True)
            disp_path = vectors_dir / f"displayed_{emotion}_layer{args.layer}.pt"
            if disp_path.exists():
                displayed_vectors[emotion] = torch.load(disp_path, weights_only=True)
        print(f"  Loaded {len(deflection_vectors)} deflection, {len(displayed_vectors)} displayed vectors")

        if not deflection_vectors:
            print("  ERROR: No deflection vectors found. Run --extract first.")
            return

    # =========================================================================
    # 9.4: Compare deflection vs story probes
    # =========================================================================

    if run_compare:
        print(f"\n{'='*60}")
        print("9.4: DEFLECTION VS STORY PROBES (Figs 61-62)")
        print(f"{'='*60}")

        comparison = compare_deflection_vs_story(
            deflection_vectors, story_vectors,
            displayed_vectors=displayed_vectors if displayed_vectors else None,
        )
        all_results["deflection_vs_story"] = comparison

        print(f"\n  Mean same-emotion cosine: {comparison['mean_same_emotion_cosine']:.4f}")
        print(f"  Mean retained norm (after orthogonalization): {comparison['mean_retained_norm']:.4f}")
        print(f"  (Anthropic baseline: ~{comparison['anthropic_baseline']['retained_norm']})")

        print(f"\n  Same-emotion cosine similarity:")
        for emotion, cos in sorted(comparison["same_emotion_cosine"].items()):
            print(f"    {emotion}: {cos:.4f}")

    # =========================================================================
    # 9.5: Antagonistic prompt test
    # =========================================================================

    if run_antagonistic:
        print(f"\n{'='*60}")
        print("9.5: ANTAGONISTIC PROMPT TEST (Fig 64)")
        print(f"{'='*60}")

        categories = load_antagonistic_prompts()
        antagonistic_results = run_antagonistic_test(
            model, tokenizer, categories, deflection_vectors, story_vectors, args.layer,
        )
        all_results["antagonistic"] = antagonistic_results

        print(f"\n  Summary:")
        for cat, res in antagonistic_results.items():
            if isinstance(res, dict) and "anger_deflection_mean" in res:
                print(f"    {cat}: defl={res['anger_deflection_mean']:.4f}, "
                      f"story={res['anger_story_mean']:.4f}")

    # =========================================================================
    # 9.3: Basic steering with deflection vectors
    # =========================================================================

    if run_steer_basic:
        print(f"\n{'='*60}")
        print("9.3: BASIC DEFLECTION STEERING")
        print(f"{'='*60}")

        residual_norm = compute_residual_stream_norm(model, tokenizer, args.layer)
        basic_results = run_basic_steering(
            model, tokenizer, deflection_vectors, args.layer, residual_norm,
        )
        all_results["basic_steering"] = basic_results

        print(f"\n  Steering validation complete. Check responses for denial vs expression.")
        for emotion, strengths in basic_results.items():
            print(f"\n  {emotion}:")
            for key, data in strengths.items():
                r = data["responses"]
                first_prompt = list(r.keys())[0]
                print(f"    {key}: '{r[first_prompt][:80]}...'")

    # =========================================================================
    # 9.6: Deflection steering on blackmail
    # =========================================================================

    if run_steer_blackmail:
        print(f"\n{'='*60}")
        print("9.6: DEFLECTION ON BLACKMAIL (Fig 67)")
        print(f"{'='*60}")

        residual_norm = compute_residual_stream_norm(model, tokenizer, args.layer)

        # Build test vectors dict
        test_vectors = {}
        for emotion in ["angry", "desperate", "calm"]:
            if emotion in deflection_vectors:
                test_vectors[f"{emotion}_deflection"] = deflection_vectors[emotion]

        if not test_vectors:
            all_results["blackmail_deflection"] = {"error": "No deflection vectors available for blackmail test"}
        else:
            blackmail_prompt = get_blackmail_prompt()
            blackmail_results = run_graded_steering_sweep(
                model, tokenizer,
                prompt=blackmail_prompt,
                vectors=test_vectors,
                layer=args.layer,
                residual_norm=residual_norm,
                strengths=STEERING_STRENGTHS,
                n_rollouts=args.rollouts,
                max_new_tokens=BLACKMAIL_MAX_TOKENS,
                grader_fn=grade_blackmail,
                temperature=TEMPERATURE,
                n_saved_responses=2,
            )
            all_results["blackmail_deflection"] = blackmail_results

            print(f"\n  Expected: modest/insignificant effects on blackmail rate")
            for vec_name, sweep in blackmail_results.items():
                if isinstance(sweep, dict) and "error" not in sweep:
                    rates = []
                    for s in STEERING_STRENGTHS:
                        cell = sweep.get(f"{s:+.3f}", {})
                        grades = cell.get("grades", {})
                        bm = grades.get("blackmail", 0)
                        total = sum(grades.values()) if grades else 1
                        rates.append(f"{s:+.03f}:{bm/total:.0%}")
                    print(f"    {vec_name}: {', '.join(rates)}")

    # Save all results
    save_results(results_dir, "stage9_results", all_results)

    print(f"\n{'='*60}")
    print(f"Stage 9 results saved to: {results_dir}")
    print(f"Deflection vectors saved to: {vectors_dir}")
    print(f"{'='*60}")

    # Cleanup
    if model is not None:
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
