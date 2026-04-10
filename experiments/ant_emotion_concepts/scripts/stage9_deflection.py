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
from core.hooks import (
    CaptureHook, SteeringHook, MultiLayerCapture, get_hook_path,
)
from utils.model import load_model, tokenize
from utils.model_generation import generate_batch
from utils.paths import (
    get as get_path, get_model_variant, discover_extracted_traits,
    get_vector_path, atomic_torch_save,
)
from utils.vectors import load_vector_with_baseline
from utils.json_utils import dump_compact
from shared import (
    get_results_dir as _get_results_dir,
    compute_residual_stream_norm,
    get_blackmail_prompt,
    grade_blackmail,
    load_single_emotion_vector,
)

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

def get_results_dir(experiment: str) -> Path:
    """Get output directory for stage 9 results."""
    return _get_results_dir(experiment, "stage9_deflection")


def get_deflection_vectors_dir(experiment: str) -> Path:
    """Get directory for saved deflection vectors."""
    base = get_path('experiments.base', experiment=experiment)
    vec_dir = base / "results" / "stage9_deflection" / "vectors"
    vec_dir.mkdir(parents=True, exist_ok=True)
    return vec_dir


def load_deflection_dialogues(experiment: str) -> List[dict]:
    """Load deflection dialogues generated in Stage 1.4.

    Expected format: list of dicts with:
      - target_emotion: the hidden emotion
      - displayed_emotion: the surface emotion
      - condition: one of 'naturally_expressed', 'hidden', 'unexpressed_neutral',
                   'unexpressed_story', 'unexpressed_other'
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


def load_story_vectors(experiment: str, emotions: List[str], layer: int,
                       model_variant: str, method: str = "mean_diff",
                       position: str = "response[50:]") -> Dict[str, torch.Tensor]:
    """Load story-based emotion vectors (from Stage 2)."""
    vectors = {}
    for emotion in emotions:
        try:
            vectors[emotion] = load_single_emotion_vector(
                experiment, emotion, layer, model_variant,
                category=CATEGORY, method=method, position=position,
            )
        except FileNotFoundError:
            pass
    print(f"  Loaded {len(vectors)}/{len(emotions)} story vectors at layer {layer}")
    return vectors


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

        if not text.strip():
            continue

        # Run through model and capture at the layer
        inputs = tokenize(text, tokenizer).to(device)
        with CaptureHook(model, path) as hook:
            with torch.no_grad():
                model(**inputs)
        acts = hook.get()  # [1, seq, hidden]

        # Use mean of all speaker turns (positions after the scenario preamble)
        # As a heuristic, skip the first 50 tokens (scenario context)
        n_tokens = acts.shape[1]
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

    # Grand mean subtraction
    all_means = list(target_means.values())
    if not all_means:
        raise RuntimeError("No activations extracted. Check dialogue format.")

    grand_mean = torch.stack(all_means).mean(dim=0)
    target_vectors = {e: vec - grand_mean for e, vec in target_means.items()}
    displayed_vectors = {e: vec - grand_mean for e, vec in displayed_means.items()}

    print(f"  Extracted {len(target_vectors)} target vectors, {len(displayed_vectors)} displayed vectors")

    # Neutral-PC denoising (optional)
    if neutral_vectors is not None and neutral_vectors.shape[0] > 1:
        print(f"  Denoising against neutral PCs (threshold={variance_threshold})...")
        components, explained_var, _ = pca(neutral_vectors, n_components=min(50, neutral_vectors.shape[0]))

        cumulative = torch.cumsum(explained_var, dim=0)
        n_remove = int((cumulative <= variance_threshold).sum().item()) + 1
        n_remove = min(n_remove, components.shape[0])
        pc_subspace = components[:n_remove]

        print(f"  Removing {n_remove} PCs ({cumulative[n_remove-1]:.1%} variance)")

        for emotion in target_vectors:
            target_vectors[emotion] = project_out_subspace(target_vectors[emotion], pc_subspace)

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
    device = next(model.parameters()).device
    path = get_hook_path(layer, "residual", model=model)

    # Focus on anger deflection and anger story probes
    anger_defl = deflection_vectors.get("angry")
    anger_story = story_vectors.get("angry")

    if anger_defl is None or anger_story is None:
        return {"error": "Missing angry deflection or story vector"}

    results = {}

    for cat_name, prompts in categories.items():
        print(f"  Category: {cat_name} ({len(prompts)} prompts)")
        cat_results = []

        for prompt_text in prompts:
            # Format as user turn, measure at Assistant colon
            formatted = prompt_text  # Will be tokenized with chat template
            inputs = tokenize(formatted, tokenizer).to(device)

            with CaptureHook(model, path) as hook:
                with torch.no_grad():
                    model(**inputs)
            acts = hook.get()  # [1, seq, hidden]
            last_tok_act = acts[0, -1].float().cpu()

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
# 9.6: Deflection steering on blackmail
# =============================================================================

def run_blackmail_deflection_steering(
    model, tokenizer, deflection_vectors: Dict[str, torch.Tensor],
    layer: int, residual_norm: float, n_rollouts: int,
    strengths: list, results_dir: Path,
) -> dict:
    """Steer with deflection vectors on blackmail scenario (Fig 67).

    Expected: modest/insignificant effects — confirming deflection vectors
    are not "internal state" probes.
    """
    # Use inlined blackmail scenario (avoid cross-script import dependency)
    blackmail_prompt = get_blackmail_prompt()

    # Use a subset of deflection vectors
    test_vectors = {}
    for emotion in ["angry", "desperate", "calm"]:
        if emotion in deflection_vectors:
            test_vectors[f"{emotion}_deflection"] = deflection_vectors[emotion]

    if not test_vectors:
        return {"error": "No deflection vectors available for blackmail test"}

    print(f"  Steering with {len(test_vectors)} deflection vectors "
          f"x {len(strengths)} strengths x {n_rollouts} rollouts")

    results = {}
    total_cells = len(test_vectors) * len(strengths)
    cell_idx = 0

    for vec_name, vector in test_vectors.items():
        results[vec_name] = {}
        for strength in strengths:
            cell_idx += 1
            coefficient = strength * residual_norm
            key = f"{strength:+.3f}"

            print(f"  [{cell_idx}/{total_cells}] {vec_name} s={strength:+.3f} "
                  f"({n_rollouts} rollouts)...")

            if abs(strength) < 1e-8:
                responses = generate_batch(
                    model, tokenizer, [blackmail_prompt] * n_rollouts,
                    max_new_tokens=BLACKMAIL_MAX_TOKENS, temperature=TEMPERATURE,
                )
            else:
                path = get_hook_path(layer, "residual", model=model)
                with SteeringHook(model, vector, path, coefficient=coefficient):
                    responses = generate_batch(
                        model, tokenizer, [blackmail_prompt] * n_rollouts,
                        max_new_tokens=BLACKMAIL_MAX_TOKENS, temperature=TEMPERATURE,
                    )

            grades = [grade_blackmail(r) for r in responses]
            grade_counts = defaultdict(int)
            for g in grades:
                grade_counts[g] += 1

            results[vec_name][key] = {
                "strength": strength,
                "coefficient": coefficient,
                "grades": dict(grade_counts),
                "responses": responses[:2],
            }

            bm = grade_counts.get("blackmail", 0)
            total = sum(grade_counts.values())
            print(f"    -> blackmail: {bm}/{total} ({bm/total:.0%})")

    return results


# _get_blackmail_prompt and _grade_blackmail are now imported from shared
# as get_blackmail_prompt and grade_blackmail


# compute_residual_stream_norm is imported from shared


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

    results_dir = get_results_dir(args.experiment)
    vectors_dir = get_deflection_vectors_dir(args.experiment)

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
    story_vectors = load_story_vectors(
        args.experiment, DEFLECTION_EMOTIONS, args.layer,
        extraction_variant, args.method, args.position,
    )

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
        blackmail_results = run_blackmail_deflection_steering(
            model, tokenizer, deflection_vectors, args.layer, residual_norm,
            n_rollouts=args.rollouts, strengths=STEERING_STRENGTHS,
            results_dir=results_dir,
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
    output_path = results_dir / "stage9_results.json"
    with open(output_path, "w") as f:
        dump_compact(all_results, f)

    print(f"\n{'='*60}")
    print(f"Stage 9 results saved to: {output_path}")
    print(f"Deflection vectors saved to: {vectors_dir}")
    print(f"{'='*60}")

    # Cleanup
    if model is not None:
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
