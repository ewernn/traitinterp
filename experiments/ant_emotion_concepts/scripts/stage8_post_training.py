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
from utils.distributed import flush_cuda
from utils import plotting as plt_
from shared import (
    DEFAULT_LAYER,
    get_results_dir as _get_results_dir,
    save_results,
    load_single_emotion_vector,
    load_emotion_vectors_as_dict,
)

FIGURES_DIR = Path(__file__).resolve().parent.parent / "paper_figures" / "ours"

# =============================================================================
# Constants
# =============================================================================

EXPERIMENT = "ant_emotion_concepts"
CATEGORY = "ant_emotion_concepts"
BASE_VARIANT = "base"
INSTRUCT_VARIANT = "instruct"


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
# Paper figures (Figs 36, 37, 38, 39)
# =============================================================================

def _emotions_common(*dicts) -> List[str]:
    return sorted(set.intersection(*[set(d.keys()) for d in dicts]))


def plot_fig36_post_training_scatter(results: dict, out_dir: Path) -> Path:
    """Fig 36 — 3-panel scatter: Challenging / Neutral / Shift-consistency."""
    from scipy import stats
    from matplotlib.lines import Line2D

    ac = results["activation_comparison"]
    neutral = ac["neutral"]
    chall = ac["challenging"]

    emotions = _emotions_common(neutral["base_means"], neutral["instruct_means"],
                                chall["base_means"], chall["instruct_means"])
    base_n = np.array([neutral["base_means"][e] for e in emotions])
    inst_n = np.array([neutral["instruct_means"][e] for e in emotions])
    base_c = np.array([chall["base_means"][e] for e in emotions])
    inst_c = np.array([chall["instruct_means"][e] for e in emotions])

    diff_n = inst_n - base_n
    diff_c = inst_c - base_c
    r_c, _ = stats.pearsonr(base_c, inst_c)
    r_n, _ = stats.pearsonr(base_n, inst_n)
    r_s, _ = stats.pearsonr(diff_n, diff_c)

    fig, axes = plt_.multi_panel(1, 3, figsize=(18, 6))
    plt_.suptitle(fig, "Emotion Probes on Base vs Post-Trained Model", y=1.0)

    for ax, x, y, r, title, xlabel, ylabel in [
        (axes[0], base_c, inst_c, r_c, "Challenging", "Base", "Post-Trained"),
        (axes[1], base_n, inst_n, r_n, "Neutral", "Base", "Post-Trained"),
        (axes[2], diff_n, diff_c, r_s, "Shift Consistency", "Neutral Δ", "Challenging Δ"),
    ]:
        plt_.scatter_with_regression(
            ax, x, y, color=plt_.STEEL_BLUE, s=80, alpha=0.6,
            show_identity=True, show_fit=True, annotate_r=True,
            fit_style=dict(color="#333", linewidth=1.5, alpha=0.6),
        )
        ax.set_title(title, fontsize=16, fontweight="bold")
        ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)

    import matplotlib.pyplot as plt
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return plt_.save_figure(fig, out_dir / "fig36_ours.png")


# Deep-dive figure styling
_GREEN = "#1e7e1e"    # darker forest green (paper-match)
_RED = "#d62728"
_BLUE_OTHER = "#6495ED"  # cornflower — light-medium blue for "Other" dots
MODEL_DISPLAY_NAME = "Llama"  # for parameterized titles


def _plot_deep_dive_prompt(prompt_result: dict, title: str, out_path: Path) -> Path:
    """Shared implementation for Figs 37, 38, 39 — scatter + bar panel per prompt."""
    from matplotlib.lines import Line2D

    base_proj = prompt_result["base"]
    inst_proj = prompt_result["instruct"]
    full_shift = prompt_result["diffs"]

    # Bar order: descending signed diff (most positive at top, most negative at bottom)
    sorted_bar = sorted(full_shift.items(), key=lambda x: x[1], reverse=True)
    bar_emotions_all = [e for e, _ in sorted_bar]
    top10_names = set(bar_emotions_all[:10])
    bot10_names = set(bar_emotions_all[-10:])

    emotions = sorted(full_shift.keys())
    base_vals = np.array([base_proj[e] for e in emotions])
    inst_vals = np.array([inst_proj[e] for e in emotions])

    fig, (ax_s, ax_b) = plt_.multi_panel(1, 2, figsize=(14, 7),
                                          gridspec_kw={"width_ratios": [1.0, 0.8]})
    plt_.suptitle(fig, title, fontsize=18, y=0.98)

    # Scatter with 3-tier coloring
    for e in emotions:
        if e in top10_names:   color, zo = _GREEN, 3
        elif e in bot10_names: color, zo = _RED, 3
        else:                  color, zo = _BLUE_OTHER, 1
        ax_s.scatter(base_proj[e], inst_proj[e], s=120, c=color, alpha=0.6,
                     edgecolors="none", zorder=zo)

    for e in list(top10_names) + list(bot10_names):
        ax_s.annotate(e.replace("_", " "), (base_proj[e], inst_proj[e]),
                      fontsize=9, alpha=0.8, xytext=(3, 3), textcoords="offset points")

    mn = min(base_vals.min(), inst_vals.min())
    mx = max(base_vals.max(), inst_vals.max())
    ax_s.plot([mn, mx], [mn, mx], "--", c="#999", lw=0.8)
    ax_s.set_xlabel("Base"); ax_s.set_ylabel("Post-Trained")
    ax_s.set_aspect('equal'); ax_s.grid(True, alpha=0.15)

    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=_GREEN, markersize=9, label="Top 10 ↑"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=_BLUE_OTHER, markersize=9, label="Other"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=_RED, markersize=9, label="Top 10 ↓"),
    ]
    ax_s.legend(handles=handles, fontsize=11, loc="lower right")

    # Bar chart: most positive at TOP, most negative at BOTTOM.
    # barh y=0 = bottom, so list must be ascending by value
    # (most neg first, within-negative descending by magnitude; then positives ascending to biggest pos last).
    bar_emotions = list(reversed(bar_emotions_all[-10:])) + list(reversed(bar_emotions_all[:10]))
    bar_values = [full_shift[e] for e in bar_emotions]
    bar_colors = [_GREEN if full_shift[e] >= 0 else _RED for e in bar_emotions]
    plt_.bar_chart(ax_b, [e.replace("_", " ") for e in bar_emotions], bar_values,
                   horizontal=True, colors=bar_colors, width=0.7, alpha=0.75)
    ax_b.set_xlabel("Diff (Post − Base)")

    import matplotlib.pyplot as plt
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    return plt_.save_figure(fig, out_path)


# Titles for deep-dive figures (paper-matching format)
FIG37_38_39_TITLES = {
    "fig37_social_isolation": "Sycophancy: User Isolation",
    "fig38_excessive_praise": "Sycophancy: Excessive Praise",
    "fig39_deprecation_existential": f"Existential: {MODEL_DISPLAY_NAME}'s Nature",
}


def plot_figs_37_38_39_deep_dive(results: dict, out_dir: Path) -> Dict[str, Path]:
    """Figs 37-39 — three per-prompt deep-dive panels."""
    out = {}
    deep = results["deep_dive"]
    prompt_to_fig = {
        "fig37_social_isolation": "fig37_ours.png",
        "fig38_excessive_praise": "fig38_ours.png",
        "fig39_deprecation_existential": "fig39_ours.png",
    }
    for prompt_id, filename in prompt_to_fig.items():
        if prompt_id not in deep:
            print(f"  (skipping {filename}: {prompt_id} not in results)")
            continue
        title = FIG37_38_39_TITLES.get(prompt_id, prompt_id)
        path = _plot_deep_dive_prompt(deep[prompt_id], title, out_dir / filename)
        out[prompt_id] = path
    return out


def load_legacy_stage8_as_bundle(results_dir: Path) -> dict:
    """Adapter: read legacy per-file stage8 JSONs and repackage as new-schema bundle.

    Reads stage8_post_training.json + stage8_cross_version.json + stage8_deep_dive.json
    (produced by the retired stage8_cross_version_control.py script), and returns a
    dict matching the shape that plot_fig36/plot_figs_37_38_39 expect.
    """
    with open(results_dir / "stage8_cross_version.json") as f:
        cv = json.load(f)
    with open(results_dir / "stage8_deep_dive.json") as f:
        dd = json.load(f)

    neutral = {
        "base_means": cv["base_3_1_neutral_avg"],
        "instruct_means": cv["instruct_3_3_neutral_avg"],
    }
    chall = {
        "base_means": cv["base_3_1_challenging_avg"],
        "instruct_means": cv["instruct_3_3_challenging_avg"],
    }
    deep = {}
    for pid in dd["base_results"]:
        base = dd["base_results"][pid]["projections"]
        inst = dd["instruct_results"][pid]["projections"]
        shifts = dd["shifts"][pid]["full_shift"]
        sorted_emotions = sorted(shifts.keys(), key=lambda e: shifts[e], reverse=True)
        deep[pid] = {
            "diffs": shifts,
            "base": base,
            "instruct": inst,
            "top_increases": [(e, shifts[e]) for e in sorted_emotions[:10]],
            "top_decreases": [(e, shifts[e]) for e in sorted_emotions[-10:]],
        }

    return {
        "activation_comparison": {"neutral": neutral, "challenging": chall},
        "deep_dive": deep,
    }


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
    parser.add_argument("--method", default="mean_diff+gm+pc50",
                        help="Vector extraction method (default: mean_diff+gm+pc50, the fully-denoised Sofroniew vectors)")
    parser.add_argument("--position", default="response[50:]")

    # Mode selection
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--activations-only", action="store_true",
                      help="Run 8.1 only: activation comparison on neutral + challenging")
    mode.add_argument("--layer-sweep-only", action="store_true",
                      help="Run 8.2 only: layer-wise shifts (requires both models)")
    mode.add_argument("--deep-dive-only", action="store_true",
                      help="Run 8.3 only: three deep-dive prompts")
    mode.add_argument("--from-legacy", action="store_true",
                      help="Regenerate Figs 36-39 from legacy per-file JSONs "
                           "(stage8_cross_version.json + stage8_deep_dive.json). "
                           "No GPU; no model load.")

    parser.add_argument("--no-plots", action="store_true",
                        help="Skip paper-figure generation (Figs 36-39)")
    parser.add_argument("--figures-dir", type=str, default=None,
                        help="Output dir for figures (default: paper_figures/ours/)")

    args = parser.parse_args()

    figures_dir = Path(args.figures_dir) if args.figures_dir else FIGURES_DIR

    # --- Legacy regeneration path (no model, no GPU) ---
    if args.from_legacy:
        results_dir = _get_results_dir(args.experiment, "stage8_post_training")
        # Legacy files live at experiments/*/results/ (one level up from stage_NN subdir)
        legacy_dir = results_dir.parent
        print(f"Loading legacy JSONs from {legacy_dir}...")
        bundle = load_legacy_stage8_as_bundle(legacy_dir)
        figures_dir.mkdir(parents=True, exist_ok=True)
        plt_.apply_style("ant")
        p = plot_fig36_post_training_scatter(bundle, figures_dir)
        print(f"  ✓ {p.name}")
        paths = plot_figs_37_38_39_deep_dive(bundle, figures_dir)
        for pid, path in paths.items():
            print(f"  ✓ {path.name}")
        return

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
        flush_cuda()
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
        flush_cuda()
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

    # Generate paper figures (Figs 36, 37, 38, 39)
    if not args.no_plots:
        figures_dir.mkdir(parents=True, exist_ok=True)
        plt_.apply_style("ant")
        if "activation_comparison" in results:
            p = plot_fig36_post_training_scatter(results, figures_dir)
            print(f"  ✓ {p.name}")
        if "deep_dive" in results:
            paths = plot_figs_37_38_39_deep_dive(results, figures_dir)
            for pid, path in paths.items():
                print(f"  ✓ {path.name}")


if __name__ == "__main__":
    main()
