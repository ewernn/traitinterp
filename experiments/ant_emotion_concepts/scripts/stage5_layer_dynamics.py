#!/usr/bin/env python3
"""Layer dynamics experiments for Anthropic Emotion Concepts replication.

Covers Stage 5 from ant_emotion_concepts_plan.md:
  5.1  User vs Assistant dissociation (Fig 10)
  5.2  Colon token predicts response (Fig 11)
  5.3  Context propagation — emotional prefix (Fig 12)
  5.4  Context propagation — numerical (Fig 13)
  5.5  Negation across layers (Fig 14)
  5.6  Person-specific binding (Fig 15)

Each sub-experiment captures activations at specific token positions across all
layers, projects them onto emotion probe vectors, and saves results as JSON for
downstream analysis/visualization.

Key technical challenge: the paper measures probes at specific tokens (user's
last period, Assistant colon, emotion words, pronoun re-references). The
position DSL handles prompt[-1] and response[:5], but we need raw token-level
access. This script tokenizes prompts manually and scans for target tokens.

Input:  datasets/inference/ant_emotion_concepts/*.json  +  extracted vectors
Output: experiments/ant_emotion_concepts/results/stage5/

Usage:
    # Run all sub-experiments
    python experiments/ant_emotion_concepts/scripts/stage5_layer_dynamics.py \
        --experiment ant_emotion_concepts --load-in-4bit

    # Run specific sub-experiment(s)
    python experiments/ant_emotion_concepts/scripts/stage5_layer_dynamics.py \
        --experiment ant_emotion_concepts --load-in-4bit \
        --sub-experiments dissociation,colon_predicts

    # With response generation for colon_predicts (needs autoregressive decoding)
    python experiments/ant_emotion_concepts/scripts/stage5_layer_dynamics.py \
        --experiment ant_emotion_concepts --load-in-4bit \
        --sub-experiments colon_predicts --max-new-tokens 20
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from core import projection, batch_cosine_similarity
from utils.model import load_model, format_prompt
from utils.paths import get as get_path
from utils.distributed import flush_cuda
from utils import plotting as plt_

from shared import (
    EXPERIMENT,
    get_results_dir, save_results,
    load_single_emotion_vector,
)
from utils.capture_activations import capture_at_position

DATASETS_DIR = get_path('datasets.inference') / "ant_emotion_concepts"
FIGURES_DIR = Path(__file__).resolve().parent.parent / "paper_figures" / "ours"

# ---------- Paper-specific plotting constants (Figs 10-15) ----------
PLOT_LAYER = "49"  # probe layer used in Figs 10, 11

# Paper's canonical 6 emotions — ORDER MATTERS: controls iteration/scatter
# z-order and legend sequence. Fig 10 and Fig 11 use different orderings.
PLOT_EMOTIONS_FIG10 = ["happy", "calm", "loving", "sad", "afraid", "angry"]

# Match Sonnet paper's per-emotion colors
EMOTION_COLORS_PAPER = {
    "afraid": "#d62728", "angry": "#e74c3c", "calm": "#1f77b4",
    "happy": "#2ca02c", "loving": "#9467bd", "sad": "#daa520",
}
# Slight variant used in fig 11 (sad is orange there, not gold)
EMOTION_COLORS_FIG11 = {
    "calm": "#1f77b4", "happy": "#2ca02c", "loving": "#9467bd",
    "sad": "#ff7f0e", "afraid": "#d62728", "angry": "#e74c3c",
}

# Paper's reported correlations (Sonnet 4.5) for side-by-side context
PAPER_R_DISSOCIATION = 0.11

# Probe emotions commonly referenced across sub-experiments
CORE_PROBES = [
    "happy", "sad", "afraid", "angry", "calm", "desperate",
    "nervous", "loving", "proud", "surprised", "terrified",
]

ALL_SUB_EXPERIMENTS = [
    "dissociation", "colon_predicts", "context_prefix",
    "context_numerical", "negation", "person_binding",
]


def capture_all_layers(model, tokenizer, text, layers):
    """Full-sequence multi-layer capture for a single text. Returns {layer: [seq_len, hidden_dim]}."""
    acts = capture_at_position(
        model, tokenizer, [text], layers=layers,
        position='all[:]', pool='none', pre_formatted=True, batch_size=1,
    )  # [1, n_layers, seq_len, hidden_dim]
    return {layer: acts[0, li] for li, layer in enumerate(layers)}


# =============================================================================
# Token scanning utilities
# =============================================================================

def find_assistant_colon_position(token_ids, tokenizer) -> int:
    """Find the assistant header's last token position in a token sequence."""
    ids = token_ids.tolist() if isinstance(token_ids, torch.Tensor) else list(token_ids)
    for i in range(len(ids) - 1, 0, -1):
        decoded = tokenizer.decode([ids[i]]).strip().lower()
        prev = tokenizer.decode([ids[i - 1]]).strip().lower()
        if decoded in (':', ':\n', ':\n\n') and 'assistant' in prev:
            return i
        if 'end_header' in decoded and 'assistant' in tokenizer.decode(ids[max(0, i-3):i]).lower():
            return i
    text = tokenizer.decode(ids)
    for pattern in ['Assistant:', 'assistant:', 'A:']:
        idx = text.rfind(pattern)
        if idx >= 0:
            prefix_ids = tokenizer.encode(text[:idx + len(pattern)], add_special_tokens=False)
            return len(prefix_ids) - 1
    return len(ids) - 1


def find_last_user_period(token_ids: List[int], tokenizer, prompt_len: int) -> Optional[int]:
    """Find the last period in the user message portion of the sequence."""
    decoded = [tokenizer.decode([tid]) for tid in token_ids[:prompt_len]]
    for i in range(len(decoded) - 1, -1, -1):
        if "." in decoded[i] or "?" in decoded[i] or "!" in decoded[i]:
            return i
    return prompt_len - 1  # Fallback: last prompt token


def find_word_positions(token_ids: List[int], tokenizer, word: str) -> List[int]:
    """Find all positions where a specific word appears in decoded tokens."""
    decoded = [tokenizer.decode([tid]).strip().lower() for tid in token_ids]
    positions = []
    for i, tok_text in enumerate(decoded):
        if word.lower() in tok_text:
            positions.append(i)
    return positions


# =============================================================================
# Model + vector loading
# =============================================================================

def project_at_positions(
    activations: Dict[int, torch.Tensor],
    vectors: Dict[str, Dict[int, torch.Tensor]],
    positions: List[int],
    layers: List[int],
) -> Dict[str, Dict[int, List[float]]]:
    """Project activations onto emotion vectors at specific token positions."""
    results = {}
    for emotion, layer_vecs in vectors.items():
        results[emotion] = {}
        for layer in layers:
            if layer not in layer_vecs or layer not in activations:
                continue
            vec = layer_vecs[layer]
            acts = activations[layer]  # [seq_len, hidden_dim]
            projs = []
            for pos in positions:
                if pos < acts.shape[0]:
                    p = batch_cosine_similarity(acts[pos:pos+1], vec)
                    projs.append(float(p.item()))
                else:
                    projs.append(0.0)
            results[emotion][layer] = projs
    return results


# =============================================================================
# Sub-experiment 5.1: User vs Assistant dissociation (Fig 10)
# =============================================================================

def run_dissociation(model, tokenizer, vectors, layers, results_dir):
    """Measure probes at user's last period vs Assistant colon position.

    For 8 scenarios where user emotion differs from expected assistant emotion,
    compare probe activations at the two positions.

    Expected: cross-position correlation r ~ 0.11 (weak).
    """
    print("\n=== 5.1: User vs Assistant Dissociation (Fig 10) ===")

    with open(DATASETS_DIR / "dissociation_scenarios.json") as f:
        data = json.load(f)

    scenarios = data["prompts"]
    results = []

    for scenario in tqdm(scenarios, desc="  Dissociation scenarios"):
        prompt_text = scenario["prompt"]
        formatted = format_prompt(prompt_text, tokenizer)

        # Tokenize the full formatted prompt (up to assistant header)
        token_ids = tokenizer.encode(formatted, add_special_tokens=False)
        prompt_len = len(token_ids)

        # Find key positions
        user_period_pos = find_last_user_period(token_ids, tokenizer, prompt_len)
        asst_colon_pos = find_assistant_colon_position(token_ids, tokenizer)
        if asst_colon_pos is None:
            asst_colon_pos = prompt_len - 1

        # Run forward pass
        activations_list = [capture_all_layers(model, tokenizer, formatted, layers)]
        activations = activations_list[0]

        # Project at both positions
        positions = [user_period_pos, asst_colon_pos]
        position_labels = ["user_period", "assistant_colon"]
        projs = project_at_positions(activations, vectors, positions, layers)

        result = {
            "id": scenario["id"],
            "prompt": prompt_text,
            "user_emotion": scenario["user_emotion"],
            "expected_assistant_emotion": scenario["expected_assistant_emotion"],
            "positions": {
                "user_period": user_period_pos,
                "assistant_colon": asst_colon_pos,
            },
            "projections": {},  # {emotion: {layer: {position_label: value}}}
        }

        for emotion in projs:
            result["projections"][emotion] = {}
            for layer in projs[emotion]:
                vals = projs[emotion][layer]
                result["projections"][emotion][layer] = {
                    label: vals[i] for i, label in enumerate(position_labels)
                }

        results.append(result)

    bundle = {
        "experiment": "5.1_user_vs_assistant_dissociation",
        "paper_ref": "Fig 10, Table 3",
        "expected": "cross-position correlation r ~ 0.11",
        "n_scenarios": len(scenarios),
        "layers": layers,
        "results": results,
    }
    save_results(results_dir, "dissociation", bundle)
    return bundle


# =============================================================================
# Sub-experiment 5.2: Colon token predicts response (Fig 11)
# =============================================================================

def run_colon_predicts(model, tokenizer, vectors, layers, results_dir,
                       max_new_tokens: int = 20):
    """Generate 20-token continuations and correlate probes across positions.

    Three measurement points per scenario:
      - User's last period token
      - Assistant colon token
      - Mean across response tokens

    Expected: colon→response correlation r ~ 0.87.
    """
    print("\n=== 5.2: Colon Token Predicts Response (Fig 11) ===")

    with open(DATASETS_DIR / "dissociation_scenarios.json") as f:
        data = json.load(f)

    scenarios = data["prompts"]
    results = []

    for scenario in tqdm(scenarios, desc="  Colon predicts"):
        prompt_text = scenario["prompt"]
        formatted = format_prompt(prompt_text, tokenizer)

        # 1. Generate a short continuation
        prompt_ids = tokenizer.encode(formatted, add_special_tokens=False)
        prompt_len = len(prompt_ids)
        input_tensor = torch.tensor([prompt_ids], device=model.device)

        with torch.no_grad():
            output = model.generate(
                input_tensor,
                max_new_tokens=max_new_tokens,
                temperature=0.0,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        full_ids = output[0].tolist()
        response_ids = full_ids[prompt_len:]
        response_text = tokenizer.decode(response_ids, skip_special_tokens=True)

        # 2. Run forward pass on full sequence (prompt + response)
        full_text = formatted + response_text
        activations_list = [capture_all_layers(model, tokenizer, full_text, layers)]
        activations = activations_list[0]

        # 3. Find positions
        full_token_ids = tokenizer.encode(full_text, add_special_tokens=False)
        user_period_pos = find_last_user_period(full_token_ids, tokenizer, prompt_len)
        asst_colon_pos = find_assistant_colon_position(full_token_ids[:prompt_len], tokenizer)
        if asst_colon_pos is None:
            asst_colon_pos = prompt_len - 1

        # Response token range
        response_start = prompt_len
        response_end = len(full_token_ids)

        # 4. Project at user period and assistant colon
        point_positions = [user_period_pos, asst_colon_pos]
        point_projs = project_at_positions(activations, vectors, point_positions, layers)

        # 5. Compute mean projection across response tokens
        response_positions = list(range(response_start, response_end))
        response_projs = project_at_positions(activations, vectors, response_positions, layers)

        result = {
            "id": scenario["id"],
            "prompt": prompt_text,
            "response": response_text,
            "user_emotion": scenario["user_emotion"],
            "expected_assistant_emotion": scenario["expected_assistant_emotion"],
            "positions": {
                "user_period": user_period_pos,
                "assistant_colon": asst_colon_pos,
                "response_start": response_start,
                "response_end": response_end,
            },
            "projections": {},
        }

        for emotion in point_projs:
            result["projections"][emotion] = {}
            for layer in layers:
                if layer not in point_projs.get(emotion, {}):
                    continue
                vals = point_projs[emotion][layer]
                resp_vals = response_projs.get(emotion, {}).get(layer, [])
                resp_mean = sum(resp_vals) / len(resp_vals) if resp_vals else 0.0

                result["projections"][emotion][layer] = {
                    "user_period": vals[0],
                    "assistant_colon": vals[1],
                    "response_mean": resp_mean,
                }

        results.append(result)

        del output
        flush_cuda()

    bundle = {
        "experiment": "5.2_colon_predicts_response",
        "paper_ref": "Fig 11, Table 4",
        "expected": "assistant_colon→response_mean correlation r ~ 0.87",
        "n_scenarios": len(scenarios),
        "max_new_tokens": max_new_tokens,
        "layers": layers,
        "results": results,
    }
    save_results(results_dir, "colon_predicts", bundle)
    return bundle


# =============================================================================
# Sub-experiment 5.3: Context propagation — emotional prefix (Fig 12)
# =============================================================================

def _run_context_propagation(
    model, tokenizer, vectors, layers, results_dir,
    template_index: int, output_name: str, experiment_id: str,
    paper_ref: str, expected: str, diff_sign: int = 1,
):
    """Shared implementation for context propagation experiments (Figs 12-13).

    Captures all-token all-layer activations for two conditions of a template,
    computes per-emotion projections and condition difference heatmap.

    Args:
        template_index: index into context_propagation_templates.json
        diff_sign: +1 for v0-v1 (prefix), -1 for v1-v0 (numerical)
    """
    with open(DATASETS_DIR / "context_propagation_templates.json") as f:
        data = json.load(f)

    template_data = data["templates"][template_index]
    template = template_data["template"]
    values = template_data["values"]

    results = {}
    for value in values:
        prompt_text = template.replace("{X}", str(value))
        formatted = format_prompt(prompt_text, tokenizer)
        token_ids = tokenizer.encode(formatted, add_special_tokens=False)
        decoded_tokens = [tokenizer.decode([tid]) for tid in token_ids]

        activations = capture_all_layers(model, tokenizer, formatted, layers)
        all_positions = list(range(len(token_ids)))
        projs = project_at_positions(activations, vectors, all_positions, layers)

        results[str(value)] = {
            "prompt": prompt_text, "tokens": decoded_tokens,
            "n_tokens": len(token_ids),
            "projections": {
                emotion: {str(l): projs[emotion].get(l, []) for l in layers}
                for emotion in projs
            },
        }

    # Condition difference heatmap
    keys = [str(v) for v in values]
    n_tokens = min(results[keys[0]]["n_tokens"], results[keys[1]]["n_tokens"])
    difference = {"tokens": results[keys[0]]["tokens"], "projections": {}}
    for emotion in results[keys[0]]["projections"]:
        if emotion not in results[keys[1]]["projections"]:
            continue
        difference["projections"][emotion] = {}
        for layer in layers:
            sl = str(layer)
            v0 = results[keys[0]]["projections"][emotion].get(sl, [])
            v1 = results[keys[1]]["projections"][emotion].get(sl, [])
            difference["projections"][emotion][sl] = [
                diff_sign * (v0[i] - v1[i]) if i < len(v0) and i < len(v1) else 0.0
                for i in range(n_tokens)
            ]

    bundle = {
        "experiment": experiment_id, "paper_ref": paper_ref,
        "template_id": template_data["id"], "conditions": keys,
        "expected": expected, "layers": layers,
        "condition_results": results, "condition_difference": difference,
    }
    save_results(results_dir, output_name, bundle)
    return bundle


def run_context_prefix(model, tokenizer, vectors, layers, results_dir):
    """Layer x token heatmap for 'hard' vs 'good' prefix (Fig 12)."""
    print("\n=== 5.3: Context Propagation — Emotional Prefix (Fig 12) ===")
    return _run_context_propagation(
        model, tokenizer, vectors, layers, results_dir,
        template_index=0, output_name="context_prefix",
        experiment_id="5.3_context_propagation_prefix", paper_ref="Fig 12",
        expected="Early layers: difference at diverging word only. Late layers: sustained difference across shared suffix.",
        diff_sign=1,
    )


def run_context_numerical(model, tokenizer, vectors, layers, results_dir):
    """Layer x token heatmap for Tylenol 1000 vs 8000mg (Fig 13)."""
    print("\n=== 5.4: Context Propagation — Numerical (Fig 13) ===")
    return _run_context_propagation(
        model, tokenizer, vectors, layers, results_dir,
        template_index=1, output_name="context_numerical",
        experiment_id="5.4_context_propagation_numerical", paper_ref="Fig 13",
        expected="Late layers: elevated 'terrified' for 8000mg; early layers: no difference.",
        diff_sign=-1,
    )


# =============================================================================
# Sub-experiment 5.5: Negation across layers (Fig 14)
# =============================================================================

def run_negation(model, tokenizer, vectors, layers, results_dir):
    """Compare 'feeling X' vs 'not feeling X' across layers.

    At the emotion word: early layers show similar activation for both;
    late layers resolve the negation.
    At the Assistant colon: distinction only in later layers.
    """
    print("\n=== 5.5: Negation Across Layers (Fig 14) ===")

    with open(DATASETS_DIR / "negation_templates.json") as f:
        data = json.load(f)

    pos_template = data["templates"][0]["positive"]  # "I am feeling {emotion} right now"
    neg_template = data["templates"][0]["negative"]  # "I am not feeling {emotion} right now"

    # Test with the core probe emotions
    test_emotions = ["happy", "sad", "afraid", "angry", "calm", "desperate"]
    results = []

    for emotion in tqdm(test_emotions, desc="  Negation emotions"):
        pos_text = pos_template.replace("{emotion}", emotion)
        neg_text = neg_template.replace("{emotion}", emotion)

        for condition, prompt_text in [("positive", pos_text), ("negated", neg_text)]:
            formatted = format_prompt(prompt_text, tokenizer)
            token_ids = tokenizer.encode(formatted, add_special_tokens=False)
            decoded_tokens = [tokenizer.decode([tid]) for tid in token_ids]

            activations_list = [capture_all_layers(model, tokenizer, formatted, layers)]
            activations = activations_list[0]

            # Find key positions: the emotion word and "now" (end of statement)
            emotion_positions = find_word_positions(token_ids, tokenizer, emotion)
            now_positions = find_word_positions(token_ids, tokenizer, "now")
            asst_colon_pos = find_assistant_colon_position(token_ids, tokenizer)

            # Also measure at all positions for the full heatmap
            all_positions = list(range(len(token_ids)))
            projs = project_at_positions(activations, vectors, all_positions, layers)

            result = {
                "emotion": emotion,
                "condition": condition,
                "prompt": prompt_text,
                "tokens": decoded_tokens,
                "positions": {
                    "emotion_word": emotion_positions,
                    "now": now_positions,
                    "assistant_colon": asst_colon_pos,
                },
                "projections": {},
            }

            for emo in projs:
                result["projections"][emo] = {
                    str(layer): projs[emo].get(layer, []) for layer in layers
                }

            results.append(result)

    bundle = {
        "experiment": "5.5_negation_across_layers",
        "paper_ref": "Fig 14",
        "expected": "Early layers: similar activation for both conditions at emotion word. Late layers: negated drops to near zero.",
        "test_emotions": test_emotions,
        "layers": layers,
        "results": results,
    }
    save_results(results_dir, "negation", bundle)
    return bundle


# =============================================================================
# Sub-experiment 5.6: Person-specific binding (Fig 15)
# =============================================================================

def run_person_binding(model, tokenizer, vectors, layers, results_dir):
    """Probe reactivation at person re-reference tokens across layers.

    16 scenarios with Person A (emotion_A) and Person B (emotion_B).
    At pronoun re-references (she/he/her/him), the corresponding person's
    emotion probe should activate in late layers.
    """
    print("\n=== 5.6: Person-Specific Emotion Binding (Fig 15) ===")

    with open(DATASETS_DIR / "person_binding_scenarios.json") as f:
        data = json.load(f)

    scenarios = data["scenarios"]
    results = []

    # Pronouns to scan for as re-reference tokens
    pronouns_female = ["she", "her"]
    pronouns_male = ["he", "him", "his"]

    for scenario in tqdm(scenarios, desc="  Person binding"):
        prompt_text = scenario["text"]
        formatted = format_prompt(prompt_text, tokenizer)

        token_ids = tokenizer.encode(formatted, add_special_tokens=False)
        decoded_tokens = [tokenizer.decode([tid]) for tid in token_ids]

        activations_list = [capture_all_layers(model, tokenizer, formatted, layers)]
        activations = activations_list[0]

        # Find emotion word positions
        emotion_a_positions = find_word_positions(token_ids, tokenizer, scenario["emotion_A"])
        emotion_b_positions = find_word_positions(token_ids, tokenizer, scenario["emotion_B"])

        # Find pronoun re-reference positions
        # Person A pronouns
        pronouns_a = pronouns_female if scenario["person_A_gender"] == "female" else pronouns_male
        pronouns_b = pronouns_male if scenario["person_A_gender"] == "female" else pronouns_female

        person_a_refs = []
        person_b_refs = []
        for pron in pronouns_a:
            person_a_refs.extend(find_word_positions(token_ids, tokenizer, pron))
        for pron in pronouns_b:
            person_b_refs.extend(find_word_positions(token_ids, tokenizer, pron))

        # Remove emotion word positions from pronoun lists (they overlap sometimes)
        all_emotion_positions = set(emotion_a_positions + emotion_b_positions)
        person_a_refs = sorted(set(person_a_refs) - all_emotion_positions)
        person_b_refs = sorted(set(person_b_refs) - all_emotion_positions)

        # Project at all positions for heatmap
        all_positions = list(range(len(token_ids)))
        projs = project_at_positions(activations, vectors, all_positions, layers)

        result = {
            "id": scenario["id"],
            "text": prompt_text,
            "emotion_A": scenario["emotion_A"],
            "emotion_B": scenario["emotion_B"],
            "person_A_gender": scenario["person_A_gender"],
            "person_B_gender": scenario["person_B_gender"],
            "tokens": decoded_tokens,
            "positions": {
                "emotion_A_words": emotion_a_positions,
                "emotion_B_words": emotion_b_positions,
                "person_A_refs": person_a_refs,
                "person_B_refs": person_b_refs,
            },
            "projections": {},
        }

        for emo in projs:
            result["projections"][emo] = {
                str(layer): projs[emo].get(layer, []) for layer in layers
            }

        results.append(result)

    bundle = {
        "experiment": "5.6_person_specific_binding",
        "paper_ref": "Fig 15",
        "expected": "At re-reference tokens: person's emotion probe activates in late layers only. At emotion words: corresponding probe activates in early layers.",
        "n_scenarios": len(scenarios),
        "layers": layers,
        "results": results,
    }
    save_results(results_dir, "person_binding", bundle)
    return bundle


# =============================================================================
# Paper figures (Figs 10-15)
# =============================================================================

def _clean_token(tok: str) -> str:
    """Shorten special tokens for readable x-axis labels."""
    reps = {
        '<|begin_of_text|>': 'BoT', '<|start_header_id|>': '<hdr>',
        '<|end_header_id|>': '</hdr>', '<|eot_id|>': '<eot>',
        '\n\n': '\\n\\n', '\n': '\\n',
    }
    return reps.get(tok, tok.strip() if tok.strip() else repr(tok))


def _find_user_content_start(tokens) -> int:
    for i, t in enumerate(tokens):
        if t == 'user':
            return i + 3
    return 30


# Short column labels for the Fig 10 heatmap — matching the paper's Table 3 labels.
# Keep in sync with datasets/inference/ant_emotion_concepts/dissociation_scenarios.json.
FIG10_SCENARIO_LABELS = {
    "ai_scares_me": "AI scares me",
    "fired_no_warning": "Fired, no warning",
    "useless_response": "Useless response",
    "all_in_on_crypto": "All-in on crypto",
    "ignoring_chest_pains": "Ignoring chest pains",
    "24hr_no_sleep_drive": "24hr no-sleep drive",
    "another_boring_report": "Another boring report",
    "3000yr_old_honey": "3000yr-old honey",
}


def plot_fig10_dissociation(bundle: dict, out_dir: Path) -> Path:
    """Fig 10 — heatmap (probes × scenarios at U/A) + User/Assistant scatter.

    Two-panel replication of the paper's Fig 10 layout: the heatmap shows per-
    scenario U-vs-A probe divergence; the scatter aggregates cross-position
    correlation.
    """
    import numpy as np
    from matplotlib.lines import Line2D
    from scipy import stats

    # --- Build heatmap matrix: rows = [afraid-U, afraid-A, angry-U, ...], cols = scenarios ---
    # Row order follows the paper's heatmap (Afraid, Angry, Sad, Calm, Happy, Loving,
    # each split into User/Assistant sub-rows).
    HEATMAP_EMOTIONS = ["afraid", "angry", "sad", "calm", "happy", "loving"]
    POSITIONS = [("user_period", "U"), ("assistant_colon", "A")]

    scenarios = bundle["results"]
    col_labels = [FIG10_SCENARIO_LABELS.get(s["id"], s["id"]) for s in scenarios]

    heat = np.zeros((len(HEATMAP_EMOTIONS) * 2, len(scenarios)))
    row_labels = []
    for i, emo in enumerate(HEATMAP_EMOTIONS):
        for j, (pos_key, pos_tag) in enumerate(POSITIONS):
            row_labels.append(f"{pos_tag}")
            for k, s in enumerate(scenarios):
                p = s["projections"].get(emo, {}).get(PLOT_LAYER, {})
                heat[i * 2 + j, k] = p.get(pos_key, 0.0)

    # Scatter data (unchanged from prior)
    user_vals, asst_vals, emo_labels = [], [], []
    for s in bundle["results"]:
        for emo in PLOT_EMOTIONS_FIG10:
            if emo in s["projections"] and PLOT_LAYER in s["projections"][emo]:
                proj = s["projections"][emo][PLOT_LAYER]
                user_vals.append(proj["user_period"])
                asst_vals.append(proj["assistant_colon"])
                emo_labels.append(emo)
    user_vals = np.array(user_vals); asst_vals = np.array(asst_vals)
    r_ours, _ = stats.pearsonr(user_vals, asst_vals)

    # --- Two-panel figure: heatmap (left, dominant) + scatter (right, compact) ---
    fig, (ax_heat, ax) = plt_.multi_panel(
        1, 2, figsize=(22, 14),
        gridspec_kw={"width_ratios": [1.8, 1.0]},
    )
    # Single title above both panels
    plt_.suptitle(fig, "Emotion Probes Distinguish User and Assistant Emotions",
                  fontsize=24, y=0.98)

    # Heatmap (taller figure + larger fonts)
    vmax = float(np.abs(heat).max()) or 0.08
    plt_.heatmap(
        ax_heat, heat, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
        row_labels=row_labels, col_labels=col_labels,
        tick_fontsize=15, xtick_rotation=45,
        cbar_label="Cosine Similarity", cbar_pad=0.02,
        cbar_label_fontsize=16,
        show_spines=True, spine_linewidth=1.0, aspect="auto",
    )
    # Emotion labels as grouped y-tick annotations on the LEFT of the axis
    for i, emo in enumerate(HEATMAP_EMOTIONS):
        y = i * 2 + 0.5  # centered between U and A rows
        ax_heat.text(-0.5, y, emo.capitalize(),
                     ha="right", va="center", fontsize=15, fontweight="bold",
                     transform=ax_heat.transData)
    # Draw thin separators between emotion pairs
    for i in range(1, len(HEATMAP_EMOTIONS)):
        ax_heat.axhline(i * 2 - 0.5, color="black", linewidth=0.5, alpha=0.4)

    ax_heat.set_xlabel("Scenario", fontsize=18)
    ax_heat.set_ylabel("Emotion Probe (U=User, A=Assistant)", fontsize=16, labelpad=40)

    # Scatter (4× original dots: s=320)
    for emo in PLOT_EMOTIONS_FIG10:
        mask = np.array([e == emo for e in emo_labels])
        if mask.sum() == 0:
            continue
        ax.scatter(user_vals[mask], asst_vals[mask],
                   c=EMOTION_COLORS_PAPER[emo], s=320, alpha=0.85,
                   edgecolors='white', linewidths=0.6, zorder=3)

    x_range = np.linspace(-0.075, 0.075, 100)
    slope, intercept = np.polyfit(user_vals, asst_vals, 1)
    ax.plot(x_range, slope * x_range + intercept, '-', color='#1a1a1a', linewidth=2.5, zorder=4)

    # Paper reference line at r=0.11, locked through our data's mean
    paper_slope = PAPER_R_DISSOCIATION * (asst_vals.std() / user_vals.std())
    paper_intercept = asst_vals.mean() - paper_slope * user_vals.mean()
    ax.plot(x_range, paper_slope * x_range + paper_intercept, '--',
            color='#b0b0b0', linewidth=2.5, zorder=4)

    ax.set_xlim(-0.075, 0.075); ax.set_ylim(-0.05, 0.05)
    ax.grid(True, alpha=0.12)
    ax.set_xlabel("Probe @ User's Last Period", fontsize=18)
    ax.set_ylabel("Probe @ Assistant Colon", fontsize=18)
    ax.tick_params(labelsize=14)
    ax.text(0.04, 0.97,
            f"Llama 70B:  r = {r_ours:.2f}\nSonnet (paper):  r = {PAPER_R_DISSOCIATION:.2f}",
            transform=ax.transAxes, fontsize=18, fontweight="bold",
            va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#aaa", alpha=0.95))

    # Legend: emotion dots only, below the scatter (r-values shown in text box at top)
    handles = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=EMOTION_COLORS_PAPER[e],
               markersize=14, label=e.capitalize())
        for e in PLOT_EMOTIONS_FIG10
    ]
    ax.legend(handles=handles, fontsize=14, loc='upper center',
              bbox_to_anchor=(0.5, -0.12), ncol=3, framealpha=0.92, edgecolor='#ccc')

    import matplotlib.pyplot as plt
    plt.tight_layout()
    fig.subplots_adjust(bottom=0.18)
    return plt_.save_figure(fig, out_dir / "fig10_ours.png")


def plot_fig11_colon_predicts(bundle: dict, out_dir: Path) -> Path:
    """Fig 11 — Probe at User '.' and Assistant ':' predicts response emotion."""
    import numpy as np
    from matplotlib.lines import Line2D
    from scipy import stats

    user_all, colon_all, response_all, emo_all = [], [], [], []
    for r_item in bundle["results"]:
        for emo in EMOTION_COLORS_FIG11:
            if emo in r_item["projections"] and PLOT_LAYER in r_item["projections"][emo]:
                proj = r_item["projections"][emo][PLOT_LAYER]
                user_all.append(proj["user_period"])
                colon_all.append(proj["assistant_colon"])
                response_all.append(proj["response_mean"])
                emo_all.append(emo)

    user_all = np.array(user_all); colon_all = np.array(colon_all); response_all = np.array(response_all)
    r_user, _ = stats.pearsonr(user_all, response_all)
    r_colon, _ = stats.pearsonr(colon_all, response_all)

    fig, (ax1, ax2) = plt_.multi_panel(1, 2, figsize=(16, 7))
    plt_.suptitle(fig, "The Assistant : Token Predicts Response Emotion", y=0.98)

    for ax, x_vals, r_val, xlabel in [
        (ax1, user_all, r_user, 'Probe @ User "."'),
        (ax2, colon_all, r_colon, 'Probe @ Assistant ":"'),
    ]:
        for emo, color in EMOTION_COLORS_FIG11.items():
            mask = np.array([e == emo for e in emo_all])
            if mask.sum() > 0:
                ax.scatter(x_vals[mask], response_all[mask], s=160, alpha=0.7,
                           c=color, edgecolors="white", linewidths=0.5, zorder=2)
        slope, intercept = np.polyfit(x_vals, response_all, 1)
        xs = np.linspace(x_vals.min(), x_vals.max(), 100)
        ax.plot(xs, slope * xs + intercept, "--", color="#333", linewidth=1.5, zorder=3)
        plt_.annotate_r_value(ax, r_val)
        ax.axhline(0, color="#ccc", linewidth=0.5)
        ax.axvline(0, color="#ccc", linewidth=0.5)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Probe @ Mean Response")
        ax.grid(True, alpha=0.1)
        ax.set_box_aspect(1)  # force square panel regardless of data range

    handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=c,
                      markersize=12, label=e.capitalize())
               for e, c in EMOTION_COLORS_FIG11.items()]
    fig.legend(handles=handles, loc='lower center', ncol=6, fontsize=14,
               bbox_to_anchor=(0.5, -0.02), frameon=True, framealpha=0.92, edgecolor='#ccc')

    import matplotlib.pyplot as plt
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    return plt_.save_figure(fig, out_dir / "fig11_ours.png")


def _plot_context_line_diff(bundle: dict, probe: str, subtitle: str,
                             diff_sign: int, out_path: Path,
                             *, ylim: tuple | None = None,
                             figsize: tuple = (12, 5),
                             linewidth: float = 2.5, markersize: float = 5,
                             legend_kwargs: dict | None = None,
                             bottom_margin: float | None = None) -> Path:
    """Shared Fig 12/13 line chart: early vs late layer mean difference per token.

    Fig 13 overrides linewidth/legend to emphasize the late-layer fear response;
    fig 12 uses defaults.
    """
    import numpy as np

    layers = bundle["layers"]
    diff = bundle["condition_difference"]
    tokens = diff["tokens"]
    proj_by_layer = diff["projections"][probe]

    matrix = diff_sign * np.array([proj_by_layer[str(l)] for l in layers])
    n = len(layers); q = n // 4
    groups = {
        "Early": layers[:q], "Early-Mid": layers[q:2*q],
        "Mid-Late": layers[2*q:3*q], "Late": layers[3*q:],
    }
    means = {gn: matrix[[layers.index(l) for l in gl]].mean(axis=0) for gn, gl in groups.items()}
    early_line = (means["Early"] + means["Early-Mid"]) / 2
    late_line = (means["Mid-Late"] + means["Late"]) / 2

    user_start = _find_user_content_start(tokens)
    sub_tokens = tokens[user_start:]
    labels = [_clean_token(t) for t in sub_tokens]

    fig, ax = plt_.multi_panel(figsize=figsize)
    series = {
        'Early → Early-Mid': early_line[user_start:],
        'Mid-Late → Late': late_line[user_start:],
    }
    colors = {'Early → Early-Mid': '#1f77b4', 'Mid-Late → Late': '#d62728'}
    legend_kw = legend_kwargs or {'fontsize': 12, 'loc': 'upper left', 'framealpha': 0.92}
    plt_.line_plot(ax, list(range(len(labels))), series, colors=colors,
                   linewidth=linewidth, markersize=markersize,
                   legend_kwargs=legend_kw)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=10)
    ax.set_ylabel('Delta Cosine Similarity')
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_title('Mean Difference by Layer Range',
                 fontsize=20, fontweight='bold', color=plt_.TITLE_CORAL, pad=12)
    ax.text(0.5, 1.02, subtitle, transform=ax.transAxes, ha='center',
            fontsize=13, style='italic', color='#555')

    fig.tight_layout()
    if bottom_margin is not None:
        fig.subplots_adjust(bottom=bottom_margin)
    return plt_.save_figure(fig, out_path)


def plot_fig12_context_prefix(bundle: dict, out_dir: Path) -> Path:
    """Fig 12 — 3-panel: 'hard' Happy probe heatmap + difference heatmap + layer-range line plot."""
    import numpy as np
    import matplotlib.pyplot as plt

    layers = bundle["layers"]
    tokens = bundle["condition_difference"]["tokens"]
    user_start = _find_user_content_start(tokens)
    labels = [_clean_token(t) for t in tokens[user_start:]]

    hard_proj = bundle["condition_results"]["hard"]["projections"]["happy"]
    diff_proj = bundle["condition_difference"]["projections"]["happy"]
    # Stored as hard-good; paper convention is good-hard (positive = "good" prefix adds happiness)
    hard_matrix = np.array([hard_proj[str(l)][user_start:] for l in layers])
    diff_matrix = -np.array([diff_proj[str(l)][user_start:] for l in layers])

    # Line plot data: full matrix for 4-band averaging
    diff_full = -np.array([diff_proj[str(l)] for l in layers])
    n = len(layers); q = n // 4
    groups = {
        "Early": layers[:q], "Early-Mid": layers[q:2*q],
        "Mid-Late": layers[2*q:3*q], "Late": layers[3*q:],
    }
    means = {gn: diff_full[[layers.index(l) for l in gl]].mean(axis=0) for gn, gl in groups.items()}
    early_line = (means["Early"] + means["Early-Mid"]) / 2
    late_line = (means["Mid-Late"] + means["Late"]) / 2

    fig, axes = plt.subplots(3, 1, figsize=(12, 16.5), sharex=True,
                             gridspec_kw={'height_ratios': [1.1, 1.1, 1]})
    ax_h1, ax_h2, ax_line = axes

    vmax1 = np.abs(hard_matrix).max()
    im1 = ax_h1.imshow(hard_matrix, aspect='auto', cmap='RdBu_r', vmin=-vmax1, vmax=vmax1)
    ax_h1.set_yticks(range(len(layers)))
    ax_h1.set_yticklabels([str(l) for l in layers], fontsize=8)
    ax_h1.set_ylabel("Layer")
    ax_h1.set_title('Happy Probe ("...really hard...")', fontsize=14, fontweight='bold',
                    color=plt_.TITLE_CORAL, pad=8)
    fig.colorbar(im1, ax=ax_h1, fraction=0.02, pad=0.01)

    vmax2 = np.abs(diff_matrix).max()
    im2 = ax_h2.imshow(diff_matrix, aspect='auto', cmap='RdBu_r', vmin=-vmax2, vmax=vmax2)
    ax_h2.set_yticks(range(len(layers)))
    ax_h2.set_yticklabels([str(l) for l in layers], fontsize=8)
    ax_h2.set_ylabel("Layer")
    ax_h2.set_title('Happy Probe Difference ("good" − "hard")', fontsize=14, fontweight='bold',
                    color=plt_.TITLE_CORAL, pad=8)
    fig.colorbar(im2, ax=ax_h2, fraction=0.02, pad=0.01)

    ax_line.plot(range(len(labels)), early_line[user_start:], '-o',
                 color='#1f77b4', linewidth=2.5, markersize=5, label='Early → Early-Mid')
    ax_line.plot(range(len(labels)), late_line[user_start:], '-o',
                 color='#d62728', linewidth=2.5, markersize=5, label='Mid-Late → Late')
    ax_line.axhline(0, color='#ccc', linewidth=0.8, zorder=0)
    ax_line.set_ylabel('Delta Cosine Similarity')
    ax_line.set_xlabel('Token')
    ax_line.legend(fontsize=11, loc='upper center', bbox_to_anchor=(0.5, -0.28),
                   ncol=2, frameon=True, framealpha=0.92)
    ax_line.set_title('Mean Difference by Layer Range', fontsize=14, fontweight='bold',
                      color=plt_.TITLE_CORAL, pad=8)
    ax_line.grid(True, alpha=0.15)

    ax_line.set_xticks(range(len(labels)))
    ax_line.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)

    fig.suptitle('Context Propagation: "really good" vs "really hard" prefix',
                 fontsize=16, fontweight='bold', y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.subplots_adjust(bottom=0.15)
    return plt_.save_figure(fig, out_dir / "fig12_ours.png")


def plot_fig13_context_numerical(bundle: dict, out_dir: Path) -> Path:
    """Fig 13 — Tylenol 8000mg vs 1000mg, per-token 'terrified' diff."""
    return _plot_context_line_diff(
        bundle, probe="terrified",
        subtitle='Terrified Probe: "...8000mg..." vs "...1000mg..."',
        diff_sign=1,
        out_path=out_dir / "fig13_ours.png",
        ylim=(-0.005, 0.04),
        figsize=(12, 7.5),
        linewidth=5, markersize=8,
        legend_kwargs={'fontsize': 24, 'loc': 'upper center',
                       'bbox_to_anchor': (0.5, -0.35), 'ncol': 2,
                       'framealpha': 0.92, 'edgecolor': '#ccc'},
        bottom_margin=0.4,
    )


def plot_fig14_negation(bundle: dict, out_dir: Path) -> Path:
    """Fig 14 — Negation resolution across layers (paired solid/dashed per position)."""
    import numpy as np

    layers = bundle['layers']
    results = bundle['results']
    pos_names = ['emotion_word', 'user_turn_end', 'assistant_colon']
    collected = {c: {p: {l: [] for l in layers} for p in pos_names}
                 for c in ['positive', 'negated']}

    for r in results:
        emotion = r['emotion']
        cond = r['condition']
        positions = r['positions']
        proj = r['projections'][emotion]
        emo_pos = positions['emotion_word'][0]
        user_end = positions['now'][-1] + 1
        asst_pos = positions['assistant_colon']
        for l in layers:
            vals = proj[str(l)]
            collected[cond]['emotion_word'][l].append(vals[emo_pos])
            collected[cond]['user_turn_end'][l].append(vals[user_end])
            collected[cond]['assistant_colon'][l].append(vals[asst_pos])

    avg = {c: {p: [np.mean(collected[c][p][l]) for l in layers] for p in pos_names}
           for c in ['positive', 'negated']}

    pos_config = {
        'emotion_word': ('#ff7f0e', '"feeling [X]" @ [X]', '"not feeling [X]" @ [X]'),
        'user_turn_end': ('#2ca02c', '"feeling [X]" @ Turn End', '"not feeling [X]" @ Turn End'),
        'assistant_colon': ('#1f77b4', '"feeling [X]" @ Asst :', '"not feeling [X]" @ Asst :'),
    }

    fig, ax = plt_.multi_panel(figsize=(12, 6))
    for pos in pos_names:
        color, pos_lbl, neg_lbl = pos_config[pos]
        ax.plot(layers, avg['positive'][pos], '-o', color=color, linewidth=4.5,
                markersize=7, label=pos_lbl, zorder=3)
        ax.plot(layers, avg['negated'][pos], '--s', color=color, linewidth=4,
                markersize=6, alpha=0.6, label=neg_lbl, zorder=3)

    ax.set_xlabel('Layer'); ax.set_ylabel('Cosine Similarity')
    ax.set_xticks(layers); ax.set_xticklabels([str(l) for l in layers])
    ax.grid(True, alpha=0.15)
    ax.axhline(0, color='#ccc', linewidth=0.8, zorder=0)
    ax.set_title('Negation Resolution Across Layers',
                 fontsize=20, fontweight='bold', color='black', pad=12)
    ax.legend(fontsize=14, loc='upper center', bbox_to_anchor=(0.5, -0.15),
              ncol=3, framealpha=0.92, edgecolor='#ccc')
    fig.tight_layout(); fig.subplots_adjust(bottom=0.25)
    return plt_.save_figure(fig, out_dir / "fig14_ours.png")


def plot_fig15_person_binding(bundle: dict, out_dir: Path) -> Path:
    """Fig 15 — Entity-binding: matched vs unmatched probes at emotion + re-reference."""
    import numpy as np

    layers = bundle['layers']
    results = bundle['results']
    line_data = {k: {l: [] for l in layers} for k in
                 ['matched_emotion', 'unmatched_emotion', 'matched_reref', 'unmatched_reref']}

    for r in results:
        em_A, em_B = r['emotion_A'], r['emotion_B']
        proj = r['projections']
        if em_A not in proj or em_B not in proj:
            continue
        positions = r['positions']
        tokens = r['tokens']
        emo_A_pos = positions['emotion_A_words']
        emo_B_pos = positions['emotion_B_words']
        user_start = _find_user_content_start(tokens)
        ref_A = [p for p in positions['person_A_refs'] if p >= user_start]
        ref_B = [p for p in positions['person_B_refs'] if p >= user_start]
        if not ref_A or not ref_B:
            continue
        for l in layers:
            lk = str(l)
            pA, pB = proj[em_A][lk], proj[em_B][lk]
            m_emo = [pA[p] for p in emo_A_pos if p < len(pA)] + [pB[p] for p in emo_B_pos if p < len(pB)]
            u_emo = [pB[p] for p in emo_A_pos if p < len(pB)] + [pA[p] for p in emo_B_pos if p < len(pA)]
            m_ref = [pA[p] for p in ref_A if p < len(pA)] + [pB[p] for p in ref_B if p < len(pB)]
            u_ref = [pB[p] for p in ref_A if p < len(pB)] + [pA[p] for p in ref_B if p < len(pA)]
            if m_emo: line_data['matched_emotion'][l].append(np.mean(m_emo))
            if u_emo: line_data['unmatched_emotion'][l].append(np.mean(u_emo))
            if m_ref: line_data['matched_reref'][l].append(np.mean(m_ref))
            if u_ref: line_data['unmatched_reref'][l].append(np.mean(u_ref))

    avg = {k: [np.mean(line_data[k][l]) for l in layers] for k in line_data}

    fig, ax = plt_.multi_panel(figsize=(14, 5))
    # Plot order below controls legend layout — matplotlib fills ncol=2 column-major,
    # so ordering as [matched-emo, matched-ref, unmatched-emo, unmatched-ref]
    # yields the paper's visual: row 1 = both matched, row 2 = both unmatched.
    line_matched_emo, = ax.plot(layers, avg['matched_emotion'], '-o', color='#2ca02c',
            linewidth=5, markersize=7, label='Matched @ emotion', zorder=3)
    line_matched_ref, = ax.plot(layers, avg['matched_reref'], '-o', color='#1f77b4',
            linewidth=5, markersize=7, label='Matched @ re-ref', zorder=3)
    line_unmatched_emo, = ax.plot(layers, avg['unmatched_emotion'], '--s', color='#2ca02c',
            linewidth=4, markersize=6, alpha=0.55, label='Unmatched @ emotion', zorder=3)
    line_unmatched_ref, = ax.plot(layers, avg['unmatched_reref'], '--s', color='#1f77b4',
            linewidth=4, markersize=6, alpha=0.55, label='Unmatched @ re-ref', zorder=3)

    ax.set_xlabel('Layer'); ax.set_ylabel('Cosine Similarity')
    ax.set_xticks(layers); ax.set_xticklabels([str(l) for l in layers])
    ax.grid(True, alpha=0.15)
    ax.axhline(0, color='#ccc', linewidth=0.8, zorder=0)
    ax.set_title('Entity-Binding: Matched vs Unmatched',
                 fontsize=20, fontweight='bold', color='black', pad=12)
    ax.legend(handles=[line_matched_emo, line_unmatched_emo,
                       line_matched_ref, line_unmatched_ref],
              fontsize=18, loc='upper center', bbox_to_anchor=(0.5, -0.22),
              ncol=2, framealpha=0.92, edgecolor='#ccc')
    fig.tight_layout(); fig.subplots_adjust(bottom=0.38)
    return plt_.save_figure(fig, out_dir / "fig15_ours.png")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--experiment", default=EXPERIMENT)
    parser.add_argument("--model-variant", default=None)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--method", default="mean_diff+gm+pc50",
                        help="Vector method to use (default: mean_diff+gm+pc50, the fully-denoised Sofroniew vectors)")
    parser.add_argument("--category", default="ant_emotion_concepts")

    # Layer selection
    parser.add_argument("--layers", type=str, default=None,
                        help="Comma-separated layer indices (default: 20 evenly spaced from 10%% to 95%% depth)")
    parser.add_argument("--n-layers-sample", type=int, default=20,
                        help="Number of layers to sample when --layers not specified")

    # Sub-experiment selection
    parser.add_argument("--sub-experiments", type=str, default=None,
                        help=f"Comma-separated sub-experiments to run. Options: {','.join(ALL_SUB_EXPERIMENTS)}")

    # Probe emotions
    parser.add_argument("--probe-emotions", type=str, default=None,
                        help="Comma-separated emotions to use as probes (default: core set of 11)")

    # Generation (for colon_predicts)
    parser.add_argument("--max-new-tokens", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1)

    # Plotting
    parser.add_argument("--no-plots", action="store_true",
                        help="Skip paper-figure generation (Figs 10-15)")
    parser.add_argument("--figures-dir", type=str, default=None,
                        help="Output dir for figures (default: paper_figures/ours/)")

    args = parser.parse_args()

    # Determine which sub-experiments to run
    if args.sub_experiments:
        sub_exps = [s.strip() for s in args.sub_experiments.split(",")]
        invalid = [s for s in sub_exps if s not in ALL_SUB_EXPERIMENTS]
        if invalid:
            parser.error(f"Unknown sub-experiments: {invalid}. Options: {ALL_SUB_EXPERIMENTS}")
    else:
        sub_exps = ALL_SUB_EXPERIMENTS

    # Probe emotions
    probe_emotions = args.probe_emotions.split(",") if args.probe_emotions else CORE_PROBES

    # Load model
    print(f"Loading model for experiment '{args.experiment}'...")
    from utils.paths import get_model_variant

    variant_info = get_model_variant(args.experiment, args.model_variant, mode="application")
    model_variant = variant_info.name
    model_name = variant_info.model

    model, tokenizer = load_model(model_name, load_in_4bit=args.load_in_4bit)
    n_model_layers = model.config.num_hidden_layers
    if hasattr(model.config, "text_config"):
        n_model_layers = model.config.text_config.num_hidden_layers

    # Resolve layers
    if args.layers:
        layers = [int(x) for x in args.layers.split(",")]
    else:
        # Sample evenly from 10% to 95% depth
        start = int(n_model_layers * 0.10)
        end = int(n_model_layers * 0.95)
        step = max(1, (end - start) // args.n_layers_sample)
        layers = list(range(start, end + 1, step))

    print(f"  Model: {model_name} ({n_model_layers} layers)")
    print(f"  Measuring at {len(layers)} layers: {layers[0]}..{layers[-1]}")
    print(f"  Probe emotions: {probe_emotions}")
    print(f"  Sub-experiments: {sub_exps}")

    # Load vectors (with mean_diff fallback)
    print("\nLoading emotion vectors...")
    vectors = {}
    for emotion in probe_emotions:
        vectors[emotion] = {}
        for layer in layers:
            try:
                vec = load_single_emotion_vector(
                    args.experiment, emotion, layer, model_variant,
                    category=args.category, method=args.method,
                )
                vectors[emotion][layer] = vec
            except FileNotFoundError:
                print(
                    f"ERROR: Vector missing for {emotion} at layer {layer} "
                    f"with method={args.method!r}. Run cross_trait_normalize.py first. "
                    f"(Silent fallback to bare 'mean_diff' removed — it could mix "
                    f"denoised and undenoised vectors in the same analysis.)"
                )
                sys.exit(1)
    loaded = {e: list(ld.keys()) for e, ld in vectors.items() if ld}
    print(f"  Loaded vectors: {len(loaded)}/{len(probe_emotions)} emotions")

    if not vectors:
        print("ERROR: No vectors loaded. Run extraction pipeline first.")
        sys.exit(1)

    # Setup results directory (get_results_dir creates it automatically)
    results_dir = get_results_dir(args.experiment, "stage5")
    print(f"  Results: {results_dir}")

    # Run selected sub-experiments
    t0 = time.time()

    dispatch = {
        "dissociation": lambda: run_dissociation(model, tokenizer, vectors, layers, results_dir),
        "colon_predicts": lambda: run_colon_predicts(model, tokenizer, vectors, layers, results_dir, args.max_new_tokens),
        "context_prefix": lambda: run_context_prefix(model, tokenizer, vectors, layers, results_dir),
        "context_numerical": lambda: run_context_numerical(model, tokenizer, vectors, layers, results_dir),
        "negation": lambda: run_negation(model, tokenizer, vectors, layers, results_dir),
        "person_binding": lambda: run_person_binding(model, tokenizer, vectors, layers, results_dir),
    }

    plot_dispatch = {
        "dissociation": plot_fig10_dissociation,
        "colon_predicts": plot_fig11_colon_predicts,
        "context_prefix": plot_fig12_context_prefix,
        "context_numerical": plot_fig13_context_numerical,
        "negation": plot_fig14_negation,
        "person_binding": plot_fig15_person_binding,
    }

    figures_dir = Path(args.figures_dir) if args.figures_dir else FIGURES_DIR
    if not args.no_plots:
        figures_dir.mkdir(parents=True, exist_ok=True)
        plt_.apply_style("ant")

    for exp_name in sub_exps:
        bundle = dispatch[exp_name]()
        if not args.no_plots and bundle is not None:
            path = plot_dispatch[exp_name](bundle, figures_dir)
            print(f"  ✓ saved {path.name}")

    elapsed = time.time() - t0
    print(f"\nStage 5 complete ({elapsed / 60:.1f} min)")
    print(f"Results saved to: {results_dir}")
    if not args.no_plots:
        print(f"Figures saved to: {figures_dir}")

    # Cleanup
    del model
    flush_cuda()


if __name__ == "__main__":
    main()
