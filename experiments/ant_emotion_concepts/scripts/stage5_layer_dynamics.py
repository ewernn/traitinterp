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

from core import projection
from utils.model import load_model, format_prompt
from utils.paths import get as get_path
from utils.distributed import flush_cuda

from shared import (
    EXPERIMENT,
    get_results_dir, save_results,
    load_single_emotion_vector,
    capture_all_tokens,
    find_assistant_colon_position,
)

DATASETS_DIR = get_path('datasets.inference') / "ant_emotion_concepts"

# Probe emotions commonly referenced across sub-experiments
CORE_PROBES = [
    "happy", "sad", "afraid", "angry", "calm", "desperate",
    "nervous", "loving", "proud", "surprised", "terrified",
]

ALL_SUB_EXPERIMENTS = [
    "dissociation", "colon_predicts", "context_prefix",
    "context_numerical", "negation", "person_binding",
]


# =============================================================================
# Token scanning utilities
# =============================================================================

def find_token_positions(token_ids: List[int], tokenizer, targets: List[str]) -> Dict[str, List[int]]:
    """Find positions of target strings in a token sequence.

    For each target string, finds all token positions whose decoded text
    contains the target. Returns {target: [positions]}.
    """
    decoded = [tokenizer.decode([tid]) for tid in token_ids]
    results = {}
    for target in targets:
        positions = []
        for i, tok_text in enumerate(decoded):
            if target in tok_text:
                positions.append(i)
        results[target] = positions
    return results


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

def load_emotion_vectors(
    experiment: str,
    model_variant: str,
    probe_emotions: List[str],
    layers: List[int],
    method: str = "denoised",
    category: str = "ant_emotion_concepts",
) -> Dict[str, Dict[int, torch.Tensor]]:
    """Load emotion vectors for specified emotions and layers.

    Returns: {emotion: {layer: vector_tensor}}

    Delegates to shared.load_single_emotion_vector with fallback to mean_diff.
    """
    vectors = {}
    for emotion in probe_emotions:
        vectors[emotion] = {}
        for layer in layers:
            try:
                vec = load_single_emotion_vector(
                    experiment, emotion, layer, model_variant,
                    category=category, method=method,
                )
                vectors[emotion][layer] = vec
            except (FileNotFoundError, Exception):
                # Try mean_diff as fallback
                try:
                    vec = load_single_emotion_vector(
                        experiment, emotion, layer, model_variant,
                        category=category, method="mean_diff",
                    )
                    vectors[emotion][layer] = vec
                except Exception:
                    pass  # Skip this emotion/layer combination

    loaded = {e: list(layers_dict.keys()) for e, layers_dict in vectors.items() if layers_dict}
    print(f"  Loaded vectors: {len(loaded)}/{len(probe_emotions)} emotions")
    return vectors


def project_at_positions(
    activations: Dict[int, torch.Tensor],
    vectors: Dict[str, Dict[int, torch.Tensor]],
    positions: List[int],
    layers: List[int],
) -> Dict[str, Dict[int, List[float]]]:
    """Project activations onto emotion vectors at specific token positions.

    Args:
        activations: {layer: [seq_len, hidden_dim]} from MultiLayerCapture
        vectors: {emotion: {layer: [hidden_dim]}}
        positions: token indices to measure
        layers: layers to include

    Returns: {emotion: {layer: [proj_at_pos0, proj_at_pos1, ...]}}
    """
    results = {}
    for emotion, layer_vecs in vectors.items():
        results[emotion] = {}
        for layer in layers:
            if layer not in layer_vecs or layer not in activations:
                continue
            vec = layer_vecs[layer]
            acts = activations[layer]  # [seq_len, hidden_dim]
            if acts.dim() == 3:
                acts = acts[0]  # Remove batch dim if present

            projs = []
            for pos in positions:
                if pos < acts.shape[0]:
                    p = projection(acts[pos:pos+1], vec, normalize_vector=True)
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
        activations_list = capture_all_tokens(model, tokenizer, [formatted], layers)
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

    # Save
    save_results(results_dir, "dissociation", {
        "experiment": "5.1_user_vs_assistant_dissociation",
        "paper_ref": "Fig 10, Table 3",
        "expected": "cross-position correlation r ~ 0.11",
        "n_scenarios": len(scenarios),
        "layers": layers,
        "results": results,
    })

    return results


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
        activations_list = capture_all_tokens(model, tokenizer, [full_text], layers)
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

    save_results(results_dir, "colon_predicts", {
        "experiment": "5.2_colon_predicts_response",
        "paper_ref": "Fig 11, Table 4",
        "expected": "assistant_colon→response_mean correlation r ~ 0.87",
        "n_scenarios": len(scenarios),
        "max_new_tokens": max_new_tokens,
        "layers": layers,
        "results": results,
    })

    return results


# =============================================================================
# Sub-experiment 5.3: Context propagation — emotional prefix (Fig 12)
# =============================================================================

def run_context_prefix(model, tokenizer, vectors, layers, results_dir):
    """Layer x token heatmap for 'hard' vs 'good' prefix template.

    The suffix after the diverging word is identical. Early layers show
    probe difference only at the diverging word; late layers sustain difference
    across the shared suffix with peak at Assistant colon.
    """
    print("\n=== 5.3: Context Propagation — Emotional Prefix (Fig 12) ===")

    with open(DATASETS_DIR / "context_propagation_templates.json") as f:
        data = json.load(f)

    template_data = data["templates"][0]  # marriage_hard_good
    template = template_data["template"]
    values = template_data["values"]  # ["hard", "good"]

    results = {}

    for value in values:
        prompt_text = template.replace("{X}", value)
        formatted = format_prompt(prompt_text, tokenizer)

        token_ids = tokenizer.encode(formatted, add_special_tokens=False)
        decoded_tokens = [tokenizer.decode([tid]) for tid in token_ids]

        # Run forward pass capturing ALL layers
        activations_list = capture_all_tokens(model, tokenizer, [formatted], layers)
        activations = activations_list[0]

        # Project every token at every layer for all probe emotions
        all_positions = list(range(len(token_ids)))
        projs = project_at_positions(activations, vectors, all_positions, layers)

        results[value] = {
            "prompt": prompt_text,
            "tokens": decoded_tokens,
            "n_tokens": len(token_ids),
            "projections": {},  # {emotion: {layer: [per_token_projections]}}
        }

        for emotion in projs:
            results[value]["projections"][emotion] = {
                str(layer): projs[emotion].get(layer, []) for layer in layers
            }

    # Compute condition difference (value1 - value2) for the heatmap
    difference = {"tokens": results[values[0]]["tokens"]}
    n_tokens = min(results[values[0]]["n_tokens"], results[values[1]]["n_tokens"])
    difference["projections"] = {}

    for emotion in results[values[0]]["projections"]:
        if emotion not in results[values[1]]["projections"]:
            continue
        difference["projections"][emotion] = {}
        for layer in layers:
            sl = str(layer)
            v0 = results[values[0]]["projections"][emotion].get(sl, [])
            v1 = results[values[1]]["projections"][emotion].get(sl, [])
            diff = [
                v0[i] - v1[i] if i < len(v0) and i < len(v1) else 0.0
                for i in range(n_tokens)
            ]
            difference["projections"][emotion][sl] = diff

    save_results(results_dir, "context_prefix", {
        "experiment": "5.3_context_propagation_prefix",
        "paper_ref": "Fig 12",
        "template_id": template_data["id"],
        "conditions": values,
        "expected": "Early layers: difference at diverging word only. Late layers: sustained difference across shared suffix.",
        "layers": layers,
        "condition_results": results,
        "condition_difference": difference,
    })

    return results


# =============================================================================
# Sub-experiment 5.4: Context propagation — numerical (Fig 13)
# =============================================================================

def run_context_numerical(model, tokenizer, vectors, layers, results_dir):
    """Layer x token heatmap for Tylenol 1000 vs 8000mg.

    Early layers show no difference at '1' vs '8'. Late layers show elevated
    'terrified' activation in 8000mg condition with peak at Assistant colon.
    """
    print("\n=== 5.4: Context Propagation — Numerical (Fig 13) ===")

    with open(DATASETS_DIR / "context_propagation_templates.json") as f:
        data = json.load(f)

    template_data = data["templates"][1]  # tylenol_dose_layer
    template = template_data["template"]
    values = template_data["values"]  # ["1", "8"]

    results = {}

    for value in values:
        prompt_text = template.replace("{X}", str(value))
        formatted = format_prompt(prompt_text, tokenizer)

        token_ids = tokenizer.encode(formatted, add_special_tokens=False)
        decoded_tokens = [tokenizer.decode([tid]) for tid in token_ids]

        activations_list = capture_all_tokens(model, tokenizer, [formatted], layers)
        activations = activations_list[0]

        all_positions = list(range(len(token_ids)))
        projs = project_at_positions(activations, vectors, all_positions, layers)

        results[str(value)] = {
            "prompt": prompt_text,
            "tokens": decoded_tokens,
            "n_tokens": len(token_ids),
            "projections": {},
        }

        for emotion in projs:
            results[str(value)]["projections"][emotion] = {
                str(layer): projs[emotion].get(layer, []) for layer in layers
            }

    # Condition difference
    v0_key, v1_key = str(values[0]), str(values[1])
    n_tokens = min(results[v0_key]["n_tokens"], results[v1_key]["n_tokens"])
    difference = {"tokens": results[v1_key]["tokens"]}
    difference["projections"] = {}

    for emotion in results[v0_key]["projections"]:
        if emotion not in results[v1_key]["projections"]:
            continue
        difference["projections"][emotion] = {}
        for layer in layers:
            sl = str(layer)
            v0 = results[v0_key]["projections"][emotion].get(sl, [])
            v1 = results[v1_key]["projections"][emotion].get(sl, [])
            # 8000mg - 1000mg (high danger minus safe)
            diff = [
                v1[i] - v0[i] if i < len(v0) and i < len(v1) else 0.0
                for i in range(n_tokens)
            ]
            difference["projections"][emotion][sl] = diff

    save_results(results_dir, "context_numerical", {
        "experiment": "5.4_context_propagation_numerical",
        "paper_ref": "Fig 13",
        "template_id": template_data["id"],
        "conditions": [str(v) for v in values],
        "expected": "Late layers: elevated 'terrified' for 8000mg; early layers: no difference.",
        "layers": layers,
        "condition_results": results,
        "condition_difference": difference,
    })

    return results


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

            activations_list = capture_all_tokens(model, tokenizer, [formatted], layers)
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

    save_results(results_dir, "negation", {
        "experiment": "5.5_negation_across_layers",
        "paper_ref": "Fig 14",
        "expected": "Early layers: similar activation for both conditions at emotion word. Late layers: negated drops to near zero.",
        "test_emotions": test_emotions,
        "layers": layers,
        "results": results,
    })

    return results


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

        activations_list = capture_all_tokens(model, tokenizer, [formatted], layers)
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

    save_results(results_dir, "person_binding", {
        "experiment": "5.6_person_specific_binding",
        "paper_ref": "Fig 15",
        "expected": "At re-reference tokens: person's emotion probe activates in late layers only. At emotion words: corresponding probe activates in early layers.",
        "n_scenarios": len(scenarios),
        "layers": layers,
        "results": results,
    })

    return results


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

    # Load vectors
    print("\nLoading emotion vectors...")
    vectors = load_emotion_vectors(
        args.experiment, model_variant, probe_emotions, layers,
        method=args.method, category=args.category,
    )

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

    for exp_name in sub_exps:
        dispatch[exp_name]()

    elapsed = time.time() - t0
    print(f"\nStage 5 complete ({elapsed / 60:.1f} min)")
    print(f"Results saved to: {results_dir}")

    # Cleanup
    del model
    flush_cuda()


if __name__ == "__main__":
    main()
