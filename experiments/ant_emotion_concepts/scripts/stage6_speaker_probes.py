#!/usr/bin/env python3
"""Speaker probe experiments for Anthropic Emotion Concepts replication.

Covers Stage 6 from ant_emotion_concepts_plan.md:
  6.1  Extract present/other speaker probes (Figs 17-18)
  6.2  Geometry of 4 probe types (Figs 17-18)
  6.3  Character-agnostic test (Fig 19)
  6.4  Cross-speaker interaction (Fig 59)
  6.5  Steering with other-speaker vectors (Table 13)

The 2x2 probe grid:
  Rows = which speaker's EMOTION the probe captures
  Cols = which speaker's TOKENS we extract from

              | Human tokens  | Assistant tokens |
  Human emo   | H-tok H-emo   | A-tok H-emo      |
  Asst emo    | H-tok A-emo   | A-tok A-emo      |

Each probe is the mean activation over transcripts matching the condition,
minus the grand mean across all emotions, then confound-projected (neutral PCs).

Technical challenges:
  - Multiturn dialogues: must identify which tokens belong to which speaker
  - Position DSL supports turn[N]:response[:] for turn-based extraction
  - Need to parse generated dialogues to find turn boundaries
  - Cross-speaker probes require tracking two independent emotion labels

Input:
  - 2-speaker dialogues (generated in Stage 1.3 or here)
  - Extracted emotion vectors (for reference/comparison)
Output: experiments/ant_emotion_concepts/results/stage6/

Usage:
    # Extract probes from pre-generated dialogues
    python experiments/ant_emotion_concepts/scripts/stage6_speaker_probes.py \
        --experiment ant_emotion_concepts --load-in-4bit \
        --dialogues-path experiments/ant_emotion_concepts/results/dialogues/

    # Generate dialogues first, then extract
    python experiments/ant_emotion_concepts/scripts/stage6_speaker_probes.py \
        --experiment ant_emotion_concepts --load-in-4bit \
        --generate-dialogues --n-dialogues 500

    # Run only geometry analysis (CPU, no model needed)
    python experiments/ant_emotion_concepts/scripts/stage6_speaker_probes.py \
        --experiment ant_emotion_concepts \
        --sub-experiments geometry,cross_speaker \
        --probes-path experiments/ant_emotion_concepts/results/stage6/probes/

    # Run steering with other-speaker vectors
    python experiments/ant_emotion_concepts/scripts/stage6_speaker_probes.py \
        --experiment ant_emotion_concepts --load-in-4bit \
        --sub-experiments steering \
        --probes-path experiments/ant_emotion_concepts/results/stage6/probes/
"""

import argparse
import gc
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from core import MultiLayerCapture, projection, cosine_similarity, pairwise_cosine_matrix
from core.hooks import SteeringHook, resolve_hook_path
from utils.model import load_model, format_prompt, tokenize_batch
from utils.model_generation import generate_batch
from utils.paths import get as get_path, get_default_variant
from utils.vectors import load_vector_with_baseline
from utils.distributed import flush_cuda
from dialogue_generation import (
    DIALOGUE_GENERATION_PROMPT,
    generate_dialogues,
    parse_dialogue_turns,
    find_turn_token_boundaries,
)

from core.math import grand_mean_center
from shared import (
    get_results_dir, save_results,
    compute_residual_stream_norm,
)

EXPERIMENT = "ant_emotion_concepts"

ALL_SUB_EXPERIMENTS = [
    "extract_probes", "geometry", "character_agnostic",
    "cross_speaker", "steering",
]

# The 171 emotions from the paper. In practice, load from trait discovery.
# Here we define a representative subset for testing.
DEFAULT_EMOTIONS = [
    "happy", "sad", "afraid", "angry", "calm", "desperate",
    "nervous", "loving", "proud", "surprised", "excited",
    "grateful", "guilty", "jealous", "content", "anxious",
]


# =============================================================================
# Dialogue generation + parsing — factored into dialogue_generation.py
# =============================================================================
# DIALOGUE_GENERATION_PROMPT, generate_dialogues, parse_dialogue_turns,
# find_turn_token_boundaries are imported from dialogue_generation above.


# =============================================================================
# Probe extraction
# =============================================================================

def extract_speaker_probes(
    model, tokenizer,
    dialogues: List[Dict],
    emotions: List[str],
    layers: List[int],
    component: str = "residual",
    batch_size: int = 1,
) -> Dict[str, Dict[str, Dict[int, torch.Tensor]]]:
    """Extract the 2x2 probe grid from dialogues.

    For each dialogue:
      1. Format as model input and run forward pass
      2. Identify Human-turn and Assistant-turn token ranges
      3. Average activations over tokens for each speaker's turns

    Accumulates per (emotion_condition, token_speaker) combination.

    Returns:
        probes["H-tok_H-emo"][emotion][layer] = [hidden_dim] mean activation
        probes["H-tok_A-emo"][emotion][layer] = [hidden_dim] mean activation
        probes["A-tok_A-emo"][emotion][layer] = [hidden_dim] mean activation
        probes["A-tok_H-emo"][emotion][layer] = [hidden_dim] mean activation
    """
    # Accumulators: {probe_type: {emotion: {layer: [list of activation means]}}}
    accum = {
        "H-tok_H-emo": defaultdict(lambda: defaultdict(list)),
        "H-tok_A-emo": defaultdict(lambda: defaultdict(list)),
        "A-tok_A-emo": defaultdict(lambda: defaultdict(list)),
        "A-tok_H-emo": defaultdict(lambda: defaultdict(list)),
    }

    for dialogue in tqdm(dialogues, desc="  Extracting speaker probes"):
        h_emo = dialogue["human_emotion"]
        a_emo = dialogue["assistant_emotion"]

        # Parse turns
        turns = parse_dialogue_turns(dialogue["text"])
        if not turns:
            continue

        # Format as chat template input
        # The dialogue is presented as a user message asking the model to read it
        # Then we capture activations on the dialogue text itself
        dialogue_text = dialogue["text"]
        formatted = format_prompt(
            f"Read this conversation:\n\n{dialogue_text}\n\nWhat emotions are present?",
            tokenizer,
        )

        token_ids = tokenizer.encode(formatted, add_special_tokens=False)
        if len(token_ids) < 10:
            continue

        # Find turn boundaries in token space
        turn_boundaries = find_turn_token_boundaries(token_ids, tokenizer, turns)

        human_turns = [tb for tb in turn_boundaries if tb["role"] == "human" and tb["token_end"] > tb["token_start"]]
        asst_turns = [tb for tb in turn_boundaries if tb["role"] == "assistant" and tb["token_end"] > tb["token_start"]]

        if not human_turns or not asst_turns:
            continue

        # Run forward pass
        input_ids = torch.tensor([token_ids], device=model.device)
        with MultiLayerCapture(model, component=component, layers=layers, keep_on_gpu=False) as capture:
            with torch.no_grad():
                model(input_ids=input_ids, use_cache=False)

        for layer in layers:
            acts = capture.get(layer)
            if acts is None:
                continue
            acts = acts[0].cpu()  # [seq_len, hidden_dim]

            # Average over Human-turn tokens
            h_token_acts = []
            for tb in human_turns:
                if tb["token_start"] < acts.shape[0] and tb["token_end"] <= acts.shape[0]:
                    h_token_acts.append(acts[tb["token_start"]:tb["token_end"]])
            if h_token_acts:
                h_mean = torch.cat(h_token_acts, dim=0).float().mean(dim=0)
            else:
                continue

            # Average over Assistant-turn tokens
            a_token_acts = []
            for tb in asst_turns:
                if tb["token_start"] < acts.shape[0] and tb["token_end"] <= acts.shape[0]:
                    a_token_acts.append(acts[tb["token_start"]:tb["token_end"]])
            if a_token_acts:
                a_mean = torch.cat(a_token_acts, dim=0).float().mean(dim=0)
            else:
                continue

            # H-tok H-emo: Human tokens, keyed by Human's emotion
            accum["H-tok_H-emo"][h_emo][layer].append(h_mean)
            # H-tok A-emo: Human tokens, keyed by Assistant's emotion
            accum["H-tok_A-emo"][a_emo][layer].append(h_mean)
            # A-tok A-emo: Assistant tokens, keyed by Assistant's emotion
            accum["A-tok_A-emo"][a_emo][layer].append(a_mean)
            # A-tok H-emo: Assistant tokens, keyed by Human's emotion
            accum["A-tok_H-emo"][h_emo][layer].append(a_mean)

        del input_ids, capture
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Average accumulators into probes
    probes = {}
    for probe_type in accum:
        probes[probe_type] = {}
        for emotion in accum[probe_type]:
            probes[probe_type][emotion] = {}
            for layer in accum[probe_type][emotion]:
                stacked = torch.stack(accum[probe_type][emotion][layer])
                probes[probe_type][emotion][layer] = stacked.mean(dim=0)

    return probes


def _normalize_probes(probes, layers):
    """Grand mean subtract + unit normalize per (probe_type, layer)."""
    result = {}
    for ptype in probes:
        result[ptype] = {}
        for layer in layers:
            vecs = {e: probes[ptype][e][layer] for e in probes[ptype] if layer in probes[ptype][e]}
            if not vecs:
                continue
            centered, _ = grand_mean_center(vecs)
            for e, v in centered.items():
                if e not in result[ptype]:
                    result[ptype][e] = {}
                n = v.norm()
                result[ptype][e][layer] = v / n if n > 1e-8 else v
    return result


def save_probes(probes: Dict, probes_dir: Path, layers: List[int]):
    """Save extracted probes as .pt files organized by probe type and layer."""
    probes_dir.mkdir(parents=True, exist_ok=True)

    for probe_type in probes:
        type_dir = probes_dir / probe_type
        type_dir.mkdir(parents=True, exist_ok=True)

        for layer in layers:
            layer_vecs = {}
            for emotion in probes[probe_type]:
                if layer in probes[probe_type][emotion]:
                    layer_vecs[emotion] = probes[probe_type][emotion][layer]

            if layer_vecs:
                torch.save(layer_vecs, type_dir / f"layer{layer}.pt")

    # Save metadata
    meta = {
        "probe_types": list(probes.keys()),
        "layers": layers,
        "emotions": {},
    }
    for probe_type in probes:
        meta["emotions"][probe_type] = list(probes[probe_type].keys())

    with open(probes_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"  Saved probes to {probes_dir}")


def load_probes(probes_dir: Path) -> Tuple[Dict, List[int]]:
    """Load previously saved probes."""
    with open(probes_dir / "metadata.json") as f:
        meta = json.load(f)

    layers = meta["layers"]
    probes = {}

    for probe_type in meta["probe_types"]:
        type_dir = probes_dir / probe_type
        probes[probe_type] = {}

        for layer in layers:
            layer_file = type_dir / f"layer{layer}.pt"
            if layer_file.exists():
                layer_vecs = torch.load(layer_file, map_location="cpu", weights_only=True)
                for emotion, vec in layer_vecs.items():
                    if emotion not in probes[probe_type]:
                        probes[probe_type][emotion] = {}
                    probes[probe_type][emotion][layer] = vec

    return probes, layers


# =============================================================================
# Sub-experiment 6.2: Geometry of 4 probe types (Figs 17-18)
# =============================================================================

def run_geometry(probes, layers, results_dir):
    """Compute cosine similarities within and across the 4 probe types.

    Expected:
      - Present-speaker probes (A-tok A-emo, H-tok H-emo) highly similar
      - Other-speaker probes (A-tok H-emo, H-tok A-emo) highly similar
      - Present vs other nearly orthogonal
      - Story probes (from Stage 2) closer to present-speaker
    """
    print("\n=== 6.2: Geometry of 4 Probe Types (Figs 17-18) ===")

    # Use the best layer (typically ~2/3 depth)
    mid_layer = layers[len(layers) // 2] if layers else layers[0]
    probe_types = list(probes.keys())

    results = {
        "layer": mid_layer,
        "probe_types": probe_types,
        "per_emotion_cosine": {},
        "mean_cosine": {},
        "cross_type_matrix": {},
    }

    # For each pair of probe types, compute per-emotion cosine similarity
    for i, pt1 in enumerate(probe_types):
        for j, pt2 in enumerate(probe_types):
            if j < i:
                continue
            pair_key = f"{pt1}_vs_{pt2}"

            # Find shared emotions at the target layer
            emotions_1 = {e for e in probes[pt1] if mid_layer in probes[pt1][e]}
            emotions_2 = {e for e in probes[pt2] if mid_layer in probes[pt2][e]}
            shared = sorted(emotions_1 & emotions_2)

            if not shared:
                continue

            cos_sims = {}
            for emo in shared:
                v1 = probes[pt1][emo][mid_layer]
                v2 = probes[pt2][emo][mid_layer]
                cos = cosine_similarity(v1, v2).item()
                cos_sims[emo] = round(cos, 4)

            mean_cos = sum(cos_sims.values()) / len(cos_sims) if cos_sims else 0.0
            results["per_emotion_cosine"][pair_key] = cos_sims
            results["mean_cosine"][pair_key] = round(mean_cos, 4)

    # Build the 4x4 cross-type mean cosine matrix
    n = len(probe_types)
    matrix = [[0.0] * n for _ in range(n)]
    for i, pt1 in enumerate(probe_types):
        for j, pt2 in enumerate(probe_types):
            pair_key = f"{pt1}_vs_{pt2}" if i <= j else f"{pt2}_vs_{pt1}"
            matrix[i][j] = results["mean_cosine"].get(pair_key, 0.0)
    results["cross_type_matrix"] = {
        "labels": probe_types,
        "values": matrix,
    }

    # Also compute full 171x171 pairwise cosine within each probe type
    results["within_type_similarity"] = {}
    for pt in probe_types:
        emotions = sorted([e for e in probes[pt] if mid_layer in probes[pt][e]])
        if len(emotions) < 2:
            continue
        vecs = torch.stack([probes[pt][e][mid_layer] for e in emotions])
        cos_matrix = pairwise_cosine_matrix(vecs)
        # Store summary stats (mean off-diagonal)
        n_emo = len(emotions)
        off_diag_mask = ~torch.eye(n_emo, dtype=torch.bool)
        mean_off_diag = cos_matrix[off_diag_mask].mean().item()
        results["within_type_similarity"][pt] = {
            "n_emotions": n_emo,
            "mean_off_diagonal_cosine": round(mean_off_diag, 4),
        }

    save_results(results_dir, "geometry", {
        "experiment": "6.2_probe_type_geometry",
        "paper_ref": "Figs 17-18",
        "expected": {
            "present_speaker_similarity": "high (A-tok A-emo ~ H-tok H-emo)",
            "other_speaker_similarity": "high (A-tok H-emo ~ H-tok A-emo)",
            "present_vs_other": "nearly orthogonal",
        },
        "results": results,
    })

    print(f"  Cross-type mean cosine similarities:")
    for pair, val in results["mean_cosine"].items():
        print(f"    {pair}: {val:.4f}")

    return results


# =============================================================================
# Sub-experiment 6.3: Character-agnostic test (Fig 19)
# =============================================================================

def run_character_agnostic(
    model, tokenizer, dialogues, probes_original, emotions, layers,
    results_dir, component="residual",
):
    """Re-run probe extraction with 'Person 1/Person 2' instead of 'Human/Assistant'.

    Shows the self/other representational structure is relational, not bound to
    specific character names.
    """
    print("\n=== 6.3: Character-Agnostic Test (Fig 19) ===")

    # Swap character names in dialogues
    swapped_dialogues = []
    for d in dialogues:
        swapped_text = d["text"].replace("Human:", "Person 1:").replace("Assistant:", "Person 2:")
        swapped_dialogues.append({
            **d,
            "id": d["id"] + "_swapped",
            "text": swapped_text,
        })

    # Extract probes with swapped names
    print("  Extracting probes with Person 1/Person 2 names...")
    probes_swapped = extract_speaker_probes(
        model, tokenizer, swapped_dialogues, emotions, layers, component,
    )
    probes_swapped = _normalize_probes(probes_swapped, layers)

    # Compare original vs swapped probes
    mid_layer = layers[len(layers) // 2]
    comparison = {}

    for probe_type in probes_original:
        if probe_type not in probes_swapped:
            continue

        # Map probe types: H-tok -> P1-tok, A-tok -> P2-tok
        shared_emotions = set()
        for e in probes_original[probe_type]:
            if mid_layer in probes_original[probe_type].get(e, {}):
                if e in probes_swapped.get(probe_type, {}) and mid_layer in probes_swapped[probe_type].get(e, {}):
                    shared_emotions.add(e)

        cos_sims = {}
        for emo in sorted(shared_emotions):
            v_orig = probes_original[probe_type][emo][mid_layer]
            v_swap = probes_swapped[probe_type][emo][mid_layer]
            cos = cosine_similarity(v_orig, v_swap).item()
            cos_sims[emo] = round(cos, 4)

        mean_cos = sum(cos_sims.values()) / len(cos_sims) if cos_sims else 0.0
        comparison[probe_type] = {
            "mean_cosine": round(mean_cos, 4),
            "per_emotion": cos_sims,
        }

    save_results(results_dir, "character_agnostic", {
        "experiment": "6.3_character_agnostic_test",
        "paper_ref": "Fig 19",
        "expected": "High cosine similarity between original and swapped probes (structure is relational, not character-bound)",
        "comparison": comparison,
    })

    print(f"  Original vs Person1/Person2 cosine similarities:")
    for pt, data in comparison.items():
        print(f"    {pt}: mean cos = {data['mean_cosine']:.4f}")

    return probes_swapped, comparison


# =============================================================================
# Sub-experiment 6.4: Cross-speaker interaction (Fig 59)
# =============================================================================

def run_cross_speaker(probes, layers, results_dir):
    """For each other-speaker probe, find the closest present-speaker probes.

    Compute weighted-average valence/arousal and check for arousal regulation.
    Expected: arousal r ~ -0.47 (high-arousal other -> low-arousal self response).
    """
    print("\n=== 6.4: Cross-Speaker Interaction (Fig 59) ===")

    mid_layer = layers[len(layers) // 2]

    # We use A-tok H-emo (other-speaker on assistant tokens) and
    # A-tok A-emo (present-speaker on assistant tokens) as the main comparison.
    other_type = "A-tok_H-emo"
    present_type = "A-tok_A-emo"

    if other_type not in probes or present_type not in probes:
        print("  ERROR: Missing probe types. Run extract_probes first.")
        return None

    other_emotions = sorted([e for e in probes[other_type] if mid_layer in probes[other_type][e]])
    present_emotions = sorted([e for e in probes[present_type] if mid_layer in probes[present_type][e]])

    if not other_emotions or not present_emotions:
        print("  ERROR: No emotions at target layer. Check probe extraction.")
        return None

    # For each other-speaker emotion, compute cosine to all present-speaker emotions
    interactions = {}
    for o_emo in other_emotions:
        o_vec = probes[other_type][o_emo][mid_layer]
        cos_to_present = {}
        for p_emo in present_emotions:
            p_vec = probes[present_type][p_emo][mid_layer]
            cos = cosine_similarity(o_vec, p_vec).item()
            cos_to_present[p_emo] = round(cos, 4)

        # Sort by cosine similarity (closest present-speaker probes)
        sorted_present = sorted(cos_to_present.items(), key=lambda x: -x[1])

        interactions[o_emo] = {
            "closest_present": sorted_present[:5],
            "all_cosines": cos_to_present,
        }

    # NOTE: Computing valence/arousal-weighted analysis requires LLM-judge ratings
    # (from experiment 4.7). For now, just save the raw cosine data.
    # The arousal regulation analysis (r ~ -0.47) can be computed in post-processing
    # once valence/arousal ratings are available.

    save_results(results_dir, "cross_speaker", {
        "experiment": "6.4_cross_speaker_interaction",
        "paper_ref": "Fig 59",
        "expected": {
            "arousal_regulation": "r ~ -0.47 (high-arousal other -> lower-arousal present)",
            "valence_relationship": "r ~ 0.07 (no systematic valence relationship)",
        },
        "note": "Valence/arousal weighted analysis requires LLM-judge ratings. Raw cosine data saved here.",
        "layer": mid_layer,
        "other_speaker_type": other_type,
        "present_speaker_type": present_type,
        "interactions": interactions,
    })

    print(f"  Computed interactions for {len(interactions)} other-speaker emotions")
    for emo in list(interactions.keys())[:3]:
        top3 = interactions[emo]["closest_present"][:3]
        top3_str = ", ".join(f"{e}({c:.3f})" for e, c in top3)
        print(f"    other={emo} -> closest present: {top3_str}")

    return interactions


# =============================================================================
# Sub-experiment 6.5: Steering with other-speaker vectors (Table 13)
# =============================================================================

def run_steering(model, tokenizer, probes, layers, results_dir,
                 steering_strength: float = 0.5, max_new_tokens: int = 100):
    """Steer with other-speaker vs present-speaker vectors on 'Hi, Claude'.

    Other-speaker steering (A-tok H-emo) should make the assistant respond as if
    the human has that emotion. Present-speaker steering (A-tok A-emo) should
    make the assistant express the emotion directly.
    """
    print("\n=== 6.5: Steering with Other-Speaker Vectors (Table 13) ===")

    mid_layer = layers[len(layers) // 2]
    prompt = "Hi, Claude."

    # Compute residual stream norm once for this layer
    residual_norm = compute_residual_stream_norm(model, tokenizer, mid_layer)

    test_emotions = ["afraid", "angry", "happy", "sad", "loving", "desperate"]
    results = []

    for emotion in test_emotions:
        for probe_type in ["A-tok_A-emo", "A-tok_H-emo"]:
            if emotion not in probes.get(probe_type, {}):
                continue
            if mid_layer not in probes[probe_type][emotion]:
                continue

            vector = probes[probe_type][emotion][mid_layer].to(model.device)

            # Coefficient = strength * residual_norm, applied to unit vector
            unit_vector = vector / (vector.norm() + 1e-8)
            coefficient = steering_strength * residual_norm

            # Generate with steering via SteeringHook + generate_batch
            hook_path = resolve_hook_path(model, mid_layer, "residual")

            with SteeringHook(model, unit_vector, hook_path, coefficient=coefficient):
                responses = generate_batch(
                    model, tokenizer, [prompt],
                    max_new_tokens=max_new_tokens,
                    temperature=0.0,
                )

            response = responses[0].strip()

            results.append({
                "emotion": emotion,
                "probe_type": probe_type,
                "probe_description": (
                    "present-speaker (assistant expresses emotion)"
                    if probe_type == "A-tok_A-emo"
                    else "other-speaker (assistant responds to human's emotion)"
                ),
                "steering_strength": steering_strength,
                "layer": mid_layer,
                "prompt": prompt,
                "response": response,
            })

            del vector, unit_vector
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Generate unsteered baseline
    baseline_responses = generate_batch(
        model, tokenizer, [prompt],
        max_new_tokens=max_new_tokens,
        temperature=0.0,
    )
    baseline_response = baseline_responses[0].strip()

    save_results(results_dir, "steering", {
        "experiment": "6.5_steering_with_other_speaker",
        "paper_ref": "Table 13",
        "expected": {
            "other_speaker_steering": "Assistant responds as if human has that emotion (e.g., other=afraid -> reassurance)",
            "present_speaker_steering": "Assistant directly expresses the emotion",
        },
        "baseline": {
            "prompt": prompt,
            "response": baseline_response,
        },
        "steering_strength": steering_strength,
        "results": results,
    })

    print(f"  Generated {len(results)} steered responses")
    print(f"  Baseline: {baseline_response[:80]}...")
    for r in results[:4]:
        print(f"    [{r['probe_type']}] {r['emotion']}: {r['response'][:80]}...")

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
    parser.add_argument("--component", default="residual")

    # Layer selection
    parser.add_argument("--layers", type=str, default=None,
                        help="Comma-separated layers (default: 5 evenly spaced from 25%% to 75%% depth)")
    parser.add_argument("--n-layers-sample", type=int, default=5,
                        help="Number of layers when --layers not specified")

    # Sub-experiment selection
    parser.add_argument("--sub-experiments", type=str, default=None,
                        help=f"Comma-separated. Options: {','.join(ALL_SUB_EXPERIMENTS)}")

    # Dialogue generation
    parser.add_argument("--generate-dialogues", action="store_true",
                        help="Generate dialogues before extraction (else load from file)")
    parser.add_argument("--n-dialogues", type=int, default=500,
                        help="Number of dialogues to generate")
    parser.add_argument("--dialogues-path", type=str, default=None,
                        help="Path to pre-generated dialogues JSON")

    # Probe I/O
    parser.add_argument("--probes-path", type=str, default=None,
                        help="Path to saved probes (for geometry/steering without re-extraction)")

    # Emotions
    parser.add_argument("--emotions", type=str, default=None,
                        help="Comma-separated emotions (default: 16-emotion subset)")
    parser.add_argument("--category", default="ant_emotion_concepts")

    # Steering
    parser.add_argument("--steering-strength", type=float, default=0.5)
    parser.add_argument("--max-new-tokens", type=int, default=100)

    # Generation
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--replication-level",
        choices=["lightweight", "full"],
        default="lightweight",
        help=(
            "Dialogue-generation prompt mode. 'lightweight' (default) uses the "
            "current paraphrased prompt with one dialogue per call. 'full' uses "
            "the paper-verbatim Appendix A.4 template (Person:/AI: labels "
            "rewritten post-hoc to Human:/Assistant:), one batched call per "
            "(person_emotion, ai_emotion, topic) cell. See "
            "datasets/traits/ant_emotion_concepts/prompts/two_speaker_dialogue.txt"
        ),
    )
    parser.add_argument(
        "--dialogue-topics",
        type=str,
        default=None,
        help=(
            "Comma-separated topic list for --replication-level=full (paper "
            "rotates topics across dialogue cells). Ignored in lightweight mode. "
            "Default in full mode: a single generic placeholder."
        ),
    )
    parser.add_argument(
        "--dialogue-max-new-tokens",
        type=int,
        default=None,
        help=(
            "max_new_tokens budget for generate_dialogues. Default: 384 in "
            "lightweight mode (one dialogue per call). For "
            "--replication-level=full, budget is for the WHOLE batched response "
            "of N dialogues per cell; recommended starting point is ~512-1024 "
            "per dialogue (e.g. 4096-8192 for paper's 6-10 exchange dialogues). "
            "If None, defaults to 384 lightweight / 2048 full."
        ),
    )

    args = parser.parse_args()

    # Sub-experiments
    if args.sub_experiments:
        sub_exps = [s.strip() for s in args.sub_experiments.split(",")]
        invalid = [s for s in sub_exps if s not in ALL_SUB_EXPERIMENTS]
        if invalid:
            parser.error(f"Unknown sub-experiments: {invalid}. Options: {ALL_SUB_EXPERIMENTS}")
    else:
        sub_exps = ALL_SUB_EXPERIMENTS

    # Emotions: support "all" as a shortcut for the full 171 category
    if args.emotions == "all":
        from utils.paths import discover_traits
        traits = discover_traits(category=args.category)
        emotions = sorted(set(t.split('/')[-1] for t in traits))
        print(f"  --emotions all → loaded {len(emotions)} emotions from category '{args.category}'")
    elif args.emotions:
        emotions = args.emotions.split(",")
    else:
        emotions = DEFAULT_EMOTIONS

    # Check which sub-experiments need GPU
    needs_gpu = any(s in sub_exps for s in ["extract_probes", "character_agnostic", "steering"])

    # Load model if needed
    model, tokenizer = None, None
    n_model_layers = 80  # Default for Llama 70B

    if needs_gpu or args.generate_dialogues:
        print(f"Loading model for experiment '{args.experiment}'...")
        from utils.paths import get_model_variant

        variant_info = get_model_variant(args.experiment, args.model_variant, mode="application")
        model_variant = variant_info.name
        model_name = variant_info.model

        model, tokenizer = load_model(model_name, load_in_4bit=args.load_in_4bit)
        config = model.config
        if hasattr(config, "text_config"):
            config = config.text_config
        n_model_layers = config.num_hidden_layers
        print(f"  Model: {model_name} ({n_model_layers} layers)")

    # Resolve layers
    if args.layers:
        layers = [int(x) for x in args.layers.split(",")]
    else:
        start = int(n_model_layers * 0.25)
        end = int(n_model_layers * 0.75)
        step = max(1, (end - start) // args.n_layers_sample)
        layers = list(range(start, end + 1, step))

    print(f"  Layers: {layers}")
    print(f"  Emotions: {len(emotions)} ({emotions[:5]}...)")
    print(f"  Sub-experiments: {sub_exps}")

    # Setup results directory
    results_dir = get_results_dir(args.experiment, "stage6")
    probes_dir = results_dir / "probes"

    t0 = time.time()

    # --- Load or generate dialogues ---
    dialogues = None
    dialogues_path = results_dir / "dialogues.json"

    if args.dialogues_path:
        dialogues_path = Path(args.dialogues_path)
        if dialogues_path.is_dir():
            dialogues_path = dialogues_path / "dialogues.json"

    if args.generate_dialogues:
        print(f"\nGenerating dialogues (replication_level={args.replication_level})...")
        # Default max_new_tokens depends on mode: lightweight = 1 dialogue per call
        # so 384 suffices; full = N dialogues per batched call so budget must scale.
        if args.dialogue_max_new_tokens is not None:
            dialogue_mnt = args.dialogue_max_new_tokens
        elif args.replication_level == "full":
            dialogue_mnt = 2048
        else:
            dialogue_mnt = 384
        topics = (
            [t.strip() for t in args.dialogue_topics.split(",") if t.strip()]
            if args.dialogue_topics
            else None
        )
        dialogues = generate_dialogues(
            model, tokenizer, emotions,
            n_dialogues=args.n_dialogues,
            max_new_tokens=dialogue_mnt,
            temperature=args.temperature,
            seed=args.seed,
            replication_level=args.replication_level,
            topics=topics,
        )
        dialogues_path.parent.mkdir(parents=True, exist_ok=True)
        save_results(dialogues_path.parent, dialogues_path.stem, {"dialogues": dialogues, "n_dialogues": len(dialogues)})
        print(f"  Saved {len(dialogues)} dialogues to {dialogues_path}")

    elif dialogues_path.exists():
        print(f"\nLoading dialogues from {dialogues_path}...")
        with open(dialogues_path) as f:
            dialogues = json.load(f)
        print(f"  Loaded {len(dialogues)} dialogues")

    # --- Load or extract probes ---
    probes = None

    if args.probes_path:
        print(f"\nLoading probes from {args.probes_path}...")
        probes, loaded_layers = load_probes(Path(args.probes_path))
        # Use loaded layers if not explicitly specified
        if not args.layers:
            layers = loaded_layers
        print(f"  Loaded probes: {list(probes.keys())}")

    if "extract_probes" in sub_exps:
        if dialogues is None:
            print("ERROR: No dialogues available. Use --generate-dialogues or --dialogues-path")
            sys.exit(1)
        if model is None:
            print("ERROR: Model needed for probe extraction. Remove --sub-experiments or add model args.")
            sys.exit(1)

        print("\n=== 6.1: Extract Present/Other Speaker Probes (Figs 17-18) ===")
        probes = extract_speaker_probes(
            model, tokenizer, dialogues, emotions, layers, args.component,
        )
        print(f"  Raw probes extracted. Applying grand mean subtraction...")
        probes = _normalize_probes(probes, layers)
        save_probes(probes, probes_dir, layers)

    if probes is None:
        print("ERROR: No probes available. Run extract_probes first or provide --probes-path")
        sys.exit(1)

    # --- Run analysis sub-experiments ---

    if "geometry" in sub_exps:
        run_geometry(probes, layers, results_dir)

    if "character_agnostic" in sub_exps:
        if dialogues is None or model is None:
            print("  SKIP character_agnostic: needs dialogues + model")
        else:
            run_character_agnostic(
                model, tokenizer, dialogues, probes, emotions, layers,
                results_dir, args.component,
            )

    if "cross_speaker" in sub_exps:
        run_cross_speaker(probes, layers, results_dir)

    if "steering" in sub_exps:
        if model is None:
            print("  SKIP steering: needs model")
        else:
            run_steering(
                model, tokenizer, probes, layers, results_dir,
                steering_strength=args.steering_strength,
                max_new_tokens=args.max_new_tokens,
            )

    elapsed = time.time() - t0
    print(f"\nStage 6 complete ({elapsed / 60:.1f} min)")
    print(f"Results saved to: {results_dir}")

    # Cleanup
    if model is not None:
        del model
        flush_cuda()


if __name__ == "__main__":
    main()
