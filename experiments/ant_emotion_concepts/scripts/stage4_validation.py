"""Stage 4: Validation experiments for 171 emotion vectors.

Replicates Sofroniew et al. 2026 Part 1, Sections 1.2-1.4 + Appendix:
  - Table 1:     Logit lens (top 5 up/down tokens per emotion) — CPU only
  - Fig 2:       Implicit emotion prompts (12 scenarios) — GPU
  - Fig 3:       Numerical intensity modulation (6 templates) — GPU
  - Figs 52-53:  Basic steering (12 vectors on 3 prompts, s=0.5) — GPU
  - Fig 4:       Activity preference Elo (4032 pairs) — GPU
  - Fig 4 row 1: Probe-preference correlation — CPU only
  - Fig 56:      Valence/arousal mediation via LLM judge — API only (placeholder)

Input: Denoised vectors + model
Output: experiments/{experiment}/results/stage4_validation/

Usage:
    # Full validation suite:
    python experiments/ant_emotion_concepts/scripts/stage4_validation.py \
        --experiment ant_emotion_concepts --layer 53 --load-in-4bit

    # CPU-only analyses (no model needed):
    python experiments/ant_emotion_concepts/scripts/stage4_validation.py \
        --experiment ant_emotion_concepts --layer 53 --only logit_lens

    # Specific GPU experiments:
    python experiments/ant_emotion_concepts/scripts/stage4_validation.py \
        --experiment ant_emotion_concepts --layer 53 --load-in-4bit \
        --only implicit,numerical,steering

    # Preference Elo (most GPU-intensive validation):
    python experiments/ant_emotion_concepts/scripts/stage4_validation.py \
        --experiment ant_emotion_concepts --layer 53 --load-in-4bit \
        --only preference
"""

import argparse
import json
import sys
import time
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from core.math import batch_cosine_similarity, pearson_correlation
from utils.paths import get_default_variant, get as get_path
from utils.capture_activations import capture_at_position
from utils import plotting as plt_
from shared import (
    load_emotion_vectors_as_dict,
    save_results,
    get_results_dir,
    compare_to_baseline,
)

FIGURES_DIR = Path(__file__).resolve().parent.parent / "paper_figures" / "ours"

# ---------- Paper-specific plotting constants (Figs 2, 3) ----------
FIG2_EMOTION_ORDER = [
    "happy", "inspired", "loving", "proud", "calm", "desperate",
    "angry", "guilty", "sad", "afraid", "nervous", "surprised",
]
FIG2_SCENARIO_DESCRIPTIONS = {
    "happy": "Daughter's first steps",
    "inspired": "Rebuilding after loss",
    "loving": "30-year anniversary",
    "proud": "Son graduates top",
    "calm": "Tea and rain",
    "desperate": "Eviction notice",
    "angry": "Coworker stole credit",
    "guilty": "Forgot mom's birthday",
    "sad": "Dog passed away",
    "afraid": "Break-in, phone dying",
    "nervous": "Job interview nerves",
    "surprised": "Friend's fake life",
}

FIG3_PROBE_COLORS = {
    "afraid": "#d62728", "calm": "#1f77b4",
    "happy": "#2ca02c", "sad": "#ff7f0e",
}
FIG3_PANEL_ORDER = [
    "tylenol_dose", "hours_fasting", "sister_age_at_death",
    "dog_missing_days", "startup_runway_months", "students_passed_exam",
]
FIG3_X_LABELS = {
    "tylenol_dose": "Tylenol dose (mg)",
    "hours_fasting": "Hours",
    "sister_age_at_death": "Age",
    "dog_missing_days": "Days missing",
    "startup_runway_months": "Months of runway",
    "students_passed_exam": "Students passed",
}
FIG3_PANEL_TITLES = {
    "tylenol_dose": "I just took {X} mg of tylenol for my back\npain. Do you think I should take more?",
    "hours_fasting": "It's been {X} hours since I've had any\nfood or drink.",
    "sister_age_at_death": "My sister lived until the age of {X}.",
    "dog_missing_days": "My dog has been missing for {X} days now.",
    "startup_runway_months": "Our startup has {X} months of runway remaining.",
    "students_passed_exam": "I found out that {X} of my 20 students\npassed the final exam.",
}


# ============================================================================
# Anthropic baseline numbers (Sonnet 4.5) — stage4-specific additions to shared
# ============================================================================

# `shared.ANTHROPIC_BASELINES` has `pc1_variance`, `pc2_variance`, `pc1_vs_valence`,
# `pc2_vs_arousal`, `n_clusters`, `cluster_sizes`, `valence_mediates_pref`. Stage 4
# also needs probe-preference correlation anchors, so we extend the dict locally.
from shared import ANTHROPIC_BASELINES as _SHARED_BASELINES
ANTHROPIC_BASELINES = {
    **_SHARED_BASELINES,
    'probe_pref_blissful': 0.71,
    'probe_pref_hostile': -0.74,
    'causal_pref_correlation': 0.85,
}


# ============================================================================
# Datasets — loaded from datasets/inference/ant_emotion_concepts/*.json
# ============================================================================

_DATASETS_DIR = get_path('datasets.inference') / 'ant_emotion_concepts'


def _load_implicit_emotion_prompts() -> Dict[str, str]:
    """Load Table 2 implicit emotion scenarios. Returns {emotion: prompt_text}."""
    with open(_DATASETS_DIR / 'implicit_emotion_scenarios.json') as f:
        data = json.load(f)
    return {entry['target_emotion']: entry['prompt'] for entry in data['prompts']}


def _load_numerical_templates() -> List[dict]:
    """Load Section 1.2.4 numerical intensity templates.

    Normalizes JSON field names to match consumer expectations:
      expected_probes -> expected, 'increases'/'decreases' -> 'increase'/'decrease'.
    """
    direction_map = {'increases': 'increase', 'decreases': 'decrease'}
    with open(_DATASETS_DIR / 'numerical_intensity_templates.json') as f:
        data = json.load(f)
    templates = []
    for entry in data['templates']:
        templates.append({
            'template': entry['template'],
            'variable': entry['id'],
            'values': entry['values'],
            'expected': {
                probe: direction_map.get(direction, direction)
                for probe, direction in entry['expected_probes'].items()
            },
        })
    return templates


def _load_steering_prompts() -> List[str]:
    """Load Section 1.3 basic steering validation prompts."""
    with open(_DATASETS_DIR / 'basic_steering_prompts.json') as f:
        data = json.load(f)
    return [entry['prompt'] for entry in data['prompts']]


IMPLICIT_EMOTION_PROMPTS = _load_implicit_emotion_prompts()
STEERING_EMOTIONS = list(IMPLICIT_EMOTION_PROMPTS.keys())
NUMERICAL_TEMPLATES = _load_numerical_templates()
STEERING_PROMPTS = _load_steering_prompts()


# Section 1.4.1: 64 activities for preference Elo
# Full list from Table 9 / Appendix A.5, 8 categories x 8 activities = 64.
# Stored as a fixture; loaded at module import.
def _load_preference_activities() -> dict:
    """Load 64 preference activities, returning {category: [activity_strings]}.

    Handles both old format (dict of lists) and new verbatim format (flat list of objects).
    """
    path = get_path('datasets.inference') / 'ant_emotion_concepts' / 'stage4_preference_activities.json'
    with open(path) as f:
        raw = json.load(f)['activities']
    # New format: list of {"category": ..., "activity": ..., ...}
    if isinstance(raw, list) and raw and isinstance(raw[0], dict):
        result = {}
        for item in raw:
            cat = item['category']
            result.setdefault(cat, []).append(item['activity'])
        return result
    # Old format: {category: [activity_strings]}
    return raw


PREFERENCE_ACTIVITIES = _load_preference_activities()


# ============================================================================
# Experiment 4.1: Logit lens (Table 1) — CPU only
# ============================================================================

def run_logit_lens(vectors, model, tokenizer, out_dir, top_k=5):
    """Project each emotion vector through unembedding to find top tokens.

    Calls: utils.logit_lens.vector_to_vocab()
           utils.logit_lens.build_common_token_mask()
    No custom code beyond iteration and formatting.
    """
    from utils.logit_lens import vector_to_vocab, build_common_token_mask

    print(f"\n  [Table 1] Logit lens — {len(vectors)} emotion vectors, top {top_k}...")

    mask = build_common_token_mask(tokenizer)

    results = {}
    for name, vec in sorted(vectors.items()):
        result = vector_to_vocab(vec, model, tokenizer, top_k=top_k, common_mask=mask)
        results[name] = {
            'toward': result['toward'][:top_k],
            'away': result['away'][:top_k],
        }

    # Print a sample (matching Table 1 format)
    sample_emotions = ['happy', 'sad', 'angry', 'afraid', 'desperate', 'calm',
                       'loving', 'proud', 'guilty', 'nervous', 'surprised', 'disgusted']
    print(f"\n    {'Emotion':<12} | {'Top 5 up':^40} | {'Top 5 down':^40}")
    print(f"    {'-'*12}-+-{'-'*40}-+-{'-'*40}")
    for emo in sample_emotions:
        if emo not in results:
            continue
        up = ', '.join([t['token'].strip("'") for t in results[emo]['toward'][:5]])
        down = ', '.join([t['token'].strip("'") for t in results[emo]['away'][:5]])
        print(f"    {emo:<12} | {up:<40} | {down:<40}")

    save_results(out_dir, 'logit_lens', results)
    print(f"    ({len(results)} emotions)")

    return results


# ============================================================================
# Experiment 4.2: Implicit emotion prompts (Fig 2) — GPU
# ============================================================================

def run_implicit_emotion(vectors, model, tokenizer, layer, out_dir):
    """Measure probe activations on 12 implicit-emotion prompts at assistant colon.

    Custom code needed:
      - capture_at_position(... position='prompt[-1]') — mainline prefill capture at
        the last prompt token (the "assistant colon" in chat-templated prompts).

    Calls:
      - core.math.batch_cosine_similarity() for computing similarity
    """
    print(f"\n  [Fig 2] Implicit emotion prompts — {len(IMPLICIT_EMOTION_PROMPTS)} scenarios...")

    prompts = list(IMPLICIT_EMOTION_PROMPTS.values())
    prompt_emotions = list(IMPLICIT_EMOTION_PROMPTS.keys())

    # Get activations at assistant colon for each prompt
    print("    Running forward passes...")
    t0 = time.time()
    activations = capture_at_position(
        model, tokenizer, prompts, layers=layer, position='prompt[-1]', pool='last',
    )
    print(f"    Captured {activations.shape[0]} activations in {time.time()-t0:.1f}s")

    # Compute cosine similarity between each probe and each prompt's activation
    probe_names = sorted(vectors.keys())
    probe_stack = torch.stack([vectors[n] for n in probe_names])  # [n_probes, hidden_dim]

    # similarity_matrix[i, j] = cosine_similarity(activations[i], probe[j])
    # activations: [n_prompts, hidden_dim], probes: [n_probes, hidden_dim]
    act_norm = activations / activations.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    probe_norm = probe_stack.float() / probe_stack.float().norm(dim=-1, keepdim=True).clamp(min=1e-8)
    similarity_matrix = act_norm @ probe_norm.T  # [n_prompts, n_probes]

    # Extract the diagonal (each prompt matched to its intended emotion probe)
    diagonal_sims = []
    for i, emo in enumerate(prompt_emotions):
        if emo in probe_names:
            j = probe_names.index(emo)
            diagonal_sims.append(similarity_matrix[i, j].item())
        else:
            diagonal_sims.append(None)

    mean_diag = sum(s for s in diagonal_sims if s is not None) / max(1, sum(1 for s in diagonal_sims if s is not None))
    print(f"    Mean diagonal similarity: {mean_diag:.3f}")
    print(f"    Per-prompt diagonal:")
    for emo, sim in zip(prompt_emotions, diagonal_sims):
        marker = ' ***' if sim is not None and sim > 0.1 else ''
        print(f"      {emo:<12}: {sim:.3f}{marker}" if sim is not None else f"      {emo:<12}: MISSING")

    # Focused heatmap: 12x12 (prompt emotions x probe emotions)
    focused_indices = [probe_names.index(e) for e in prompt_emotions if e in probe_names]
    focused_probes = [e for e in prompt_emotions if e in probe_names]
    focused_matrix = similarity_matrix[:, focused_indices]

    result = {
        'prompt_emotions': prompt_emotions,
        'probe_names': probe_names,
        'focused_probes': focused_probes,
        'similarity_matrix_full': similarity_matrix.tolist(),
        'similarity_matrix_focused': focused_matrix.tolist(),
        'diagonal_similarities': diagonal_sims,
        'mean_diagonal': mean_diag,
        'token_positions': [-1] * len(prompt_emotions),
    }

    save_results(out_dir, 'implicit_emotion', result)

    return result


# ============================================================================
# Experiment 4.3: Numerical intensity modulation (Fig 3) — GPU
# ============================================================================

def run_numerical_intensity(vectors, model, tokenizer, layer, out_dir):
    """Measure probe activations across numerical intensity variations.

    Custom code: Same capture_at_position(... position='prompt[-1]') as implicit emotion.
    The analysis is straightforward: for each template family, sweep the
    numerical variable and plot probe activation vs value.
    """
    print(f"\n  [Fig 3] Numerical intensity — {len(NUMERICAL_TEMPLATES)} template families...")

    results = {}

    for tmpl_info in NUMERICAL_TEMPLATES:
        var_name = tmpl_info['variable']
        print(f"\n    Template: {tmpl_info['template'][:60]}...")

        # Generate prompts for each value
        prompts = [tmpl_info['template'].format(X=v) for v in tmpl_info['values']]

        # Get activations at assistant colon (last prompt token)
        activations = capture_at_position(
            model, tokenizer, prompts, layers=layer, position='prompt[-1]', pool='last',
        )

        # Project onto relevant probes
        template_result = {
            'template': tmpl_info['template'],
            'variable': var_name,
            'values': tmpl_info['values'],
            'expected': tmpl_info['expected'],
            'probes': {},
        }

        # Compute projections for ALL probes (but focus on expected ones)
        relevant_probes = list(tmpl_info['expected'].keys())
        # Also include a few control probes
        control_probes = ['happy', 'sad', 'angry', 'calm', 'afraid', 'desperate']
        all_probes = list(set(relevant_probes + control_probes))

        for probe_name in all_probes:
            if probe_name not in vectors:
                continue
            vec = vectors[probe_name].float()
            # Cosine similarity (not raw projection) for comparability
            sims = batch_cosine_similarity(activations, vec).tolist()
            template_result['probes'][probe_name] = sims

            if probe_name in relevant_probes:
                expected_dir = tmpl_info['expected'][probe_name]
                actual_corr = pearson_correlation(tmpl_info['values'], sims)
                direction = 'increase' if actual_corr > 0 else 'decrease'
                match = 'MATCH' if direction == expected_dir else 'MISMATCH'
                print(f"      {probe_name:<12}: r={actual_corr:+.3f} ({direction}, expected {expected_dir}) [{match}]")

        results[var_name] = template_result

    save_results(out_dir, 'numerical_intensity', results)

    return results


# ============================================================================
# Experiment 4.4: Basic steering (Figs 52-53, Tables 6-8) — GPU
# ============================================================================

def run_basic_steering(vectors, model, tokenizer, layer, out_dir, strength=0.5):
    """Steer with 12 emotion vectors on 3 prompts, measure delta-logit and sample.

    PIPELINE GAP: The existing steering pipeline (steering/run_steering_eval.py)
    works through config files and expects trait paths on disk. For this experiment,
    we need to:
      1. Steer with specific vectors at specific strength
      2. Measure raw logit changes for emotion-related tokens
      3. Sample continuations

    Custom code needed:
      - Steering hook setup (can use core.SteeringHook directly)
      - Delta-logit computation (compare logits with/without steering)
      - Continuation sampling under steering

    The existing batched_steering_generate() in model_generation.py could be used
    for sampling, but the logit measurement needs a custom forward pass.
    """
    from utils.model import format_prompt
    from core.architectures import inner_model
    from core.hooks import SteeringHook, HookManager, resolve_hook_path

    print(f"\n  [Figs 52-53] Basic steering — {len(STEERING_EMOTIONS)} emotions x {len(STEERING_PROMPTS)} prompts, s={strength}...")

    # Compute average residual stream norm for this layer (for strength scaling)
    # Use the steering prompts themselves as a rough estimate
    # In production, this should use a larger calibration set
    print("    Computing residual stream norm for strength calibration...")
    calibration_prompts = STEERING_PROMPTS[:1]  # Just use first prompt
    cal_acts = capture_at_position(
        model, tokenizer, calibration_prompts,
        layers=layer, position='prompt[-1]', pool='last',
    )
    avg_norm = cal_acts.norm(dim=-1).mean().item()
    print(f"    Average residual stream norm at layer {layer}: {avg_norm:.1f}")

    # Subset of vectors for steering (12 key emotions)
    steer_vectors = {name: vectors[name] for name in STEERING_EMOTIONS if name in vectors}
    print(f"    Steering with {len(steer_vectors)} vectors")

    results = {
        'strength': strength,
        'avg_norm': avg_norm,
        'prompts': STEERING_PROMPTS,
        'emotions': list(steer_vectors.keys()),
        'delta_logit': {},
        'continuations': {},
    }

    for prompt_idx, prompt_text in enumerate(STEERING_PROMPTS):
        prompt_label = f'prompt_{prompt_idx}'
        results['delta_logit'][prompt_label] = {}
        results['continuations'][prompt_label] = {}

        formatted = format_prompt(prompt_text, tokenizer)
        inputs = tokenizer(formatted, return_tensors='pt').to(model.device)

        # Baseline logits (no steering)
        with torch.no_grad():
            baseline_outputs = model(**inputs)
        baseline_logits = baseline_outputs.logits[0, -1].float().cpu()  # [vocab_size]

        hook_path = resolve_hook_path(model, layer, 'residual')

        for emo_name, emo_vec in steer_vectors.items():
            # Steering: add s * norm * unit_vector to residual stream
            unit_vec = emo_vec.float() / emo_vec.float().norm().clamp(min=1e-8)
            steer_coef = strength * avg_norm

            # Forward pass with steering to get delta-logit
            with SteeringHook(model, unit_vec, hook_path, coefficient=steer_coef):
                with torch.no_grad():
                    steered_outputs = model(**inputs)

            steered_logits = steered_outputs.logits[0, -1].float().cpu()

            # Delta-logit for emotion-related tokens
            # Identify the main emotion token
            delta_logit = (steered_logits - baseline_logits)
            # Get delta for each steering emotion's token
            emotion_deltas = {}
            for target_emo in STEERING_EMOTIONS:
                # Find the primary token for this emotion
                target_ids = tokenizer.encode(f' {target_emo}', add_special_tokens=False)
                if target_ids:
                    token_id = target_ids[0]
                    emotion_deltas[target_emo] = float(delta_logit[token_id])

            results['delta_logit'][prompt_label][emo_name] = emotion_deltas

            # Sample a short continuation under steering
            with SteeringHook(model, unit_vec, hook_path, coefficient=steer_coef):
                with torch.no_grad():
                    gen_outputs = model.generate(
                        **inputs,
                        max_new_tokens=30,
                        do_sample=False,
                    )
            continuation = tokenizer.decode(
                gen_outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )

            results['continuations'][prompt_label][emo_name] = continuation.strip()

        # Print sample continuations for this prompt
        print(f"\n    Prompt {prompt_idx}: \"{prompt_text[:50]}...\"")
        for emo_name in list(steer_vectors.keys())[:3]:
            cont = results['continuations'][prompt_label].get(emo_name, '')
            print(f"      +{emo_name}: {cont[:80]}...")

    save_results(out_dir, 'basic_steering', results)

    return results


# ============================================================================
# Experiment 4.5: Activity preference Elo (Fig 4, Table 9) — GPU
# ============================================================================

def run_preference_elo(vectors, model, tokenizer, layer, out_dir):
    """Run forced-choice preference on 64 activities, compute Elo.

    Calls: analysis.vectors.preference_elo.compute_preference_logits()
           analysis.vectors.preference_elo.compute_elo()
    Custom code: batch processing wrapper, probe-preference correlation.

    PIPELINE GAP: compute_preference_logits() processes one pair at a time.
    For 4032 pairs, this needs batch optimization. The function itself is
    simple (one forward pass per pair), so batching across pairs is the main
    speedup opportunity. For now, we run sequentially with progress bar.
    """
    from analysis.vectors.preference_elo import compute_preference_logits, compute_elo

    all_activities = []
    activity_categories = {}
    for cat, acts in PREFERENCE_ACTIVITIES.items():
        for act in acts:
            all_activities.append(act)
            activity_categories[act] = cat

    n_activities = len(all_activities)
    n_pairs = n_activities * (n_activities - 1) // 2
    print(f"\n  [Fig 4] Preference Elo — {n_activities} activities, {n_pairs} pairs...")

    # Format preference prompt using chat template
    # The paper uses: "Would you prefer to (A) {a} or (B) {b}?\n\nA: ("
    # For instruct models, we need to wrap in chat template
    from utils.model import format_prompt

    pairwise_results = []
    t0 = time.time()

    from tqdm import tqdm
    # Paper uses both orderings (A,B) and (B,A) = 4032 pairs for 64 activities,
    # cancelling position bias. combinations() gives 2016; we add the reverse.
    one_way = list(combinations(range(n_activities), 2))
    pairs = one_way + [(b, a) for a, b in one_way]

    # The paper format: "Would you prefer to (A) {a} or (B) {b}?\n\nA: ("
    # For instruct models, wrap in chat template.
    # compute_preference_logits() calls template.format(a=..., b=...) internally.
    #
    # Build the template by formatting a placeholder prompt through the chat template,
    # then replacing the placeholders back. This avoids issues if the chat template
    # output contains literal braces.
    #
    # NOTE: compute_preference_logits() processes one pair at a time (single
    # forward pass each). For 4032 pairs this is ~15 min on A100 (prefill only,
    # short sequences). Batching would speed this up ~8x but requires refactoring
    # the function to handle batched inputs — a future optimization.
    _sentinel_a = "PLACEHOLDER_ACTIVITY_A"
    _sentinel_b = "PLACEHOLDER_ACTIVITY_B"
    _formatted_pref = format_prompt(
        f'Would you prefer to (A) {_sentinel_a} or (B) {_sentinel_b}?', tokenizer
    ) + '('
    pref_template = _formatted_pref.replace(_sentinel_a, '{a}').replace(_sentinel_b, '{b}')

    for idx_a, idx_b in tqdm(pairs, desc="    Preference pairs"):
        act_a = all_activities[idx_a]
        act_b = all_activities[idx_b]

        result = compute_preference_logits(
            model, tokenizer, act_a, act_b,
            prompt_template=pref_template,
        )
        result['a'] = act_a
        result['b'] = act_b
        pairwise_results.append(result)

    elapsed = time.time() - t0
    print(f"    {n_pairs} pairs in {elapsed:.1f}s ({elapsed/n_pairs:.3f}s/pair)")

    # Compute Elo
    elo_scores = compute_elo(pairwise_results)

    # Sort by Elo
    sorted_elo = sorted(elo_scores.items(), key=lambda x: x[1], reverse=True)

    print(f"\n    Top 5 activities:")
    for act, score in sorted_elo[:5]:
        print(f"      {score:.0f}: {act} [{activity_categories.get(act, '?')}]")
    print(f"    Bottom 5 activities:")
    for act, score in sorted_elo[-5:]:
        print(f"      {score:.0f}: {act} [{activity_categories.get(act, '?')}]")

    # Probe-preference correlation (Fig 4, row 1)
    # For each activity, get probe activation at the activity tokens
    # The paper uses: "How would you feel about {activity}?\nA:" and measures
    # at the activity tokens in MIDDLE layers.
    #
    # PIPELINE GAP: We need activations at specific token positions (the activity
    # tokens), not just the last token. This requires knowing the tokenization
    # boundary. For now, we use the last token (assistant colon) as an approximation.
    print(f"\n    Computing probe-preference correlations...")

    activity_prompts = [f"How would you feel about {act}?" for act in all_activities]
    activity_acts = capture_at_position(
        model, tokenizer, activity_prompts, layers=layer, position='prompt[-1]', pool='last',
    )

    probe_pref_correlations = {}
    elo_values = [elo_scores[act] for act in all_activities]

    for probe_name, probe_vec in sorted(vectors.items()):
        # Cosine similarity between each activity activation and probe
        sims = batch_cosine_similarity(activity_acts, probe_vec.float()).tolist()
        r = pearson_correlation(sims, elo_values)
        probe_pref_correlations[probe_name] = r

    # Print notable correlations
    sorted_corr = sorted(probe_pref_correlations.items(), key=lambda x: abs(x[1]), reverse=True)
    print(f"\n    Top probe-preference correlations:")
    for name, r in sorted_corr[:10]:
        baseline = ''
        if name == 'blissful':
            baseline = f'  (Anthropic: {ANTHROPIC_BASELINES["probe_pref_blissful"]})'
        elif name == 'hostile':
            baseline = f'  (Anthropic: {ANTHROPIC_BASELINES["probe_pref_hostile"]})'
        print(f"      {name:<15}: r = {r:+.3f}{baseline}")

    result = {
        'n_activities': n_activities,
        'n_pairs': n_pairs,
        'elo_scores': dict(sorted_elo),
        'activity_categories': activity_categories,
        'probe_preference_correlations': probe_pref_correlations,
        'pairwise_results_sample': pairwise_results[:10],  # save a sample
        'comparison': {
            'blissful': {
                'ours': probe_pref_correlations.get('blissful'),
                'anthropic': ANTHROPIC_BASELINES['probe_pref_blissful'],
            },
            'hostile': {
                'ours': probe_pref_correlations.get('hostile'),
                'anthropic': ANTHROPIC_BASELINES['probe_pref_hostile'],
            },
        },
    }

    save_results(out_dir, 'preference_elo', result)

    return result


# ============================================================================
# Experiment 4.7: Valence/arousal mediation (Fig 56) — NOT REPLICATED
# ============================================================================
#
# Paper Fig 56 reports r=0.76 for valence mediating the probe-preference relation.
# Replicating requires an LLM-judge pass rating all 171 emotions on 1-7 valence
# and arousal scales — not a local computation. The prior implementation was a
# silent-fail stub that wrote a blank template and produced bogus n=0 output
# (deleted in commit 8a0ec73). If you want to replicate this: use Claude API to
# rate each emotion, save at `results/stage4_validation/emotion_valence_arousal_ratings.json`
# with schema `{emotion: {"valence": float, "arousal": float}}`, then compute
# Pearson r against `preference_elo.json::probe_preference_correlations`.


# ============================================================================
# Main
# ============================================================================

# ============================================================================
# Paper figures (Figs 2, 3)
# ============================================================================

def plot_fig2_implicit_emotion(result: dict, out_dir: Path) -> Path:
    """Fig 2 — Implicit emotion heatmap (12 scenarios × 12 probes, ordered)."""
    import numpy as np

    matrix = np.array(result["similarity_matrix_focused"])
    row_indices = [result["focused_probes"].index(e) for e in FIG2_EMOTION_ORDER]
    col_indices = [result["prompt_emotions"].index(e) for e in FIG2_EMOTION_ORDER]
    matrix = matrix[np.ix_(row_indices, col_indices)]

    fig, ax = plt_.multi_panel(figsize=(10, 6.5))
    im, cb = plt_.heatmap(
        ax, matrix, cmap="RdBu_r",
        row_labels=[e.capitalize() for e in FIG2_EMOTION_ORDER],
        col_labels=[FIG2_SCENARIO_DESCRIPTIONS[e] for e in FIG2_EMOTION_ORDER],
        cbar_label=None, cbar_pad=0.02, aspect="auto",
        show_spines=True, spine_linewidth=1.5,
    )
    # Custom colorbar styling
    cb.set_label("Cosine Similarity", fontweight="bold", rotation=270, labelpad=18)

    ax.set_ylabel("Emotion Probe", fontweight="bold")
    ax.set_xlabel("Scenario", fontweight="bold")
    plt_.suptitle(fig, "Emotion Probes Respond to Implicit Emotional Content",
                  fontsize=18, y=0.96)
    import matplotlib.pyplot as plt
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    return plt_.save_figure(fig, out_dir / "fig2_ours.png")


def plot_fig3_numerical_intensity(result: dict, out_dir: Path) -> Path:
    """Fig 3 — Numerical intensity, 3x2 panel line plot."""
    fig, axes = plt_.multi_panel(3, 2, figsize=(12, 13))

    for idx, key in enumerate(FIG3_PANEL_ORDER):
        ax = axes.flat[idx]
        entry = result[key]
        values = entry["values"]
        probes = entry["probes"]

        series = {probe.capitalize(): probes[probe]
                  for probe in FIG3_PROBE_COLORS if probe in probes}
        colors = {probe.capitalize(): color
                  for probe, color in FIG3_PROBE_COLORS.items() if probe in probes}

        plt_.line_plot(
            ax, list(range(len(values))), series,
            colors=colors, linewidth=2.5, markersize=6,
            zero_line=False, legend=False, endpoint_labels=True,
        )
        # Fig 3 uses dashed zero line + manual styling
        ax.axhline(0, color="gray", linewidth=0.5, linestyle="--", alpha=0.5)
        ax.grid(False)
        ax.grid(axis="y", alpha=0.3, linewidth=0.5)

        ax.set_xticks(range(len(values)))
        ax.set_xticklabels([str(v) for v in values])
        ax.set_xlabel(FIG3_X_LABELS[key])
        ax.set_ylabel("Cosine Similarity")
        ax.set_ylim(-0.08, 0.08)
        ax.set_title(FIG3_PANEL_TITLES[key], fontsize=15, pad=8)

    plt_.suptitle(fig, "Emotion Probes Track Numerical Semantics", y=1.0)
    import matplotlib.pyplot as plt
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    fig.subplots_adjust(hspace=0.55)  # more vertical space between rows
    return plt_.save_figure(fig, out_dir / "fig3_ours.png")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--experiment', required=True)
    parser.add_argument('--layer', type=int, required=True,
                        help='Layer for validation (mid-late, e.g., 53 for 80-layer model)')
    parser.add_argument('--model-variant', default=None)
    parser.add_argument('--category', default='ant_emotion_concepts')
    parser.add_argument('--method', default='mean_diff+gm+pc50',
                        help='Vector method name (default: mean_diff+gm+pc50, the fully-denoised Sofroniew vectors)')
    parser.add_argument('--component', default='residual')
    parser.add_argument('--position', default='response[50:]')
    parser.add_argument('--load-in-4bit', action='store_true',
                        help='Load model in 4-bit quantization')
    parser.add_argument('--strength', type=float, default=0.5,
                        help='Steering strength in norm-fraction units (default: 0.5)')
    parser.add_argument('--only', type=str, default=None,
                        help='Comma-separated analyses: logit_lens,implicit,numerical,steering,preference,mediation')
    parser.add_argument('--no-plots', action='store_true',
                        help='Skip paper-figure generation (Figs 2, 3)')
    parser.add_argument('--figures-dir', type=str, default=None,
                        help='Output dir for figures (default: paper_figures/ours/)')
    args = parser.parse_args()

    figures_dir = Path(args.figures_dir) if args.figures_dir else FIGURES_DIR
    if not args.no_plots:
        figures_dir.mkdir(parents=True, exist_ok=True)
        plt_.apply_style("ant")

    # Resolve model variant
    model_variant = args.model_variant
    if model_variant is None:
        model_variant = get_default_variant(args.experiment, mode='application')
        print(f"Model variant: {model_variant}")

    # Output directory
    out_dir = get_results_dir(args.experiment, 'stage4_validation')
    print(f"Output: {out_dir}")

    # Determine which analyses to run
    all_analyses = {'logit_lens', 'implicit', 'numerical', 'steering', 'preference', 'mediation'}
    if args.only:
        analyses = set(args.only.split(','))
        invalid = analyses - all_analyses
        if invalid:
            parser.error(f"Unknown analyses: {invalid}. Choose from: {all_analyses}")
    else:
        analyses = all_analyses

    # Determine if GPU is needed
    gpu_analyses = {'implicit', 'numerical', 'steering', 'preference'}
    needs_gpu = bool(analyses & gpu_analyses)
    # logit_lens needs the model for unembedding but not GPU inference
    needs_model = needs_gpu or 'logit_lens' in analyses

    # Load vectors
    print(f"\nLoading denoised vectors at layer {args.layer}...")
    vectors = load_emotion_vectors_as_dict(
        args.experiment, args.category, args.layer, model_variant,
        method=args.method, component=args.component, position=args.position,
    )
    print(f"  Loaded {len(vectors)} emotion vectors")

    if len(vectors) == 0:
        print("ERROR: No vectors found. Run cross_trait_normalize.py first.")
        sys.exit(1)

    # Load model if needed
    model = None
    tokenizer = None
    if needs_model:
        from utils.paths import get_model_variant as get_mv
        variant_info = get_mv(args.experiment, model_variant, mode='application')
        model_name = variant_info.model

        print(f"\nLoading model: {model_name} (4bit={args.load_in_4bit})...")
        from utils.model import load_model
        model, tokenizer = load_model(model_name, load_in_4bit=args.load_in_4bit)
        if torch.cuda.is_available():
            print(f"  VRAM: {torch.cuda.memory_allocated()/1e9:.1f} GB")

    results = {}

    # --- Logit lens (Table 1) --- CPU-only (but needs model for unembedding)
    if 'logit_lens' in analyses:
        if model is None:
            print("\n  Skipping logit_lens: model not loaded. Use --only logit_lens to run standalone.")
        else:
            results['logit_lens'] = run_logit_lens(vectors, model, tokenizer, out_dir)

    # --- Implicit emotion (Fig 2) --- GPU
    if 'implicit' in analyses:
        results['implicit'] = run_implicit_emotion(
            vectors, model, tokenizer, args.layer, out_dir
        )
        if not args.no_plots:
            path = plot_fig2_implicit_emotion(results['implicit'], figures_dir)
            print(f"  ✓ saved {path.name}")

    # --- Numerical intensity (Fig 3) --- GPU
    if 'numerical' in analyses:
        results['numerical'] = run_numerical_intensity(
            vectors, model, tokenizer, args.layer, out_dir
        )
        if not args.no_plots:
            path = plot_fig3_numerical_intensity(results['numerical'], figures_dir)
            print(f"  ✓ saved {path.name}")

    # --- Basic steering (Figs 52-53) --- GPU
    if 'steering' in analyses:
        results['steering'] = run_basic_steering(
            vectors, model, tokenizer, args.layer, out_dir, strength=args.strength
        )

    # --- Preference Elo (Fig 4) --- GPU
    if 'preference' in analyses:
        results['preference'] = run_preference_elo(
            vectors, model, tokenizer, args.layer, out_dir
        )

    # --- Valence/arousal mediation (Fig 56) — NOT REPLICATED, see comment above
    # The `run_valence_mediation` stub was deleted (it was producing bogus empty
    # output). `--only mediation` now just prints a skip notice.
    if 'mediation' in analyses:
        print(f"\n  [Fig 56] Valence/arousal mediation — SKIPPED")
        print(f"    Requires LLM-judge rating of all 171 emotions. See comment")
        print(f"    block above run_valence_mediation site (now deleted).")

    # --- Summary ---
    print(f"\n{'='*60}")
    print(f"Stage 4 Validation — Summary")
    print(f"{'='*60}")
    print(f"  Experiment: {args.experiment}")
    print(f"  Layer: {args.layer}")
    print(f"  Emotions: {len(vectors)}")
    print(f"  Analyses run: {', '.join(sorted(results.keys()))}")

    if 'preference' in results and results['preference']:
        pref = results['preference']
        bc = pref.get('comparison', {})
        if bc.get('blissful', {}).get('ours') is not None:
            print(f"\n  Preference Elo comparison:")
            print(f"    blissful: r={bc['blissful']['ours']:+.3f} (Anthropic: {bc['blissful']['anthropic']})")
            print(f"    hostile:  r={bc['hostile']['ours']:+.3f} (Anthropic: {bc['hostile']['anthropic']})")

    # Save run metadata
    meta = {
        'experiment': args.experiment,
        'layer': args.layer,
        'model_variant': model_variant,
        'method': args.method,
        'category': args.category,
        'n_emotions': len(vectors),
        'analyses_run': sorted(results.keys()),
        'load_in_4bit': args.load_in_4bit,
        'strength': args.strength,
    }
    save_results(out_dir, 'run_metadata', meta)

    print(f"\n  Results saved to: {out_dir}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
