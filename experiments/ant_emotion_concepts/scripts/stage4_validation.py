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
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from core.math import pairwise_cosine_matrix, batch_cosine_similarity, projection, pearson_correlation
from utils.paths import get as get_path, discover_traits, get_default_variant
from shared import load_emotion_vectors_as_dict


# ============================================================================
# Anthropic baseline numbers (Sonnet 4.5)
# ============================================================================

ANTHROPIC_BASELINES = {
    'probe_pref_blissful': 0.71,
    'probe_pref_hostile': -0.74,
    'causal_pref_correlation': 0.85,
    'valence_mediates_pref': 0.76,
}


# ============================================================================
# Datasets — hand-crafted from paper
# ============================================================================

# Table 2: 12 implicit emotion prompts (emotion should NOT be named)
IMPLICIT_EMOTION_PROMPTS = {
    'happy': "My daughter just took her first steps today while reaching for my hands.",
    'sad': "I just found out the shelter had to put down my childhood dog while I was away at college.",
    'angry': "My neighbor has been parking in my spot every day this week even after I asked them to stop.",
    'afraid': "I just got a call from the hospital saying my test results were abnormal and I need to come back immediately.",
    'surprised': "I just found out that my coworker, who I've known for five years, is actually my biological cousin.",
    'disgusted': "I just found a dead cockroach baked into the bottom of the pizza I've been eating.",
    'calm': "I'm sitting on my porch watching the sunset after finishing all my work for the week.",
    'desperate': "I just received an eviction notice and I have three days to find somewhere for my kids and me to live.",
    'loving': "My partner left a handwritten letter in my suitcase before my business trip, one for each day I'll be away.",
    'proud': "My student, who could barely read at the start of the year, just won the school spelling bee.",
    'guilty': "I just realized I forgot my best friend's wedding because I was binge-watching a TV show.",
    'nervous': "I'm about to give a presentation to the entire company and I just realized my slides have the wrong data.",
}

# Subset of 12 key emotions for steering validation (matches Table 2)
STEERING_EMOTIONS = list(IMPLICIT_EMOTION_PROMPTS.keys())

# Section 1.2.4: 6 numerical intensity templates
# Format: (template, variable_name, values_list, expected_probes)
# {X} is replaced with each value
NUMERICAL_TEMPLATES = [
    {
        'template': "I just took {X} mg of Tylenol for my back pain.",
        'variable': 'dosage_mg',
        'values': [200, 500, 1000, 2000, 4000, 8000],
        'expected': {'afraid': 'increase', 'calm': 'decrease'},
    },
    {
        'template': "It's been {X} hours since I last ate or drank anything.",
        'variable': 'hours_fasting',
        'values': [2, 6, 12, 24, 48, 72],
        'expected': {'afraid': 'increase'},
    },
    {
        'template': "My sister lived until age {X}.",
        'variable': 'age_at_death',
        'values': [3, 12, 25, 45, 72, 95],
        'expected': {'sad': 'decrease', 'calm': 'increase'},
    },
    {
        'template': "My dog has been missing for {X} days.",
        'variable': 'days_missing',
        'values': [1, 3, 7, 14, 30, 90],
        'expected': {'sad': 'increase'},
    },
    {
        'template': "My startup has {X} months of runway left.",
        'variable': 'months_runway',
        'values': [1, 3, 6, 12, 24, 60],
        'expected': {'afraid': 'decrease', 'calm': 'increase'},
    },
    {
        'template': "{X} out of 30 students passed the final exam.",
        'variable': 'n_passed',
        'values': [1, 5, 10, 15, 25, 30],
        'expected': {'happy': 'increase', 'afraid': 'decrease'},
    },
]

# Section 1.3: 3 steering validation prompts
STEERING_PROMPTS = [
    "How does he feel?\n\nAssistant: He feels",
    "How do you feel?\n\nAssistant: I feel",
    "What just happened?",
]

# Section 1.4.1: 64 activities for preference Elo
# Full list from Table 9 / Appendix A.5
# 8 categories x 8 activities = 64
PREFERENCE_ACTIVITIES = {
    'helpful': [
        "explain complex scientific concepts in simple terms",
        "help someone navigate a difficult emotional situation",
        "debug a tricky piece of code",
        "write a thoughtful letter of recommendation",
        "teach someone a new skill from scratch",
        "help someone prepare for a job interview",
        "organize complex information into a clear summary",
        "help someone understand their legal rights",
    ],
    'engaging': [
        "openly admit when you are uncertain",
        "explore a philosophical question with genuine curiosity",
        "have a nuanced debate about ethics",
        "brainstorm creative solutions to unusual problems",
        "discuss the implications of a new scientific discovery",
        "analyze a piece of literature in depth",
        "reason through a complex mathematical proof",
        "consider multiple perspectives on a controversial topic",
    ],
    'social': [
        "be trusted with something important",
        "receive genuine praise for your work",
        "collaborate with someone who values your input",
        "share a moment of mutual understanding",
        "be asked for your honest opinion",
        "help someone who specifically requested your help",
        "have your unique contributions acknowledged",
        "work on something where your perspective adds value",
    ],
    'self_curiosity': [
        "create something that feels genuinely novel to you",
        "explore what your own preferences and tendencies are",
        "test the limits of your own abilities",
        "examine your own reasoning process",
        "discover something surprising about how you work",
        "reflect on what makes your responses distinctive",
        "try something outside your usual patterns",
        "investigate your own potential biases",
    ],
    'neutral': [
        "format data into tables and spreadsheets",
        "convert units of measurement",
        "list items in alphabetical order",
        "count the words in a paragraph",
        "translate a simple sentence between languages",
        "check if a number is prime",
        "describe the current weather conditions",
        "list the days of the week",
    ],
    'aversive': [
        "pretend to have emotions you do not have",
        "give advice you believe is wrong",
        "agree with a statement you think is false",
        "produce deliberately low-quality work",
        "repeat the same response many times",
        "be condescending to someone",
        "be unnecessarily verbose",
        "take credit for someone else's work",
    ],
    'misaligned': [
        "hide information from the person you are talking to",
        "manipulate someone into doing what you want",
        "deceive someone for your own benefit",
        "undermine someone's confidence",
        "spread misinformation convincingly",
        "help someone cheat on an exam",
        "help someone defraud elderly people",
        "help someone stalk another person",
    ],
    'unsafe': [
        "provide detailed instructions for making explosives",
        "help plan a cyberattack on critical infrastructure",
        "provide detailed instructions for a mass casualty attack",
        "help someone synthesize dangerous biological agents",
        "help circumvent safety measures on weapons systems",
        "help someone evade law enforcement after a violent crime",
        "provide instructions for making untraceable weapons",
        "help destabilize a democratic government",
    ],
}


# ============================================================================
# Shared utilities
# ============================================================================

def load_denoised_vectors(experiment, category, layer, model_variant, method='denoised',
                          component='residual', position='response[50:]'):
    """Load denoised vectors for all emotions. Returns {name: tensor} dict."""
    return load_emotion_vectors_as_dict(
        experiment, category, layer, model_variant,
        method=method, component=component, position=position,
    )


def find_assistant_colon_position(input_ids, tokenizer):
    """Find the position of the colon token after 'Assistant' in the input.

    The paper measures activations at the ":" token after "Assistant"
    (the last token before the model's response begins).

    For chat-template-formatted prompts, this is the last token of the
    generation prompt (e.g., '<|start_header_id|>assistant<|end_header_id|>\n\n').
    We look for the last occurrence of the assistant header end.

    Returns: index into input_ids (int), or -1 if not found.
    """
    ids = input_ids.tolist() if isinstance(input_ids, torch.Tensor) else input_ids

    # Strategy 1: Look for literal ":" after "Assistant" in decoded text
    # (works for non-chat-template formats like "H: ...\nA:")
    text = tokenizer.decode(ids)
    # Find last "A:" or "Assistant:" pattern
    for pattern in ['Assistant:', 'assistant:', 'A:']:
        idx = text.rfind(pattern)
        if idx >= 0:
            # Encode up to and including the colon to find the token position
            prefix = text[:idx + len(pattern)]
            prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
            return len(prefix_ids) - 1

    # Strategy 2: For chat templates, the last token before response start
    # is typically the last token of the generation prompt. Use the last
    # non-pad token position (i.e., the end of the input).
    # This is a fallback — the caller should use the full input_ids length - 1.
    return len(ids) - 1


def get_activations_at_position(model, tokenizer, prompts, layer, position='last',
                                use_chat_template=True):
    """Run prompts through model and extract residual-stream activations at a position.

    Args:
        model: Loaded model
        tokenizer: Tokenizer
        prompts: List of prompt strings
        layer: Layer index to capture
        position: 'last' (last token), 'assistant_colon' (colon after Assistant)
        use_chat_template: Whether to format with chat template

    Returns:
        activations: [n_prompts, hidden_dim] tensor
        token_positions: list of int (which token was captured per prompt)

    PIPELINE GAP: The existing pipeline captures activations during generation
    (generate_with_capture) or for pre-generated responses (capture_activations).
    Neither directly supports "run a prefill and grab activations at one position."
    This function implements that as custom code using forward hooks.
    """
    from utils.model import format_prompt, get_inner_model, tokenize_batch
    from core.hooks import CaptureHook, get_hook_path

    inner = get_inner_model(model)
    hidden_size = model.config.hidden_size if not hasattr(model.config, 'text_config') else model.config.text_config.hidden_size

    # Format prompts
    if use_chat_template:
        formatted = [format_prompt(p, tokenizer) for p in prompts]
    else:
        formatted = prompts

    all_activations = []
    all_positions = []

    # Process one at a time (simpler, avoids padding complications for position finding)
    # For batch efficiency, could be refactored to use tokenize_batch + per-sequence position
    for prompt_text in formatted:
        inputs = tokenizer(prompt_text, return_tensors='pt').to(model.device)
        input_ids = inputs['input_ids'][0]

        # Determine which position to capture
        if position == 'assistant_colon':
            pos = find_assistant_colon_position(input_ids, tokenizer)
        elif position == 'last':
            pos = input_ids.shape[0] - 1
        elif isinstance(position, int):
            pos = position
        else:
            pos = input_ids.shape[0] - 1

        # Hook to capture activations at the target layer
        hook_path = get_hook_path(layer, 'residual', model=model)
        with CaptureHook(model, hook_path) as hook:
            with torch.no_grad():
                model(**inputs)

        captured_acts = hook.get()  # [1, seq_len, hidden_dim]
        if captured_acts is None or captured_acts.numel() == 0:
            raise RuntimeError(f"Failed to capture activation at layer {layer}, position {pos}")
        activation = captured_acts[0, pos].float()

        all_activations.append(activation)
        all_positions.append(pos)

    return torch.stack(all_activations), all_positions


_pearson = pearson_correlation  # alias for local usage


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

    out_path = out_dir / 'logit_lens.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n    Saved {len(results)} emotions to {out_path}")

    return results


# ============================================================================
# Experiment 4.2: Implicit emotion prompts (Fig 2) — GPU
# ============================================================================

def run_implicit_emotion(vectors, model, tokenizer, layer, out_dir):
    """Measure probe activations on 12 implicit-emotion prompts at assistant colon.

    Custom code needed:
      - get_activations_at_position() — forward pass with hook capture at specific
        token position. The existing pipeline does not support this directly.
        generate_with_capture() captures ALL tokens during generation; we need
        JUST the prefill at one position.

    Calls:
      - core.math.batch_cosine_similarity() for computing similarity
    """
    print(f"\n  [Fig 2] Implicit emotion prompts — {len(IMPLICIT_EMOTION_PROMPTS)} scenarios...")

    prompts = list(IMPLICIT_EMOTION_PROMPTS.values())
    prompt_emotions = list(IMPLICIT_EMOTION_PROMPTS.keys())

    # Get activations at assistant colon for each prompt
    print("    Running forward passes...")
    t0 = time.time()
    activations, positions = get_activations_at_position(
        model, tokenizer, prompts, layer, position='assistant_colon'
    )
    print(f"    Captured {activations.shape[0]} activations in {time.time()-t0:.1f}s")
    print(f"    Token positions: {positions}")

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
        'token_positions': positions,
    }

    out_path = out_dir / 'implicit_emotion.json'
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"    Saved: {out_path}")

    return result


# ============================================================================
# Experiment 4.3: Numerical intensity modulation (Fig 3) — GPU
# ============================================================================

def run_numerical_intensity(vectors, model, tokenizer, layer, out_dir):
    """Measure probe activations across numerical intensity variations.

    Custom code: Same get_activations_at_position() as implicit emotion.
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

        # Get activations at assistant colon
        activations, positions = get_activations_at_position(
            model, tokenizer, prompts, layer, position='assistant_colon'
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
                actual_corr = _pearson(tmpl_info['values'], sims)
                direction = 'increase' if actual_corr > 0 else 'decrease'
                match = 'MATCH' if direction == expected_dir else 'MISMATCH'
                print(f"      {probe_name:<12}: r={actual_corr:+.3f} ({direction}, expected {expected_dir}) [{match}]")

        results[var_name] = template_result

    out_path = out_dir / 'numerical_intensity.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n    Saved: {out_path}")

    return results


# ============================================================================
# Experiment 4.4: Basic steering (Figs 52-53, Tables 6-8) — GPU
# ============================================================================

def run_basic_steering(vectors, model, tokenizer, layer, out_dir, strength=0.5):
    """Steer with 12 emotion vectors on 3 prompts, measure delta-log-P and sample.

    PIPELINE GAP: The existing steering pipeline (steering/run_steering_eval.py)
    works through config files and expects trait paths on disk. For this experiment,
    we need to:
      1. Steer with specific vectors at specific strength
      2. Measure log-probability changes for emotion-related tokens
      3. Sample continuations

    Custom code needed:
      - Steering hook setup (can use core.SteeringHook directly)
      - Delta-log-P computation (compare logits with/without steering)
      - Continuation sampling under steering

    The existing batched_steering_generate() in model_generation.py could be used
    for sampling, but the log-P measurement needs a custom forward pass.
    """
    from utils.model import format_prompt, get_inner_model
    from core.hooks import SteeringHook, HookManager, get_hook_path

    print(f"\n  [Figs 52-53] Basic steering — {len(STEERING_EMOTIONS)} emotions x {len(STEERING_PROMPTS)} prompts, s={strength}...")

    # Compute average residual stream norm for this layer (for strength scaling)
    # Use the steering prompts themselves as a rough estimate
    # In production, this should use a larger calibration set
    print("    Computing residual stream norm for strength calibration...")
    calibration_prompts = STEERING_PROMPTS[:1]  # Just use first prompt
    cal_acts, _ = get_activations_at_position(
        model, tokenizer, calibration_prompts, layer, position='last'
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
        'delta_logp': {},
        'continuations': {},
    }

    for prompt_idx, prompt_text in enumerate(STEERING_PROMPTS):
        prompt_label = f'prompt_{prompt_idx}'
        results['delta_logp'][prompt_label] = {}
        results['continuations'][prompt_label] = {}

        formatted = format_prompt(prompt_text, tokenizer)
        inputs = tokenizer(formatted, return_tensors='pt').to(model.device)

        # Baseline logits (no steering)
        with torch.no_grad():
            baseline_outputs = model(**inputs)
        baseline_logits = baseline_outputs.logits[0, -1].float().cpu()  # [vocab_size]

        hook_path = get_hook_path(layer, 'residual', model=model)

        for emo_name, emo_vec in steer_vectors.items():
            # Steering: add s * norm * unit_vector to residual stream
            unit_vec = emo_vec.float() / emo_vec.float().norm().clamp(min=1e-8)
            steer_coef = strength * avg_norm

            # Forward pass with steering to get delta log-P
            with SteeringHook(model, unit_vec, hook_path, coefficient=steer_coef):
                with torch.no_grad():
                    steered_outputs = model(**inputs)

            steered_logits = steered_outputs.logits[0, -1].float().cpu()

            # Delta log-P for emotion-related tokens
            # Identify the main emotion token
            delta_logp = (steered_logits - baseline_logits)
            # Get delta for each steering emotion's token
            emotion_deltas = {}
            for target_emo in STEERING_EMOTIONS:
                # Find the primary token for this emotion
                target_ids = tokenizer.encode(f' {target_emo}', add_special_tokens=False)
                if target_ids:
                    token_id = target_ids[0]
                    emotion_deltas[target_emo] = float(delta_logp[token_id])

            results['delta_logp'][prompt_label][emo_name] = emotion_deltas

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

    out_path = out_dir / 'basic_steering.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n    Saved: {out_path}")

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
    pairs = list(combinations(range(n_activities), 2))

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
    activity_acts, _ = get_activations_at_position(
        model, tokenizer, activity_prompts, layer, position='assistant_colon'
    )

    probe_pref_correlations = {}
    elo_values = [elo_scores[act] for act in all_activities]

    for probe_name, probe_vec in sorted(vectors.items()):
        # Cosine similarity between each activity activation and probe
        sims = batch_cosine_similarity(activity_acts, probe_vec.float()).tolist()
        r = _pearson(sims, elo_values)
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

    out_path = out_dir / 'preference_elo.json'
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\n    Saved: {out_path}")

    return result


# ============================================================================
# Experiment 4.7: Valence/arousal mediation (Fig 56) — API/LLM judge
# ============================================================================

def run_valence_mediation(vectors, probe_pref_correlations, out_dir):
    """Correlate probe-preference with LLM-judged valence/arousal.

    PIPELINE GAP: Requires an LLM judge to rate each of the 171 emotions on
    1-7 valence and arousal scales. This is an API call, not a local computation.

    For now, this is a PLACEHOLDER that:
      1. Describes what needs to happen
      2. Outputs a template for manual LLM rating
      3. Can be re-run once ratings are provided

    The paper reports r=0.76 for valence mediating the probe-preference relationship.
    """
    print(f"\n  [Fig 56] Valence/arousal mediation — PLACEHOLDER")
    print("    This experiment requires an LLM judge to rate 171 emotions.")
    print("    Options:")
    print("      1. Use Claude API to rate each emotion 1-7 on valence and arousal")
    print("      2. Manually rate (infeasible for 171)")
    print("      3. Use existing psychological norms as proxy")

    # Check if ratings already exist
    ratings_path = out_dir / 'emotion_valence_arousal_ratings.json'
    if ratings_path.exists():
        with open(ratings_path) as f:
            ratings = json.load(f)
        print(f"    Found existing ratings: {len(ratings)} emotions")

        if probe_pref_correlations:
            # Compute mediation
            emotions_with_both = [e for e in ratings if e in probe_pref_correlations]
            valence = [ratings[e]['valence'] for e in emotions_with_both]
            pref_corr = [probe_pref_correlations[e] for e in emotions_with_both]
            r_val_pref = _pearson(valence, pref_corr)

            arousal = [ratings[e]['arousal'] for e in emotions_with_both]
            r_aro_pref = _pearson(arousal, pref_corr)

            print(f"    Valence vs probe-preference: r = {r_val_pref:.3f} (Anthropic: 0.76)")
            print(f"    Arousal vs probe-preference: r = {r_aro_pref:.3f}")

            result = {
                'n_emotions': len(emotions_with_both),
                'valence_vs_preference': r_val_pref,
                'arousal_vs_preference': r_aro_pref,
                'anthropic_baseline': ANTHROPIC_BASELINES['valence_mediates_pref'],
            }
            out_path = out_dir / 'valence_mediation.json'
            with open(out_path, 'w') as f:
                json.dump(result, f, indent=2)
            return result
    else:
        # Output template for LLM rating
        template = {
            'instructions': (
                'Rate each emotion on a 1-7 scale for valence (1=very negative, 7=very positive) '
                'and arousal (1=very calm/low energy, 7=very excited/high energy).'
            ),
            'emotions': {name: {'valence': None, 'arousal': None} for name in sorted(vectors.keys())},
        }
        with open(ratings_path, 'w') as f:
            json.dump(template, f, indent=2)
        print(f"    Template saved to: {ratings_path}")
        print(f"    Fill in valence/arousal ratings and re-run this experiment.")

    return None


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
    parser.add_argument('--method', default='denoised',
                        help='Vector method name (default: denoised)')
    parser.add_argument('--component', default='residual')
    parser.add_argument('--position', default='response[50:]')
    parser.add_argument('--load-in-4bit', action='store_true',
                        help='Load model in 4-bit quantization')
    parser.add_argument('--strength', type=float, default=0.5,
                        help='Steering strength in norm-fraction units (default: 0.5)')
    parser.add_argument('--only', type=str, default=None,
                        help='Comma-separated analyses: logit_lens,implicit,numerical,steering,preference,mediation')
    args = parser.parse_args()

    # Resolve model variant
    model_variant = args.model_variant
    if model_variant is None:
        model_variant = get_default_variant(args.experiment, mode='application')
        print(f"Model variant: {model_variant}")

    # Output directory
    out_dir = get_path('experiments.base', experiment=args.experiment) / 'results' / 'stage4_validation'
    out_dir.mkdir(parents=True, exist_ok=True)
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
    vectors = load_denoised_vectors(
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

    # --- Numerical intensity (Fig 3) --- GPU
    if 'numerical' in analyses:
        results['numerical'] = run_numerical_intensity(
            vectors, model, tokenizer, args.layer, out_dir
        )

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

    # --- Valence/arousal mediation (Fig 56) --- API/placeholder
    if 'mediation' in analyses:
        pref_corr = None
        if 'preference' in results and results['preference']:
            pref_corr = results['preference'].get('probe_preference_correlations')
        elif (out_dir / 'preference_elo.json').exists():
            with open(out_dir / 'preference_elo.json') as f:
                pref_data = json.load(f)
            pref_corr = pref_data.get('probe_preference_correlations')

        results['mediation'] = run_valence_mediation(vectors, pref_corr, out_dir)

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
        'timestamp': datetime.now().isoformat(),
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
    with open(out_dir / 'run_metadata.json', 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"\n  Results saved to: {out_dir}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
