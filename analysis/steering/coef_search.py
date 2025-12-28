"""
Coefficient search for steering evaluation.

Input:
    - model, tokenizer: Loaded model
    - layer_data: Layer info with vectors and base coefficients
    - questions: Evaluation questions
    - judge: TraitJudge instance

Output:
    - Finds optimal coefficients via adaptive search
    - Saves results incrementally

Usage:
    from analysis.steering.coef_search import adaptive_search_layer, batched_adaptive_search
"""

import time

import torch
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm

from core import SteeringHook, get_hook_path
from analysis.steering.steer import BatchedLayerSteeringHook
from utils.generation import generate_batch, calculate_max_batch_size
from analysis.steering.results import find_existing_run_index, save_results, save_responses
from utils.judge import TraitJudge
from utils.model import format_prompt
from utils.vectors import MIN_COHERENCE


async def evaluate_single_config(
    model,
    tokenizer,
    vector: torch.Tensor,
    layer: int,
    coef: float,
    questions: List[str],
    trait_name: str,
    trait_definition: str,
    judge: TraitJudge,
    use_chat_template: bool,
    component: str,
    max_new_tokens: int = 256,
) -> Tuple[Dict, List[Dict]]:
    """Evaluate a single (layer, coefficient) config with batched generation."""
    desc = f"L{layer} c{coef:.0f}"

    # Format all questions
    formatted = [format_prompt(q, tokenizer, use_chat_template=use_chat_template) for q in questions]

    # Generate all responses in batch with steering
    print(f"  Generating {len(questions)} responses for {desc}...")
    with SteeringHook(model, vector, get_hook_path(layer, component), coefficient=coef):
        responses = generate_batch(model, tokenizer, formatted, max_new_tokens=max_new_tokens)

    all_qa_pairs = list(zip(questions, responses))

    # Score
    print(f"  Scoring {len(all_qa_pairs)} responses...")
    all_scores = await judge.score_steering_batch(all_qa_pairs, trait_name, trait_definition)

    trait_scores = [s["trait_score"] for s in all_scores if s["trait_score"] is not None]
    coherence_scores = [s["coherence_score"] for s in all_scores if s.get("coherence_score") is not None]

    result = {
        "trait_mean": sum(trait_scores) / len(trait_scores) if trait_scores else None,
        "coherence_mean": sum(coherence_scores) / len(coherence_scores) if coherence_scores else None,
        "n": len(trait_scores),
    }

    responses = [
        {"question": q, "response": r, "trait_score": s["trait_score"], "coherence_score": s.get("coherence_score")}
        for (q, r), s in zip(all_qa_pairs, all_scores)
    ]

    trait_str = f"{result['trait_mean']:.1f}" if result['trait_mean'] else "N/A"
    coh_str = f"{result['coherence_mean']:.1f}" if result['coherence_mean'] else "N/A"
    print(f"  {desc}: trait={trait_str}, coherence={coh_str}, n={result['n']}")

    return result, responses


async def evaluate_and_save(
    model, tokenizer, vector, layer, coef,
    questions, trait_name, trait_definition, judge, use_chat_template, component,
    results, experiment, trait, vector_experiment, method, position="response[:]"
):
    """Evaluate a single config and save to results."""
    config = {
        "layers": [layer],
        "methods": [method],
        "coefficients": [coef],
        "component": component,
    }

    # Skip if already exists
    if find_existing_run_index(results, config) is not None:
        print(f"  L{layer} c{coef:.0f}: Already evaluated, skipping")
        return

    print(f"\n  Evaluating L{layer} c{coef:.0f}...")
    result, responses = await evaluate_single_config(
        model, tokenizer, vector, layer, coef,
        questions, trait_name, trait_definition, judge, use_chat_template, component
    )

    timestamp = datetime.now().isoformat()
    save_responses(responses, experiment, trait, position, config, timestamp)

    run_data = {
        "config": config,
        "result": result,
        "timestamp": timestamp,
    }

    results["runs"].append(run_data)
    save_results(results, experiment, trait, position)


async def adaptive_search_layer(
    model, tokenizer, vector, layer, base_coef,
    questions, trait_name, trait_definition, judge, use_chat_template, component,
    results, experiment, trait, vector_experiment, method,
    position: str = "response[:]",
    n_steps: int = 8,
    threshold: float = MIN_COHERENCE,
    up_mult: float = 1.3,
    down_mult: float = 0.85,
    momentum: float = 0.0,  # 0.0 = no momentum, 0.7 = typical momentum
    max_new_tokens: int = 256,
):
    """Run adaptive search for a single layer, saving each result."""
    print(f"\n--- Layer {layer} (base_coef={base_coef:.0f}) ---")
    print(f"Step |  Coef  | Trait | Coherence | Action")
    print("-----|--------|-------|-----------|-------")

    coef = base_coef * 0.7  # Start at 0.7x base
    velocity = 1.0  # Multiplicative velocity for momentum
    history = []

    for step in range(n_steps):
        # Evaluate
        config = {
            "layers": [layer],
            "methods": [method],
            "coefficients": [coef],
            "component": component,
        }

        existing_idx = find_existing_run_index(results, config)
        if existing_idx is not None:
            # Use existing result
            result = results["runs"][existing_idx]["result"]
            trait_score = result.get("trait_mean") or 0
            coherence = result.get("coherence_mean") or 0
            print(f"  {step+1}  | {coef:>6.1f} | {trait_score:>5.1f} | {coherence:>9.1f} | (cached)")
        else:
            result, responses = await evaluate_single_config(
                model, tokenizer, vector, layer, coef,
                questions, trait_name, trait_definition, judge, use_chat_template, component,
                max_new_tokens=max_new_tokens
            )

            trait_score = result.get("trait_mean") or 0
            coherence = result.get("coherence_mean") or 0

            # Save
            timestamp = datetime.now().isoformat()
            save_responses(responses, experiment, trait, position, config, timestamp)

            run_data = {
                "config": config,
                "result": result,
                "timestamp": timestamp,
            }

            results["runs"].append(run_data)
            save_results(results, experiment, trait, position)

            # Print progress
            if coherence < threshold:
                action = f"×{down_mult}"
            else:
                action = f"×{up_mult}"
            if step == n_steps - 1:
                action = "(done)"

            marker = "★" if coherence >= threshold and trait_score > 80 else ""
            print(f"  {step+1}  | {coef:>6.1f} | {trait_score:>5.1f} | {coherence:>9.1f} | {action} {marker}")

        history.append((coef, trait_score, coherence))

        # Decide next coefficient
        direction = up_mult if coherence >= threshold else down_mult

        if momentum > 0:
            # Smooth updates with momentum
            velocity = momentum * velocity + (1 - momentum) * direction
            coef *= velocity
        else:
            # Original behavior: direct multiplicative update
            coef *= direction

    # Report best
    valid = [(c, t, coh) for c, t, coh in history if coh >= threshold]
    if valid:
        best_coef, best_trait, best_coh = max(valid, key=lambda x: x[1])
        print(f"✓ Best: coef={best_coef:.1f} (trait={best_trait:.1f}, coherence={best_coh:.1f})")
    else:
        best_coef, best_trait, best_coh = max(history, key=lambda x: x[2])
        print(f"⚠ No coef met threshold. Best coherence: coef={best_coef:.1f}")


async def batched_adaptive_search(
    model,
    tokenizer,
    layer_data: List[Dict],  # [{layer, vector, base_coef}, ...]
    questions: List[str],
    trait_name: str,
    trait_definition: str,
    judge: TraitJudge,
    use_chat_template: bool,
    component: str,
    results: Dict,
    experiment: str,
    trait: str,
    vector_experiment: str,
    method: str,
    position: str = "response[:]",
    n_steps: int = 8,
    threshold: float = MIN_COHERENCE,
    up_mult: float = 1.3,
    down_mult: float = 0.85,
    max_batch_layers: Optional[int] = None,
    max_new_tokens: int = 256,
    momentum: float = 0.0,  # 0.0 = no momentum, 0.7 = typical momentum
):
    """
    Run adaptive search for multiple layers in parallel batches.

    All layers step together, but each follows its own coefficient trajectory
    based on its coherence results.

    Args:
        momentum: Smoothing factor for coefficient updates (0.0-1.0).
            0.0 = no momentum (original behavior, direct up/down mult)
            0.7 = typical momentum (smooths oscillations between up/down)
            Higher values = more inertia, slower direction changes
    """
    n_questions = len(questions)
    n_layers = len(layer_data)

    # Calculate max layers per batch based on VRAM
    if max_batch_layers is None:
        # Estimate max_seq_len: prompt (~100 tokens) + output
        max_seq_len = 100 + max_new_tokens
        max_batch_size = calculate_max_batch_size(model, max_seq_len, mode='generation')
        max_batch_layers = max(1, max_batch_size // n_questions)

    print(f"\nBatched adaptive search: {n_layers} layers, {n_questions} questions")
    print(f"Max layers per batch: {max_batch_layers}")

    # Format all questions once
    formatted_questions = [
        format_prompt(q, tokenizer, use_chat_template=use_chat_template)
        for q in questions
    ]

    # Initialize state for each layer
    layer_states = []
    for ld in layer_data:
        layer_states.append({
            "layer": ld["layer"],
            "vector": ld["vector"],
            "coef": ld["base_coef"] * 0.7,  # Start at 0.7x base
            "velocity": 1.0,  # Multiplicative velocity for momentum
            "history": [],
            "done": False,
        })

    # Process in batches of layers
    for step in range(n_steps):
        print(f"\n--- Step {step + 1}/{n_steps} ---")

        # Split layers into batches
        active_states = [s for s in layer_states if not s["done"]]
        if not active_states:
            print("All layers done (cached)")
            break

        for batch_start in range(0, len(active_states), max_batch_layers):
            batch_states = active_states[batch_start:batch_start + max_batch_layers]

            # Check which configs already have cached results
            cached_indices = []
            uncached_states = []
            for i, state in enumerate(batch_states):
                config = {
                    "layers": [state["layer"]],
                    "methods": [method],
                    "coefficients": [state["coef"]],
                    "component": component,
                }
                existing_idx = find_existing_run_index(results, config)
                if existing_idx is not None:
                    cached_indices.append((i, existing_idx))
                else:
                    uncached_states.append((i, state))

            # Report cached results
            for i, existing_idx in cached_indices:
                state = batch_states[i]
                result = results["runs"][existing_idx]["result"]
                trait_score = result.get("trait_mean") or 0
                coherence = result.get("coherence_mean") or 0
                state["history"].append((state["coef"], trait_score, coherence))
                print(f"  L{state['layer']:2d} c{state['coef']:>6.1f}: trait={trait_score:5.1f}, coh={coherence:5.1f} (cached)")

            # Generate for uncached configs
            if uncached_states:
                # Build batched prompts: [q1_layer1, q2_layer1, ..., q1_layer2, q2_layer2, ...]
                batched_prompts = []
                for _, state in uncached_states:
                    batched_prompts.extend(formatted_questions)

                # Build steering configs: (layer, vector, coef, (batch_start, batch_end))
                steering_configs = []
                for idx, (_, state) in enumerate(uncached_states):
                    batch_slice_start = idx * n_questions
                    batch_slice_end = (idx + 1) * n_questions
                    steering_configs.append((
                        state["layer"],
                        state["vector"],
                        state["coef"],
                        (batch_slice_start, batch_slice_end)
                    ))

                # Generate all at once
                print(f"  Generating {len(batched_prompts)} responses ({len(uncached_states)} layers × {n_questions} questions)...", end=" ", flush=True)
                t0 = time.time()
                with BatchedLayerSteeringHook(model, steering_configs, component=component):
                    all_responses = generate_batch(model, tokenizer, batched_prompts, max_new_tokens=max_new_tokens)
                gen_time = time.time() - t0
                print(f"({gen_time:.1f}s)")

                # Score all responses
                all_qa_pairs = []
                for idx, (_, state) in enumerate(uncached_states):
                    start = idx * n_questions
                    end = (idx + 1) * n_questions
                    for q, r in zip(questions, all_responses[start:end]):
                        all_qa_pairs.append((q, r))

                print(f"  Scoring {len(all_qa_pairs)} responses...", end=" ", flush=True)
                t0 = time.time()
                all_scores = await judge.score_steering_batch(all_qa_pairs, trait_name, trait_definition)
                score_time = time.time() - t0
                print(f"({score_time:.1f}s)")

                # Process results per layer
                for idx, (_, state) in enumerate(uncached_states):
                    start = idx * n_questions
                    end = (idx + 1) * n_questions
                    layer_scores = all_scores[start:end]
                    layer_responses = all_responses[start:end]

                    trait_scores = [s["trait_score"] for s in layer_scores if s["trait_score"] is not None]
                    coherence_scores = [s["coherence_score"] for s in layer_scores if s.get("coherence_score") is not None]

                    trait_mean = sum(trait_scores) / len(trait_scores) if trait_scores else 0
                    coherence_mean = sum(coherence_scores) / len(coherence_scores) if coherence_scores else 0

                    state["history"].append((state["coef"], trait_mean, coherence_mean))

                    # Save results
                    config = {
                        "layers": [state["layer"]],
                        "methods": [method],
                        "coefficients": [state["coef"]],
                        "component": component,
                    }
                    result = {
                        "trait_mean": trait_mean,
                        "coherence_mean": coherence_mean,
                        "n": len(trait_scores),
                    }
                    timestamp = datetime.now().isoformat()

                    responses_data = [
                        {"question": q, "response": r, "trait_score": s["trait_score"], "coherence_score": s.get("coherence_score")}
                        for q, r, s in zip(questions, layer_responses, layer_scores)
                    ]
                    save_responses(responses_data, experiment, trait, position, config, timestamp)

                    run_data = {
                        "config": config,
                        "result": result,
                        "timestamp": timestamp,
                    }
                    results["runs"].append(run_data)

                    marker = "★" if coherence_mean >= threshold and trait_mean > 80 else ""
                    print(f"  L{state['layer']:2d} c{state['coef']:>6.1f}: trait={trait_mean:5.1f}, coh={coherence_mean:5.1f} {marker}")

                # Save after each batch
                save_results(results, experiment, trait, position)

        # Update coefficients for next step
        # Binary control: push up while coherence >= threshold, back off when below
        for state in layer_states:
            if state["history"]:
                _, _, last_coherence = state["history"][-1]
                direction = up_mult if last_coherence >= threshold else down_mult

                if momentum > 0:
                    # Smooth updates with momentum
                    state["velocity"] = momentum * state["velocity"] + (1 - momentum) * direction
                    state["coef"] *= state["velocity"]
                else:
                    # Direct multiplicative update
                    state["coef"] *= direction

    # Report best per layer
    print(f"\n{'='*60}")
    print("Best per layer:")
    print(f"{'='*60}")
    for state in layer_states:
        valid = [(c, t, coh) for c, t, coh in state["history"] if coh >= threshold]
        if valid:
            best_coef, best_trait, best_coh = max(valid, key=lambda x: x[1])
            print(f"  L{state['layer']:2d}: coef={best_coef:.1f}, trait={best_trait:.1f}, coh={best_coh:.1f}")
        else:
            if state["history"]:
                best_coef, best_trait, best_coh = max(state["history"], key=lambda x: x[2])
                print(f"  L{state['layer']:2d}: coef={best_coef:.1f} (no valid, best coh={best_coh:.1f})")
            else:
                print(f"  L{state['layer']:2d}: no results")
