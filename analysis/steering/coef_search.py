"""
Coefficient search for steering evaluation.

Input:
    - backend: GenerationBackend instance with model access
    - layer_data: Layer info with vectors and base coefficients
    - questions: Evaluation questions
    - judge: TraitJudge instance

Output:
    - Finds optimal coefficients via adaptive search
    - Saves results incrementally

Usage:
    from analysis.steering.coef_search import adaptive_search_layer, batched_adaptive_search
"""

import gc
import time
from typing import Dict, List, Literal, Optional, Tuple, TYPE_CHECKING

import torch
from datetime import datetime
from tqdm import tqdm

from core import SteeringHook, get_hook_path, VectorSpec, batched_steering_generate, multi_trait_batched_steering_generate
from utils.generation import generate_batch, calculate_max_batch_size
from analysis.steering.results import append_run, save_responses, find_cached_run, is_better_result
from utils.judge import TraitJudge
from utils.model import format_prompt, tokenize_batch
from utils.vectors import MIN_COHERENCE

if TYPE_CHECKING:
    from core import GenerationBackend


async def evaluate_single_config(
    backend: "GenerationBackend",
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
    eval_prompt: Optional[str] = None,
    relevance_check: bool = True,
) -> Tuple[Dict, List[Dict]]:
    """Evaluate a single (layer, coefficient) config with batched generation.

    Uses escape hatch: backend.model for SteeringHook.
    """
    # Access model and tokenizer for hooks and generation
    model = backend.model
    tokenizer = backend.tokenizer

    desc = f"L{layer} c{coef:.0f}"

    # Format all questions
    formatted = [format_prompt(q, tokenizer, use_chat_template=use_chat_template) for q in questions]

    # Generate all responses in batch with steering (escape hatch: hook needs direct model access)
    print(f"  Generating {len(questions)} responses for {desc}...")
    with SteeringHook(model, vector, get_hook_path(layer, component, model=model), coefficient=coef):
        responses = generate_batch(model, tokenizer, formatted, max_new_tokens=max_new_tokens)

    all_qa_pairs = list(zip(questions, responses))

    # Score
    print(f"  Scoring {len(all_qa_pairs)} responses...")
    all_scores = await judge.score_steering_batch(all_qa_pairs, trait_name, trait_definition, eval_prompt=eval_prompt, relevance_check=relevance_check)

    trait_scores = [s["trait_score"] for s in all_scores if s["trait_score"] is not None]
    coherence_scores = [s["coherence_score"] for s in all_scores if s.get("coherence_score") is not None]

    result = {
        "trait_mean": sum(trait_scores) / len(trait_scores) if trait_scores else None,
        "coherence_mean": sum(coherence_scores) / len(coherence_scores) if coherence_scores else None,
        "n": len(trait_scores),
    }

    responses = [
        {"prompt": q, "response": r, "system_prompt": None, "trait_score": s["trait_score"], "coherence_score": s.get("coherence_score")}
        for (q, r), s in zip(all_qa_pairs, all_scores)
    ]

    trait_str = f"{result['trait_mean']:.1f}" if result['trait_mean'] else "N/A"
    coh_str = f"{result['coherence_mean']:.1f}" if result['coherence_mean'] else "N/A"
    print(f"  {desc}: trait={trait_str}, coherence={coh_str}, n={result['n']}")

    return result, responses


async def adaptive_search_layer(
    backend: "GenerationBackend", vector, layer, base_coef,
    questions, trait_name, trait_definition, judge, use_chat_template, component,
    cached_runs, experiment, trait, model_variant, vector_experiment, method,
    position: str = "response[:]",
    prompt_set: str = "steering",
    n_steps: int = 8,
    threshold: float = MIN_COHERENCE,
    up_mult: float = 1.3,
    down_mult: float = 0.85,
    start_mult: float = 0.7,
    momentum: float = 0.0,  # 0.0 = no momentum, 0.7 = typical momentum
    max_new_tokens: int = 256,
    eval_prompt: Optional[str] = None,
    save_mode: str = "best",
    coherence_threshold: float = MIN_COHERENCE,
    relevance_check: bool = True,
    direction: Literal["positive", "negative"] = "positive",
    trait_judge: Optional[str] = None,
):
    """Run adaptive search for a single layer, saving each result.

    Uses evaluate_single_config which uses escape hatch for SteeringHook.

    Args:
        direction: "positive" for inducing trait (coef > 0), "negative" for suppressing (coef < 0)
    """
    sign = 1 if direction == "positive" else -1
    print(f"\n--- Layer {layer} (base_coef={base_coef:.0f}, direction={direction}) ---")
    print(f"Step |  Coef  | Trait | Coherence | Action")
    print("-----|--------|-------|-----------|-------")

    coef = base_coef * start_mult * sign  # Start with direction-appropriate sign
    velocity = 1.0  # Multiplicative velocity for momentum
    history = []
    best_for_layer = None  # Track best for save_mode="best"

    for step in range(n_steps):
        # Evaluate
        spec = VectorSpec(layer=layer, component=component, position=position, method=method, weight=coef)
        config = {"vectors": [spec.to_dict()]}

        cached_result = find_cached_run(cached_runs, config)
        if cached_result is not None:
            # Use existing result
            trait_score = cached_result.get("trait_mean") or 0
            coherence = cached_result.get("coherence_mean") or 0
            print(f"  {step+1}  | {coef:>6.1f} | {trait_score:>5.1f} | {coherence:>9.1f} | (cached)")
        else:
            result, responses = await evaluate_single_config(
                backend, vector, layer, coef,
                questions, trait_name, trait_definition, judge, use_chat_template, component,
                max_new_tokens=max_new_tokens, eval_prompt=eval_prompt,
                relevance_check=relevance_check
            )

            trait_score = result.get("trait_mean") or 0
            coherence = result.get("coherence_mean") or 0

            timestamp = datetime.now().isoformat()

            # Always append to results.jsonl
            append_run(experiment, trait, model_variant, config, result, position, prompt_set, trait_judge=trait_judge)
            cached_runs.append({"config": config, "result": result, "timestamp": timestamp})

            # Handle response saving based on save_mode
            if save_mode == "all":
                save_responses(responses, experiment, trait, model_variant, position, prompt_set, config, timestamp)
            elif save_mode == "best":
                # Track best: direction-aware comparison
                if is_better_result(best_for_layer, trait_score, coherence, coherence_threshold, direction):
                    best_for_layer = {
                        "trait_mean": trait_score,
                        "coherence_mean": coherence,
                        "valid": coherence >= coherence_threshold,
                        "responses": responses,
                        "config": config,
                        "timestamp": timestamp,
                    }
            # save_mode == "none": don't save responses

            # Print progress
            if coherence < threshold:
                action = f"×{down_mult}"
            else:
                action = f"×{up_mult}"
            if step == n_steps - 1:
                action = "(done)"

            print(f"  {step+1}  | {coef:>6.1f} | {trait_score:>5.1f} | {coherence:>9.1f} | {action}")

        history.append((coef, trait_score, coherence))

        # Decide next coefficient
        mult = up_mult if coherence >= threshold else down_mult

        if momentum > 0:
            # Smooth updates with momentum
            velocity = momentum * velocity + (1 - momentum) * mult
            coef *= velocity
        else:
            # Original behavior: direct multiplicative update
            coef *= mult

    # Save best for this layer (if tracking)
    if save_mode == "best" and best_for_layer and best_for_layer.get("responses"):
        save_responses(
            best_for_layer["responses"], experiment, trait, model_variant,
            position, prompt_set, best_for_layer["config"], best_for_layer["timestamp"]
        )

    # Report best (direction-aware: positive maximizes trait, negative minimizes)
    valid = [(c, t, coh) for c, t, coh in history if coh >= threshold]
    if valid:
        best_coef, best_trait, best_coh = max(valid, key=lambda x: x[1] * sign)
        print(f"✓ Best: coef={best_coef:.1f} (trait={best_trait:.1f}, coherence={best_coh:.1f})")
    else:
        best_coef, best_trait, best_coh = max(history, key=lambda x: x[2])
        print(f"⚠ No coef met threshold. Best coherence: coef={best_coef:.1f}")


async def batched_adaptive_search(
    backend: "GenerationBackend",
    layer_data: List[Dict],  # [{layer, vector, base_coef}, ...]
    questions: List[str],
    trait_name: str,
    trait_definition: str,
    judge: TraitJudge,
    use_chat_template: bool,
    component: str,
    cached_runs: List[Dict],
    experiment: str,
    trait: str,
    model_variant: str,
    vector_experiment: str,
    method: str,
    position: str = "response[:]",
    prompt_set: str = "steering",
    n_steps: int = 8,
    threshold: float = MIN_COHERENCE,
    up_mult: float = 1.3,
    down_mult: float = 0.85,
    start_mult: float = 0.7,
    max_batch_layers: Optional[int] = None,
    max_new_tokens: int = 256,
    momentum: float = 0.0,  # 0.0 = no momentum, 0.7 = typical momentum
    eval_prompt: Optional[str] = None,
    save_mode: str = "best",
    coherence_threshold: float = MIN_COHERENCE,
    relevance_check: bool = True,
    direction: Literal["positive", "negative"] = "positive",
    trait_judge: Optional[str] = None,
):
    """
    Run adaptive search for multiple layers in parallel batches.

    All layers step together, but each follows its own coefficient trajectory
    based on its coherence results.

    Uses escape hatch: backend.model for batched_steering_generate.

    Args:
        backend: GenerationBackend with model access
        cached_runs: List of previously evaluated runs (for resume capability)
        momentum: Smoothing factor for coefficient updates (0.0-1.0).
            0.0 = no momentum (original behavior, direct up/down mult)
            0.7 = typical momentum (smooths oscillations between up/down)
            Higher values = more inertia, slower direction changes
        direction: "positive" for inducing trait (coef > 0), "negative" for suppressing (coef < 0)
    """
    sign = 1 if direction == "positive" else -1

    # Access model and tokenizer for hooks and generation
    model = backend.model
    tokenizer = backend.tokenizer
    n_questions = len(questions)
    n_layers = len(layer_data)

    # Format all questions once (moved up from below)
    formatted_questions = [
        format_prompt(q, tokenizer, use_chat_template=use_chat_template)
        for q in questions
    ]

    # Calculate max layers per batch based on VRAM
    if max_batch_layers is None:
        if not formatted_questions:
            raise ValueError("No questions to tokenize. Cannot calculate batch size.")

        # Calculate max_seq_len from actual tokenized questions (single batch call)
        batch_result = tokenize_batch(formatted_questions, tokenizer)
        max_prompt_len = max(batch_result['lengths'])

        max_seq_len = max_prompt_len + max_new_tokens
        max_batch_size = calculate_max_batch_size(model, max_seq_len, mode='generation')
        max_batch_layers = max(1, max_batch_size // n_questions)
        print(f"\nBatched adaptive search: {n_layers} layers, {n_questions} questions, direction={direction}")
        print(f"Max layers per batch: {max_batch_layers} (max_prompt_len={max_prompt_len}, max_new_tokens={max_new_tokens}, max_batch_size={max_batch_size})")
    else:
        print(f"\nBatched adaptive search: {n_layers} layers, {n_questions} questions, direction={direction}")
        print(f"Max layers per batch: {max_batch_layers} (user-specified)")

    # Initialize state for each layer
    layer_states = []
    for ld in layer_data:
        layer_states.append({
            "layer": ld["layer"],
            "vector": ld["vector"],
            "coef": ld["base_coef"] * start_mult * sign,  # Start with direction-appropriate sign
            "velocity": 1.0,  # Multiplicative velocity for momentum
            "history": [],
            "done": False,
            "best_result": None,  # Track best for save_mode="best"
            "best_responses": None,
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
            cached_configs = []
            uncached_states = []
            for i, state in enumerate(batch_states):
                spec = VectorSpec(layer=state["layer"], component=component, position=position, method=method, weight=state["coef"])
                config = {"vectors": [spec.to_dict()]}
                cached_result = find_cached_run(cached_runs, config)
                if cached_result is not None:
                    cached_configs.append((i, cached_result))
                else:
                    uncached_states.append((i, state))

            # Report cached results
            for i, cached_result in cached_configs:
                state = batch_states[i]
                trait_score = cached_result.get("trait_mean") or 0
                coherence = cached_result.get("coherence_mean") or 0
                state["history"].append((state["coef"], trait_score, coherence))
                print(f"  L{state['layer']:2d} c{state['coef']:>6.1f}: trait={trait_score:5.1f}, coh={coherence:5.1f} (cached)")

            # Generate for uncached configs
            if uncached_states:
                # Build configs for batched generation: (layer, vector, coef)
                generation_configs = [(state["layer"], state["vector"], state["coef"]) for _, state in uncached_states]

                # Generate all at once using primitive
                print(f"  Generating {len(uncached_states) * n_questions} responses ({len(uncached_states)} layers × {n_questions} questions)...", end=" ", flush=True)
                t0 = time.time()
                all_responses = batched_steering_generate(
                    model, tokenizer, formatted_questions, generation_configs,
                    component=component, max_new_tokens=max_new_tokens
                )
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
                all_scores = await judge.score_steering_batch(all_qa_pairs, trait_name, trait_definition, eval_prompt=eval_prompt, relevance_check=relevance_check)
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
                    spec = VectorSpec(layer=state["layer"], component=component, position=position, method=method, weight=state["coef"])
                    config = {"vectors": [spec.to_dict()]}
                    result = {
                        "trait_mean": trait_mean,
                        "coherence_mean": coherence_mean,
                        "n": len(trait_scores),
                    }
                    timestamp = datetime.now().isoformat()

                    responses_data = [
                        {"prompt": q, "response": r, "system_prompt": None, "trait_score": s["trait_score"], "coherence_score": s.get("coherence_score")}
                        for q, r, s in zip(questions, layer_responses, layer_scores)
                    ]

                    # Always append to JSONL
                    append_run(experiment, trait, model_variant, config, result, position, prompt_set, trait_judge=trait_judge)
                    cached_runs.append({"config": config, "result": result, "timestamp": timestamp})

                    # Handle response saving based on save_mode
                    if save_mode == "all":
                        save_responses(responses_data, experiment, trait, model_variant, position, prompt_set, config, timestamp)
                    elif save_mode == "best":
                        # Track best: direction-aware comparison
                        if is_better_result(state.get("best_result"), trait_mean, coherence_mean, coherence_threshold, direction):
                            state["best_result"] = {
                                "trait_mean": trait_mean,
                                "coherence_mean": coherence_mean,
                                "valid": coherence_mean >= coherence_threshold,
                                "config": config,
                                "timestamp": timestamp,
                            }
                            state["best_responses"] = responses_data
                    # save_mode == "none": don't save responses

                    print(f"  L{state['layer']:2d} c{state['coef']:>6.1f}: trait={trait_mean:5.1f}, coh={coherence_mean:5.1f}")

                # Free memory after batch
                del all_responses, all_qa_pairs, all_scores
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # Update coefficients for next step
        # Binary control: push up while coherence >= threshold, back off when below
        for state in layer_states:
            if state["history"]:
                _, _, last_coherence = state["history"][-1]
                mult = up_mult if last_coherence >= threshold else down_mult

                if momentum > 0:
                    # Smooth updates with momentum
                    state["velocity"] = momentum * state["velocity"] + (1 - momentum) * mult
                    state["coef"] *= state["velocity"]
                else:
                    # Direct multiplicative update
                    state["coef"] *= mult

    # Save best responses for each layer (if tracking)
    if save_mode == "best":
        for state in layer_states:
            if state.get("best_responses") and state.get("best_result"):
                save_responses(
                    state["best_responses"], experiment, trait, model_variant,
                    position, prompt_set, state["best_result"]["config"], state["best_result"]["timestamp"]
                )

    # Report best per layer (direction-aware: positive maximizes trait, negative minimizes)
    print(f"\n{'='*60}")
    print("Best per layer:")
    print(f"{'='*60}")
    for state in layer_states:
        valid = [(c, t, coh) for c, t, coh in state["history"] if coh >= threshold]
        if valid:
            best_coef, best_trait, best_coh = max(valid, key=lambda x: x[1] * sign)
            print(f"  L{state['layer']:2d}: coef={best_coef:.1f}, trait={best_trait:.1f}, coh={best_coh:.1f}")
        else:
            if state["history"]:
                best_coef, best_trait, best_coh = max(state["history"], key=lambda x: x[2])
                print(f"  L{state['layer']:2d}: coef={best_coef:.1f} (no valid, best coh={best_coh:.1f})")
            else:
                print(f"  L{state['layer']:2d}: no results")


def _group_configs_by_batch_size(states: List[Dict], max_batch_size: int) -> List[List[Dict]]:
    """Group config states into sub-batches where total prompts <= max_batch_size."""
    batches = []
    current_batch = []
    current_count = 0

    for state in states:
        n_q = len(state["formatted_questions"])
        if current_count + n_q > max_batch_size and current_batch:
            batches.append(current_batch)
            current_batch = []
            current_count = 0
        current_batch.append(state)
        current_count += n_q

    if current_batch:
        batches.append(current_batch)

    return batches


async def _score_multi_trait_batch(
    judge: TraitJudge,
    all_qa_pairs: List[Tuple[str, str]],
    pair_traits: List[Dict],
    relevance_check: bool,
) -> List[Dict]:
    """Score QA pairs grouped by trait for efficient judge routing.

    Groups pairs by (trait_name, trait_definition, eval_prompt), scores each group
    via judge.score_steering_batch, and reassembles in original order.
    """
    from collections import defaultdict

    groups = defaultdict(list)
    for idx, (qa, trait_ctx) in enumerate(zip(all_qa_pairs, pair_traits)):
        key = (trait_ctx["trait_name"], trait_ctx["trait_definition"], trait_ctx.get("eval_prompt"))
        groups[key].append((idx, qa))

    results = [None] * len(all_qa_pairs)
    for (trait_name, trait_definition, eval_prompt), items in groups.items():
        indices, pairs = zip(*items)
        scores = await judge.score_steering_batch(
            list(pairs), trait_name, trait_definition,
            eval_prompt=eval_prompt, relevance_check=relevance_check,
        )
        for idx, score in zip(indices, scores):
            results[idx] = score

    return results


def _process_config_result(
    state: Dict,
    scores: List[Dict],
    responses: List[str],
    component: str,
    position: str,
    method: str,
    model_variant: str,
    prompt_set: str,
    save_mode: str,
    coherence_threshold: float,
    direction: str,
    trait_judge: Optional[str],
):
    """Process scores for a single config state — compute means, save results, track best."""
    questions = state["questions"]

    trait_scores = [s["trait_score"] for s in scores if s["trait_score"] is not None]
    coherence_scores = [s["coherence_score"] for s in scores if s.get("coherence_score") is not None]

    trait_mean = sum(trait_scores) / len(trait_scores) if trait_scores else 0
    coherence_mean = sum(coherence_scores) / len(coherence_scores) if coherence_scores else 0

    state["history"].append((state["coef"], trait_mean, coherence_mean))

    spec = VectorSpec(layer=state["layer"], component=component, position=position, method=method, weight=state["coef"])
    config = {"vectors": [spec.to_dict()]}
    result = {"trait_mean": trait_mean, "coherence_mean": coherence_mean, "n": len(trait_scores)}
    timestamp = datetime.now().isoformat()

    responses_data = [
        {"prompt": q, "response": r, "system_prompt": None,
         "trait_score": s["trait_score"], "coherence_score": s.get("coherence_score")}
        for q, r, s in zip(questions, responses, scores)
    ]

    # Save to trait-specific JSONL
    append_run(state["experiment"], state["trait"], model_variant, config, result, position, prompt_set, trait_judge=trait_judge)
    state["cached_runs"].append({"config": config, "result": result, "timestamp": timestamp})

    if save_mode == "all":
        save_responses(responses_data, state["experiment"], state["trait"], model_variant, position, prompt_set, config, timestamp)
    elif save_mode == "best":
        if is_better_result(state.get("best_result"), trait_mean, coherence_mean, coherence_threshold, direction):
            state["best_result"] = {
                "trait_mean": trait_mean, "coherence_mean": coherence_mean,
                "valid": coherence_mean >= coherence_threshold,
                "config": config, "timestamp": timestamp,
            }
            state["best_responses"] = responses_data

    return trait_mean, coherence_mean


async def multi_trait_batched_adaptive_search(
    backend: "GenerationBackend",
    trait_configs: List[Dict],
    judge: TraitJudge,
    use_chat_template: bool,
    component: str,
    model_variant: str,
    method: str,
    position: str = "response[:]",
    prompt_set: str = "steering",
    n_steps: int = 8,
    threshold: float = MIN_COHERENCE,
    up_mult: float = 1.3,
    down_mult: float = 0.85,
    start_mult: float = 0.7,
    momentum: float = 0.0,
    max_new_tokens: int = 256,
    save_mode: str = "best",
    coherence_threshold: float = MIN_COHERENCE,
    relevance_check: bool = True,
    direction: Literal["positive", "negative"] = "positive",
    trait_judge: Optional[str] = None,
):
    """
    Run adaptive search for multiple traits × layers in parallel batches.

    Like batched_adaptive_search but the config space is trait×layer instead of
    just layer. Each config independently follows its own coefficient trajectory.
    Different traits can have different question sets (heterogeneous batch sizes).

    Args:
        trait_configs: List of dicts, one per trait, each containing:
            trait, trait_name, trait_definition, eval_prompt, questions,
            formatted_questions, layer_data, cached_runs, experiment, vector_experiment
    """
    sign = 1 if direction == "positive" else -1
    model = backend.model
    tokenizer = backend.tokenizer

    # Initialize config states: one per (trait, layer) pair
    config_states = []
    for tc in trait_configs:
        for ld in tc["layer_data"]:
            config_states.append({
                "layer": ld["layer"],
                "vector": ld["vector"],
                "coef": ld["base_coef"] * start_mult * sign,
                "velocity": 1.0,
                "history": [],
                "done": False,
                "best_result": None,
                "best_responses": None,
                # Trait context
                "trait": tc["trait"],
                "trait_name": tc["trait_name"],
                "trait_definition": tc["trait_definition"],
                "eval_prompt": tc["eval_prompt"],
                "questions": tc["questions"],
                "formatted_questions": tc["formatted_questions"],
                "cached_runs": tc["cached_runs"],
                "experiment": tc["experiment"],
                "vector_experiment": tc["vector_experiment"],
            })

    n_configs = len(config_states)
    n_traits = len(trait_configs)
    total_questions = sum(len(tc["questions"]) for tc in trait_configs)

    # Calculate max batch size
    all_formatted = [q for tc in trait_configs for q in tc["formatted_questions"]]
    if not all_formatted:
        raise ValueError("No questions across any trait. Cannot run adaptive search.")

    batch_result = tokenize_batch(all_formatted, tokenizer)
    max_prompt_len = max(batch_result['lengths'])
    max_seq_len = max_prompt_len + max_new_tokens
    max_batch_size = calculate_max_batch_size(model, max_seq_len, mode='generation')

    print(f"\nMulti-trait batched adaptive search: {n_traits} traits, {n_configs} configs ({n_traits}×layers), direction={direction}")
    print(f"Total questions per full step: {total_questions * len(trait_configs[0]['layer_data'])} | max_batch_size={max_batch_size}")

    for step in range(n_steps):
        active_states = [s for s in config_states if not s["done"]]
        if not active_states:
            print("All configs done (cached)")
            break

        # Split cached vs uncached
        uncached_states = []
        for state in active_states:
            spec = VectorSpec(layer=state["layer"], component=component, position=position, method=method, weight=state["coef"])
            config = {"vectors": [spec.to_dict()]}
            cached_result = find_cached_run(state["cached_runs"], config)
            if cached_result is not None:
                trait_score = cached_result.get("trait_mean") or 0
                coherence = cached_result.get("coherence_mean") or 0
                state["history"].append((state["coef"], trait_score, coherence))
            else:
                uncached_states.append(state)

        n_cached = len(active_states) - len(uncached_states)
        print(f"\n--- Step {step + 1}/{n_steps}: {len(active_states)} active, {n_cached} cached, {len(uncached_states)} to generate ---")

        if uncached_states:
            batches = _group_configs_by_batch_size(uncached_states, max_batch_size)

            for batch_idx, batch_states in enumerate(batches):
                batch_prompts = sum(len(s["formatted_questions"]) for s in batch_states)
                batch_traits = len(set(s["trait"] for s in batch_states))
                print(f"  Batch {batch_idx+1}/{len(batches)}: {len(batch_states)} configs, {batch_prompts} prompts, {batch_traits} traits...", end=" ", flush=True)

                # Generate with per-config prompts
                t0 = time.time()
                generation_configs = [
                    (s["layer"], s["vector"], s["coef"], s["formatted_questions"])
                    for s in batch_states
                ]
                per_config_responses = multi_trait_batched_steering_generate(
                    model, tokenizer, generation_configs,
                    component=component, max_new_tokens=max_new_tokens,
                )
                gen_time = time.time() - t0
                print(f"gen={gen_time:.1f}s", end=" ", flush=True)

                # Build QA pairs with trait context for per-trait judge routing
                all_qa_pairs = []
                pair_traits = []
                for state, responses in zip(batch_states, per_config_responses):
                    for q, r in zip(state["questions"], responses):
                        all_qa_pairs.append((q, r))
                        pair_traits.append(state)

                t0 = time.time()
                all_scores = await _score_multi_trait_batch(judge, all_qa_pairs, pair_traits, relevance_check)
                score_time = time.time() - t0
                print(f"score={score_time:.1f}s")

                # Slice scores back per config and process
                offset = 0
                for state, responses in zip(batch_states, per_config_responses):
                    n_q = len(state["questions"])
                    state_scores = all_scores[offset:offset + n_q]
                    offset += n_q

                    trait_mean, coherence_mean = _process_config_result(
                        state, state_scores, responses, component, position,
                        method, model_variant, prompt_set, save_mode,
                        coherence_threshold, direction, trait_judge,
                    )
                    print(f"    {state['trait']} L{state['layer']:2d} c{state['coef']:>6.1f}: trait={trait_mean:5.1f}, coh={coherence_mean:5.1f}")

                # Free memory
                del per_config_responses, all_qa_pairs, all_scores
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # Update coefficients
        for state in config_states:
            if state["history"]:
                _, _, last_coherence = state["history"][-1]
                mult = up_mult if last_coherence >= threshold else down_mult
                if momentum > 0:
                    state["velocity"] = momentum * state["velocity"] + (1 - momentum) * mult
                    state["coef"] *= state["velocity"]
                else:
                    state["coef"] *= mult

    # Save best responses per config
    if save_mode == "best":
        for state in config_states:
            if state.get("best_responses") and state.get("best_result"):
                save_responses(
                    state["best_responses"], state["experiment"], state["trait"],
                    model_variant, position, prompt_set,
                    state["best_result"]["config"], state["best_result"]["timestamp"],
                )

    # Report best per trait×layer
    print(f"\n{'='*60}")
    print("Best per trait×layer:")
    print(f"{'='*60}")
    current_trait = None
    for state in config_states:
        if state["trait"] != current_trait:
            current_trait = state["trait"]
            print(f"\n  {current_trait}:")
        valid = [(c, t, coh) for c, t, coh in state["history"] if coh >= threshold]
        if valid:
            best_coef, best_trait, best_coh = max(valid, key=lambda x: x[1] * sign)
            print(f"    L{state['layer']:2d}: coef={best_coef:.1f}, trait={best_trait:.1f}, coh={best_coh:.1f}")
        elif state["history"]:
            best_coef, best_trait, best_coh = max(state["history"], key=lambda x: x[2])
            print(f"    L{state['layer']:2d}: coef={best_coef:.1f} (no valid, best coh={best_coh:.1f})")
        else:
            print(f"    L{state['layer']:2d}: no results")
