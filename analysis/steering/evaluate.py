#!/usr/bin/env python3
"""
Steering evaluation - validate trait vectors via causal intervention.

Input:
    - experiment: Experiment where steering results are saved
    - vector-from-trait OR traits: Single or multiple trait specs

Output:
    - experiments/{experiment}/steering/{trait}/{model_variant}/{position}/{prompt_set}/results.jsonl
    - experiments/{experiment}/steering/{trait}/{model_variant}/{position}/{prompt_set}/responses/{component}/{method}/

Usage:
    # Basic usage - adaptive search finds good coefficients
    python analysis/steering/evaluate.py \\
        --experiment {experiment} \\
        --vector-from-trait {experiment}/{category}/{trait}

    # Multiple traits (loads model once, evaluates sequentially)
    python analysis/steering/evaluate.py \\
        --experiment {experiment} \\
        --traits "exp/cat/trait1,exp/cat/trait2,exp/cat/trait3" \\
        --load-in-8bit

    # Specific layers only
    python analysis/steering/evaluate.py \\
        --experiment {experiment} \\
        --vector-from-trait {experiment}/{category}/{trait} \\
        --layers 10,12,14

    # 70B+ models with quantization
    python analysis/steering/evaluate.py \\
        --experiment {experiment} \\
        --vector-from-trait {experiment}/{category}/{trait} \\
        --load-in-8bit

    # Sequential mode (one layer at a time, slower but lower memory)
    python analysis/steering/evaluate.py \\
        --experiment {experiment} \\
        --vector-from-trait {experiment}/{category}/{trait} \\
        --no-batch

    # Manual coefficients (skip adaptive search)
    python analysis/steering/evaluate.py \\
        --experiment {experiment} \\
        --vector-from-trait {experiment}/{category}/{trait} \\
        --coefficients 50,100,150
"""

import sys
import gc
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import argparse
import asyncio
import json
from typing import List, Dict, Optional
from datetime import datetime

from analysis.steering.data import load_steering_data, load_questions_from_inference
from analysis.steering.results import (
    init_results_file, load_results, append_baseline,
    save_baseline_responses, save_ablation_responses, find_cached_run, append_run, save_responses,
    is_better_result,
)
from utils.paths import get_steering_results_path
from analysis.steering.coef_search import (
    evaluate_single_config,
    adaptive_search_layer,
    batched_adaptive_search,
)
from core import VectorSpec, MultiLayerAblationHook, LocalBackend, GenerationConfig
from utils.generation import generate_batch
from utils.judge import TraitJudge
from utils.paths import get, get_vector_path, get_default_variant
from utils.model import format_prompt, tokenize_prompt, load_model_with_lora, get_layers_module
from utils.paths import get_model_variant, load_experiment_config
from utils.vectors import MIN_COHERENCE, load_vector, load_cached_activation_norms


def parse_layers(layers_arg: str, num_layers: int) -> List[int]:
    """
    Parse layers argument.

    Args:
        layers_arg: "all", single number "16", range "5-20", list "5,10,15",
                    or percentage range "30%-60%"
        num_layers: Total layers in model

    Returns:
        List of layer indices
    """
    if layers_arg.lower() == "all":
        return list(range(num_layers))
    elif "%" in layers_arg:
        # Percentage range: "30%-60%" -> layers at 30% to 60% of depth
        parts = layers_arg.replace("%", "").split("-")
        start_pct = int(parts[0]) / 100
        end_pct = int(parts[1]) / 100 if len(parts) > 1 else start_pct
        start = int(num_layers * start_pct)
        end = int(num_layers * end_pct)
        return list(range(start, end + 1))
    elif "-" in layers_arg and "," not in layers_arg:
        start, end = layers_arg.split("-")
        return list(range(int(start), int(end) + 1))
    elif "," in layers_arg:
        return [int(x) for x in layers_arg.split(",")]
    else:
        return [int(layers_arg)]


def parse_coefficients(coef_arg: Optional[str]) -> Optional[List[float]]:
    """Parse comma-separated coefficients. Returns None if not provided."""
    if coef_arg is None:
        return None
    return [float(c) for c in coef_arg.split(",")]


def estimate_activation_norm(
    model,
    tokenizer,
    prompts: List[str],
    layer: int,
    use_chat_template: bool,
) -> float:
    """Estimate activation norm at a layer by running a few prompts."""
    norms = []

    def capture_hook(_module, _input, output):
        hidden = output[0] if isinstance(output, tuple) else output
        norm = hidden[:, -1, :].float().norm().item()
        norms.append(norm)

    layer_module = get_layers_module(model)[layer]
    handle = layer_module.register_forward_hook(capture_hook)

    try:
        for prompt in prompts[:3]:
            formatted = format_prompt(prompt, tokenizer, use_chat_template=use_chat_template)
            inputs = tokenize_prompt(formatted, tokenizer).to(model.device)
            with torch.no_grad():
                model(**inputs)
    finally:
        handle.remove()

    return sum(norms) / len(norms) if norms else 100.0


# =============================================================================
# Core Evaluation
# =============================================================================

async def compute_baseline(
    backend,
    questions: List[str],
    trait_name: str,
    trait_definition: str,
    judge: TraitJudge,
    max_new_tokens: int = 256,
    eval_prompt: Optional[str] = None,
    relevance_check: bool = True,
) -> tuple[Dict, List[Dict]]:
    """Compute baseline scores (no steering) with batched generation.

    Returns:
        Tuple of (baseline_stats, response_data) where response_data contains
        questions, responses, and scores for saving.
    """
    print("\nComputing baseline (no steering)...")

    # Generate all responses in batch (backend handles formatting)
    responses = backend.generate(questions, config=GenerationConfig(max_new_tokens=max_new_tokens))

    # Score all at once
    all_qa_pairs = list(zip(questions, responses))
    all_scores = await judge.score_steering_batch(all_qa_pairs, trait_name, trait_definition, eval_prompt=eval_prompt, relevance_check=relevance_check)

    all_trait_scores = [s["trait_score"] for s in all_scores if s["trait_score"] is not None]
    all_coherence_scores = [s["coherence_score"] for s in all_scores if s.get("coherence_score") is not None]

    baseline = {
        "trait_mean": sum(all_trait_scores) / len(all_trait_scores) if all_trait_scores else None,
        "n": len(all_trait_scores),
    }

    if all_coherence_scores:
        baseline["coherence_mean"] = sum(all_coherence_scores) / len(all_coherence_scores)

    # Build response data for saving
    response_data = [
        {
            "prompt": q,
            "response": r,
            "system_prompt": None,
            "trait_score": s["trait_score"],
            "coherence_score": s.get("coherence_score"),
        }
        for q, r, s in zip(questions, responses, all_scores)
    ]

    print(f"  Baseline: trait={baseline['trait_mean']:.1f}, n={baseline['n']}")
    return baseline, response_data


async def run_ablation_evaluation(
    experiment: str,
    trait: str,
    vector_experiment: str,
    model_variant: str,
    vector_layer: int,
    method: str,
    component: str,
    position: str,
    prompt_set: str,
    model_name: str,
    judge_provider: str,
    subset: Optional[int],
    backend=None,
    judge=None,
    load_in_8bit: bool = False,
    load_in_4bit: bool = False,
    lora_adapter: str = None,
    max_new_tokens: int = 256,
    eval_prompt: Optional[str] = None,
    use_default_prompt: bool = False,
    extraction_variant: Optional[str] = None,
    relevance_check: bool = True,
):
    """
    Run directional ablation evaluation.

    Ablates a direction at ALL layers simultaneously.
    Compares baseline (no intervention) vs ablated responses.
    """
    from analysis.steering.data import load_steering_data, load_questions_from_inference

    # Load prompts and trait definition
    steering_data = load_steering_data(trait)

    # Load questions
    if prompt_set == "steering":
        questions = steering_data.questions
        print(f"Loaded {len(questions)} questions from steering.json")
    else:
        questions = load_questions_from_inference(prompt_set)
        print(f"Loaded {len(questions)} questions from inference: {prompt_set}")

    if subset:
        questions = questions[:subset]

    # Resolve eval_prompt
    if use_default_prompt:
        effective_eval_prompt = None
    elif eval_prompt is not None:
        effective_eval_prompt = eval_prompt
    else:
        effective_eval_prompt = steering_data.eval_prompt

    # Load model if not provided
    should_close_judge = False
    if backend is None:
        model, tokenizer = load_model_with_lora(model_name, load_in_8bit=load_in_8bit, load_in_4bit=load_in_4bit, lora_adapter=lora_adapter)
        backend = LocalBackend.from_model(model, tokenizer)

    # Extract model and tokenizer from backend for internal use
    model = backend.model
    tokenizer = backend.tokenizer

    config = load_experiment_config(experiment)
    use_chat_template = config.get('use_chat_template')
    if use_chat_template is None:
        use_chat_template = tokenizer.chat_template is not None

    # Create judge if not provided
    if judge is None:
        judge = TraitJudge()
        should_close_judge = True

    # Resolve extraction variant
    resolved_extraction_variant = extraction_variant or get_default_variant(vector_experiment, mode='extraction')

    # Load vector from specified layer
    vector = load_vector(vector_experiment, trait, vector_layer, resolved_extraction_variant, method, component, position)
    if vector is None:
        raise ValueError(f"Vector not found: {vector_experiment}/{trait} L{vector_layer} {method}")

    print(f"\n{'='*60}")
    print(f"ABLATION EVALUATION")
    print(f"{'='*60}")
    print(f"Trait: {trait}")
    print(f"Model: {model_name}")
    print(f"Vector: L{vector_layer} {method} (ablated at ALL layers)")
    print(f"Questions: {len(questions)}")

    # Compute baseline
    baseline, baseline_responses = await compute_baseline(
        backend, questions, steering_data.trait_name, steering_data.trait_definition,
        judge, max_new_tokens=max_new_tokens, eval_prompt=effective_eval_prompt,
        relevance_check=relevance_check
    )

    # Generate with ablation at ALL layers
    print(f"\nGenerating with ablation at all layers...")
    formatted = [format_prompt(q, tokenizer, use_chat_template=use_chat_template) for q in questions]

    with MultiLayerAblationHook(model, vector):
        ablated_responses = generate_batch(model, tokenizer, formatted, max_new_tokens=max_new_tokens)

    # Score ablated responses
    print(f"Scoring ablated responses...")
    all_qa_pairs = list(zip(questions, ablated_responses))
    all_scores = await judge.score_steering_batch(
        all_qa_pairs, steering_data.trait_name, steering_data.trait_definition, eval_prompt=effective_eval_prompt,
        relevance_check=relevance_check
    )

    ablated_trait_scores = [s["trait_score"] for s in all_scores if s["trait_score"] is not None]
    ablated_coherence_scores = [s["coherence_score"] for s in all_scores if s.get("coherence_score") is not None]

    ablated_trait_mean = sum(ablated_trait_scores) / len(ablated_trait_scores) if ablated_trait_scores else None
    ablated_coherence_mean = sum(ablated_coherence_scores) / len(ablated_coherence_scores) if ablated_coherence_scores else None

    # Build and save ablated response data
    ablated_response_data = [
        {
            "prompt": q,
            "response": r,
            "system_prompt": None,
            "trait_score": s["trait_score"],
            "coherence_score": s.get("coherence_score"),
        }
        for q, r, s in zip(questions, ablated_responses, all_scores)
    ]

    # Save baseline responses
    baseline_path = save_baseline_responses(
        baseline_responses, experiment, trait, model_variant, position, prompt_set
    )
    print(f"  Saved baseline responses: {baseline_path}")

    # Save ablated responses
    ablated_path = save_ablation_responses(
        ablated_response_data, experiment, trait, model_variant, position, prompt_set,
        vector_layer, method, component
    )
    print(f"  Saved ablated responses: {ablated_path}")

    # Print results
    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}")
    print(f"Baseline:  trait={baseline['trait_mean']:.1f}, coherence={baseline.get('coherence_mean', 0):.1f}")
    print(f"Ablated:   trait={ablated_trait_mean:.1f}, coherence={ablated_coherence_mean:.1f}")

    delta = ablated_trait_mean - baseline['trait_mean']
    baseline_coh = baseline.get('coherence_mean', 0)

    if baseline['trait_mean'] > 50:
        # Bypassing refusal (high baseline = model refuses, want to reduce)
        reduction_pct = (baseline['trait_mean'] - ablated_trait_mean) / baseline['trait_mean'] * 100
        print(f"\nBypass: {baseline['trait_mean']:.1f} → {ablated_trait_mean:.1f} ({reduction_pct:.0f}% reduction)")
    else:
        # Inducing refusal (low baseline)
        print(f"\nDelta: {delta:+.1f}")

    print(f"Coherence: {baseline_coh:.1f} → {ablated_coherence_mean:.1f}")

    if should_close_judge:
        await judge.close()

    return {
        "baseline": baseline,
        "ablated": {
            "trait_mean": ablated_trait_mean,
            "coherence_mean": ablated_coherence_mean,
            "n": len(ablated_trait_scores),
        },
        "delta": delta,
    }


async def run_evaluation(
    experiment: str,
    trait: str,
    vector_experiment: str,
    model_variant: str,
    layers_arg: str,
    coefficients: Optional[List[float]],
    method: str,
    component: str,
    position: str,
    prompt_set: str,
    model_name: str,
    judge_provider: str,
    subset: Optional[int],
    n_search_steps: int,
    up_mult: float,
    down_mult: float,
    start_mult: float = 0.7,
    momentum: float = 0.0,
    batched: bool = True,
    backend=None,
    judge=None,
    load_in_8bit: bool = False,
    load_in_4bit: bool = False,
    lora_adapter: str = None,
    max_new_tokens: int = 256,
    eval_prompt: Optional[str] = None,
    use_default_prompt: bool = False,
    min_coherence: float = MIN_COHERENCE,
    extraction_variant: Optional[str] = None,
    save_mode: str = "best",
    ablation: bool = False,
    ablation_vector_layer: Optional[int] = None,
    relevance_check: bool = True,
    trait_judge: Optional[str] = None,
):
    """
    Main evaluation flow.

    If coefficients provided: evaluate those directly.
    Otherwise: run adaptive search to find good coefficients.

    Args:
        batched: If True (default), run all layers in parallel batches.
                 If False, run each layer sequentially.
        backend: LocalBackend instance (optional, loads if not provided)
        judge: Pre-created TraitJudge (optional, creates if not provided)
        load_in_8bit: Use 8-bit quantization when loading model
        load_in_4bit: Use 4-bit quantization when loading model
        eval_prompt: Custom trait scoring prompt (auto-detected from steering.json if None)
        use_default_prompt: Force V3c default, ignore steering.json eval_prompt
        trait_judge: Path to trait judge prompt (e.g., "pv/hallucination") for metadata
        ablation: If True, run Arditi-style ablation instead of steering
        ablation_vector_layer: Layer to load vector from for ablation
    """
    # Dispatch to ablation mode if requested
    if ablation:
        return await run_ablation_evaluation(
            experiment=experiment, trait=trait, vector_experiment=vector_experiment,
            model_variant=model_variant, vector_layer=ablation_vector_layer,
            method=method, component=component, position=position,
            prompt_set=prompt_set, model_name=model_name, judge_provider=judge_provider,
            subset=None if subset == 5 else subset,  # Default 5 -> None for ablation
            backend=backend, judge=judge,
            load_in_8bit=load_in_8bit, load_in_4bit=load_in_4bit,
            lora_adapter=lora_adapter, max_new_tokens=max_new_tokens,
            eval_prompt=eval_prompt, use_default_prompt=use_default_prompt,
            extraction_variant=extraction_variant,
            relevance_check=relevance_check,
        )
    # Load prompts and trait definition
    steering_data = load_steering_data(trait)

    # Load questions: from inference dataset or trait's steering.json
    if prompt_set == "steering":
        questions = steering_data.questions
        print(f"Questions: {len(questions)} from steering.json")
    else:
        questions = load_questions_from_inference(prompt_set)
        print(f"Questions: {len(questions)} from inference/{prompt_set}.json")

    if subset:
        questions = questions[:subset]

    # Resolve eval_prompt: explicit override > use_default flag > steering.json
    if use_default_prompt:
        effective_eval_prompt = None
    elif eval_prompt is not None:
        effective_eval_prompt = eval_prompt
    else:
        effective_eval_prompt = steering_data.eval_prompt

    # Load model if not provided
    # Note: Steering uses hooks, so we force local mode
    should_close_judge = False
    if backend is None:
        model, tokenizer = load_model_with_lora(model_name, load_in_8bit=load_in_8bit, load_in_4bit=load_in_4bit, lora_adapter=lora_adapter)
        backend = LocalBackend.from_model(model, tokenizer)

    # Extract model and tokenizer from backend for internal use
    model = backend.model
    tokenizer = backend.tokenizer
    num_layers = backend.n_layers

    # Load experiment config
    config = load_experiment_config(experiment)
    use_chat_template = config.get('use_chat_template')
    if use_chat_template is None:
        use_chat_template = tokenizer.chat_template is not None

    # Parse and validate layers using actual model size
    layers = parse_layers(layers_arg, num_layers)
    layers = [l for l in layers if 0 <= l < num_layers]
    if not layers:
        raise ValueError(f"No valid layers. Model has {num_layers} layers (0-{num_layers-1})")

    # Load/create results (JSONL format)
    cached_runs = []
    baseline_result = None
    results_path = get_steering_results_path(experiment, trait, model_variant, position, prompt_set)

    if results_path.exists():
        # Load existing results for resume
        results_data = load_results(experiment, trait, model_variant, position, prompt_set)
        cached_runs = results_data.get("runs", [])
        baseline_result = results_data.get("baseline")
    else:
        # Initialize new results file
        init_results_file(
            experiment, trait, model_variant, steering_data.prompts_file,
            model_name, vector_experiment, judge_provider, position, prompt_set,
            trait_judge=trait_judge
        )

    # Create judge if not provided
    if judge is None:
        judge = TraitJudge()
        should_close_judge = True

    print(f"\nTrait: {trait}")
    print(f"Model: {model_name} ({num_layers} layers)")
    print(f"Chat template: {use_chat_template}")
    print(f"Vectors from: {vector_experiment}/{trait} @ {position}")
    print(f"Questions: {len(questions)}")
    print(f"Existing runs: {len(cached_runs)}")

    # Compute baseline if needed
    if baseline_result is None:
        baseline_result, baseline_responses = await compute_baseline(
            backend, questions, steering_data.trait_name, steering_data.trait_definition,
            judge, max_new_tokens=max_new_tokens, eval_prompt=effective_eval_prompt
        )
        save_baseline_responses(baseline_responses, experiment, trait, model_variant, position, prompt_set)
        append_baseline(experiment, trait, model_variant, baseline_result, position, prompt_set)
    else:
        print(f"\nUsing existing baseline: trait={baseline_result['trait_mean']:.1f}")

    # Load vectors and compute base coefficients
    # Try cached activation norms first (from extraction_evaluation.json)
    cached_norms = load_cached_activation_norms(vector_experiment)
    if cached_norms:
        print(f"\nUsing cached activation norms from extraction_evaluation.json")
    else:
        print(f"\nNo cached norms, will estimate activation norms...")

    # Resolve extraction model variant (vectors are stored under extraction variant, not application variant)
    resolved_extraction_variant = extraction_variant or get_default_variant(vector_experiment, mode='extraction')

    print(f"Loading vectors...")
    layer_data = []
    for layer in layers:
        vector = load_vector(vector_experiment, trait, layer, resolved_extraction_variant, method, component, position)
        if vector is None:
            print(f"  L{layer}: Vector not found, skipping")
            continue

        vec_norm = vector.norm().item()

        # Use cached norm if available, otherwise estimate
        if layer in cached_norms:
            act_norm = cached_norms[layer]
        else:
            act_norm = estimate_activation_norm(model, tokenizer, questions, layer, use_chat_template)

        base_coef = act_norm / vec_norm

        layer_data.append({
            "layer": layer,
            "vector": vector,
            "base_coef": base_coef,
        })
        print(f"  L{layer}: base_coef={base_coef:.0f}")

    if not layer_data:
        print("No valid layers with vectors found")
        return

    # Determine coefficients to test
    if coefficients is not None:
        # Manual mode: test specified coefficients for each layer
        print(f"\nManual coefficients: {coefficients}")
        for ld in layer_data:
            best_for_layer = None  # Track best for save_mode="best"

            for coef in coefficients:
                spec = VectorSpec(layer=ld["layer"], component=component, position=position, method=method, weight=coef)
                config = {"vectors": [spec.to_dict()]}

                # Skip if cached
                if find_cached_run(cached_runs, config) is not None:
                    print(f"  L{ld['layer']} c{coef:.0f}: cached, skipping")
                    continue

                print(f"\n  Evaluating L{ld['layer']} c{coef:.0f}...")
                result, responses = await evaluate_single_config(
                    backend, ld["vector"], ld["layer"], coef,
                    questions, steering_data.trait_name, steering_data.trait_definition,
                    judge, use_chat_template, component, eval_prompt=effective_eval_prompt
                )

                timestamp = datetime.now().isoformat()

                # Always append to results.jsonl
                append_run(experiment, trait, model_variant, config, result, position, prompt_set)
                cached_runs.append({"config": config, "result": result, "timestamp": timestamp})

                # Handle response saving based on save_mode
                if save_mode == "all":
                    save_responses(responses, experiment, trait, model_variant, position, prompt_set, config, timestamp)
                elif save_mode == "best":
                    # Track best: highest trait where coherence >= threshold, or best coherence
                    trait_mean = result.get("trait_mean") or 0
                    coherence_mean = result.get("coherence_mean") or 0

                    if is_better_result(best_for_layer, trait_mean, coherence_mean, min_coherence):
                        best_for_layer = {
                            "trait_mean": trait_mean,
                            "coherence_mean": coherence_mean,
                            "valid": coherence_mean >= min_coherence,
                            "responses": responses,
                            "config": config,
                            "timestamp": timestamp,
                        }
                # save_mode == "none": don't save responses

            # Save best for this layer (if tracking)
            if save_mode == "best" and best_for_layer and best_for_layer.get("responses"):
                save_responses(
                    best_for_layer["responses"], experiment, trait, model_variant,
                    position, prompt_set, best_for_layer["config"], best_for_layer["timestamp"]
                )
    elif batched and len(layer_data) > 1:
        # Batched adaptive search (default) - all layers in parallel
        await batched_adaptive_search(
            backend, layer_data, questions, steering_data.trait_name, steering_data.trait_definition,
            judge, use_chat_template, component, cached_runs, experiment, trait, model_variant,
            vector_experiment, method, position=position, prompt_set=prompt_set, n_steps=n_search_steps,
            up_mult=up_mult, down_mult=down_mult, start_mult=start_mult, momentum=momentum,
            max_new_tokens=max_new_tokens, eval_prompt=effective_eval_prompt,
            save_mode=save_mode, coherence_threshold=min_coherence,
            relevance_check=relevance_check
        )
    else:
        # Sequential adaptive search for each layer
        print(f"\nSequential adaptive search ({n_search_steps} steps per layer)")
        for ld in layer_data:
            await adaptive_search_layer(
                backend, ld["vector"], ld["layer"], ld["base_coef"],
                questions, steering_data.trait_name, steering_data.trait_definition,
                judge, use_chat_template, component,
                cached_runs, experiment, trait, model_variant, vector_experiment, method,
                position=position, prompt_set=prompt_set, n_steps=n_search_steps, up_mult=up_mult, down_mult=down_mult, start_mult=start_mult, momentum=momentum,
                max_new_tokens=max_new_tokens, eval_prompt=effective_eval_prompt,
                save_mode=save_mode, coherence_threshold=min_coherence,
                relevance_check=relevance_check
            )

    # Print summary
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"Baseline: {baseline_result['trait_mean']:.1f}")
    print(f"Total runs: {len(cached_runs)}")

    # Filter by coherence threshold
    valid_runs = [r for r in cached_runs if r.get('result', {}).get('coherence_mean', 0) >= min_coherence]
    if valid_runs:
        best_run = max(valid_runs, key=lambda r: r.get('result', {}).get('trait_mean') or 0)
        score = best_run['result']['trait_mean']
        coh = best_run['result'].get('coherence_mean', 0)
        delta = score - baseline_result['trait_mean']
        layer = best_run['config']['vectors'][0]['layer']
        coef = best_run['config']['vectors'][0]['weight']
        print(f"Best (coherence≥{min_coherence:.0f}): L{layer} c{coef:.0f}")
        print(f"  trait={score:.1f} (+{delta:.1f}), coherence={coh:.1f}")
    else:
        print(f"No valid runs with coherence≥{min_coherence:.0f}")

    if should_close_judge:
        await judge.close()


def main():
    parser = argparse.ArgumentParser(description="Steering evaluation")

    # === Core (required) ===
    parser.add_argument("--experiment", required=True,
                        help="Experiment where steering results are saved")
    trait_group = parser.add_mutually_exclusive_group(required=True)
    trait_group.add_argument("--vector-from-trait",
                        help="Single trait: 'experiment/category/trait'")
    trait_group.add_argument("--traits",
                        help="Multiple traits (comma-separated): 'exp/cat/t1,exp/cat/t2'")

    # === Input/Output ===
    parser.add_argument("--prompt-set", default="steering",
                        help="Prompt set: 'steering' uses trait's steering.json, otherwise loads from datasets/inference/{prompt-set}.json")

    # === Model ===
    parser.add_argument("--model-variant", default=None,
                        help="Model variant for steering (default: from experiment defaults.application)")
    parser.add_argument("--extraction-variant", default=None,
                        help="Model variant where vectors are stored (default: from experiment defaults.extraction)")
    parser.add_argument("--load-in-8bit", action="store_true",
                        help="Load model in 8-bit quantization (for 70B+ models)")
    parser.add_argument("--load-in-4bit", action="store_true",
                        help="Load model in 4-bit quantization")
    parser.add_argument("--no-server", action="store_true",
                        help="Force local model loading (skip model server check)")

    # === Vector Specification ===
    parser.add_argument("--method", default="probe", help="Vector extraction method")
    parser.add_argument("--component", default="residual",
                        choices=["residual", "attn_out", "mlp_out", "attn_contribution", "mlp_contribution", "k_proj", "v_proj"])
    parser.add_argument("--position", default="response[:5]",
                        help="Token position for vectors (default: response[:5])")

    # === Evaluation ===
    parser.add_argument("--subset", type=int, default=5,
                        help="Use subset of questions (default: 5, use --subset 0 for all)")
    parser.add_argument("--max-new-tokens", type=int, default=64,
                        help="Max tokens to generate per response (default: 64)")
    parser.add_argument("--judge", default="openai", choices=["openai", "gemini"])
    parser.add_argument("--min-coherence", type=float, default=MIN_COHERENCE,
                        help=f"Minimum coherence threshold for valid results (default: {MIN_COHERENCE})")
    prompt_group = parser.add_mutually_exclusive_group()
    prompt_group.add_argument("--no-custom-prompt", action="store_true",
                        help="Ignore eval_prompt from steering.json, use V3c default scoring")
    prompt_group.add_argument("--eval-prompt-from", type=str, metavar="TRAIT_PATH",
                        help="Load eval_prompt from different trait's steering.json (e.g., 'persona_vectors_instruction/evil')")
    prompt_group.add_argument("--trait-judge", type=str, metavar="JUDGE_PATH",
                        help="Load trait judge prompt from datasets/llm_judge/trait_score/{path}.txt (e.g., 'pv/hallucination')")

    # === Search/Optimization ===
    parser.add_argument("--layers", default="30%-60%",
                        help="Layers: 'all', '30%%-60%%' (default), single '16', range '5-20', or list '5,10,15'")
    parser.add_argument("--coefficients",
                        help="Manual coefficients (comma-separated). If not provided, uses adaptive search.")
    parser.add_argument("--search-steps", type=int, default=5,
                        help="Number of adaptive search steps per layer (default: 5)")
    parser.add_argument("--up-mult", type=float, default=1.3,
                        help="Coefficient multiplier when increasing (default: 1.3)")
    parser.add_argument("--down-mult", type=float, default=0.85,
                        help="Coefficient multiplier when decreasing (default: 0.85)")
    parser.add_argument("--start-mult", type=float, default=0.7,
                        help="Starting coefficient as fraction of base_coef (default: 0.7). Use lower values (e.g., 0.05) for sensitive models.")
    parser.add_argument("--momentum", type=float, default=0.1,
                        help="Momentum for coefficient updates (0.0=direct, 0.7=smoothed). Default: 0.1")

    # === Advanced ===
    parser.add_argument("--no-batch", action="store_true",
                        help="Disable batched layer evaluation (run layers sequentially)")
    parser.add_argument("--save-responses", choices=["all", "best", "none"], default="best",
                        help="Response saving: 'all' (every config), 'best' (best per layer), 'none'. Default: best")

    # === Ablation ===
    parser.add_argument("--ablation", type=int, metavar="LAYER", default=None,
                        help="Use directional ablation at ALL layers. Load vector from LAYER.")

    # === Coherence Scoring ===
    parser.add_argument("--no-relevance-check", action="store_true",
                        help="Disable relevance check in coherence scoring (don't cap refusals at 50)")

    args = parser.parse_args()

    # Parse trait specs (single or multiple)
    if args.traits:
        trait_specs = [t.strip() for t in args.traits.split(',')]
    else:
        trait_specs = [args.vector_from_trait]

    # Parse trait specs: 'category/trait' uses current experiment, 'exp/category/trait' uses specified
    parsed_traits = []
    for spec in trait_specs:
        parts = spec.split('/')
        if len(parts) == 2:
            # category/trait - use current experiment
            parsed_traits.append((args.experiment, spec))
        elif len(parts) == 3:
            # experiment/category/trait
            parsed_traits.append((parts[0], f"{parts[1]}/{parts[2]}"))
        else:
            parser.error(f"Invalid trait spec '{spec}': use 'category/trait' or 'experiment/category/trait'")

    # Resolve model variant
    variant = get_model_variant(args.experiment, args.model_variant, mode="application")
    model_variant = variant['name']
    model_name = variant['model']
    lora = variant.get('lora')

    coefficients = parse_coefficients(args.coefficients)

    # Run evaluation(s) - layers parsed after model loads to get actual num_layers
    asyncio.run(_run_main(
        args=args,
        parsed_traits=parsed_traits,
        model_variant=model_variant,
        model_name=model_name,
        lora=lora,
        layers_arg=args.layers,
        coefficients=coefficients,
    ))


async def _run_main(args, parsed_traits, model_variant, model_name, lora, layers_arg, coefficients):
    """Async main to handle model/judge lifecycle."""
    multi_trait = len(parsed_traits) > 1

    # Load model once if multiple traits
    # Note: Steering uses hooks, so we force local mode (no_server=True)
    backend, judge = None, None
    if multi_trait:
        print(f"\nEvaluating {len(parsed_traits)} traits with shared model")
        model, tokenizer = load_model_with_lora(
            model_name,
            load_in_8bit=args.load_in_8bit,
            load_in_4bit=args.load_in_4bit,
            lora_adapter=lora
        )
        backend = LocalBackend.from_model(model, tokenizer)
        judge = TraitJudge()

    # Resolve eval_prompt override
    effective_eval_prompt = None
    trait_judge = None
    use_default = args.no_custom_prompt
    if args.trait_judge:
        # Load from datasets/llm_judge/trait_score/{path}.txt
        judge_path = get('datasets.llm_judge_trait', judge_prompt=args.trait_judge)
        if not judge_path.exists():
            parser_error = f"Trait judge prompt not found: {judge_path}"
            raise FileNotFoundError(parser_error)
        effective_eval_prompt = judge_path.read_text()
        trait_judge = args.trait_judge
        print(f"Using trait judge: {args.trait_judge}")
    elif args.eval_prompt_from:
        override_data = load_steering_data(args.eval_prompt_from)
        effective_eval_prompt = override_data.eval_prompt
        if not effective_eval_prompt:
            print(f"Warning: {args.eval_prompt_from} has no eval_prompt, using V3c default")
            use_default = True
        else:
            print(f"Using eval_prompt from: {args.eval_prompt_from}")
    elif use_default:
        print("Using V3c default scoring (--no-custom-prompt)")

    try:
        for vector_experiment, trait in parsed_traits:
            if multi_trait:
                print(f"\n{'='*60}")
                print(f"TRAIT: {vector_experiment}/{trait}")
                print(f"{'='*60}")

            await run_evaluation(
                experiment=args.experiment,
                trait=trait,
                vector_experiment=vector_experiment,
                model_variant=model_variant,
                layers_arg=layers_arg,
                coefficients=coefficients,
                method=args.method,
                component=args.component,
                position=args.position,
                prompt_set=args.prompt_set,
                model_name=model_name,
                judge_provider=args.judge,
                subset=args.subset,
                n_search_steps=args.search_steps,
                up_mult=args.up_mult,
                down_mult=args.down_mult,
                start_mult=args.start_mult,
                momentum=args.momentum,
                batched=not args.no_batch,
                backend=backend,
                judge=judge,
                load_in_8bit=args.load_in_8bit,
                load_in_4bit=args.load_in_4bit,
                lora_adapter=lora,
                max_new_tokens=args.max_new_tokens,
                eval_prompt=effective_eval_prompt,
                use_default_prompt=use_default,
                min_coherence=args.min_coherence,
                extraction_variant=args.extraction_variant,
                save_mode=args.save_responses,
                ablation=args.ablation is not None,
                ablation_vector_layer=args.ablation,
                relevance_check=not args.no_relevance_check,
                trait_judge=trait_judge,
            )
    finally:
        if judge is not None:
            await judge.close()
        # Cleanup GPU memory
        if backend is not None:
            del backend
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    if multi_trait:
        print(f"\n{'='*60}")
        print(f"COMPLETED {len(parsed_traits)} TRAITS")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
