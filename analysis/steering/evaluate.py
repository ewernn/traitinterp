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

    # Re-score existing responses with current judge (no GPU needed)
    python analysis/steering/evaluate.py \\
        --experiment {experiment} \\
        --rescore {category}/{trait}
"""

import sys
import gc
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import argparse
import asyncio
from typing import List, Dict, Optional
from datetime import datetime

from analysis.steering.data import load_steering_data, load_questions_from_inference
from analysis.steering.results import (
    init_results_file, load_results, append_baseline,
    save_baseline_responses, save_ablation_responses, find_cached_run, append_run, save_responses,
    is_better_result,
)
from utils.paths import get_steering_results_path, get_steering_dir
from analysis.steering.coef_search import (
    adaptive_search_layer,
    batched_adaptive_search,
)
from core import VectorSpec, MultiLayerAblationHook, LocalBackend, GenerationConfig, batched_steering_generate
from core.hooks import get_hook_path
from utils.generation import generate_batch
from utils.judge import TraitJudge
from utils.paths import get, get_default_variant
from utils.model import format_prompt, tokenize_prompt, load_model_with_lora, get_layers_module
from utils.paths import get_model_variant, load_experiment_config
from utils.vectors import MIN_COHERENCE, load_vector, load_cached_activation_norms
from utils.layers import parse_layers


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
    component: str = "residual",
) -> float:
    """Estimate activation norm at a layer for a specific component by running a few prompts."""
    norms = []

    def capture_hook(_module, _input, output):
        hidden = output[0] if isinstance(output, tuple) else output
        norm = hidden[:, -1, :].float().norm().item()
        norms.append(norm)

    hook_path = get_hook_path(layer, component, model=model)
    module = model
    for attr in hook_path.split('.'):
        module = getattr(module, attr)
    handle = module.register_forward_hook(capture_hook)

    try:
        for prompt in prompts[:3]:
            formatted = format_prompt(prompt, tokenizer, use_chat_template=use_chat_template)
            inputs = tokenize_prompt(formatted, tokenizer)
            inputs = {k: v.to(model.device) if hasattr(v, 'to') else v for k, v in inputs.items()}
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
    bnb_4bit_quant_type: str = "nf4",
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
        model, tokenizer = load_model_with_lora(model_name, load_in_8bit=load_in_8bit, load_in_4bit=load_in_4bit, bnb_4bit_quant_type=bnb_4bit_quant_type, lora_adapter=lora_adapter)
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
    bnb_4bit_quant_type: str = "nf4",
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
    direction: str = "positive",
    force: bool = False,
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
        direction: "positive" (induce trait, coef>0) or "negative" (suppress trait, coef<0)
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
            bnb_4bit_quant_type=bnb_4bit_quant_type,
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
        model, tokenizer = load_model_with_lora(model_name, load_in_8bit=load_in_8bit, load_in_4bit=load_in_4bit, bnb_4bit_quant_type=bnb_4bit_quant_type, lora_adapter=lora_adapter)
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

    if force and results_path.exists():
        import shutil
        steering_dir = get_steering_dir(experiment, trait, model_variant, position, prompt_set)
        shutil.rmtree(steering_dir)
        print(f"--force: cleared {steering_dir}")

    if results_path.exists():
        # Load existing results for resume
        results_data = load_results(experiment, trait, model_variant, position, prompt_set)
        cached_runs = results_data.get("runs", [])
        baseline_result = results_data.get("baseline")
        header_direction = results_data.get("direction", "positive")
        if header_direction != direction:
            print(f"\n⚠️  Warning: CLI direction '{direction}' differs from results file direction '{header_direction}'")
            print(f"   Results header will retain original direction. Consider starting fresh if this is intentional.")
    else:
        # Initialize new results file
        init_results_file(
            experiment, trait, model_variant, steering_data.prompts_file,
            model_name, vector_experiment, judge_provider, position, prompt_set,
            trait_judge=trait_judge, direction=direction
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
        append_baseline(experiment, trait, model_variant, baseline_result, position, prompt_set, trait_judge=trait_judge)
    else:
        print(f"\nUsing existing baseline: trait={baseline_result['trait_mean']:.1f}")

    # Load vectors and compute base coefficients
    # Always use residual norms for base_coef — steering perturbation flows into residual
    # stream regardless of which component is hooked, so residual scale is the right reference
    cached_norms = load_cached_activation_norms(vector_experiment, "residual")
    if cached_norms:
        print(f"\nUsing cached activation norms from extraction_evaluation.json (residual)")
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
        if vec_norm == 0:
            print(f"  L{layer}: Zero vector, skipping")
            continue

        # Use cached norm if available, otherwise estimate
        if layer in cached_norms:
            act_norm = cached_norms[layer]
        else:
            act_norm = estimate_activation_norm(model, tokenizer, questions, layer, use_chat_template, "residual")

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
        # Manual mode: test specified coefficients for each layer (batched)
        print(f"\nManual coefficients: {coefficients}")
        n_q = len(questions)

        # Build all (layer, coef) configs, filter cached
        all_configs = []
        for ld in layer_data:
            for coef in coefficients:
                config = {"vectors": [VectorSpec(layer=ld["layer"], component=component, position=position, method=method, weight=coef).to_dict()]}
                if find_cached_run(cached_runs, config) is not None:
                    print(f"  L{ld['layer']} c{coef:.0f}: cached")
                else:
                    all_configs.append((ld, coef, config))

        if all_configs:
            print(f"\nEvaluating {len(all_configs)} configs in batch...")
            formatted = [format_prompt(q, tokenizer, use_chat_template=use_chat_template) for q in questions]

            # Generate all responses
            all_responses = batched_steering_generate(
                model, tokenizer, formatted,
                [(ld["layer"], ld["vector"], coef) for ld, coef, _ in all_configs],
                component=component, max_new_tokens=max_new_tokens
            )

            # Score all responses
            all_qa_pairs = [(q, all_responses[i * n_q + j]) for i in range(len(all_configs)) for j, q in enumerate(questions)]
            print(f"Scoring {len(all_qa_pairs)} responses...")
            all_scores = await judge.score_steering_batch(
                all_qa_pairs, steering_data.trait_name, steering_data.trait_definition,
                eval_prompt=effective_eval_prompt, relevance_check=relevance_check
            )

            # Process results
            best_per_layer = {}
            for idx, (ld, coef, config) in enumerate(all_configs):
                scores = all_scores[idx * n_q:(idx + 1) * n_q]
                resps = all_responses[idx * n_q:(idx + 1) * n_q]
                trait_scores = [s["trait_score"] for s in scores if s["trait_score"] is not None]
                coh_scores = [s["coherence_score"] for s in scores if s.get("coherence_score") is not None]

                result = {
                    "trait_mean": sum(trait_scores) / len(trait_scores) if trait_scores else None,
                    "coherence_mean": sum(coh_scores) / len(coh_scores) if coh_scores else None,
                    "n": len(trait_scores),
                }
                timestamp = datetime.now().isoformat()
                print(f"  L{ld['layer']} c{coef:.0f}: trait={result['trait_mean'] or 0:.1f}, coherence={result['coherence_mean'] or 0:.1f}, n={result['n']}")

                append_run(experiment, trait, model_variant, config, result, position, prompt_set, trait_judge=trait_judge)
                cached_runs.append({"config": config, "result": result, "timestamp": timestamp})

                # Handle response saving
                responses = [{"prompt": q, "response": r, "system_prompt": None, "trait_score": s["trait_score"], "coherence_score": s.get("coherence_score")}
                             for q, r, s in zip(questions, resps, scores)]
                if save_mode == "all":
                    save_responses(responses, experiment, trait, model_variant, position, prompt_set, config, timestamp)
                elif save_mode == "best":
                    t_mean, c_mean, layer = result.get("trait_mean") or 0, result.get("coherence_mean") or 0, ld["layer"]
                    if is_better_result(best_per_layer.get(layer), t_mean, c_mean, min_coherence, direction):
                        best_per_layer[layer] = {"responses": responses, "config": config, "timestamp": timestamp}

            # Save best per layer
            if save_mode == "best":
                for best in best_per_layer.values():
                    if best.get("responses"):
                        save_responses(best["responses"], experiment, trait, model_variant, position, prompt_set, best["config"], best["timestamp"])
    elif batched and len(layer_data) > 1:
        # Batched adaptive search (default) - all layers in parallel
        await batched_adaptive_search(
            backend, layer_data, questions, steering_data.trait_name, steering_data.trait_definition,
            judge, use_chat_template, component, cached_runs, experiment, trait, model_variant,
            vector_experiment, method, position=position, prompt_set=prompt_set, n_steps=n_search_steps,
            up_mult=up_mult, down_mult=down_mult, start_mult=start_mult, momentum=momentum,
            max_new_tokens=max_new_tokens, eval_prompt=effective_eval_prompt,
            save_mode=save_mode, coherence_threshold=min_coherence,
            relevance_check=relevance_check, direction=direction, trait_judge=trait_judge
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
                relevance_check=relevance_check, direction=direction, trait_judge=trait_judge
            )

    # Print summary
    sign = 1 if direction == "positive" else -1
    print(f"\n{'='*60}")
    print(f"Summary (direction={direction})")
    print(f"{'='*60}")
    print(f"Baseline: {baseline_result['trait_mean']:.1f}")
    print(f"Total runs: {len(cached_runs)}")

    # Filter by coherence threshold, find best using direction-aware comparison
    valid_runs = [r for r in cached_runs if r.get('result', {}).get('coherence_mean', 0) >= min_coherence]
    if valid_runs:
        best_run = max(valid_runs, key=lambda r: (r.get('result', {}).get('trait_mean') or 0) * sign)
        score = best_run['result']['trait_mean']
        coh = best_run['result'].get('coherence_mean', 0)
        delta = score - baseline_result['trait_mean']
        layer = best_run['config']['vectors'][0]['layer']
        coef = best_run['config']['vectors'][0]['weight']
        print(f"Best (coherence≥{min_coherence:.0f}): L{layer} c{coef:.0f}")
        delta_str = f"+{delta:.1f}" if delta >= 0 else f"{delta:.1f}"
        print(f"  trait={score:.1f} ({delta_str}), coherence={coh:.1f}")
    else:
        print(f"No valid runs with coherence≥{min_coherence:.0f}")

    if should_close_judge:
        await judge.close()


def discover_response_files(experiment: str, trait: str, model_variant: str,
                            position: str, prompt_set: str) -> List[Dict]:
    """Find all saved response files and parse config from path.

    Returns list of dicts with keys: path, component, method, layer, coef, timestamp, is_baseline
    """
    steering_dir = get_steering_dir(experiment, trait, model_variant, position, prompt_set)
    responses_dir = steering_dir / "responses"
    if not responses_dir.exists():
        return []

    found = []

    # Baseline
    baseline_path = responses_dir / "baseline.json"
    if baseline_path.exists():
        found.append({"path": baseline_path, "is_baseline": True})

    # Steered responses: responses/{component}/{method}/L{layer}_c{coef}_{timestamp}.json
    for component_dir in responses_dir.iterdir():
        if not component_dir.is_dir() or component_dir.name in ("ablation",):
            continue
        for method_dir in component_dir.iterdir():
            if not method_dir.is_dir():
                continue
            for f in method_dir.glob("L*_c*.json"):
                name = f.stem
                try:
                    # Parse: L14_c6.6_2026-02-03_07-10-27
                    parts = name.split("_c")
                    layer = int(parts[0][1:])  # Remove 'L'
                    rest = parts[1].split("_", 1)
                    coef = float(rest[0])
                    timestamp = rest[1] if len(rest) > 1 else ""
                except (IndexError, ValueError):
                    print(f"  Warning: couldn't parse {f.name}, skipping")
                    continue

                found.append({
                    "path": f,
                    "is_baseline": False,
                    "component": component_dir.name,
                    "method": method_dir.name,
                    "layer": layer,
                    "coef": coef,
                    "timestamp": timestamp,
                })

    return found


async def run_rescore(
    experiment: str,
    trait: str,
    model_variant: str,
    position: str,
    prompt_set: str,
    eval_prompt: Optional[str] = None,
    relevance_check: bool = True,
    trait_judge: Optional[str] = None,
    dry_run: bool = False,
):
    """Re-score existing steering responses with current judge. No GPU needed."""
    from analysis.steering.data import load_steering_data

    steering_data = load_steering_data(trait)
    trait_name = steering_data.trait_name
    trait_definition = steering_data.trait_definition

    if eval_prompt is None:
        eval_prompt = steering_data.eval_prompt

    # Discover response files
    response_files = discover_response_files(experiment, trait, model_variant, position, prompt_set)
    if not response_files:
        print(f"No response files found for {experiment}/{trait}")
        return

    n_baseline = sum(1 for f in response_files if f["is_baseline"])
    n_steered = len(response_files) - n_baseline
    mode = "DRY RUN (no writes)" if dry_run else "Re-scoring"
    print(f"\n{mode}: {n_baseline} baseline + {n_steered} steered response files")
    print(f"Trait: {trait} ({trait_name})")
    print(f"Judge: {'custom' if eval_prompt else 'default'}")

    judge = TraitJudge()

    # Re-score each file
    new_results = {}  # config_key -> result dict
    new_baseline = None

    for i, entry in enumerate(response_files):
        path = entry["path"]
        is_baseline = entry["is_baseline"]

        with open(path) as f:
            responses = json.load(f)

        qa_pairs = [(r.get("prompt", r.get("question", "")), r["response"]) for r in responses]

        label = "baseline" if is_baseline else f"L{entry['layer']} c{entry['coef']:.1f} {entry['component']}/{entry['method']}"
        print(f"  [{i+1}/{len(response_files)}] {label} ({len(qa_pairs)} responses)...", end="", flush=True)

        scores = await judge.score_steering_batch(
            qa_pairs, trait_name, trait_definition,
            eval_prompt=eval_prompt, relevance_check=relevance_check,
        )

        # Update response data in-place
        for r, s in zip(responses, scores):
            r["trait_score"] = s["trait_score"]
            r["coherence_score"] = s["coherence_score"]

        if dry_run:
            # Print per-response scores
            for j, r in enumerate(responses):
                prompt_short = r.get("prompt", r.get("question", ""))[:80]
                print(f"    [{j+1}] trait={r['trait_score']:.1f} coh={r['coherence_score']:.1f}  {prompt_short}")
        else:
            # Save updated responses
            with open(path, 'w') as f:
                json.dump(responses, f, indent=2)

        # Compute aggregates
        trait_scores = [s["trait_score"] for s in scores if s["trait_score"] is not None]
        coh_scores = [s["coherence_score"] for s in scores if s.get("coherence_score") is not None]
        result = {
            "trait_mean": sum(trait_scores) / len(trait_scores) if trait_scores else None,
            "coherence_mean": sum(coh_scores) / len(coh_scores) if coh_scores else None,
            "n": len(trait_scores),
        }

        print(f" trait={result['trait_mean']:.1f} coh={result['coherence_mean']:.1f}")

        if is_baseline:
            new_baseline = result
        else:
            # Key by (layer, component, method, rounded_coef) for matching against results.jsonl
            match_key = (entry["layer"], entry["component"], entry["method"], round(entry["coef"], 2))
            new_results[match_key] = (entry, result)

    await judge.close()

    if dry_run:
        # Print summary and exit without writing
        if new_baseline:
            print(f"\n  Baseline: trait={new_baseline['trait_mean']:.1f} coh={new_baseline['coherence_mean']:.1f}")
        for match_key, (entry, result) in sorted(new_results.items(), key=lambda x: x[0][0]):
            delta = result['trait_mean'] - new_baseline['trait_mean'] if new_baseline else 0
            print(f"  L{entry['layer']} c{entry['coef']:.1f} {entry['component']}/{entry['method']}: "
                  f"trait={result['trait_mean']:.1f} coh={result['coherence_mean']:.1f} delta={delta:+.1f}")
        print(f"\nDry run complete — no files modified.")
        return

    # Rebuild results.jsonl
    results_path = get_steering_results_path(experiment, trait, model_variant, position, prompt_set)
    if not results_path.exists():
        print(f"\nWarning: {results_path} not found, skipping results.jsonl update")
        return

    # Read existing, update scores (drop entries without response files)
    lines = []
    n_dropped = 0
    with open(results_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)

            if entry.get("type") == "header":
                entry.setdefault("eval", {})["trait_judge"] = trait_judge
                lines.append(entry)
            elif entry.get("type") == "baseline":
                if new_baseline:
                    entry["result"] = new_baseline
                    entry.setdefault("eval", {})["trait_judge"] = trait_judge
                    entry["timestamp"] = datetime.now().isoformat()
                lines.append(entry)
            else:
                # Match by (layer, component, method, rounded_coef)
                # Drop entries without matching response files (stale scores)
                vectors = entry.get("config", {}).get("vectors", [{}])
                if vectors:
                    v = vectors[0]
                    match_key = (v.get("layer"), v.get("component"), v.get("method"), round(v.get("weight", 0), 2))
                    if match_key in new_results:
                        _, new_result = new_results[match_key]
                        entry["result"] = new_result
                        entry.setdefault("eval", {})["trait_judge"] = trait_judge
                        entry["timestamp"] = datetime.now().isoformat()
                        lines.append(entry)
                    else:
                        n_dropped += 1

    with open(results_path, 'w') as f:
        for entry in lines:
            f.write(json.dumps(entry) + '\n')

    print(f"\nUpdated {results_path}")
    print(f"  Rescored: {n_baseline} baseline + {len(new_results)} steered configs"
          f"{f', dropped {n_dropped} stale entries' if n_dropped else ''}")

    # Print summary
    if new_baseline:
        print(f"  Baseline: trait={new_baseline['trait_mean']:.1f} coh={new_baseline['coherence_mean']:.1f}")


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
    trait_group.add_argument("--rescore",
                        help="Re-score existing responses with current judge (no GPU). "
                             "Takes trait path: 'category/trait'")

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
    parser.add_argument("--bnb-4bit-quant-type", default="nf4",
                        help="BnB 4-bit quant type: 'nf4' (default) or 'fp4'")
    parser.add_argument("--vector-experiment", default=None,
                        help="Experiment to load vectors from (defaults to --experiment). "
                             "Use for cross-experiment steering, e.g., FP16 vectors on quantized model.")
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
    parser.add_argument("--force", action="store_true",
                        help="Delete existing results and start fresh")
    parser.add_argument("--no-batch", action="store_true",
                        help="Disable batched layer evaluation (run layers sequentially)")
    parser.add_argument("--save-responses", choices=["all", "best", "none"], default="best",
                        help="Response saving: 'all' (every config), 'best' (best per layer), 'none'. Default: best")

    # === Ablation ===
    parser.add_argument("--ablation", type=int, metavar="LAYER", default=None,
                        help="Use directional ablation at ALL layers. Load vector from LAYER.")

    # === Coherence Scoring ===
    parser.add_argument("--no-relevance-check", action="store_true",
                        help="Disable relevance check in coherence scoring (don't cap off-topic responses at 50)")
    parser.add_argument("--dry-run", action="store_true",
                        help="With --rescore: print new scores without writing to disk")

    # === Steering Direction ===
    parser.add_argument("--direction", choices=["positive", "negative"], default=None,
                        help="Steering direction: 'positive' (induce trait, coef>0) or 'negative' (suppress trait, coef<0). "
                             "Default: positive, or inferred from --coefficients if all are negative.")

    args = parser.parse_args()

    # Handle --rescore mode (no GPU, just re-judge existing responses)
    if args.rescore:
        # Resolve model variant name (needed for path lookup, not for loading model)
        variant = get_model_variant(args.experiment, args.model_variant, mode="application")
        model_variant = variant['name']

        # Resolve eval_prompt
        effective_eval_prompt = None
        trait_judge = None
        if args.trait_judge:
            judge_path = get('datasets.llm_judge_trait', judge_prompt=args.trait_judge)
            if not judge_path.exists():
                raise FileNotFoundError(f"Trait judge prompt not found: {judge_path}")
            effective_eval_prompt = judge_path.read_text()
            trait_judge = args.trait_judge
        elif args.no_custom_prompt:
            effective_eval_prompt = None  # Force V3c default
        # else: auto-detect from steering.json (handled in run_rescore)

        # Resolve position format for path lookup
        position = args.position

        asyncio.run(run_rescore(
            experiment=args.experiment,
            trait=args.rescore,
            model_variant=model_variant,
            position=position,
            prompt_set=args.prompt_set,
            eval_prompt=effective_eval_prompt,
            relevance_check=not args.no_relevance_check,
            trait_judge=trait_judge,
            dry_run=args.dry_run,
        ))
        return

    # Parse trait specs (single or multiple)
    if args.traits:
        trait_specs = [t.strip() for t in args.traits.split(',')]
    else:
        trait_specs = [args.vector_from_trait]

    # Parse trait specs: 'category/trait' uses current experiment, 'exp/category/trait' uses specified
    # --vector-experiment overrides the experiment source for all traits
    vector_exp_default = args.vector_experiment or args.experiment
    parsed_traits = []
    for spec in trait_specs:
        parts = spec.split('/')
        if len(parts) == 2:
            # category/trait - use vector-experiment if provided, else current experiment
            parsed_traits.append((vector_exp_default, spec))
        elif len(parts) == 3:
            # experiment/category/trait - explicit experiment in spec
            parsed_traits.append((parts[0], f"{parts[1]}/{parts[2]}"))
        else:
            parser.error(f"Invalid trait spec '{spec}': use 'category/trait' or 'experiment/category/trait'")

    # Resolve model variant
    variant = get_model_variant(args.experiment, args.model_variant, mode="application")
    model_variant = variant['name']
    model_name = variant['model']
    lora = variant.get('lora')

    coefficients = parse_coefficients(args.coefficients)

    # Determine direction: explicit flag > infer from coefficients > default positive
    if args.direction:
        direction = args.direction
    elif coefficients:
        # Infer: negative only if ALL coefficients are <= 0 and at least one is < 0
        all_non_positive = all(c <= 0 for c in coefficients)
        any_negative = any(c < 0 for c in coefficients)
        direction = "negative" if (all_non_positive and any_negative) else "positive"
    else:
        direction = "positive"

    # Run evaluation(s) - layers parsed after model loads to get actual num_layers
    asyncio.run(_run_main(
        args=args,
        parsed_traits=parsed_traits,
        model_variant=model_variant,
        model_name=model_name,
        lora=lora,
        layers_arg=args.layers,
        coefficients=coefficients,
        direction=direction,
        force=args.force,
    ))


async def _run_main(args, parsed_traits, model_variant, model_name, lora, layers_arg, coefficients, direction, force=False):
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
            bnb_4bit_quant_type=args.bnb_4bit_quant_type,
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
                bnb_4bit_quant_type=args.bnb_4bit_quant_type,
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
                direction=direction,
                force=force,
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
