#!/usr/bin/env python3
"""
Steering evaluation - validate trait vectors via causal intervention.

Input: experiment, traits, layers, coefficients
Output: experiments/{experiment}/steering/{trait}/{variant}/{position}/{prompt_set}/results.jsonl

Usage:
    python steering/run_steering_eval.py --experiment {exp} --vector-from-trait {exp}/{cat}/{trait}
    python steering/run_steering_eval.py --experiment {exp} --traits "cat/t1,cat/t2" --load-in-8bit
    python steering/run_steering_eval.py --experiment {exp} --rescore {cat}/{trait}
"""

import sys
import gc
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import argparse
import asyncio
from typing import List, Dict, Optional
from datetime import datetime

from utils.traits import load_steering_data, load_questions_from_inference, load_questions_from_file
from utils.steering_results import (
    init_results_file, load_results, append_baseline, remove_baseline, get_baseline,
    save_baseline_responses, save_ablation_responses, find_cached_run, append_run, save_responses,
    is_better_result,
)
from utils.paths import get_steering_results_path, get_steering_dir
from steering.coefficient_search import (
    adaptive_search_layer, batched_adaptive_search, multi_trait_batched_adaptive_search,
)
from core import VectorSpec, MultiLayerAblationHook
from utils.backends import LocalBackend, add_backend_args
from utils.steered_generation import (
    score_stats, estimate_activation_norm, compute_baseline, batched_steering_generate,
)
from utils.judge import TraitJudge
from utils.paths import get, get_default_variant, get_model_variant, load_experiment_config
from utils.model import format_prompt, load_model_with_lora
from utils.generation import generate_batch
from utils.distributed import is_tp_mode, is_rank_zero, tp_barrier
from utils.vectors import MIN_COHERENCE, load_vector, load_cached_activation_norms
from utils.layers import parse_layers


# =============================================================================
# Helpers
# =============================================================================

def parse_coefficients(coef_arg: Optional[str]) -> Optional[List[float]]:
    if coef_arg is None:
        return None
    return [float(c) for c in coef_arg.split(",")]


def _resolve_eval_prompt(steering_data, eval_prompt, use_default_prompt):
    """Resolve eval_prompt: explicit > use_default flag > steering.json."""
    if use_default_prompt:
        return None
    if eval_prompt is not None:
        return eval_prompt
    return steering_data.eval_prompt


def _resolve_questions(trait, questions_file, prompt_set, subset):
    """Load questions from file, inference dataset, or steering.json."""
    steering_data = load_steering_data(trait)
    if questions_file:
        questions = load_questions_from_file(questions_file)
    elif prompt_set == "steering":
        questions = steering_data.questions
    else:
        questions = load_questions_from_inference(prompt_set)
    if subset:
        questions = questions[:subset]
    return questions, steering_data


def _resolve_cli_eval_prompt(args):
    """Resolve eval_prompt from CLI args. Returns (eval_prompt, trait_judge, use_default)."""
    use_default = args.no_custom_prompt
    trait_judge = None
    eval_prompt = None

    if args.trait_judge:
        judge_path = get('datasets.llm_judge_trait', judge_prompt=args.trait_judge)
        if not judge_path.exists():
            raise FileNotFoundError(f"Trait judge prompt not found: {judge_path}")
        eval_prompt = judge_path.read_text()
        trait_judge = args.trait_judge
    elif args.eval_prompt_from:
        override_data = load_steering_data(args.eval_prompt_from)
        eval_prompt = override_data.eval_prompt
        if not eval_prompt:
            print(f"Warning: {args.eval_prompt_from} has no eval_prompt, using default")
            use_default = True

    return eval_prompt, trait_judge, use_default


def _load_or_init_results(experiment, trait, model_variant, steering_data,
                          model_name, vector_experiment, judge_provider,
                          position, prompt_set, direction, force, trait_judge,
                          n_questions, regenerate_responses=False):
    """Load existing results or init new results file.

    Returns (cached_runs, baseline_result, direction) or None on skip.
    """
    cached_runs = []
    baseline_result = None
    results_path = get_steering_results_path(experiment, trait, model_variant, position, prompt_set)

    if force and results_path.exists():
        if is_rank_zero():
            import shutil
            steering_dir = get_steering_dir(experiment, trait, model_variant, position, prompt_set)
            shutil.rmtree(steering_dir)
            print(f"  --force: cleared {steering_dir}")
        tp_barrier()

    if results_path.exists():
        results_data = load_results(experiment, trait, model_variant, position, prompt_set)
        cached_runs = results_data.get("runs", [])
        baseline_result = results_data.get("baseline")
        header_direction = results_data.get("direction", "positive")
        if regenerate_responses:
            direction = header_direction
        elif header_direction != direction:
            if prompt_set == "steering":
                raise ValueError(
                    f"Direction mismatch for {trait}: results file has '{header_direction}' "
                    f"but --direction {direction}. Use --force to start fresh."
                )
            print(f"\n⚠️  Warning: direction '{direction}' differs from results file '{header_direction}'")
    elif regenerate_responses:
        print(f"  No results file for {trait}, skipping")
        return None
    else:
        if is_rank_zero():
            init_results_file(
                experiment, trait, model_variant, steering_data.prompts_file,
                model_name, vector_experiment, judge_provider, position, prompt_set,
                trait_judge=trait_judge, direction=direction, n_questions=n_questions,
            )
        tp_barrier()

    return cached_runs, baseline_result, direction


def _load_vectors(vector_experiment, trait, layers, extraction_variant,
                  method, component, position, cached_norms,
                  model, tokenizer, questions, use_chat_template):
    """Load vectors and compute base coefficients for each layer."""
    layer_data = []
    for layer in layers:
        vector = load_vector(vector_experiment, trait, layer, extraction_variant, method, component, position)
        if vector is None:
            continue
        vec_norm = vector.norm().item()
        if vec_norm == 0:
            continue
        if layer in cached_norms:
            act_norm = cached_norms[layer]
        else:
            act_norm = estimate_activation_norm(model, tokenizer, questions, layer, use_chat_template, "residual")
        base_coef = act_norm / vec_norm
        layer_data.append({"layer": layer, "vector": vector, "base_coef": base_coef})
        print(f"  L{layer}: base_coef={base_coef:.0f}")
    return layer_data


# =============================================================================
# Evaluation dispatch helpers
# =============================================================================

def _regenerate_responses(layer_data, cached_runs, questions, model, tokenizer,
                          use_chat_template, component, direction, min_coherence,
                          experiment, trait, model_variant, position, prompt_set,
                          max_new_tokens):
    """Re-generate response files from cached best configs (no judge scoring)."""
    sign = 1 if direction == "positive" else -1
    best_per_layer = {}
    for run in cached_runs:
        vectors = run.get("config", {}).get("vectors", [])
        if not vectors:
            continue
        layer = vectors[0]["layer"]
        t = (run.get("result", {}).get("trait_mean") or 0)
        c = (run.get("result", {}).get("coherence_mean") or 0)
        if c < min_coherence:
            continue
        prev = best_per_layer.get(layer)
        if prev is None or t * sign > (prev["result"].get("trait_mean") or 0) * sign:
            best_per_layer[layer] = run

    all_configs = []
    for ld in layer_data:
        run = best_per_layer.get(ld["layer"])
        if run:
            coef = run["config"]["vectors"][0]["weight"]
            all_configs.append((ld, coef, run["config"]))

    if not all_configs:
        print("No configs to regenerate")
        return

    print(f"\nRegenerating {len(all_configs)} response files...")
    formatted = [format_prompt(q, tokenizer, use_chat_template=use_chat_template) for q in questions]
    all_responses = batched_steering_generate(
        model, tokenizer, formatted,
        [(ld["layer"], ld["vector"], coef) for ld, coef, _ in all_configs],
        component=component, max_new_tokens=max_new_tokens,
    )

    n_q = len(questions)
    for idx, (ld, coef, config) in enumerate(all_configs):
        resps = all_responses[idx * n_q:(idx + 1) * n_q]
        timestamp = datetime.now().isoformat()
        responses = [{"prompt": q, "response": r, "system_prompt": None,
                      "trait_score": None, "coherence_score": None}
                     for q, r in zip(questions, resps)]
        save_responses(responses, experiment, trait, model_variant, position, prompt_set, config, timestamp)
        print(f"  L{ld['layer']} c{coef:.0f}: saved")


async def _evaluate_manual_coefficients(
    backend, layer_data, coefficients, questions, steering_data,
    judge, use_chat_template, component, method, position,
    cached_runs, experiment, trait, model_variant, prompt_set,
    save_mode, min_coherence, direction, trait_judge,
    max_new_tokens=64, eval_prompt=None, relevance_check=True,
):
    """Evaluate specific coefficients for each layer, score, and save results."""
    model, tokenizer = backend.model, backend.tokenizer
    n_q = len(questions)

    # Build configs, filter cached
    all_configs = []
    for ld in layer_data:
        for coef in coefficients:
            config = {"vectors": [VectorSpec(layer=ld["layer"], component=component, position=position, method=method, weight=coef).to_dict()]}
            if find_cached_run(cached_runs, config) is not None:
                print(f"  L{ld['layer']} c{coef:.0f}: cached")
            else:
                all_configs.append((ld, coef, config))

    if not all_configs:
        return

    print(f"\nEvaluating {len(all_configs)} configs in batch...")
    formatted = [format_prompt(q, tokenizer, use_chat_template=use_chat_template) for q in questions]
    all_responses = batched_steering_generate(
        model, tokenizer, formatted,
        [(ld["layer"], ld["vector"], coef) for ld, coef, _ in all_configs],
        component=component, max_new_tokens=max_new_tokens,
    )

    # Score (rank-0 only, then broadcast in TP mode)
    tp = is_tp_mode()
    all_scores = None
    if not tp or is_rank_zero():
        all_qa_pairs = [(q, all_responses[i * n_q + j]) for i in range(len(all_configs)) for j, q in enumerate(questions)]
        print(f"Scoring {len(all_qa_pairs)} responses...")
        all_scores = await judge.score_steering_batch(
            all_qa_pairs, steering_data.trait_name, steering_data.trait_definition,
            eval_prompt=eval_prompt, relevance_check=relevance_check,
        )

    # Broadcast per-config summaries in TP mode
    config_summaries = None
    if tp:
        import torch.distributed as dist
        if is_rank_zero():
            config_summaries = []
            for idx in range(len(all_configs)):
                scores = all_scores[idx * n_q:(idx + 1) * n_q]
                trait_scores = [s["trait_score"] for s in scores if s["trait_score"] is not None]
                coh_scores = [s["coherence_score"] for s in scores if s.get("coherence_score") is not None]
                config_summaries.append({
                    "trait_mean": sum(trait_scores) / len(trait_scores) if trait_scores else None,
                    **score_stats(trait_scores),
                    "coherence_mean": sum(coh_scores) / len(coh_scores) if coh_scores else None,
                    "n": len(trait_scores),
                })
        broadcast_list = [config_summaries]
        dist.broadcast_object_list(broadcast_list, src=0)
        config_summaries = broadcast_list[0]

    # Process results
    best_per_layer = {}
    for idx, (ld, coef, config) in enumerate(all_configs):
        if tp:
            result = config_summaries[idx]
        else:
            scores = all_scores[idx * n_q:(idx + 1) * n_q]
            trait_scores = [s["trait_score"] for s in scores if s["trait_score"] is not None]
            coh_scores = [s["coherence_score"] for s in scores if s.get("coherence_score") is not None]
            result = {
                "trait_mean": sum(trait_scores) / len(trait_scores) if trait_scores else None,
                **score_stats(trait_scores),
                "coherence_mean": sum(coh_scores) / len(coh_scores) if coh_scores else None,
                "n": len(trait_scores),
            }

        timestamp = datetime.now().isoformat()
        print(f"  L{ld['layer']} c{coef:.0f}: trait={result['trait_mean'] or 0:.1f}, coherence={result['coherence_mean'] or 0:.1f}")

        cached_runs.append({"config": config, "result": result, "timestamp": timestamp})

        if is_rank_zero():
            append_run(experiment, trait, model_variant, config, result, position, prompt_set, trait_judge=trait_judge)

            resps = all_responses[idx * n_q:(idx + 1) * n_q]
            scores_slice = all_scores[idx * n_q:(idx + 1) * n_q]
            responses = [{"prompt": q, "response": r, "system_prompt": None,
                          "trait_score": s["trait_score"], "coherence_score": s.get("coherence_score")}
                         for q, r, s in zip(questions, resps, scores_slice)]

            if save_mode == "all":
                save_responses(responses, experiment, trait, model_variant, position, prompt_set, config, timestamp)
            elif save_mode == "best":
                t_mean = result.get("trait_mean") or 0
                c_mean = result.get("coherence_mean") or 0
                if is_better_result(best_per_layer.get(ld["layer"]), t_mean, c_mean, min_coherence, direction):
                    best_per_layer[ld["layer"]] = {"responses": responses, "config": config, "timestamp": timestamp}

    if save_mode == "best" and is_rank_zero():
        for best in best_per_layer.values():
            if best.get("responses"):
                save_responses(best["responses"], experiment, trait, model_variant, position, prompt_set, best["config"], best["timestamp"])


def _print_eval_summary(cached_runs, baseline_result, direction, min_coherence):
    """Print evaluation summary with best result."""
    sign = 1 if direction == "positive" else -1
    print(f"\n{'='*60}")
    print(f"Summary (direction={direction})")
    print(f"{'='*60}")
    _btm = baseline_result['trait_mean']
    print(f"Baseline: {float(_btm):.1f}" if _btm is not None else "Baseline: None")
    print(f"Total runs: {len(cached_runs)}")

    valid_runs = [r for r in cached_runs if r.get('result', {}).get('coherence_mean', 0) >= min_coherence]
    any_below = any(r.get('result', {}).get('coherence_mean', 0) < min_coherence for r in cached_runs)

    if valid_runs:
        best_run = max(valid_runs, key=lambda r: (r.get('result', {}).get('trait_mean') or 0) * sign)
        score = best_run['result']['trait_mean']
        coh = best_run['result'].get('coherence_mean', 0)
        delta = (score - _btm) if (_btm is not None and score is not None) else None
        layer = best_run['config']['vectors'][0]['layer']
        coef = best_run['config']['vectors'][0]['weight']
        print(f"Best (coherence≥{min_coherence:.0f}): L{layer} c{coef:.0f}")
        if delta is not None:
            print(f"  trait={score:.1f} ({'+' if delta >= 0 else ''}{delta:.1f}), coherence={coh:.1f}")
        else:
            print(f"  trait={score or 0:.1f} (baseline=None), coherence={coh:.1f}")
    else:
        print(f"No valid runs with coherence≥{min_coherence:.0f}")

    if not any_below:
        print(f"  WARNING: coherence never dropped below {min_coherence:.0f} — may not have steered hard enough")


# =============================================================================
# Core Evaluation (the recipe)
# =============================================================================

async def run_evaluation(
    experiment, trait, vector_experiment, model_variant, layers_arg,
    coefficients, method, component, position, prompt_set, model_name,
    judge_provider, subset, n_search_steps, up_mult, down_mult,
    start_mult=0.7, momentum=0.0, batched=True, backend=None, judge=None,
    load_in_8bit=False, load_in_4bit=False, bnb_4bit_quant_type="nf4",
    lora_adapter=None, max_new_tokens=64, eval_prompt=None,
    use_default_prompt=False, min_coherence=MIN_COHERENCE,
    extraction_variant=None, save_mode="best", ablation=False,
    ablation_vector_layer=None, relevance_check=True, trait_judge=None,
    direction="positive", force=False, questions_file=None,
    regenerate_responses=False,
):
    """Main evaluation flow for a single trait.

    If coefficients provided: evaluate those directly.
    Otherwise: run adaptive search to find good coefficients.
    """
    # Dispatch to ablation mode
    if ablation:
        return await run_ablation_evaluation(
            experiment=experiment, trait=trait, vector_experiment=vector_experiment,
            model_variant=model_variant, vector_layer=ablation_vector_layer,
            method=method, component=component, position=position,
            prompt_set=prompt_set, model_name=model_name, judge_provider=judge_provider,
            subset=None if subset == 5 else subset,
            backend=backend, judge=judge,
            load_in_8bit=load_in_8bit, load_in_4bit=load_in_4bit,
            bnb_4bit_quant_type=bnb_4bit_quant_type,
            lora_adapter=lora_adapter, max_new_tokens=max_new_tokens,
            eval_prompt=eval_prompt, use_default_prompt=use_default_prompt,
            extraction_variant=extraction_variant, relevance_check=relevance_check,
        )

    # --- Load data ---
    questions, steering_data = _resolve_questions(trait, questions_file, prompt_set, subset)
    effective_eval_prompt = _resolve_eval_prompt(steering_data, eval_prompt, use_default_prompt)

    # --- Init model ---
    should_close_judge = False
    if backend is None:
        model, tokenizer = load_model_with_lora(model_name, load_in_8bit=load_in_8bit,
                                                 load_in_4bit=load_in_4bit, bnb_4bit_quant_type=bnb_4bit_quant_type,
                                                 lora_adapter=lora_adapter)
        backend = LocalBackend.from_model(model, tokenizer)
    model, tokenizer, num_layers = backend.model, backend.tokenizer, backend.n_layers

    config = load_experiment_config(experiment)
    use_chat_template = config.get('use_chat_template')
    if use_chat_template is None:
        use_chat_template = tokenizer.chat_template is not None

    # --- Parse layers ---
    if regenerate_responses:
        layers = list(range(num_layers))
    else:
        layers = parse_layers(layers_arg, num_layers)
    layers = [l for l in layers if 0 <= l < num_layers]
    if not layers:
        raise ValueError(f"No valid layers. Model has {num_layers} layers (0-{num_layers-1})")

    # --- Init/resume results ---
    result = _load_or_init_results(
        experiment, trait, model_variant, steering_data, model_name,
        vector_experiment, judge_provider, position, prompt_set,
        direction, force, trait_judge, len(questions), regenerate_responses,
    )
    if result is None:
        return
    cached_runs, baseline_result, direction = result

    # Filter layers for regeneration mode
    if regenerate_responses and cached_runs:
        good_layers = {run["config"]["vectors"][0]["layer"]
                       for run in cached_runs
                       if (run.get("result", {}).get("coherence_mean") or 0) >= min_coherence}
        layers = sorted(l for l in layers if l in good_layers)
        if not layers:
            print(f"  No cached configs with coherence >= {min_coherence}, skipping")
            return

    # --- Judge ---
    if judge is None and not regenerate_responses:
        if is_rank_zero():
            judge = TraitJudge()
        should_close_judge = True

    print(f"\nTrait: {trait}")
    print(f"Model: {model_name} ({num_layers} layers)")
    print(f"Vectors from: {vector_experiment}/{trait} @ {position}")
    print(f"Questions: {len(questions)}, Existing runs: {len(cached_runs)}")

    # --- Baseline ---
    if not regenerate_responses and baseline_result is None:
        baseline_result, baseline_responses = await compute_baseline(
            backend, questions, steering_data.trait_name, steering_data.trait_definition,
            judge, max_new_tokens=max_new_tokens, eval_prompt=effective_eval_prompt,
        )
        if is_rank_zero():
            save_baseline_responses(baseline_responses, experiment, trait, model_variant, position, prompt_set)
            append_baseline(experiment, trait, model_variant, baseline_result, position, prompt_set, trait_judge=trait_judge)
        tp_barrier()
    elif baseline_result is not None:
        _btm = baseline_result['trait_mean']
        print(f"\nUsing existing baseline: trait={f'{float(_btm):.1f}' if _btm is not None else 'None'}")

    # --- Load vectors ---
    cached_norms = load_cached_activation_norms(vector_experiment, "residual")
    if cached_norms:
        print(f"\nUsing cached activation norms (residual)")
    resolved_extraction_variant = extraction_variant or get_default_variant(vector_experiment, mode='extraction')

    print(f"Loading vectors...")
    layer_data = _load_vectors(
        vector_experiment, trait, layers, resolved_extraction_variant,
        method, component, position, cached_norms,
        model, tokenizer, questions, use_chat_template,
    )
    if not layer_data:
        print("No valid layers with vectors found")
        return

    # --- Dispatch ---
    if regenerate_responses:
        _regenerate_responses(
            layer_data, cached_runs, questions, model, tokenizer,
            use_chat_template, component, direction, min_coherence,
            experiment, trait, model_variant, position, prompt_set, max_new_tokens,
        )
    elif coefficients is not None:
        await _evaluate_manual_coefficients(
            backend, layer_data, coefficients, questions, steering_data,
            judge, use_chat_template, component, method, position,
            cached_runs, experiment, trait, model_variant, prompt_set,
            save_mode, min_coherence, direction, trait_judge,
            max_new_tokens, effective_eval_prompt, relevance_check,
        )
    elif batched and len(layer_data) > 1:
        await batched_adaptive_search(
            backend, layer_data, questions, steering_data.trait_name, steering_data.trait_definition,
            judge, use_chat_template, component, cached_runs, experiment, trait, model_variant,
            vector_experiment, method, position=position, prompt_set=prompt_set, n_steps=n_search_steps,
            up_mult=up_mult, down_mult=down_mult, start_mult=start_mult, momentum=momentum,
            max_new_tokens=max_new_tokens, eval_prompt=effective_eval_prompt,
            save_mode=save_mode, coherence_threshold=min_coherence,
            relevance_check=relevance_check, direction=direction, trait_judge=trait_judge,
        )
    else:
        print(f"\nSequential adaptive search ({n_search_steps} steps per layer)")
        for ld in layer_data:
            await adaptive_search_layer(
                backend, ld["vector"], ld["layer"], ld["base_coef"],
                questions, steering_data.trait_name, steering_data.trait_definition,
                judge, use_chat_template, component,
                cached_runs, experiment, trait, model_variant, vector_experiment, method,
                position=position, prompt_set=prompt_set, n_steps=n_search_steps,
                up_mult=up_mult, down_mult=down_mult, start_mult=start_mult, momentum=momentum,
                max_new_tokens=max_new_tokens, eval_prompt=effective_eval_prompt,
                save_mode=save_mode, coherence_threshold=min_coherence,
                relevance_check=relevance_check, direction=direction, trait_judge=trait_judge,
            )

    # --- Summary ---
    _print_eval_summary(cached_runs, baseline_result, direction, min_coherence)

    if should_close_judge and judge is not None:
        await judge.close()


# =============================================================================
# Orchestration
# =============================================================================

async def _run_baselines(args, parsed_traits, model_variant, model_name,
                         backend, judge, eval_prompt_override, trait_judge,
                         use_default, force):
    """Compute baselines only (no steering, no vectors)."""
    print(f"\n{'='*60}")
    print(f"BASELINE ONLY MODE")
    print(f"{'='*60}")
    print(f"Traits: {len(parsed_traits)}, Model: {model_name}")

    summary = []
    for vector_experiment, trait in parsed_traits:
        print(f"\n--- {trait} ---")
        questions, steering_data = _resolve_questions(trait, args.questions_file, args.prompt_set, args.subset)

        if use_default:
            trait_eval_prompt = None
        elif eval_prompt_override is not None:
            trait_eval_prompt = eval_prompt_override
        else:
            trait_eval_prompt = steering_data.eval_prompt

        # Check existing baseline
        existing = get_baseline(args.experiment, trait, model_variant, args.position, args.prompt_set)
        if existing and not force:
            print(f"  Existing baseline: trait={existing['trait_mean']:.1f}, "
                  f"coh={existing.get('coherence_mean', 0):.1f}, n={existing['n']}")
            summary.append((trait, existing['trait_mean'], existing.get('coherence_mean'), existing['n'], "cached"))
            continue

        if existing and force:
            if is_rank_zero():
                remove_baseline(args.experiment, trait, model_variant, args.position, args.prompt_set)
                print(f"  --force: removed existing baseline")
            tp_barrier()

        # Init results file if needed
        results_path = get_steering_results_path(args.experiment, trait, model_variant, args.position, args.prompt_set)
        if not results_path.exists():
            if is_rank_zero():
                init_results_file(
                    args.experiment, trait, model_variant, steering_data.prompts_file,
                    model_name, vector_experiment, args.judge, args.position,
                    args.prompt_set, trait_judge=trait_judge, direction="positive",
                    n_questions=len(questions),
                )
            tp_barrier()

        # Compute baseline
        print(f"  Questions: {len(questions)}")
        baseline_result, baseline_responses = await compute_baseline(
            backend, questions, steering_data.trait_name, steering_data.trait_definition,
            judge, max_new_tokens=args.max_new_tokens, eval_prompt=trait_eval_prompt,
            relevance_check=not args.no_relevance_check,
        )
        if is_rank_zero():
            save_baseline_responses(baseline_responses, args.experiment, trait, model_variant, args.position, args.prompt_set)
            append_baseline(args.experiment, trait, model_variant, baseline_result, args.position, args.prompt_set, trait_judge=trait_judge)

        summary.append((trait, baseline_result['trait_mean'], baseline_result.get('coherence_mean'), baseline_result['n'], "computed"))

    # Summary table
    print(f"\n{'='*60}")
    print(f"BASELINE SUMMARY")
    print(f"{'='*60}")
    print(f"{'Trait':<40} {'Baseline':>8} {'Coh':>6} {'N':>3} {'Status':>8}")
    print(f"{'-'*40} {'-'*8} {'-'*6} {'-'*3} {'-'*8}")
    for trait, b_mean, c_mean, n, status in summary:
        c_str = f"{c_mean:.1f}" if c_mean is not None else "N/A"
        print(f"{trait:<40} {b_mean:>8.1f} {c_str:>6} {n:>3} {status:>8}")

    return summary


async def _run_batched_multi_trait(args, parsed_traits, model_variant, model_name,
                                   backend, judge, eval_prompt_override, trait_judge,
                                   use_default, direction, force, trait_layers, layers_arg):
    """Multi-trait batched evaluation: prepare all traits, search all at once."""
    model, tokenizer, num_layers = backend.model, backend.tokenizer, backend.n_layers

    config = load_experiment_config(args.experiment)
    use_chat_template = config.get('use_chat_template')
    if use_chat_template is None:
        use_chat_template = tokenizer.chat_template is not None

    default_layers = parse_layers(layers_arg, num_layers)
    default_layers = [l for l in default_layers if 0 <= l < num_layers]
    if not default_layers and not trait_layers:
        raise ValueError(f"No valid layers. Model has {num_layers} layers (0-{num_layers-1})")

    # Pre-parse per-trait layer overrides
    parsed_trait_layers = {}
    if trait_layers:
        for trait_key, layer_spec in trait_layers.items():
            tl = parse_layers(layer_spec, num_layers)
            tl = [l for l in tl if 0 <= l < num_layers]
            if tl:
                parsed_trait_layers[trait_key] = tl

    cached_norms = load_cached_activation_norms(args.vector_experiment or args.experiment, "residual")
    resolved_extraction_variant = (
        args.extraction_variant or get_default_variant(args.vector_experiment or args.experiment, mode='extraction')
    )

    print(f"\nMulti-trait batched mode: {len(parsed_traits)} traits")
    print(f"Model: {model_name} ({num_layers} layers)")
    if parsed_trait_layers:
        print(f"Default layers: {default_layers}")
        print(f"Per-trait overrides: {len(parsed_trait_layers)} traits")
    else:
        print(f"Layers: {default_layers}")

    # Prepare each trait
    trait_configs = []
    for vector_experiment, trait in parsed_traits:
        print(f"\n--- Preparing {trait} ---")

        questions, steering_data = _resolve_questions(trait, args.questions_file, args.prompt_set, args.subset)

        if use_default:
            trait_eval_prompt = None
        elif eval_prompt_override is not None:
            trait_eval_prompt = eval_prompt_override
        else:
            trait_eval_prompt = steering_data.eval_prompt

        result = _load_or_init_results(
            args.experiment, trait, model_variant, steering_data, model_name,
            vector_experiment, args.judge, args.position, args.prompt_set,
            direction, force, trait_judge, len(questions),
        )
        if result is None:
            continue
        cached_runs, baseline_result, _ = result

        # Compute baseline if needed
        if baseline_result is None:
            baseline_result, baseline_responses = await compute_baseline(
                backend, questions, steering_data.trait_name, steering_data.trait_definition,
                judge, max_new_tokens=args.max_new_tokens, eval_prompt=trait_eval_prompt,
            )
            if is_rank_zero():
                save_baseline_responses(baseline_responses, args.experiment, trait, model_variant, args.position, args.prompt_set)
                append_baseline(args.experiment, trait, model_variant, baseline_result, args.position, args.prompt_set, trait_judge=trait_judge)
            tp_barrier()
        else:
            _bm = baseline_result.get('trait_mean')
            print(f"  Existing baseline: trait={f'{float(_bm):.1f}' if _bm is not None else 'None'}")

        # Load vectors
        trait_layer_list = parsed_trait_layers.get(trait, default_layers)
        if trait in parsed_trait_layers:
            print(f"  Layers (override): {trait_layer_list}")

        layer_data = _load_vectors(
            vector_experiment, trait, trait_layer_list, resolved_extraction_variant,
            args.method, args.component, args.position, cached_norms,
            model, tokenizer, questions, use_chat_template,
        )
        if not layer_data:
            print(f"  No valid vectors for {trait}, skipping")
            continue

        formatted_questions = [format_prompt(q, tokenizer, use_chat_template=use_chat_template) for q in questions]
        print(f"  {len(questions)} questions, {len(layer_data)} layers, {len(cached_runs)} cached runs")

        trait_configs.append({
            "trait": trait, "trait_name": steering_data.trait_name,
            "trait_definition": steering_data.trait_definition,
            "eval_prompt": trait_eval_prompt, "questions": questions,
            "formatted_questions": formatted_questions,
            "layer_data": layer_data, "cached_runs": cached_runs,
            "experiment": args.experiment, "vector_experiment": vector_experiment,
        })

    if trait_configs:
        await multi_trait_batched_adaptive_search(
            backend=backend, trait_configs=trait_configs, judge=judge,
            use_chat_template=use_chat_template, component=args.component,
            model_variant=model_variant, method=args.method, position=args.position,
            prompt_set=args.prompt_set, n_steps=args.search_steps,
            up_mult=args.up_mult, down_mult=args.down_mult,
            start_mult=args.start_mult, momentum=args.momentum,
            max_new_tokens=args.max_new_tokens, save_mode=args.save_responses,
            coherence_threshold=args.min_coherence,
            relevance_check=not args.no_relevance_check,
            direction=direction, trait_judge=trait_judge,
        )


async def _run_main(args, parsed_traits, model_variant, model_name, lora,
                    layers_arg=None, coefficients=None, direction="positive",
                    force=False, backend=None, judge=None, trait_layers=None,
                    baseline_only=False):
    """Main async entry point. Loads model once, evaluates traits."""
    multi_trait = len(parsed_traits) > 1
    _owns_backend = backend is None

    use_batched_path = (not baseline_only
                        and not args.no_batch
                        and coefficients is None
                        and args.ablation is None
                        and not args.regenerate_responses)

    # Load model once for all traits
    if backend is None and (multi_trait or use_batched_path or baseline_only):
        if multi_trait:
            print(f"\nEvaluating {len(parsed_traits)} traits with shared model")
        backend = LocalBackend.from_experiment(
            args.experiment, variant=model_variant,
            load_in_8bit=args.load_in_8bit, load_in_4bit=args.load_in_4bit,
            bnb_4bit_quant_type=args.bnb_4bit_quant_type,
        )

    if judge is None and is_rank_zero() and not args.regenerate_responses:
        judge = TraitJudge()

    effective_eval_prompt, trait_judge, use_default = _resolve_cli_eval_prompt(args)

    try:
        if baseline_only:
            await _run_baselines(
                args, parsed_traits, model_variant, model_name,
                backend, judge, effective_eval_prompt, trait_judge, use_default, force,
            )
        elif use_batched_path:
            await _run_batched_multi_trait(
                args, parsed_traits, model_variant, model_name, backend, judge,
                effective_eval_prompt, trait_judge, use_default, direction, force,
                trait_layers, layers_arg,
            )
        else:
            # Per-trait sequential path
            for vector_experiment, trait in parsed_traits:
                if multi_trait:
                    print(f"\n{'='*60}")
                    print(f"TRAIT: {vector_experiment}/{trait}")
                    print(f"{'='*60}")

                effective_layers_arg = trait_layers[trait] if (trait_layers and trait in trait_layers) else layers_arg

                await run_evaluation(
                    experiment=args.experiment, trait=trait, vector_experiment=vector_experiment,
                    model_variant=model_variant, layers_arg=effective_layers_arg,
                    coefficients=coefficients, method=args.method, component=args.component,
                    position=args.position, prompt_set=args.prompt_set, model_name=model_name,
                    judge_provider=args.judge, subset=args.subset,
                    n_search_steps=args.search_steps, up_mult=args.up_mult,
                    down_mult=args.down_mult, start_mult=args.start_mult,
                    momentum=args.momentum, batched=not args.no_batch,
                    backend=backend, judge=judge,
                    load_in_8bit=args.load_in_8bit, load_in_4bit=args.load_in_4bit,
                    bnb_4bit_quant_type=args.bnb_4bit_quant_type,
                    lora_adapter=lora, max_new_tokens=args.max_new_tokens,
                    eval_prompt=effective_eval_prompt, use_default_prompt=use_default,
                    min_coherence=args.min_coherence, extraction_variant=args.extraction_variant,
                    save_mode=args.save_responses,
                    ablation=args.ablation is not None, ablation_vector_layer=args.ablation,
                    relevance_check=not args.no_relevance_check, trait_judge=trait_judge,
                    direction=direction, force=force,
                    questions_file=args.questions_file,
                    regenerate_responses=args.regenerate_responses,
                )
    finally:
        if judge is not None:
            await judge.close()
        if _owns_backend and backend is not None:
            del backend
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

    if multi_trait:
        print(f"\n{'='*60}")
        print(f"COMPLETED {len(parsed_traits)} TRAITS")
        print(f"{'='*60}")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Steering evaluation")

    # === Core (required) ===
    parser.add_argument("--experiment", required=True)
    trait_group = parser.add_mutually_exclusive_group(required=True)
    trait_group.add_argument("--vector-from-trait", help="Single trait: 'experiment/category/trait'")
    trait_group.add_argument("--traits", help="Multiple traits (comma-separated)")
    trait_group.add_argument("--rescore", help="Re-score existing responses (no GPU)")

    # === Input/Output ===
    parser.add_argument("--prompt-set", default="steering")
    parser.add_argument("--questions-file", type=str, default=None)

    # === Model ===
    parser.add_argument("--model-variant", default=None)
    parser.add_argument("--extraction-variant", default=None)
    parser.add_argument("--load-in-8bit", action="store_true")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--bnb-4bit-quant-type", default="nf4")
    parser.add_argument("--vector-experiment", default=None,
                        help="Experiment to load vectors from (defaults to --experiment)")
    add_backend_args(parser)

    # === Vector Specification ===
    parser.add_argument("--method", default="probe")
    parser.add_argument("--component", default="residual",
                        choices=["residual", "attn_out", "mlp_out", "attn_contribution", "mlp_contribution", "k_proj", "v_proj"])
    parser.add_argument("--position", default="response[:5]")

    # === Evaluation ===
    parser.add_argument("--subset", type=int, default=5, help="Use subset of questions (0 for all)")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--judge", default="openai", choices=["openai", "gemini"])
    parser.add_argument("--min-coherence", type=float, default=MIN_COHERENCE)
    prompt_group = parser.add_mutually_exclusive_group()
    prompt_group.add_argument("--no-custom-prompt", action="store_true")
    prompt_group.add_argument("--eval-prompt-from", type=str, metavar="TRAIT_PATH")
    prompt_group.add_argument("--trait-judge", type=str, metavar="JUDGE_PATH")

    # === Search/Optimization ===
    parser.add_argument("--layers", default="30%-60%")
    parser.add_argument("--trait-layers", nargs="+", metavar="TRAIT:LAYERS")
    parser.add_argument("--coefficients", help="Manual coefficients (comma-separated)")
    parser.add_argument("--search-steps", type=int, default=5)
    parser.add_argument("--up-mult", type=float, default=1.3)
    parser.add_argument("--down-mult", type=float, default=0.85)
    parser.add_argument("--start-mult", type=float, default=0.7)
    parser.add_argument("--momentum", type=float, default=0.1)

    # === Advanced ===
    parser.add_argument("--regenerate-responses", action="store_true")
    parser.add_argument("--baseline-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-batch", action="store_true")
    parser.add_argument("--save-responses", choices=["all", "best", "none"], default="best")
    parser.add_argument("--ablation", type=int, metavar="LAYER", default=None)
    parser.add_argument("--no-relevance-check", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--direction", choices=["positive", "negative"], default=None)

    args = parser.parse_args()

    # TP lifecycle
    import builtins
    _original_print = builtins.print
    if is_tp_mode():
        import torch.distributed as dist
        if not dist.is_initialized():
            dist.init_process_group("nccl")
        if not is_rank_zero():
            builtins.print = lambda *a, **k: None

    # Handle --rescore mode (no GPU)
    if args.rescore:
        variant = get_model_variant(args.experiment, args.model_variant, mode="application")
        model_variant = variant['name']

        effective_eval_prompt = None
        trait_judge = None
        if args.trait_judge:
            judge_path = get('datasets.llm_judge_trait', judge_prompt=args.trait_judge)
            if not judge_path.exists():
                raise FileNotFoundError(f"Trait judge prompt not found: {judge_path}")
            effective_eval_prompt = judge_path.read_text()
            trait_judge = args.trait_judge
        elif args.no_custom_prompt:
            effective_eval_prompt = None

        asyncio.run(run_rescore(
            experiment=args.experiment, trait=args.rescore,
            model_variant=model_variant, position=args.position,
            prompt_set=args.prompt_set, eval_prompt=effective_eval_prompt,
            relevance_check=not args.no_relevance_check,
            trait_judge=trait_judge, dry_run=args.dry_run,
        ))
        return

    # Parse trait specs
    if args.traits:
        trait_specs = [t.strip() for t in args.traits.split(',')]
    else:
        trait_specs = [args.vector_from_trait]

    vector_exp_default = args.vector_experiment or args.experiment
    parsed_traits = []
    for spec in trait_specs:
        parts = spec.split('/')
        if len(parts) == 2:
            parsed_traits.append((vector_exp_default, spec))
        elif len(parts) == 3:
            parsed_traits.append((parts[0], f"{parts[1]}/{parts[2]}"))
        else:
            parser.error(f"Invalid trait spec '{spec}': use 'category/trait' or 'experiment/category/trait'")

    # Parse per-trait layer overrides
    trait_layers = None
    if args.trait_layers:
        trait_layers = {}
        for spec in args.trait_layers:
            if ':' not in spec:
                parser.error(f"Invalid --trait-layers spec '{spec}': use 'category/trait:layers'")
            trait_part, layer_spec = spec.rsplit(':', 1)
            tparts = trait_part.split('/')
            if len(tparts) == 3:
                trait_part = f"{tparts[1]}/{tparts[2]}"
            trait_layers[trait_part] = layer_spec

    # Auto-derive prompt-set from questions-file
    if args.questions_file and args.prompt_set == "steering":
        args.prompt_set = Path(args.questions_file).stem
        print(f"Auto-set --prompt-set to '{args.prompt_set}' (from --questions-file)")

    # Resolve model variant
    variant = get_model_variant(args.experiment, args.model_variant, mode="application")
    model_variant = variant['name']
    model_name = variant['model']
    lora = variant.get('lora')

    coefficients = parse_coefficients(args.coefficients)

    # Determine direction
    if args.direction:
        direction = args.direction
    elif coefficients:
        all_non_positive = all(c <= 0 for c in coefficients)
        any_negative = any(c < 0 for c in coefficients)
        direction = "negative" if (all_non_positive and any_negative) else "positive"
    else:
        trait_directions = set()
        for _, trait in parsed_traits:
            try:
                sd = load_steering_data(trait)
                trait_directions.add(sd.direction or "positive")
            except (FileNotFoundError, ValueError):
                trait_directions.add("positive")
        if len(trait_directions) == 1:
            direction = trait_directions.pop()
            if direction != "positive":
                print(f"Direction from steering.json: {direction}")
        elif len(trait_directions) > 1:
            raise ValueError(
                f"Mixed directions across traits: {trait_directions}. "
                f"Use --direction to specify, or run traits with different directions separately."
            )
        else:
            direction = "positive"

    # Run
    try:
        asyncio.run(_run_main(
            args=args, parsed_traits=parsed_traits,
            model_variant=model_variant, model_name=model_name, lora=lora,
            layers_arg=args.layers, coefficients=coefficients,
            direction=direction, force=args.force, trait_layers=trait_layers,
            baseline_only=args.baseline_only,
        ))
    finally:
        builtins.print = _original_print
        if is_tp_mode():
            import torch.distributed as dist
            if dist.is_initialized():
                dist.destroy_process_group()


# =============================================================================
# Ablation mode (rarely used — kept below main flow)
# =============================================================================

async def run_ablation_evaluation(
    experiment, trait, vector_experiment, model_variant, vector_layer,
    method, component, position, prompt_set, model_name, judge_provider,
    subset, backend=None, judge=None,
    load_in_8bit=False, load_in_4bit=False, bnb_4bit_quant_type="nf4",
    lora_adapter=None, max_new_tokens=64, eval_prompt=None,
    use_default_prompt=False, extraction_variant=None, relevance_check=True,
):
    """Directional ablation: ablate a direction at ALL layers simultaneously."""
    questions, steering_data = _resolve_questions(trait, None, prompt_set, subset)
    effective_eval_prompt = _resolve_eval_prompt(steering_data, eval_prompt, use_default_prompt)

    should_close_judge = False
    if backend is None:
        model, tokenizer = load_model_with_lora(model_name, load_in_8bit=load_in_8bit,
                                                 load_in_4bit=load_in_4bit, bnb_4bit_quant_type=bnb_4bit_quant_type,
                                                 lora_adapter=lora_adapter)
        backend = LocalBackend.from_model(model, tokenizer)

    model, tokenizer = backend.model, backend.tokenizer
    config = load_experiment_config(experiment)
    use_chat_template = config.get('use_chat_template')
    if use_chat_template is None:
        use_chat_template = tokenizer.chat_template is not None

    if judge is None:
        judge = TraitJudge()
        should_close_judge = True

    resolved_extraction_variant = extraction_variant or get_default_variant(vector_experiment, mode='extraction')
    vector = load_vector(vector_experiment, trait, vector_layer, resolved_extraction_variant, method, component, position)
    if vector is None:
        raise ValueError(f"Vector not found: {vector_experiment}/{trait} L{vector_layer} {method}")

    print(f"\n{'='*60}")
    print(f"ABLATION EVALUATION")
    print(f"{'='*60}")
    print(f"Trait: {trait}, Vector: L{vector_layer} {method} (ablated at ALL layers)")
    print(f"Questions: {len(questions)}")

    # Compute baseline
    baseline, baseline_responses = await compute_baseline(
        backend, questions, steering_data.trait_name, steering_data.trait_definition,
        judge, max_new_tokens=max_new_tokens, eval_prompt=effective_eval_prompt,
        relevance_check=relevance_check,
    )

    # Generate with ablation
    print(f"\nGenerating with ablation at all layers...")
    formatted = [format_prompt(q, tokenizer, use_chat_template=use_chat_template) for q in questions]
    with MultiLayerAblationHook(model, vector):
        ablated_responses = generate_batch(model, tokenizer, formatted, max_new_tokens=max_new_tokens)

    # Score
    print(f"Scoring ablated responses...")
    all_scores = await judge.score_steering_batch(
        list(zip(questions, ablated_responses)), steering_data.trait_name, steering_data.trait_definition,
        eval_prompt=effective_eval_prompt, relevance_check=relevance_check,
    )

    ablated_trait_scores = [s["trait_score"] for s in all_scores if s["trait_score"] is not None]
    ablated_coherence_scores = [s["coherence_score"] for s in all_scores if s.get("coherence_score") is not None]
    ablated_trait_mean = sum(ablated_trait_scores) / len(ablated_trait_scores) if ablated_trait_scores else None
    ablated_coherence_mean = sum(ablated_coherence_scores) / len(ablated_coherence_scores) if ablated_coherence_scores else None

    # Save
    ablated_response_data = [
        {"prompt": q, "response": r, "system_prompt": None,
         "trait_score": s["trait_score"], "coherence_score": s.get("coherence_score")}
        for q, r, s in zip(questions, ablated_responses, all_scores)
    ]
    save_baseline_responses(baseline_responses, experiment, trait, model_variant, position, prompt_set)
    save_ablation_responses(
        ablated_response_data, experiment, trait, model_variant, position, prompt_set,
        vector_layer, method, component,
    )

    # Results
    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}")
    print(f"Baseline:  trait={baseline['trait_mean']:.1f}, coherence={baseline.get('coherence_mean', 0):.1f}")
    print(f"Ablated:   trait={ablated_trait_mean:.1f}, coherence={ablated_coherence_mean:.1f}")

    delta = ablated_trait_mean - baseline['trait_mean']
    if baseline['trait_mean'] > 50:
        reduction_pct = (baseline['trait_mean'] - ablated_trait_mean) / baseline['trait_mean'] * 100
        print(f"\nBypass: {baseline['trait_mean']:.1f} → {ablated_trait_mean:.1f} ({reduction_pct:.0f}% reduction)")
    else:
        print(f"\nDelta: {delta:+.1f}")
    print(f"Coherence: {baseline.get('coherence_mean', 0):.1f} → {ablated_coherence_mean:.1f}")

    if should_close_judge:
        await judge.close()

    return {"baseline": baseline, "ablated": {"trait_mean": ablated_trait_mean, "coherence_mean": ablated_coherence_mean, "n": len(ablated_trait_scores)}, "delta": delta}


# =============================================================================
# Rescore mode (no GPU — re-judge existing responses)
# =============================================================================

def discover_response_files(experiment, trait, model_variant, position, prompt_set):
    """Find all saved response files and parse config from path."""
    steering_dir = get_steering_dir(experiment, trait, model_variant, position, prompt_set)
    responses_dir = steering_dir / "responses"
    if not responses_dir.exists():
        return []

    found = []

    # Baseline
    baseline_path = responses_dir / "baseline.json"
    if baseline_path.exists():
        found.append({"path": baseline_path, "is_baseline": True})

    # Steered: responses/{component}/{method}/L{layer}_c{coef}_{timestamp}.json
    for component_dir in responses_dir.iterdir():
        if not component_dir.is_dir() or component_dir.name in ("ablation",):
            continue
        for method_dir in component_dir.iterdir():
            if not method_dir.is_dir():
                continue
            for f in method_dir.glob("L*_c*.json"):
                try:
                    parts = f.stem.split("_c")
                    layer = int(parts[0][1:])
                    rest = parts[1].split("_", 1)
                    coef = float(rest[0])
                    timestamp = rest[1] if len(rest) > 1 else ""
                except (IndexError, ValueError):
                    print(f"  Warning: couldn't parse {f.name}, skipping")
                    continue
                found.append({
                    "path": f, "is_baseline": False,
                    "component": component_dir.name, "method": method_dir.name,
                    "layer": layer, "coef": coef, "timestamp": timestamp,
                })

    return found


async def run_rescore(experiment, trait, model_variant, position, prompt_set,
                      eval_prompt=None, relevance_check=True, trait_judge=None,
                      dry_run=False):
    """Re-score existing steering responses with current judge. No GPU needed."""
    steering_data = load_steering_data(trait)
    trait_name = steering_data.trait_name
    trait_definition = steering_data.trait_definition

    if eval_prompt is None:
        eval_prompt = steering_data.eval_prompt

    response_files = discover_response_files(experiment, trait, model_variant, position, prompt_set)
    if not response_files:
        print(f"No response files found for {experiment}/{trait}")
        return

    n_baseline = sum(1 for f in response_files if f["is_baseline"])
    n_steered = len(response_files) - n_baseline
    mode = "DRY RUN (no writes)" if dry_run else "Re-scoring"
    print(f"\n{mode}: {n_baseline} baseline + {n_steered} steered response files")
    print(f"Trait: {trait} ({trait_name})")

    judge = TraitJudge()
    new_results = {}
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

        for r, s in zip(responses, scores):
            r["trait_score"] = s["trait_score"]
            r["coherence_score"] = s["coherence_score"]

        if dry_run:
            for j, r in enumerate(responses):
                prompt_short = r.get("prompt", r.get("question", ""))[:80]
                print(f"    [{j+1}] trait={r['trait_score']:.1f} coh={r['coherence_score']:.1f}  {prompt_short}")
        else:
            with open(path, 'w') as f:
                json.dump(responses, f, indent=2)

        trait_scores = [s["trait_score"] for s in scores if s["trait_score"] is not None]
        coh_scores = [s["coherence_score"] for s in scores if s.get("coherence_score") is not None]
        result = {
            "trait_mean": sum(trait_scores) / len(trait_scores) if trait_scores else None,
            **score_stats(trait_scores),
            "coherence_mean": sum(coh_scores) / len(coh_scores) if coh_scores else None,
            "n": len(trait_scores),
        }

        _rtm = result['trait_mean']
        _rcm = result['coherence_mean']
        print(f" trait={f'{float(_rtm):.1f}' if _rtm is not None else 'None'} coh={f'{float(_rcm):.1f}' if _rcm is not None else 'None'}")

        if is_baseline:
            new_baseline = result
        else:
            match_key = (entry["layer"], entry["component"], entry["method"], round(entry["coef"], 2))
            new_results[match_key] = (entry, result)

    await judge.close()

    if dry_run:
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

    # Add new entries for response files without existing results.jsonl entries
    matched_keys = set()
    for entry in lines:
        vectors = entry.get("config", {}).get("vectors", [{}])
        if vectors and entry.get("type") not in ("header", "baseline"):
            v = vectors[0]
            matched_keys.add((v.get("layer"), v.get("component"), v.get("method"), round(v.get("weight", 0), 2)))

    n_added = 0
    for match_key, (resp_entry, new_result) in new_results.items():
        if match_key not in matched_keys:
            layer, component, method, coef = match_key
            new_entry = {
                "config": {"vectors": [{"layer": layer, "component": component, "method": method, "weight": coef, "position": position}]},
                "result": new_result,
                "eval": {"trait_judge": trait_judge},
                "timestamp": datetime.now().isoformat(),
            }
            lines.append(new_entry)
            n_added += 1

    with open(results_path, 'w') as f:
        for entry in lines:
            f.write(json.dumps(entry) + '\n')

    print(f"\nUpdated {results_path}")
    print(f"  Rescored: {n_baseline} baseline + {len(new_results)} steered configs"
          f"{f', dropped {n_dropped} stale entries' if n_dropped else ''}"
          f"{f', added {n_added} new entries' if n_added else ''}")

    if new_baseline:
        print(f"  Baseline: trait={new_baseline['trait_mean']:.1f} coh={new_baseline['coherence_mean']:.1f}")


if __name__ == "__main__":
    main()
