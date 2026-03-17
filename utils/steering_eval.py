"""
Steering evaluation library — helpers, orchestration, and modes.

Called by steering/run_steering_eval.py (thin recipe).

Input: SteeringConfig, backend, traits
Output: results.jsonl with steering scores per (layer, coefficient)
"""

import gc
import json
from typing import Dict, List, Optional, Tuple
from datetime import datetime

import torch

from core import VectorSpec, MultiLayerAblationHook
from core.kwargs_configs import SteeringConfig
from utils.traits import load_steering_data, load_questions_from_inference, load_questions_from_file
from utils.steering_results import (
    init_results_file, load_results, append_baseline, remove_baseline, get_baseline,
    save_baseline_responses, save_ablation_responses, find_cached_run, append_run, save_responses,
    is_better_result,
)
from utils.paths import get_steering_results_path, get_steering_dir, get as get_path
from utils.coefficient_search import (
    adaptive_search_layer, batched_adaptive_search, multi_trait_batched_adaptive_search,
)
from utils.steered_generation import (
    score_stats, estimate_activation_norm, compute_baseline, batched_steering_generate,
)
from utils.judge import TraitJudge
from utils.paths import get_default_variant, get_model_variant, load_experiment_config
from utils.model import format_prompt, load_model_with_lora
from utils.generation import generate_batch
from utils.distributed import is_tp_mode, is_rank_zero, tp_barrier
from utils.vectors import MIN_COHERENCE, load_vector, load_cached_activation_norms
from utils.layers import parse_layers
from utils.backends import LocalBackend


# =============================================================================
# Helpers
# =============================================================================

def parse_coefficients(coef_arg: Optional[str]) -> Optional[List[float]]:
    if coef_arg is None:
        return None
    return [float(c) for c in coef_arg.split(",")]


def resolve_eval_prompt(steering_data, eval_prompt, use_default_prompt):
    """Resolve eval_prompt: explicit > use_default flag > steering.json."""
    if use_default_prompt:
        return None
    if eval_prompt is not None:
        return eval_prompt
    return steering_data.eval_prompt


def resolve_questions(trait, questions_file, prompt_set, subset):
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


def resolve_cli_eval_prompt(args):
    """Resolve eval_prompt from CLI args. Returns (eval_prompt, trait_judge, use_default)."""
    use_default = args.no_custom_prompt
    trait_judge = None
    eval_prompt = None

    if args.trait_judge:
        judge_path = get_path('datasets.llm_judge_trait', judge_prompt=args.trait_judge)
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


def load_or_init_results(config: SteeringConfig, trait, model_variant, steering_data,
                         model_name, vector_experiment, direction, n_questions,
                         regenerate_responses=False):
    """Load existing results or init new results file."""
    cached_runs = []
    baseline_result = None
    results_path = get_steering_results_path(config.experiment, trait, model_variant, config.position, config.prompt_set)

    if config.force and results_path.exists():
        if is_rank_zero():
            import shutil
            steering_dir = get_steering_dir(config.experiment, trait, model_variant, config.position, config.prompt_set)
            shutil.rmtree(steering_dir)
            print(f"  --force: cleared {steering_dir}")
        tp_barrier()

    if results_path.exists():
        results_data = load_results(config.experiment, trait, model_variant, config.position, config.prompt_set)
        cached_runs = results_data.get("runs", [])
        baseline_result = results_data.get("baseline")
        header_direction = results_data.get("direction", "positive")
        if regenerate_responses:
            direction = header_direction
        elif header_direction != direction:
            if config.prompt_set == "steering":
                raise ValueError(
                    f"Direction mismatch for {trait}: results file has '{header_direction}' "
                    f"but --direction {direction}. Use --force to start fresh."
                )
            print(f"\n  Warning: direction '{direction}' differs from results file '{header_direction}'")
    elif regenerate_responses:
        print(f"  No results file for {trait}, skipping")
        return None
    else:
        if is_rank_zero():
            init_results_file(
                config.experiment, trait, model_variant, steering_data.prompts_file,
                model_name, vector_experiment, config.judge_provider, config.position, config.prompt_set,
                trait_judge=config.trait_judge, direction=direction, n_questions=n_questions,
            )
        tp_barrier()

    return cached_runs, baseline_result, direction


def load_vectors(vector_experiment, trait, layers, extraction_variant,
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


def print_eval_summary(cached_runs, baseline_result, direction, min_coherence):
    """Print evaluation summary with best result."""
    sign = 1 if direction == "positive" else -1
    print(f"\n{'='*60}")
    print(f"Summary (direction={direction})")
    print(f"{'='*60}")
    _btm = baseline_result['trait_mean']
    print(f"Baseline: {float(_btm):.1f}" if _btm is not None else "Baseline: None")
    print(f"Total runs: {len(cached_runs)}")

    valid_runs = [r for r in cached_runs if r.get('result', {}).get('coherence_mean', 0) >= min_coherence]
    if valid_runs:
        best_run = max(valid_runs, key=lambda r: (r.get('result', {}).get('trait_mean') or 0) * sign)
        score = best_run['result']['trait_mean']
        coh = best_run['result'].get('coherence_mean', 0)
        delta = (score - _btm) if (_btm is not None and score is not None) else None
        layer = best_run['config']['vectors'][0]['layer']
        coef = best_run['config']['vectors'][0]['weight']
        print(f"Best (coherence>={min_coherence:.0f}): L{layer} c{coef:.0f}")
        if delta is not None:
            print(f"  trait={score:.1f} ({'+' if delta >= 0 else ''}{delta:.1f}), coherence={coh:.1f}")
        else:
            print(f"  trait={score or 0:.1f} (baseline=None), coherence={coh:.1f}")
    else:
        print(f"No valid runs with coherence>={min_coherence:.0f}")
        any_below = any(r.get('result', {}).get('coherence_mean', 0) < min_coherence for r in cached_runs)
        if not any_below:
            print(f"  WARNING: coherence never dropped below {min_coherence:.0f} — may not have steered hard enough")


# =============================================================================
# Evaluation dispatch helpers
# =============================================================================

def regenerate_responses_for_trait(layer_data, cached_runs, questions, model, tokenizer,
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


async def evaluate_manual_coefficients(
    backend, layer_data, config: SteeringConfig, questions, steering_data,
    judge, use_chat_template, cached_runs, trait, model_variant, direction,
    eval_prompt=None,
):
    """Evaluate specific coefficients for each layer, score, and save results."""
    model, tokenizer = backend.model, backend.tokenizer
    n_q = len(questions)

    all_configs = []
    for ld in layer_data:
        for coef in config.coefficients:
            cfg = {"vectors": [VectorSpec(layer=ld["layer"], component=config.component, position=config.position, method=config.method, weight=coef).to_dict()]}
            if find_cached_run(cached_runs, cfg) is not None:
                print(f"  L{ld['layer']} c{coef:.0f}: cached")
            else:
                all_configs.append((ld, coef, cfg))

    if not all_configs:
        return

    print(f"\nEvaluating {len(all_configs)} configs in batch...")
    formatted = [format_prompt(q, tokenizer, use_chat_template=use_chat_template) for q in questions]
    all_responses = batched_steering_generate(
        model, tokenizer, formatted,
        [(ld["layer"], ld["vector"], coef) for ld, coef, _ in all_configs],
        component=config.component, max_new_tokens=config.max_new_tokens,
    )

    tp = is_tp_mode()
    all_scores = None
    if not tp or is_rank_zero():
        all_qa_pairs = [(q, all_responses[i * n_q + j]) for i in range(len(all_configs)) for j, q in enumerate(questions)]
        print(f"Scoring {len(all_qa_pairs)} responses...")
        all_scores = await judge.score_steering_batch(
            all_qa_pairs, steering_data.trait_name, steering_data.trait_definition,
            eval_prompt=eval_prompt, relevance_check=config.relevance_check,
        )

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

    best_per_layer = {}
    for idx, (ld, coef, cfg) in enumerate(all_configs):
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

        cached_runs.append({"config": cfg, "result": result, "timestamp": timestamp})

        if is_rank_zero():
            append_run(config.experiment, trait, model_variant, cfg, result, config.position, config.prompt_set, trait_judge=config.trait_judge)

            resps = all_responses[idx * n_q:(idx + 1) * n_q]
            scores_slice = all_scores[idx * n_q:(idx + 1) * n_q]
            responses = [{"prompt": q, "response": r, "system_prompt": None,
                          "trait_score": s["trait_score"], "coherence_score": s.get("coherence_score")}
                         for q, r, s in zip(questions, resps, scores_slice)]

            if config.save_mode == "all":
                save_responses(responses, config.experiment, trait, model_variant, config.position, config.prompt_set, cfg, timestamp)
            elif config.save_mode == "best":
                t_mean = result.get("trait_mean") or 0
                c_mean = result.get("coherence_mean") or 0
                if is_better_result(best_per_layer.get(ld["layer"]), t_mean, c_mean, config.min_coherence, direction):
                    best_per_layer[ld["layer"]] = {"responses": responses, "config": cfg, "timestamp": timestamp}

    if config.save_mode == "best" and is_rank_zero():
        for best in best_per_layer.values():
            if best.get("responses"):
                save_responses(best["responses"], config.experiment, trait, model_variant, config.position, config.prompt_set, best["config"], best["timestamp"])


# =============================================================================
# Core evaluation for a single trait
# =============================================================================

async def run_evaluation(config: SteeringConfig, trait: str, model_variant: str,
                         model_name: str, backend=None, judge=None, lora_adapter=None):
    """Main evaluation flow for a single trait."""
    vector_experiment = config.vector_experiment or config.experiment

    # Dispatch to ablation mode
    if config.ablation is not None:
        return await run_ablation_evaluation(
            config, trait, model_variant, model_name,
            backend=backend, judge=judge, lora_adapter=lora_adapter,
        )

    # Load data
    questions, steering_data = resolve_questions(trait, config.questions_file, config.prompt_set, config.subset)
    effective_eval_prompt = resolve_eval_prompt(steering_data, config.eval_prompt, config.use_default_prompt)

    # Init model
    should_close_judge = False
    if backend is None:
        model, tokenizer = load_model_with_lora(model_name, load_in_8bit=config.load_in_8bit,
                                                 load_in_4bit=config.load_in_4bit, bnb_4bit_quant_type=config.bnb_4bit_quant_type,
                                                 lora_adapter=lora_adapter)
        backend = LocalBackend.from_model(model, tokenizer)
    model, tokenizer, num_layers = backend.model, backend.tokenizer, backend.n_layers

    exp_config = load_experiment_config(config.experiment)
    use_chat_template = exp_config.get('use_chat_template')
    if use_chat_template is None:
        use_chat_template = tokenizer.chat_template is not None

    # Parse layers
    if config.regenerate_responses:
        layers = list(range(num_layers))
    else:
        layers = parse_layers(config.layers_arg, num_layers)
    layers = [l for l in layers if 0 <= l < num_layers]
    if not layers:
        raise ValueError(f"No valid layers. Model has {num_layers} layers (0-{num_layers-1})")

    # Init/resume results
    result = load_or_init_results(
        config, trait, model_variant, steering_data, model_name,
        vector_experiment, config.direction, len(questions), config.regenerate_responses,
    )
    if result is None:
        return
    cached_runs, baseline_result, direction = result

    # Filter layers for regeneration mode
    if config.regenerate_responses and cached_runs:
        good_layers = {run["config"]["vectors"][0]["layer"]
                       for run in cached_runs
                       if (run.get("result", {}).get("coherence_mean") or 0) >= config.min_coherence}
        layers = sorted(l for l in layers if l in good_layers)
        if not layers:
            print(f"  No cached configs with coherence >= {config.min_coherence}, skipping")
            return

    # Judge
    if judge is None and not config.regenerate_responses:
        if is_rank_zero():
            judge = TraitJudge()
        should_close_judge = True

    print(f"\nTrait: {trait}")
    print(f"Model: {model_name} ({num_layers} layers)")
    print(f"Vectors from: {vector_experiment}/{trait} @ {config.position}")
    print(f"Questions: {len(questions)}, Existing runs: {len(cached_runs)}")

    # Baseline
    if not config.regenerate_responses and baseline_result is None:
        baseline_result, baseline_responses = await compute_baseline(
            backend, questions, steering_data.trait_name, steering_data.trait_definition,
            judge, max_new_tokens=config.max_new_tokens, eval_prompt=effective_eval_prompt,
        )
        if is_rank_zero():
            save_baseline_responses(baseline_responses, config.experiment, trait, model_variant, config.position, config.prompt_set)
            append_baseline(config.experiment, trait, model_variant, baseline_result, config.position, config.prompt_set, trait_judge=config.trait_judge)
        tp_barrier()
    elif baseline_result is not None:
        _btm = baseline_result['trait_mean']
        print(f"\nUsing existing baseline: trait={f'{float(_btm):.1f}' if _btm is not None else 'None'}")

    # Load vectors
    cached_norms = load_cached_activation_norms(vector_experiment, "residual")
    if cached_norms:
        print(f"\nUsing cached activation norms (residual)")
    resolved_extraction_variant = config.extraction_variant or get_default_variant(vector_experiment, mode='extraction')

    print(f"Loading vectors...")
    layer_data = load_vectors(
        vector_experiment, trait, layers, resolved_extraction_variant,
        config.method, config.component, config.position, cached_norms,
        model, tokenizer, questions, use_chat_template,
    )
    if not layer_data:
        print("No valid layers with vectors found")
        return

    # Dispatch
    if config.regenerate_responses:
        regenerate_responses_for_trait(
            layer_data, cached_runs, questions, model, tokenizer,
            use_chat_template, config.component, direction, config.min_coherence,
            config.experiment, trait, model_variant, config.position, config.prompt_set, config.max_new_tokens,
        )
    elif config.coefficients is not None:
        await evaluate_manual_coefficients(
            backend, layer_data, config, questions, steering_data,
            judge, use_chat_template, cached_runs, trait, model_variant, direction,
            eval_prompt=effective_eval_prompt,
        )
    elif config.batched and len(layer_data) > 1:
        await batched_adaptive_search(
            backend, layer_data, questions, steering_data.trait_name, steering_data.trait_definition,
            judge, use_chat_template, config.component, cached_runs, config.experiment, trait, model_variant,
            vector_experiment, config.method, position=config.position, prompt_set=config.prompt_set,
            n_steps=config.n_steps, up_mult=config.up_mult, down_mult=config.down_mult,
            start_mult=config.start_mult, momentum=config.momentum,
            max_new_tokens=config.max_new_tokens, eval_prompt=effective_eval_prompt,
            save_mode=config.save_mode, coherence_threshold=config.min_coherence,
            relevance_check=config.relevance_check, direction=direction, trait_judge=config.trait_judge,
        )
    else:
        print(f"\nSequential adaptive search ({config.n_steps} steps per layer)")
        for ld in layer_data:
            await adaptive_search_layer(
                backend, ld["vector"], ld["layer"], ld["base_coef"],
                questions, steering_data.trait_name, steering_data.trait_definition,
                judge, use_chat_template, config.component,
                cached_runs, config.experiment, trait, model_variant, vector_experiment, config.method,
                position=config.position, prompt_set=config.prompt_set, n_steps=config.n_steps,
                up_mult=config.up_mult, down_mult=config.down_mult, start_mult=config.start_mult,
                momentum=config.momentum, max_new_tokens=config.max_new_tokens,
                eval_prompt=effective_eval_prompt, save_mode=config.save_mode,
                coherence_threshold=config.min_coherence, relevance_check=config.relevance_check,
                direction=direction, trait_judge=config.trait_judge,
            )

    # Summary
    print_eval_summary(cached_runs, baseline_result, direction, config.min_coherence)

    if should_close_judge and judge is not None:
        await judge.close()


# =============================================================================
# Orchestration modes
# =============================================================================

async def run_baselines(config: SteeringConfig, parsed_traits, model_variant, model_name,
                        backend, judge, eval_prompt_override, trait_judge, use_default, force):
    """Compute baselines only (no steering, no vectors)."""
    print(f"\n{'='*60}")
    print(f"BASELINE ONLY MODE")
    print(f"{'='*60}")

    summary = []
    for vector_experiment, trait in parsed_traits:
        print(f"\n--- {trait} ---")
        questions, steering_data = resolve_questions(trait, config.questions_file, config.prompt_set, config.subset)

        if use_default:
            trait_eval_prompt = None
        elif eval_prompt_override is not None:
            trait_eval_prompt = eval_prompt_override
        else:
            trait_eval_prompt = steering_data.eval_prompt

        existing = get_baseline(config.experiment, trait, model_variant, config.position, config.prompt_set)
        if existing and not force:
            print(f"  Existing baseline: trait={existing['trait_mean']:.1f}, "
                  f"coh={existing.get('coherence_mean', 0):.1f}, n={existing['n']}")
            summary.append((trait, existing['trait_mean'], existing.get('coherence_mean'), existing['n'], "cached"))
            continue

        if existing and force:
            if is_rank_zero():
                remove_baseline(config.experiment, trait, model_variant, config.position, config.prompt_set)
            tp_barrier()

        results_path = get_steering_results_path(config.experiment, trait, model_variant, config.position, config.prompt_set)
        if not results_path.exists():
            if is_rank_zero():
                init_results_file(
                    config.experiment, trait, model_variant, steering_data.prompts_file,
                    model_name, vector_experiment, config.judge_provider, config.position,
                    config.prompt_set, trait_judge=trait_judge, direction="positive",
                    n_questions=len(questions),
                )
            tp_barrier()

        print(f"  Questions: {len(questions)}")
        baseline_result, baseline_responses = await compute_baseline(
            backend, questions, steering_data.trait_name, steering_data.trait_definition,
            judge, max_new_tokens=config.max_new_tokens, eval_prompt=trait_eval_prompt,
            relevance_check=config.relevance_check,
        )

        if is_rank_zero():
            save_baseline_responses(baseline_responses, config.experiment, trait, model_variant, config.position, config.prompt_set)
            append_baseline(config.experiment, trait, model_variant, baseline_result, config.position, config.prompt_set, trait_judge=trait_judge)

        summary.append((trait, baseline_result['trait_mean'], baseline_result.get('coherence_mean'), baseline_result['n'], "computed"))

    print(f"\n{'='*60}")
    print(f"BASELINE SUMMARY")
    print(f"{'='*60}")
    print(f"{'Trait':<40} {'Baseline':>8} {'Coh':>6} {'N':>3} {'Status':>8}")
    for trait, b_mean, c_mean, n, status in summary:
        c_str = f"{c_mean:.1f}" if c_mean is not None else "N/A"
        print(f"{trait:<40} {b_mean:>8.1f} {c_str:>6} {n:>3} {status:>8}")

    return summary


async def run_batched_multi_trait(config: SteeringConfig, parsed_traits, model_variant, model_name,
                                  backend, judge, eval_prompt_override, trait_judge,
                                  use_default, direction, force, trait_layers, layers_arg):
    """Multi-trait batched evaluation."""
    model, tokenizer, num_layers = backend.model, backend.tokenizer, backend.n_layers

    exp_config = load_experiment_config(config.experiment)
    use_chat_template = exp_config.get('use_chat_template')
    if use_chat_template is None:
        use_chat_template = tokenizer.chat_template is not None

    default_layers = parse_layers(layers_arg, num_layers)
    default_layers = [l for l in default_layers if 0 <= l < num_layers]

    parsed_trait_layers = {}
    if trait_layers:
        for trait_key, layer_spec in trait_layers.items():
            tl = parse_layers(layer_spec, num_layers)
            tl = [l for l in tl if 0 <= l < num_layers]
            if tl:
                parsed_trait_layers[trait_key] = tl

    vector_experiment = config.vector_experiment or config.experiment
    cached_norms = load_cached_activation_norms(vector_experiment, "residual")
    resolved_extraction_variant = config.extraction_variant or get_default_variant(vector_experiment, mode='extraction')

    print(f"\nMulti-trait batched mode: {len(parsed_traits)} traits")
    print(f"Model: {model_name} ({num_layers} layers)")
    print(f"Layers: {default_layers}")

    trait_configs = []
    for vec_exp, trait in parsed_traits:
        print(f"\n--- Preparing {trait} ---")

        questions, steering_data = resolve_questions(trait, config.questions_file, config.prompt_set, config.subset)

        if use_default:
            trait_eval_prompt = None
        elif eval_prompt_override is not None:
            trait_eval_prompt = eval_prompt_override
        else:
            trait_eval_prompt = steering_data.eval_prompt

        result = load_or_init_results(
            config, trait, model_variant, steering_data, model_name,
            vec_exp, direction, len(questions),
        )
        if result is None:
            continue
        cached_runs, baseline_result, _ = result

        if baseline_result is None:
            baseline_result, baseline_responses = await compute_baseline(
                backend, questions, steering_data.trait_name, steering_data.trait_definition,
                judge, max_new_tokens=config.max_new_tokens, eval_prompt=trait_eval_prompt,
            )
            if is_rank_zero():
                save_baseline_responses(baseline_responses, config.experiment, trait, model_variant, config.position, config.prompt_set)
                append_baseline(config.experiment, trait, model_variant, baseline_result, config.position, config.prompt_set, trait_judge=trait_judge)
            tp_barrier()
        else:
            _bm = baseline_result.get('trait_mean')
            print(f"  Existing baseline: trait={f'{float(_bm):.1f}' if _bm is not None else 'None'}")

        trait_layer_list = parsed_trait_layers.get(trait, default_layers)
        layer_data = load_vectors(
            vec_exp, trait, trait_layer_list, resolved_extraction_variant,
            config.method, config.component, config.position, cached_norms,
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
            "experiment": config.experiment, "vector_experiment": vec_exp,
        })

    if trait_configs:
        await multi_trait_batched_adaptive_search(
            backend=backend, trait_configs=trait_configs, judge=judge,
            use_chat_template=use_chat_template, component=config.component,
            model_variant=model_variant, method=config.method, position=config.position,
            prompt_set=config.prompt_set, n_steps=config.n_steps,
            up_mult=config.up_mult, down_mult=config.down_mult,
            start_mult=config.start_mult, momentum=config.momentum,
            max_new_tokens=config.max_new_tokens, save_mode=config.save_mode,
            coherence_threshold=config.min_coherence,
            relevance_check=config.relevance_check,
            direction=direction, trait_judge=config.trait_judge,
        )



def resolve_cli_eval_prompt_from_config(config: SteeringConfig):
    """Resolve eval_prompt from config fields. Returns (eval_prompt, trait_judge, use_default)."""
    if config.trait_judge:
        judge_path = get_path('datasets.llm_judge_trait', judge_prompt=config.trait_judge)
        if not judge_path.exists():
            raise FileNotFoundError(f"Trait judge prompt not found: {judge_path}")
        return judge_path.read_text(), config.trait_judge, False
    if config.use_default_prompt:
        return None, None, True
    return config.eval_prompt, None, False


# =============================================================================
# Ablation mode
# =============================================================================

async def run_ablation_evaluation(config: SteeringConfig, trait, model_variant, model_name,
                                   backend=None, judge=None, lora_adapter=None):
    """Directional ablation: ablate a direction at ALL layers simultaneously."""
    vector_experiment = config.vector_experiment or config.experiment
    questions, steering_data = resolve_questions(trait, None, config.prompt_set, config.subset if config.subset != 5 else None)
    effective_eval_prompt = resolve_eval_prompt(steering_data, config.eval_prompt, config.use_default_prompt)

    should_close_judge = False
    if backend is None:
        model, tokenizer = load_model_with_lora(model_name, load_in_8bit=config.load_in_8bit,
                                                 load_in_4bit=config.load_in_4bit,
                                                 bnb_4bit_quant_type=config.bnb_4bit_quant_type,
                                                 lora_adapter=lora_adapter)
        backend = LocalBackend.from_model(model, tokenizer)

    model, tokenizer = backend.model, backend.tokenizer
    exp_config = load_experiment_config(config.experiment)
    use_chat_template = exp_config.get('use_chat_template')
    if use_chat_template is None:
        use_chat_template = tokenizer.chat_template is not None

    if judge is None:
        judge = TraitJudge()
        should_close_judge = True

    resolved_extraction_variant = config.extraction_variant or get_default_variant(vector_experiment, mode='extraction')
    vector = load_vector(vector_experiment, trait, config.ablation, resolved_extraction_variant, config.method, config.component, config.position)
    if vector is None:
        raise ValueError(f"Vector not found: {vector_experiment}/{trait} L{config.ablation} {config.method}")

    print(f"\nABLATION: {trait}, Vector: L{config.ablation} {config.method}")

    baseline, baseline_responses = await compute_baseline(
        backend, questions, steering_data.trait_name, steering_data.trait_definition,
        judge, max_new_tokens=config.max_new_tokens, eval_prompt=effective_eval_prompt,
        relevance_check=config.relevance_check,
    )

    formatted = [format_prompt(q, tokenizer, use_chat_template=use_chat_template) for q in questions]
    with MultiLayerAblationHook(model, vector):
        ablated_responses = generate_batch(model, tokenizer, formatted, max_new_tokens=config.max_new_tokens)

    all_scores = await judge.score_steering_batch(
        list(zip(questions, ablated_responses)), steering_data.trait_name, steering_data.trait_definition,
        eval_prompt=effective_eval_prompt, relevance_check=config.relevance_check,
    )

    ablated_trait_scores = [s["trait_score"] for s in all_scores if s["trait_score"] is not None]
    ablated_coh_scores = [s["coherence_score"] for s in all_scores if s.get("coherence_score") is not None]
    ablated_trait_mean = sum(ablated_trait_scores) / len(ablated_trait_scores) if ablated_trait_scores else None
    ablated_coh_mean = sum(ablated_coh_scores) / len(ablated_coh_scores) if ablated_coh_scores else None

    ablated_data = [
        {"prompt": q, "response": r, "system_prompt": None,
         "trait_score": s["trait_score"], "coherence_score": s.get("coherence_score")}
        for q, r, s in zip(questions, ablated_responses, all_scores)
    ]
    save_baseline_responses(baseline_responses, config.experiment, trait, model_variant, config.position, config.prompt_set)
    save_ablation_responses(
        ablated_data, config.experiment, trait, model_variant, config.position, config.prompt_set,
        config.ablation, config.method, config.component,
    )

    delta = ablated_trait_mean - baseline['trait_mean']
    print(f"Baseline: trait={baseline['trait_mean']:.1f} | Ablated: trait={ablated_trait_mean:.1f} | Delta: {delta:+.1f}")

    if should_close_judge:
        await judge.close()

    return {"baseline": baseline, "ablated": {"trait_mean": ablated_trait_mean, "coherence_mean": ablated_coh_mean}, "delta": delta}


# =============================================================================
# Rescore mode
# =============================================================================

def discover_response_files(experiment, trait, model_variant, position, prompt_set):
    """Find all saved response files and parse config from path."""
    steering_dir = get_steering_dir(experiment, trait, model_variant, position, prompt_set)
    responses_dir = steering_dir / "responses"
    if not responses_dir.exists():
        return []

    found = []
    baseline_path = responses_dir / "baseline.json"
    if baseline_path.exists():
        found.append({"path": baseline_path, "is_baseline": True})

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
                    continue
                found.append({
                    "path": f, "is_baseline": False,
                    "component": component_dir.name, "method": method_dir.name,
                    "layer": layer, "coef": coef, "timestamp": timestamp,
                })

    return found


async def run_rescore(config: SteeringConfig, trait, model_variant, dry_run=False):
    """Re-score existing steering responses with current judge. No GPU needed."""
    steering_data = load_steering_data(trait)
    eval_prompt = config.eval_prompt or steering_data.eval_prompt

    response_files = discover_response_files(config.experiment, trait, model_variant, config.position, config.prompt_set)
    if not response_files:
        print(f"No response files found for {config.experiment}/{trait}")
        return

    n_baseline = sum(1 for f in response_files if f["is_baseline"])
    n_steered = len(response_files) - n_baseline
    print(f"\n{'DRY RUN' if dry_run else 'Re-scoring'}: {n_baseline} baseline + {n_steered} steered response files")

    judge = TraitJudge()
    new_results = {}
    new_baseline = None

    for i, entry in enumerate(response_files):
        with open(entry["path"]) as f:
            responses = json.load(f)

        qa_pairs = [(r.get("prompt", r.get("question", "")), r["response"]) for r in responses]
        label = "baseline" if entry["is_baseline"] else f"L{entry['layer']} c{entry['coef']:.1f}"
        print(f"  [{i+1}/{len(response_files)}] {label} ({len(qa_pairs)} responses)...", end="", flush=True)

        scores = await judge.score_steering_batch(
            qa_pairs, steering_data.trait_name, steering_data.trait_definition,
            eval_prompt=eval_prompt, relevance_check=config.relevance_check,
        )

        for r, s in zip(responses, scores):
            r["trait_score"] = s["trait_score"]
            r["coherence_score"] = s["coherence_score"]

        if not dry_run:
            with open(entry["path"], 'w') as f:
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

        if entry["is_baseline"]:
            new_baseline = result
        else:
            match_key = (entry["layer"], entry["component"], entry["method"], round(entry["coef"], 2))
            new_results[match_key] = (entry, result)

    await judge.close()

    if dry_run:
        return

    # Rebuild results.jsonl
    results_path = get_steering_results_path(config.experiment, trait, model_variant, config.position, config.prompt_set)
    if not results_path.exists():
        return

    lines = []
    with open(results_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            if entry.get("type") == "header":
                entry.setdefault("eval", {})["trait_judge"] = config.trait_judge
                lines.append(entry)
            elif entry.get("type") == "baseline":
                if new_baseline:
                    entry["result"] = new_baseline
                    entry["timestamp"] = datetime.now().isoformat()
                lines.append(entry)
            else:
                vectors = entry.get("config", {}).get("vectors", [{}])
                if vectors:
                    v = vectors[0]
                    mk = (v.get("layer"), v.get("component"), v.get("method"), round(v.get("weight", 0), 2))
                    if mk in new_results:
                        _, new_result = new_results[mk]
                        entry["result"] = new_result
                        entry["timestamp"] = datetime.now().isoformat()
                        lines.append(entry)

    # Add new entries
    matched = set()
    for e in lines:
        vectors = e.get("config", {}).get("vectors", [{}])
        if vectors and e.get("type") not in ("header", "baseline"):
            v = vectors[0]
            matched.add((v.get("layer"), v.get("component"), v.get("method"), round(v.get("weight", 0), 2)))

    for mk, (resp_entry, new_result) in new_results.items():
        if mk not in matched:
            layer, component, method, coef = mk
            lines.append({
                "config": {"vectors": [{"layer": layer, "component": component, "method": method, "weight": coef, "position": config.position}]},
                "result": new_result,
                "timestamp": datetime.now().isoformat(),
            })

    with open(results_path, 'w') as f:
        for entry in lines:
            f.write(json.dumps(entry) + '\n')

    print(f"\nUpdated {results_path}")
