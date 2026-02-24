#!/usr/bin/env python3
"""
Multi-layer steering evaluation - steer all layers simultaneously with shared coefficient.

Compares against single-layer steering by loading vectors at multiple layers,
applying them all at once with a single shared alpha, and sweeping alpha
via adaptive search.

Input:
    - experiment: Experiment with existing vectors and steering data
    - traits: Comma-separated trait paths
    - method: Extraction method (probe, rfm, mean_diff)

Output:
    - Results appended to existing results.jsonl (multi-vector configs)

Usage:
    python analysis/steering/multi_layer_evaluate.py \
        --experiment temp_llama_steering_feb18 \
        --traits "mental_state/anxiety,mental_state/confidence" \
        --method probe \
        --layers 3,6,9,12,15,18
"""

import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import argparse
import asyncio
from typing import List, Dict, Optional
from datetime import datetime

from analysis.steering.data import load_steering_data
from analysis.steering.results import (
    init_results_file, load_results, append_baseline,
    save_baseline_responses, find_cached_run, append_run, save_responses,
    is_better_result,
)
from analysis.steering.evaluate import compute_baseline, estimate_activation_norm
from utils.paths import get_steering_results_path, get_steering_dir
from core import VectorSpec, MultiLayerSteeringHook, GenerationConfig, LocalBackend
from core.hooks import get_hook_path
from utils.generation import generate_batch
from utils.judge import TraitJudge
from utils.paths import get, get_default_variant, get_model_variant, load_experiment_config
from utils.model import format_prompt, tokenize_prompt, load_model_with_lora, is_rank_zero
from utils.vectors import MIN_COHERENCE, load_vector, load_cached_activation_norms
from utils.layers import parse_layers


async def evaluate_multi_layer_config(
    backend,
    vectors_and_layers: List[tuple],  # [(layer, vector), ...]
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
) -> tuple:
    """Evaluate a multi-layer config with shared coefficient."""
    model = backend.model
    tokenizer = backend.tokenizer

    n_layers = len(vectors_and_layers)
    desc = f"Multi-L({n_layers}) c{coef:.1f}"

    formatted = [format_prompt(q, tokenizer, use_chat_template=use_chat_template) for q in questions]

    configs = [(layer, vector, coef) for layer, vector in vectors_and_layers]
    print(f"  Generating {len(questions)} responses for {desc}...")
    with MultiLayerSteeringHook(model, configs, component=component):
        responses = generate_batch(model, tokenizer, formatted, max_new_tokens=max_new_tokens)

    if is_rank_zero():
        all_qa_pairs = list(zip(questions, responses))
        print(f"  Scoring {len(all_qa_pairs)} responses...")
        all_scores = await judge.score_steering_batch(
            all_qa_pairs, trait_name, trait_definition,
            eval_prompt=eval_prompt, relevance_check=relevance_check,
        )

        trait_scores = [s["trait_score"] for s in all_scores if s["trait_score"] is not None]
        coherence_scores = [s["coherence_score"] for s in all_scores if s.get("coherence_score") is not None]

        result = {
            "trait_mean": sum(trait_scores) / len(trait_scores) if trait_scores else None,
            "coherence_mean": sum(coherence_scores) / len(coherence_scores) if coherence_scores else None,
            "n": len(trait_scores),
        }

        responses_data = [
            {"prompt": q, "response": r, "system_prompt": None,
             "trait_score": s["trait_score"], "coherence_score": s.get("coherence_score")}
            for (q, r), s in zip(all_qa_pairs, all_scores)
        ]

        trait_str = f"{result['trait_mean']:.1f}" if result['trait_mean'] else "N/A"
        coh_str = f"{result['coherence_mean']:.1f}" if result['coherence_mean'] else "N/A"
        print(f"  {desc}: trait={trait_str}, coherence={coh_str}, n={result['n']}")
    else:
        result = None
        responses_data = None

    return result, responses_data


async def multi_layer_adaptive_search(
    backend,
    vectors_and_layers: List[tuple],
    base_coef: float,
    questions: List[str],
    trait_name: str,
    trait_definition: str,
    judge: TraitJudge,
    use_chat_template: bool,
    component: str,
    method: str,
    position: str,
    cached_runs: list,
    experiment: str,
    trait: str,
    model_variant: str,
    vector_experiment: str,
    prompt_set: str = "steering",
    n_steps: int = 8,
    threshold: float = MIN_COHERENCE,
    up_mult: float = 1.3,
    down_mult: float = 0.85,
    start_mult: float = 0.7,
    max_new_tokens: int = 256,
    eval_prompt: Optional[str] = None,
    save_mode: str = "best",
    relevance_check: bool = True,
    direction: str = "positive",
):
    """Adaptive coefficient search for multi-layer steering."""
    n_layers = len(vectors_and_layers)
    sign = 1 if direction == "positive" else -1

    print(f"\n--- Multi-layer ({n_layers} layers, base_coef={base_coef:.1f}, direction={direction}) ---")
    print(f"Layers: {[l for l, _ in vectors_and_layers]}")
    print(f"Step |  Coef  | Trait | Coherence | Action")
    print("-----|--------|-------|-----------|-------")

    coef = base_coef * start_mult * sign
    history = []
    best_result = None

    for step in range(n_steps):
        # Build config dict with all vectors
        spec_list = [
            VectorSpec(layer=layer, component=component, position=position, method=method, weight=coef).to_dict()
            for layer, _ in vectors_and_layers
        ]
        config = {"vectors": spec_list}

        cached_result = find_cached_run(cached_runs, config)
        if cached_result is not None:
            trait_score = cached_result.get("trait_mean") or 0
            coherence = cached_result.get("coherence_mean") or 0
            print(f"  {step+1}  | {coef:>6.1f} | {trait_score:>5.1f} | {coherence:>9.1f} | (cached)")
        else:
            result, responses = await evaluate_multi_layer_config(
                backend, vectors_and_layers, coef, questions,
                trait_name, trait_definition, judge, use_chat_template, component,
                max_new_tokens=max_new_tokens, eval_prompt=eval_prompt,
                relevance_check=relevance_check,
            )

            trait_score = result.get("trait_mean") or 0
            coherence = result.get("coherence_mean") or 0

            timestamp = datetime.now().isoformat()
            cached_runs.append({"config": config, "result": result, "timestamp": timestamp})

            if is_rank_zero():
                append_run(experiment, trait, model_variant, config, result, position, prompt_set)

                if save_mode == "all":
                    save_responses(responses, experiment, trait, model_variant, position, prompt_set, config, timestamp)
                elif save_mode == "best":
                    if is_better_result(best_result, trait_score, coherence, threshold, direction):
                        best_result = {
                            "trait_mean": trait_score,
                            "coherence_mean": coherence,
                            "valid": coherence >= threshold,
                            "responses": responses,
                            "config": config,
                            "timestamp": timestamp,
                        }

            action = f"x{up_mult}" if coherence >= threshold else f"x{down_mult}"
            if step == n_steps - 1:
                action = "(done)"
            print(f"  {step+1}  | {coef:>6.1f} | {trait_score:>5.1f} | {coherence:>9.1f} | {action}")

        history.append((coef, trait_score, coherence))
        mult = up_mult if coherence >= threshold else down_mult
        coef *= mult

    # Save best responses
    if save_mode == "best" and best_result and best_result.get("responses") and is_rank_zero():
        save_responses(
            best_result["responses"], experiment, trait, model_variant,
            position, prompt_set, best_result["config"], best_result["timestamp"],
        )

    # Report
    valid = [(c, t, coh) for c, t, coh in history if coh >= threshold]
    if valid:
        best_coef, best_trait, best_coh = max(valid, key=lambda x: x[1] * sign)
        print(f"✓ Best: coef={best_coef:.1f} (trait={best_trait:.1f}, coherence={best_coh:.1f})")
    else:
        best_coef, best_trait, best_coh = max(history, key=lambda x: x[2])
        print(f"⚠ No coef met threshold. Best coherence: coef={best_coef:.1f}")


async def run_multi_layer_evaluation(
    experiment: str,
    trait: str,
    vector_experiment: str,
    model_variant: str,
    extraction_variant: str,
    method: str,
    component: str,
    position: str,
    layers: List[int],
    prompt_set: str,
    judge_provider: str,
    model_name: str,
    subset: Optional[int],
    backend=None,
    judge=None,
    n_search_steps: int = 8,
    up_mult: float = 1.3,
    down_mult: float = 0.85,
    start_mult: float = 0.7,
    max_new_tokens: int = 256,
    eval_prompt: Optional[str] = None,
    use_default_prompt: bool = False,
    min_coherence: float = MIN_COHERENCE,
    save_mode: str = "best",
    relevance_check: bool = True,
    direction: str = "positive",
    force: bool = False,
):
    """Run multi-layer steering evaluation for one trait."""
    steering_data = load_steering_data(trait)
    should_close_judge = False

    if prompt_set == "steering":
        questions = steering_data.questions
    else:
        from analysis.steering.data import load_questions_from_inference
        questions = load_questions_from_inference(prompt_set)

    if subset:
        questions = questions[:subset]

    if use_default_prompt:
        effective_eval_prompt = None
    elif eval_prompt is not None:
        effective_eval_prompt = eval_prompt
    else:
        effective_eval_prompt = steering_data.eval_prompt

    config = load_experiment_config(experiment)
    use_chat_template = config.get('use_chat_template')
    if backend is not None:
        model = backend.model
        tokenizer = backend.tokenizer
        if use_chat_template is None:
            use_chat_template = tokenizer.chat_template is not None
    else:
        model_info = get_model_variant(experiment, model_variant)
        model, tokenizer = load_model_with_lora(model_info['model'])
        backend = LocalBackend.from_model(model, tokenizer)
        if use_chat_template is None:
            use_chat_template = tokenizer.chat_template is not None

    model = backend.model
    tokenizer = backend.tokenizer

    # Load/create results
    cached_runs = []
    baseline_result = None
    results_path = get_steering_results_path(experiment, trait, model_variant, position, prompt_set)

    if force and results_path.exists():
        # Don't delete — just clear cached runs so we re-evaluate
        pass

    if results_path.exists():
        results_data = load_results(experiment, trait, model_variant, position, prompt_set)
        cached_runs = results_data.get("runs", [])
        baseline_result = results_data.get("baseline")
        header_direction = results_data.get("direction", "positive")
        if header_direction != direction and prompt_set == "steering":
            raise ValueError(
                f"Direction mismatch for {trait}: results file has '{header_direction}' "
                f"but --direction {direction} was requested. "
                f"Use --force to clear results and start fresh."
            )
    else:
        if is_rank_zero():
            init_results_file(
                experiment, trait, model_variant, steering_data.prompts_file,
                model_name, vector_experiment, judge_provider, position, prompt_set,
                direction=direction,
            )

    if judge is None:
        judge = TraitJudge()
        should_close_judge = True

    print(f"\n{'='*60}")
    print(f"MULTI-LAYER STEERING: {trait}")
    print(f"{'='*60}")
    print(f"Method: {method}")
    print(f"Layers: {layers}")
    print(f"Questions: {len(questions)}")

    # Compute baseline if needed
    if baseline_result is None:
        baseline_result, baseline_responses = await compute_baseline(
            backend, questions, steering_data.trait_name, steering_data.trait_definition,
            judge, max_new_tokens=max_new_tokens, eval_prompt=effective_eval_prompt,
        )
        if is_rank_zero():
            save_baseline_responses(baseline_responses, experiment, trait, model_variant, position, prompt_set)
            append_baseline(experiment, trait, model_variant, baseline_result, position, prompt_set)
    else:
        print(f"\nUsing existing baseline: trait={baseline_result['trait_mean']:.1f}")

    # Load vectors at all requested layers
    resolved_extraction_variant = extraction_variant or get_default_variant(vector_experiment, mode='extraction')
    cached_norms = load_cached_activation_norms(vector_experiment, "residual")

    vectors_and_layers = []
    base_coefs = []
    for layer in layers:
        vector = load_vector(vector_experiment, trait, layer, resolved_extraction_variant, method, component, position)
        if vector is None:
            print(f"  L{layer}: Vector not found, skipping")
            continue

        vec_norm = vector.norm().item()
        if vec_norm == 0:
            print(f"  L{layer}: Zero vector, skipping")
            continue

        if layer in cached_norms:
            act_norm = cached_norms[layer]
        else:
            act_norm = estimate_activation_norm(model, tokenizer, questions, layer, use_chat_template, "residual")

        base_coefs.append(act_norm / vec_norm)
        vectors_and_layers.append((layer, vector))
        print(f"  L{layer}: loaded (act_norm={act_norm:.0f}, vec_norm={vec_norm:.3f})")

    if not vectors_and_layers:
        print("No valid layers with vectors found")
        return

    n_layers = len(vectors_and_layers)
    # Key: divide base_coef by number of layers since effects accumulate
    mean_base_coef = sum(base_coefs) / len(base_coefs)
    multi_base_coef = mean_base_coef / n_layers
    print(f"\nMulti-layer base_coef: {mean_base_coef:.0f} / {n_layers} layers = {multi_base_coef:.1f}")

    await multi_layer_adaptive_search(
        backend, vectors_and_layers, multi_base_coef,
        questions, steering_data.trait_name, steering_data.trait_definition,
        judge, use_chat_template, component, method, position,
        cached_runs, experiment, trait, model_variant, vector_experiment,
        prompt_set=prompt_set, n_steps=n_search_steps,
        threshold=min_coherence, up_mult=up_mult, down_mult=down_mult,
        start_mult=start_mult, max_new_tokens=max_new_tokens,
        eval_prompt=effective_eval_prompt, save_mode=save_mode,
        relevance_check=relevance_check, direction=direction,
    )

    # Print summary
    sign = 1 if direction == "positive" else -1
    print(f"\n{'='*60}")
    print(f"Summary (multi-layer, {n_layers} layers, {method})")
    print(f"{'='*60}")
    print(f"Baseline: {baseline_result['trait_mean']:.1f}")

    valid_runs = [
        r for r in cached_runs
        if r.get('result', {}).get('coherence_mean', 0) >= min_coherence
        and len(r.get('config', {}).get('vectors', [])) > 1
    ]
    if valid_runs:
        best_run = max(valid_runs, key=lambda r: (r.get('result', {}).get('trait_mean') or 0) * sign)
        score = best_run['result']['trait_mean']
        coh = best_run['result'].get('coherence_mean', 0)
        coef = best_run['config']['vectors'][0]['weight']
        delta = score - baseline_result['trait_mean']
        delta_str = f"+{delta:.1f}" if delta >= 0 else f"{delta:.1f}"
        print(f"Best multi-layer: coef={coef:.1f}, trait={score:.1f} ({delta_str}), coherence={coh:.1f}")
    else:
        print(f"No valid multi-layer runs with coherence>={min_coherence:.0f}")

    if should_close_judge:
        await judge.close()


async def main():
    parser = argparse.ArgumentParser(description="Multi-layer steering evaluation")

    parser.add_argument("--experiment", required=True)
    parser.add_argument("--traits", required=True, help="Comma-separated: 'cat/t1,cat/t2'")
    parser.add_argument("--method", default="probe")
    parser.add_argument("--component", default="residual")
    parser.add_argument("--position", default="response[:5]")
    parser.add_argument("--layers", default="30%-60%",
                        help="Layers to steer simultaneously (default: 30%%-60%%)")
    parser.add_argument("--prompt-set", default="steering")
    parser.add_argument("--subset", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--judge", default="openai", choices=["openai", "gemini"])
    parser.add_argument("--search-steps", type=int, default=8)
    parser.add_argument("--up-mult", type=float, default=1.3)
    parser.add_argument("--down-mult", type=float, default=0.85)
    parser.add_argument("--start-mult", type=float, default=0.7)
    parser.add_argument("--min-coherence", type=float, default=MIN_COHERENCE)
    parser.add_argument("--save-responses", choices=["all", "best", "none"], default="best")
    parser.add_argument("--direction", choices=["positive", "negative"], default="positive")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--model-variant", default=None)
    parser.add_argument("--extraction-variant", default=None)
    parser.add_argument("--vector-experiment", default=None)
    parser.add_argument("--no-server", action="store_true")
    parser.add_argument("--load-in-8bit", action="store_true")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--no-relevance-check", action="store_true")

    args = parser.parse_args()

    # Parse traits
    traits = [t.strip() for t in args.traits.split(",")]

    # Load experiment config
    exp_config = load_experiment_config(args.experiment)

    # Resolve model variant
    model_variant = args.model_variant or get_default_variant(args.experiment, mode='application')
    model_info = get_model_variant(args.experiment, model_variant)
    model_name = model_info['model']
    vector_experiment = args.vector_experiment or args.experiment

    # Resolve layers
    num_layers = exp_config.get('num_layers')
    if num_layers is None:
        # Try to get from model config
        import yaml
        model_configs = Path('config/models')
        for f in model_configs.glob('*.yaml'):
            cfg = yaml.safe_load(f.read_text())
            if cfg.get('huggingface_id') == model_name:
                num_layers = cfg['num_hidden_layers']
                break
    if num_layers is None:
        num_layers = 32  # Default for 8B models

    layers = parse_layers(args.layers, num_layers)

    # Load model once for all traits
    print(f"Loading model: {model_name}")
    model, tokenizer = load_model_with_lora(
        model_name,
        load_in_8bit=args.load_in_8bit,
        load_in_4bit=args.load_in_4bit,
    )
    backend = LocalBackend.from_model(model, tokenizer)
    judge = TraitJudge()

    extraction_variant = args.extraction_variant or get_default_variant(vector_experiment, mode='extraction')

    for trait in traits:
        try:
            await run_multi_layer_evaluation(
                experiment=args.experiment,
                trait=trait,
                vector_experiment=vector_experiment,
                model_variant=model_variant,
                extraction_variant=extraction_variant,
                method=args.method,
                component=args.component,
                position=args.position,
                layers=layers,
                prompt_set=args.prompt_set,
                judge_provider=args.judge,
                model_name=model_name,
                subset=args.subset,
                backend=backend,
                judge=judge,
                n_search_steps=args.search_steps,
                up_mult=args.up_mult,
                down_mult=args.down_mult,
                start_mult=args.start_mult,
                max_new_tokens=args.max_new_tokens,
                min_coherence=args.min_coherence,
                save_mode=args.save_responses,
                relevance_check=not args.no_relevance_check,
                direction=args.direction,
                force=args.force,
            )
        except Exception as e:
            print(f"\nERROR on {trait}: {e}")
            import traceback
            traceback.print_exc()
            continue

    await judge.close()
    print("\nDone.")


if __name__ == "__main__":
    asyncio.run(main())
