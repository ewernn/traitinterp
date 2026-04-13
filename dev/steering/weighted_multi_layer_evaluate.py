#!/usr/bin/env python3
"""
Weighted multi-layer steering evaluation with pluggable weight sources.

Selects (layer, component) hooks and weights via one of four strategies,
then runs multi-layer coefficient search with a single shared scalar.

Input: Weight source strategy, experiment/trait identifiers
Output: Steering results in standard format (results.jsonl + responses/)

Usage:
    # Probe weights (L1 logistic regression coefficients)
    python steering/weighted_multi_layer_evaluate.py \
        --experiment {experiment} --traits {traits} --weight-source probe

    # Causal weights (per-component steering delta)
    python steering/weighted_multi_layer_evaluate.py \
        --experiment {experiment} --traits {traits} --weight-source causal

    # Intersection (both probe and causal signal required)
    python steering/weighted_multi_layer_evaluate.py \
        --experiment {experiment} --traits {traits} --weight-source intersection

    # Top-K by causal delta, equal weight
    python steering/weighted_multi_layer_evaluate.py \
        --experiment {experiment} --traits {traits} --weight-source topk --top-k 4
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import asyncio

from dev.steering.multi_layer_evaluate import run_multi_layer_evaluation
from dev.steering.weight_sources import build_weighted_configs, STRATEGIES
from utils.backends import LocalBackend
from utils.judge import TraitJudge
from utils.model import load_model_with_lora
from utils.paths import get_default_variant, get_model_variant, load_experiment_config
from utils.vectors import MIN_COHERENCE


async def main():
    parser = argparse.ArgumentParser(description="Weighted multi-layer steering evaluation")
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--traits", required=True, help="Comma-separated traits")
    parser.add_argument("--weight-source", required=True, choices=STRATEGIES,
                        help="Weight strategy: probe, causal, intersection, topk")
    parser.add_argument("--method", default="mean_diff")
    parser.add_argument("--position", default="response[:5]")
    parser.add_argument("--weight-threshold", type=float, default=0.01,
                        help="Probe weight threshold as fraction of total (default: 0.01)")
    parser.add_argument("--delta-threshold", type=float, default=5.0,
                        help="Minimum causal delta for causal/intersection (default: 5.0)")
    parser.add_argument("--top-k", type=int, default=5,
                        help="Number of hooks for topk strategy (default: 5)")
    parser.add_argument("--prompt-set", default="steering")
    parser.add_argument("--subset", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    # --judge removed: only OpenAI logprob scoring is supported (TraitJudge in utils/judge.py)
    parser.add_argument("--search-steps", type=int, default=8)
    parser.add_argument("--min-coherence", type=float, default=MIN_COHERENCE)
    parser.add_argument("--save-responses", choices=["all", "best", "none"], default="best")
    parser.add_argument("--direction", choices=["positive", "negative"], default="positive")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--model-variant", default=None)
    parser.add_argument("--extraction-variant", default=None)
    parser.add_argument("--vector-experiment", default=None,
                        help="Experiment to load vectors from (default: same as --experiment)")
    parser.add_argument("--steering-variant", default=None,
                        help="Model variant for reading causal results (default: --model-variant)")
    args = parser.parse_args()

    traits = [t.strip() for t in args.traits.split(",")]

    exp_config = load_experiment_config(args.experiment)
    model_variant = args.model_variant or get_default_variant(args.experiment, mode='application')
    model_info = get_model_variant(args.experiment, model_variant)
    model_name = model_info.model
    vector_experiment = args.vector_experiment or args.experiment
    extraction_variant = args.extraction_variant or get_default_variant(vector_experiment, mode='extraction')
    steering_variant = args.steering_variant or model_variant

    print(f"Loading model: {model_name}")
    model, tokenizer = load_model_with_lora(model_name)
    backend = LocalBackend.from_model(model, tokenizer)
    judge = TraitJudge()

    for trait in traits:
        print(f"\n{'='*60}")
        print(f"WEIGHTED STEERING [{args.weight_source.upper()}]: {trait}")
        print(f"{'='*60}")

        vectors_and_layers, per_hook_weights, base_coef = build_weighted_configs(
            source=args.weight_source,
            experiment=vector_experiment,
            trait=trait,
            extraction_variant=extraction_variant,
            method=args.method,
            position=args.position,
            weight_threshold=args.weight_threshold,
            delta_threshold=args.delta_threshold,
            min_coherence=args.min_coherence,
            top_k=args.top_k,
            model_variant=steering_variant,
            prompt_set=args.prompt_set,
            steering_experiment=args.experiment,
            device=model.device,
        )

        if not vectors_and_layers:
            print(f"  No valid configs for {trait}, skipping")
            continue

        try:
            await run_multi_layer_evaluation(
                experiment=args.experiment,
                trait=trait,
                vector_experiment=vector_experiment,
                model_variant=model_variant,
                extraction_variant=extraction_variant,
                method=args.method,
                component="residual",  # default, overridden by per-entry components
                position=args.position,
                layers=[],
                prompt_set=args.prompt_set,
                model_name=model_name,
                subset=args.subset,
                backend=backend,
                judge=judge,
                n_search_steps=args.search_steps,
                max_new_tokens=args.max_new_tokens,
                min_coherence=args.min_coherence,
                save_mode=args.save_responses,
                relevance_check=True,
                direction=args.direction,
                force=args.force,
                vectors_and_layers_override=vectors_and_layers,
                per_hook_weights=per_hook_weights,
                base_coef_override=base_coef,
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
