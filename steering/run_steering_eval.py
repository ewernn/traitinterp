#!/usr/bin/env python3
"""
Steering evaluation — validate trait vectors via causal intervention.

Modes:
    default          Batched multi-trait coefficient search (most common)
    --baseline-only  Score unsteered responses only
    --coefficients   Evaluate specific coefficients (no search)
    --ablation L     Remove direction at all layers, measure delta
    --rescore TRAIT  Re-judge existing responses (no GPU)

Usage:
    python steering/run_steering_eval.py --experiment {exp} --traits "cat/trait1,cat/trait2"
    python steering/run_steering_eval.py --experiment {exp} --traits cat/trait --baseline-only
    python steering/run_steering_eval.py --experiment {exp} --rescore cat/trait
"""

import sys
import asyncio
import argparse
import gc
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from core.kwargs_configs import SteeringConfig
from utils.steering_eval import (
    run_baselines, run_batched_multi_trait, run_evaluation,
    run_ablation_evaluation, run_rescore, parse_coefficients,
    resolve_cli_eval_prompt_from_config,
)
from utils.traits import load_steering_data
from utils.backends import LocalBackend, add_backend_args
from utils.paths import get_model_variant
from utils.distributed import is_tp_mode, is_rank_zero
from utils.vectors import MIN_COHERENCE
from utils.judge import TraitJudge


# =============================================================================
# Recipes — one per mode, dispatch visible below
# =============================================================================

async def recipe_baselines(config, parsed_traits, backend, judge):
    """Score unsteered responses for each trait. No steering, no vectors."""
    eval_prompt, trait_judge, use_default = resolve_cli_eval_prompt_from_config(config)
    await run_baselines(config, parsed_traits, backend=backend, judge=judge,
                        eval_prompt_override=eval_prompt, trait_judge=trait_judge,
                        use_default=use_default, force=config.force)


async def recipe_batched(config, parsed_traits, backend, judge, layers_arg, trait_layers, direction):
    """Main path: search coefficients for multiple traits × layers in parallel batches."""
    eval_prompt, trait_judge, use_default = resolve_cli_eval_prompt_from_config(config)
    await run_batched_multi_trait(config, parsed_traits, backend=backend, judge=judge,
                                  eval_prompt_override=eval_prompt, trait_judge=trait_judge,
                                  use_default=use_default, direction=direction, force=config.force,
                                  trait_layers=trait_layers, layers_arg=layers_arg)


async def recipe_sequential(config, parsed_traits, backend, judge, layers_arg, trait_layers, lora):
    """Fallback: evaluate traits one at a time. For --no-batch, --coefficients, --regenerate-responses."""
    for vector_experiment, trait in parsed_traits:
        if len(parsed_traits) > 1:
            print(f"\n{'='*60}\nTRAIT: {vector_experiment}/{trait}\n{'='*60}")

        effective_layers = trait_layers[trait] if (trait_layers and trait in trait_layers) else layers_arg
        trait_config = SteeringConfig(**{**config.__dict__, 'layers_arg': effective_layers})
        await run_evaluation(trait_config, trait, backend=backend, judge=judge, lora_adapter=lora)


async def recipe_ablation(config, parsed_traits, backend, judge, lora):
    """Remove a direction at ALL layers, measure trait delta."""
    for _, trait in parsed_traits:
        await run_ablation_evaluation(config, trait, backend=backend, judge=judge, lora_adapter=lora)


# =============================================================================
# Entry point — resolve config, dispatch to recipe
# =============================================================================

async def run(config, parsed_traits, model_variant, model_name, lora,
              layers_arg, trait_layers, direction):
    """Load model once, dispatch to the right recipe."""

    # Rescore: no GPU needed, early return
    # (handled in main() before calling run())

    backend = LocalBackend.from_experiment(
        config.experiment, variant=model_variant,
        load_in_8bit=config.load_in_8bit, load_in_4bit=config.load_in_4bit,
        bnb_4bit_quant_type=config.bnb_4bit_quant_type,
    )
    judge = TraitJudge() if is_rank_zero() and not config.regenerate_responses else None

    try:
        if config.baseline_only:
            await recipe_baselines(config, parsed_traits, backend, judge)

        elif config.ablation is not None:
            await recipe_ablation(config, parsed_traits, backend, judge, lora)

        elif config.batched and config.coefficients is None and not config.regenerate_responses:
            await recipe_batched(config, parsed_traits, backend, judge,
                                 layers_arg, trait_layers, direction)
        else:
            await recipe_sequential(config, parsed_traits, backend, judge,
                                    layers_arg, trait_layers, lora)
    finally:
        if judge is not None:
            await judge.close()
        del backend
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    if len(parsed_traits) > 1:
        print(f"\n{'='*60}\nCOMPLETED {len(parsed_traits)} TRAITS\n{'='*60}")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Steering evaluation")

    # Core
    parser.add_argument("--experiment", required=True)
    trait_group = parser.add_mutually_exclusive_group(required=True)
    trait_group.add_argument("--vector-from-trait", help="'experiment/category/trait'")
    trait_group.add_argument("--traits", help="Comma-separated traits")
    trait_group.add_argument("--rescore", help="Re-score existing responses (no GPU)")

    # I/O
    parser.add_argument("--prompt-set", default="steering")
    parser.add_argument("--questions-file", type=str, default=None)

    # Model
    parser.add_argument("--model-variant", default=None)
    parser.add_argument("--extraction-variant", default=None)
    parser.add_argument("--load-in-8bit", action="store_true")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--bnb-4bit-quant-type", default="nf4")
    parser.add_argument("--vector-experiment", default=None)
    add_backend_args(parser)

    # Vector
    parser.add_argument("--method", default="probe")
    parser.add_argument("--component", default="residual")
    parser.add_argument("--position", default="response[:5]")

    # Evaluation
    parser.add_argument("--subset", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--judge", default="openai", choices=["openai", "gemini"])
    parser.add_argument("--min-coherence", type=float, default=MIN_COHERENCE)
    prompt_group = parser.add_mutually_exclusive_group()
    prompt_group.add_argument("--no-custom-prompt", action="store_true")
    prompt_group.add_argument("--eval-prompt-from", type=str, metavar="TRAIT_PATH")
    prompt_group.add_argument("--trait-judge", type=str, metavar="JUDGE_PATH")

    # Search
    parser.add_argument("--layers", default="30%-60%")
    parser.add_argument("--trait-layers", nargs="+", metavar="TRAIT:LAYERS")
    parser.add_argument("--coefficients", help="Manual coefficients (comma-separated)")
    parser.add_argument("--search-steps", type=int, default=5)
    parser.add_argument("--up-mult", type=float, default=1.3)
    parser.add_argument("--down-mult", type=float, default=0.85)
    parser.add_argument("--start-mult", type=float, default=0.7)
    parser.add_argument("--momentum", type=float, default=0.1)

    # Advanced
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

    # Resolve model
    variant = get_model_variant(args.experiment, args.model_variant, mode="application")
    model_variant = variant['name']
    model_name = variant['model']
    lora = variant.get('lora')

    # Build config
    config = SteeringConfig(
        experiment=args.experiment,
        vector_experiment=args.vector_experiment,
        extraction_variant=args.extraction_variant,
        layers_arg=args.layers,
        coefficients=parse_coefficients(args.coefficients),
        n_steps=args.search_steps,
        up_mult=args.up_mult, down_mult=args.down_mult,
        start_mult=args.start_mult, momentum=args.momentum,
        method=args.method, component=args.component, position=args.position,
        max_new_tokens=args.max_new_tokens, min_coherence=args.min_coherence,
        subset=args.subset, relevance_check=not args.no_relevance_check,
        judge_provider=args.judge, prompt_set=args.prompt_set,
        use_default_prompt=args.no_custom_prompt,
        trait_judge=getattr(args, 'trait_judge', None),
        save_mode=args.save_responses, force=args.force,
        load_in_8bit=args.load_in_8bit, load_in_4bit=args.load_in_4bit,
        bnb_4bit_quant_type=args.bnb_4bit_quant_type,
        batched=not args.no_batch,
        regenerate_responses=args.regenerate_responses,
        baseline_only=args.baseline_only,
        questions_file=args.questions_file,
        ablation=args.ablation,
    )

    if args.eval_prompt_from:
        config.eval_prompt = load_steering_data(args.eval_prompt_from).eval_prompt
    if args.questions_file and args.prompt_set == "steering":
        config.prompt_set = Path(args.questions_file).stem

    try:
        # Rescore: no GPU, no model
        if args.rescore:
            asyncio.run(run_rescore(config, args.rescore, model_variant, dry_run=args.dry_run))
            return

        # Parse traits
        trait_specs = [t.strip() for t in args.traits.split(',')] if args.traits else [args.vector_from_trait]
        vec_exp = args.vector_experiment or args.experiment
        parsed_traits = []
        for spec in trait_specs:
            parts = spec.split('/')
            if len(parts) == 2:
                parsed_traits.append((vec_exp, spec))
            elif len(parts) == 3:
                parsed_traits.append((parts[0], f"{parts[1]}/{parts[2]}"))
            else:
                parser.error(f"Invalid: '{spec}'")

        # Parse per-trait layer overrides
        trait_layers = None
        if args.trait_layers:
            trait_layers = {}
            for spec in args.trait_layers:
                if ':' not in spec:
                    parser.error(f"Invalid --trait-layers: '{spec}'")
                tp, ls = spec.rsplit(':', 1)
                parts = tp.split('/')
                trait_layers[f"{parts[1]}/{parts[2]}" if len(parts) == 3 else tp] = ls

        # Resolve direction
        if args.direction:
            direction = args.direction
        elif config.coefficients:
            direction = "negative" if all(c <= 0 for c in config.coefficients) and any(c < 0 for c in config.coefficients) else "positive"
        else:
            dirs = set()
            for _, t in parsed_traits:
                try:
                    dirs.add(load_steering_data(t).direction or "positive")
                except (FileNotFoundError, ValueError):
                    dirs.add("positive")
            if len(dirs) > 1:
                raise ValueError(f"Mixed directions: {dirs}. Use --direction.")
            direction = dirs.pop() if dirs else "positive"
        config.direction = direction

        # Run
        asyncio.run(run(config, parsed_traits, model_variant, model_name, lora,
                        layers_arg=args.layers, trait_layers=trait_layers, direction=direction))
    finally:
        builtins.print = _original_print
        if is_tp_mode():
            import torch.distributed as dist
            if dist.is_initialized():
                dist.destroy_process_group()


if __name__ == "__main__":
    main()
