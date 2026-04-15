#!/usr/bin/env python3
"""
Extraction pipeline: scenarios → responses → vectors → logit lens → evaluate.

Stages (use --only-stage 3,4 to run specific stages):
    1: generate          --rollouts, --temperature      Model generates responses
    2: vet responses     --vet-responses                LLM judge checks quality (off by default)
  3+4: extract vectors   --methods, --layers            Forward pass → trait vectors
    5: logit lens                                       Project residual vectors → vocab tokens
    6: evaluate                                         Quality metrics on held-out

Usage:
    python extraction/run_extraction_pipeline.py --experiment gemma-2-2b --traits category/trait
    python extraction/run_extraction_pipeline.py --experiment gemma-2-2b --category epistemic
    python extraction/run_extraction_pipeline.py --experiment gemma-2-2b --only-stage 3,4 --layers 25,30,35
"""

import sys
import time
import argparse
import warnings
from pathlib import Path
from typing import List

warnings.filterwarnings("ignore", message=".*penalty.*deprecated.*", category=FutureWarning)
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from core.kwargs_configs import ExtractionConfig
from utils.distributed import is_rank_zero, tp_lifecycle, flush_cuda
from utils.backends import add_backend_args
from utils.paths import discover_traits
from utils.vram import format_duration
from utils.extraction import (
    init_backend, fill_extraction_defaults, per_trait_config,
    generate_responses, vet_responses, extract_vectors_for_trait,
    run_logit_lens, evaluate_extraction,
)


# =============================================================================
# Recipe
# =============================================================================

def run_pipeline(config: ExtractionConfig, traits: List[str], cli_overrides: set = None):
    """Generate → vet → extract → logit_lens → evaluate."""
    cli_overrides = cli_overrides or set()
    backend, variant_name, use_chat_template = init_backend(config)
    fill_extraction_defaults(config)

    for trait in traits:
        print(f"\n--- {trait} ---")
        with per_trait_config(config, trait, cli_overrides):
            generate_responses(config, trait, variant_name, backend, use_chat_template)
            if config.vet_responses:
                stats = vet_responses(config, trait, variant_name, backend, use_chat_template)
                if not stats.passed:
                    print(f"  SKIP: {stats.pos_passed} pos, {stats.neg_passed} neg passed vetting")
                    continue
            extract_vectors_for_trait(config, trait, variant_name, backend)
            if config.logit_lens:
                run_logit_lens(config, trait, variant_name, backend)

    evaluate_extraction(config, traits, variant_name)
    del backend
    flush_cuda()


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Extraction pipeline")
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--traits", type=str)
    parser.add_argument("--category", type=str)
    parser.add_argument("--only-stage", type=lambda s: [int(x) for x in s.split(',')], dest='only_stages')
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--methods", default="probe")

    # Generation
    parser.add_argument("--rollouts", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=None,
                       help="Random seed for reproducible sampling (T>0)")
    parser.add_argument("--max-new-tokens", type=int, default=None)

    # Vetting
    parser.add_argument("--vet-responses", action="store_true",
                       help="Enable response quality vetting (off by default)")
    parser.add_argument("--pos-threshold", type=int, default=60)
    parser.add_argument("--neg-threshold", type=int, default=40)
    parser.add_argument("--max-concurrent", type=int, default=100)
    parser.add_argument("--paired-filter", action="store_true")
    parser.add_argument("--adaptive", action="store_true")

    # Logit lens
    parser.add_argument("--no-logit-lens", action="store_true",
                       help="Skip stage 5 (logit lens). Defaults to running.")

    # Extraction
    parser.add_argument("--model-variant", default=None)
    parser.add_argument("--component", default="residual")
    parser.add_argument("--position", default=None,
                       help="Extraction position (default: response[:5] for base, response[:] for instruct)")
    parser.add_argument("--layers", type=str, default=None)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--save-activations", action="store_true")

    # Model
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--bnb-4bit-quant-type", default="nf4")
    parser.add_argument("--base-model", action="store_true", dest="base_model_override")
    parser.add_argument("--it-model", action="store_true", dest="it_model_override")
    add_backend_args(parser)

    args = parser.parse_args()

    traits = args.traits.split(',') if args.traits else discover_traits(category=args.category)
    if not traits:
        raise ValueError("No traits found")

    # Parse layers (needs model config to resolve "best" / "all" / etc.)
    parsed_layers = None
    if args.layers:
        from utils.layers import parse_layers
        from utils.paths import load_experiment_config
        exp_config = load_experiment_config(args.experiment)
        vname = args.model_variant or exp_config.get('defaults', {}).get('extraction', 'base')
        model_name = exp_config['model_variants'][vname]['model']
        from transformers import AutoConfig
        mc = AutoConfig.from_pretrained(model_name)
        if hasattr(mc, 'text_config'):
            mc = mc.text_config
        parsed_layers = parse_layers(args.layers, mc.num_hidden_layers)

    # Track which YAML-overridable fields the CLI explicitly set
    _yaml_flag_map = {
        'position': ('--position',),
        'max_new_tokens': ('--max-new-tokens',),
        'methods': ('--methods',),
        'temperature': ('--temperature',),
        'rollouts': ('--rollouts',),
    }
    cli_overrides = {
        field for field, flags in _yaml_flag_map.items()
        if any(f in sys.argv for f in flags)
    }

    config = ExtractionConfig(
        experiment=args.experiment,
        model_variant=args.model_variant,
        only_stages=set(args.only_stages) if args.only_stages else None,
        force=args.force,
        save_activations=args.save_activations,
        methods=args.methods.split(','),
        component=args.component,
        position=args.position,
        layers=parsed_layers,
        rollouts=args.rollouts,
        temperature=args.temperature,
        seed=args.seed,
        max_new_tokens=args.max_new_tokens,
        vet_responses=args.vet_responses,
        pos_threshold=args.pos_threshold,
        neg_threshold=args.neg_threshold,
        max_concurrent=args.max_concurrent,
        paired_filter=args.paired_filter,
        adaptive=args.adaptive,
        logit_lens=not args.no_logit_lens,
        val_split=args.val_split,
        load_in_4bit=args.load_in_4bit,
        bnb_4bit_quant_type=args.bnb_4bit_quant_type,
        base_model=True if args.base_model_override else (False if args.it_model_override else None),
    )

    with tp_lifecycle():
        t = time.time()
        run_pipeline(config, traits, cli_overrides=cli_overrides)
        if is_rank_zero():
            print(f"\nComplete ({format_duration(time.time() - t)})")
            print(f"Tip: python steering/run_steering_eval.py "
                  f"--experiment {config.experiment} --traits {','.join(traits)}")


if __name__ == "__main__":
    main()
