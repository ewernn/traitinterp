#!/usr/bin/env python3
"""
Extraction pipeline: scenarios → responses → activations → vectors.

Stages (use --only-stage 3,4 to run specific stages):
    1: generate          --rollouts, --temperature      Model generates responses
    2: vet responses     --no-vet-responses to skip     LLM judge checks quality
  3+4: extract vectors   --methods, --layers             Forward pass → trait vectors
    6: evaluate                                          Quality metrics on held-out

Usage:
    python extraction/run_extraction_pipeline.py --experiment gemma-2-2b --traits category/trait
    python extraction/run_extraction_pipeline.py --experiment gemma-2-2b --category epistemic
    python extraction/run_extraction_pipeline.py --experiment gemma-2-2b --only-stage 3,4 --layers 25,30,35
"""

import sys
import json
import time
import argparse
import warnings
from datetime import datetime
from pathlib import Path
from typing import List

warnings.filterwarnings("ignore", message=".*penalty.*deprecated.*", category=FutureWarning)
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()


from core.kwargs_configs import ExtractionConfig, VettingStats
from utils.paths import (
    get as get_path, get_activation_metadata_path, get_activation_path,
    get_activation_dir, get_vector_dir, get_model_variant, discover_traits,
    content_hash,
)
from utils.distributed import is_rank_zero, tp_barrier, tp_lifecycle, flush_cuda
from utils.backends import LocalBackend, add_backend_args
from utils.model_registry import is_base_model
from utils.vram import format_duration
from utils.traits import load_scenarios
from utils.model import format_prompt
from utils.generation import generate_batch
from utils.extract_vectors import (
    extract_activations_for_trait, extract_vectors_for_trait,
    resolve_max_new_tokens, load_llm_judge_position,
)
from utils.preextraction_vetting import vet_responses as _vet_responses_raw


# =============================================================================
# Recipe
# =============================================================================

def run_pipeline(config: ExtractionConfig, traits: List[str]):
    """Generate → vet → extract → evaluate."""
    backend, variant_name, use_chat_template = _init_backend(config)

    for trait in traits:
        print(f"\n--- {trait} ---")

        generate_responses(config, trait, variant_name, backend, use_chat_template)

        if config.vet_responses:
            stats = vet_responses(config, trait, variant_name)
            if not stats.passed:
                print(f"  SKIP: {stats.pos_passed} pos, {stats.neg_passed} neg passed vetting")
                continue

        position = config.position
        if config.adaptive:
            llm_pos = load_llm_judge_position(config.experiment, trait, variant_name)
            if llm_pos:
                position = llm_pos
                print(f"  Adaptive position: {position}")

        extract_vectors(config, trait, variant_name, backend, position)

    evaluate(config, traits, variant_name)

    del backend
    flush_cuda()


# =============================================================================
# Stage implementations
# =============================================================================

def _init_backend(config: ExtractionConfig):
    """Load model backend. Returns (backend, variant_name, use_chat_template)."""
    variant = get_model_variant(config.experiment, config.model_variant, mode="extraction")
    is_base = config.base_model if config.base_model is not None else is_base_model(variant.model)
    backend = LocalBackend.from_experiment(
        config.experiment, variant=variant.name,
        load_in_8bit=config.load_in_8bit, load_in_4bit=config.load_in_4bit,
        bnb_4bit_quant_type=config.bnb_4bit_quant_type,
    )
    use_chat_template = not is_base and backend.tokenizer.chat_template is not None
    return backend, variant.name, use_chat_template


def _run_stage(config, stage_num):
    """Check if a stage should run based on --only-stage."""
    return not config.only_stages or stage_num in config.only_stages


def generate_responses(config: ExtractionConfig, trait: str, variant_name: str,
                       backend, use_chat_template: bool):
    """Stage 1: Generate model responses from scenarios."""
    if not _run_stage(config, 1):
        return

    responses_path = get_path("extraction.responses", experiment=config.experiment,
                               trait=trait, model_variant=variant_name)
    if not config.force and (responses_path / "pos.json").exists() and (responses_path / "neg.json").exists():
        return

    print(f"  [1] Generating responses...")
    max_new_tokens = resolve_max_new_tokens(config.position, config.max_new_tokens)
    model, tokenizer = backend.model, backend.tokenizer

    try:
        scenarios = load_scenarios(trait)
    except FileNotFoundError as e:
        print(f"    ERROR: {e}")
        return

    responses_path.mkdir(parents=True, exist_ok=True)

    for label in ['positive', 'negative']:
        results = []
        formatted = [
            format_prompt(s['prompt'], tokenizer, use_chat_template=use_chat_template,
                         system_prompt=s.get('system_prompt'))
            for s in scenarios[label]
        ]
        for _ in range(config.rollouts):
            responses = (
                [''] * len(formatted) if max_new_tokens == 0
                else generate_batch(model, tokenizer, formatted, max_new_tokens, config.temperature)
            )
            for scenario, response in zip(scenarios[label], responses):
                results.append({
                    'prompt': scenario['prompt'], 'response': response,
                    'system_prompt': scenario.get('system_prompt'),
                })

        if is_rank_zero():
            with open(responses_path / f'{label[:3]}.json', 'w') as f:
                json.dump(results, f, indent=2)
        print(f"    {label}: {len(results)} responses")

    if is_rank_zero():
        from utils.traits import get_scenario_path
        trait_dir = get_path('datasets.trait', trait=trait)
        with open(responses_path / 'metadata.json', 'w') as f:
            json.dump({
                'model': model.config.name_or_path, 'experiment': config.experiment,
                'trait': trait, 'max_new_tokens': max_new_tokens,
                'chat_template': use_chat_template, 'rollouts': config.rollouts,
                'temperature': config.temperature, 'timestamp': datetime.now().isoformat(),
                'input_hashes': {
                    'positive': content_hash(get_scenario_path(trait, 'positive')),
                    'negative': content_hash(get_scenario_path(trait, 'negative')),
                    'definition': content_hash(trait_dir / 'definition.txt'),
                },
            }, f, indent=2)

    tp_barrier()
    flush_cuda()


def vet_responses(config: ExtractionConfig, trait: str, variant_name: str) -> VettingStats:
    """Stage 2: LLM judge checks response quality. Returns VettingStats."""
    if not _run_stage(config, 2):
        return VettingStats.skip()

    scores_file = (
        get_path("extraction.trait", experiment=config.experiment, trait=trait, model_variant=variant_name)
        / "vetting" / "response_scores.json"
    )

    if not scores_file.exists() or config.force:
        if is_rank_zero():
            print(f"  [2] Vetting responses...")
            _vet_responses_raw(
                config.experiment, trait, variant_name,
                config.pos_threshold, config.neg_threshold, config.max_concurrent,
                estimate_trait_tokens=config.adaptive,
            )
        tp_barrier()

    if not scores_file.exists():
        return VettingStats.skip()

    with open(scores_file) as f:
        s = json.load(f).get('summary', {})

    return VettingStats(
        pos_passed=s.get('positive_passed', 0),
        pos_total=s.get('positive_passed', 0) + s.get('positive_failed', 0),
        neg_passed=s.get('negative_passed', 0),
        neg_total=s.get('negative_passed', 0) + s.get('negative_failed', 0),
    )


def _has_activations(config, trait, variant_name, position):
    """Check if activations exist (stacked or per-layer)."""
    metadata = get_activation_metadata_path(config.experiment, trait, variant_name, config.component, position)
    if not metadata.exists():
        return False
    stacked = get_activation_path(config.experiment, trait, variant_name, config.component, position)
    if stacked.exists():
        return True
    act_dir = get_activation_dir(config.experiment, trait, variant_name, config.component, position)
    return any(act_dir.glob("train_layer*.pt"))


def _has_vectors(config, trait, variant_name, position):
    """Check if vectors exist for all methods."""
    return all(
        list(get_vector_dir(config.experiment, trait, m, variant_name,
                            config.component, position).glob("layer*.pt"))
        for m in config.methods
    )


def extract_vectors(config: ExtractionConfig, trait: str, variant_name: str,
                    backend, position: str):
    """Stages 3+4: Forward pass → activations → trait vectors (in-memory by default)."""
    cached_activations = None

    if _run_stage(config, 3) and (config.force or not _has_activations(config, trait, variant_name, position)):
        print(f"  [3] Extracting activations...")
        cached_activations = extract_activations_for_trait(
            config.experiment, trait, variant_name, backend, config.val_split,
            position=position, component=config.component,
            use_vetting_filter=config.vet_responses, paired_filter=config.paired_filter,
            layers=config.layers,
            pos_threshold=config.pos_threshold, neg_threshold=config.neg_threshold,
            save_activations=config.save_activations,
        )
        tp_barrier()

    if _run_stage(config, 4) and (config.force or not _has_vectors(config, trait, variant_name, position)):
        print(f"  [4] Extracting vectors...")
        extract_vectors_for_trait(
            config.experiment, trait, variant_name, config.methods,
            layers=config.layers, component=config.component, position=position,
            activations=cached_activations,
        )


def evaluate(config: ExtractionConfig, traits: List[str], variant_name: str):
    """Stage 6: Quality metrics on held-out validation data."""
    if not _run_stage(config, 6):
        return

    eval_path = get_path("extraction_eval.evaluation", experiment=config.experiment)
    if eval_path.exists() and not config.force:
        return

    from analysis.vectors.extraction_evaluation import main as run_eval
    print(f"\n[6] Evaluating ({len(traits)} traits)...")
    run_eval(config.experiment, model_variant=variant_name,
             methods=",".join(config.methods), component=config.component, position=config.position)


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
    parser.add_argument("--max-new-tokens", type=int, default=None)

    # Vetting
    parser.add_argument("--no-vet-responses", action="store_true", help="Skip response quality vetting")
    parser.add_argument("--pos-threshold", type=int, default=60)
    parser.add_argument("--neg-threshold", type=int, default=40)
    parser.add_argument("--max-concurrent", type=int, default=100)
    parser.add_argument("--paired-filter", action="store_true")
    parser.add_argument("--adaptive", action="store_true")

    # Extraction
    parser.add_argument("--model-variant", default=None)
    parser.add_argument("--component", default="residual")
    parser.add_argument("--position", default="response[:5]")
    parser.add_argument("--layers", type=str, default=None)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--save-activations", action="store_true")

    # Model
    parser.add_argument("--load-in-8bit", action="store_true")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--bnb-4bit-quant-type", default="nf4")
    parser.add_argument("--base-model", action="store_true", dest="base_model_override")
    parser.add_argument("--it-model", action="store_true", dest="it_model_override")
    add_backend_args(parser)

    args = parser.parse_args()

    traits = args.traits.split(',') if args.traits else discover_traits(category=args.category)
    if not traits:
        raise ValueError("No traits found")

    # Parse layers
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
        max_new_tokens=args.max_new_tokens,
        vet_responses=not args.no_vet_responses,
        pos_threshold=args.pos_threshold,
        neg_threshold=args.neg_threshold,
        max_concurrent=args.max_concurrent,
        paired_filter=args.paired_filter,
        adaptive=args.adaptive,
        val_split=args.val_split,
        load_in_8bit=args.load_in_8bit,
        load_in_4bit=args.load_in_4bit,
        bnb_4bit_quant_type=args.bnb_4bit_quant_type,
        base_model=True if args.base_model_override else (False if args.it_model_override else None),
    )

    with tp_lifecycle():
        t = time.time()
        run_pipeline(config, traits)
        if is_rank_zero():
            print(f"\nComplete ({format_duration(time.time() - t)})")
            print(f"Tip: python steering/run_steering_eval.py "
                  f"--experiment {config.experiment} --traits {','.join(traits)}")


if __name__ == "__main__":
    main()
