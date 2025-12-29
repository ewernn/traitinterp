#!/usr/bin/env python3
"""
Extraction pipeline: responses → activations → vectors → evaluation.

Stages:
    0: vet_scenarios   - LLM judges if scenarios match trait
    1: generate        - Generate model responses to scenarios
    2: vet_responses   - LLM judges if responses match trait
    3: activations     - Extract activations from responses
    4: vectors         - Train probe/gradient/mean_diff vectors
    5: evaluation      - Evaluate vectors on held-out data

Usage:
    python extraction/run_pipeline.py --experiment gemma-2-2b --traits category/trait
    python extraction/run_pipeline.py --experiment gemma-2-2b --category epistemic
    python extraction/run_pipeline.py --experiment gemma-2-2b  # all traits
    python extraction/run_pipeline.py --experiment gemma-2-2b --only-stage 4  # vectors only
"""

import sys
import gc
import json
import argparse
import warnings
from pathlib import Path

import torch
from typing import List, Optional, Set
from dotenv import load_dotenv

warnings.filterwarnings("ignore", message=".*penalty.*deprecated.*", category=FutureWarning)
load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.paths import (
    get as get_path,
    discover_traits,
    get_activation_metadata_path,
    get_vector_dir,
)
from utils.model import load_model
from utils.model_registry import is_base_model
from extraction.generate_responses import generate_responses_for_trait
from extraction.extract_activations import extract_activations_for_trait
from extraction.extract_vectors import extract_vectors_for_trait
from extraction.vet_scenarios import vet_scenarios
from extraction.vet_responses import vet_responses
from analysis.vectors.extraction_evaluation import main as run_evaluation

STAGES = {
    0: 'vet_scenarios',
    1: 'generate',
    2: 'vet_responses',
    3: 'activations',
    4: 'vectors',
    5: 'evaluation',
}


def run_pipeline(
    experiment: str,
    extraction_model: str,
    traits: List[str],
    only_stages: Optional[Set[int]] = None,
    force: bool = False,
    methods: List[str] = None,
    vet: bool = True,
    rollouts: int = 1,
    temperature: float = 0.0,
    batch_size: int = 8,
    val_split: float = 0.2,
    base_model: Optional[bool] = None,
    pos_threshold: int = 60,
    neg_threshold: int = 40,
    component: str = 'residual',
    position: str = 'response[:]',
    load_in_8bit: bool = False,
    load_in_4bit: bool = False,
    max_new_tokens: int = 64,
):
    """Execute extraction pipeline."""
    methods = methods or ['mean_diff', 'probe', 'gradient']
    if base_model is None:
        base_model = is_base_model(extraction_model)

    def should_run(stage: int) -> bool:
        return only_stages is None or stage in only_stages

    # Only load model if needed
    needs_model = should_run(1) or should_run(3)

    print("=" * 60)
    print(f"EXTRACTION PIPELINE | {experiment}")
    if only_stages:
        stage_names = [STAGES[s] for s in sorted(only_stages)]
        print(f"Stages: {', '.join(stage_names)}")
    print(f"Model: {extraction_model} | {'BASE' if base_model else 'IT'} ({max_new_tokens} tokens)")
    print(f"Traits: {len(traits)}")
    print("=" * 60)

    model, tokenizer = None, None
    if needs_model:
        model, tokenizer = load_model(extraction_model, load_in_8bit=load_in_8bit, load_in_4bit=load_in_4bit)
    use_chat_template = False if base_model else (model and tokenizer.chat_template is not None)

    for trait in traits:
        print(f"\n--- {trait} ---")
        vetting_path = get_path("extraction.trait", experiment=experiment, trait=trait) / "vetting"

        # Stage 0: Scenario vetting
        if should_run(0) and vet:
            if not (vetting_path / "scenario_scores.json").exists() or force:
                vet_scenarios(experiment, trait, pos_threshold, neg_threshold)

        # Stage 1: Generate responses
        if should_run(1):
            responses_path = get_path("extraction.responses", experiment=experiment, trait=trait)
            if not (responses_path / "pos.json").exists() or force:
                generate_responses_for_trait(experiment, trait, model, tokenizer, max_new_tokens,
                                             batch_size, rollouts, temperature, use_chat_template)

        # Stage 2: Response vetting
        if should_run(2) and vet:
            if not (vetting_path / "response_scores.json").exists() or force:
                vet_responses(experiment, trait, pos_threshold, neg_threshold)

        # Stage 3: Extract activations
        if should_run(3):
            activation_metadata = get_activation_metadata_path(experiment, trait, component, position)
            if not activation_metadata.exists() or force:
                extract_activations_for_trait(experiment, trait, model, tokenizer, val_split,
                                              position=position, component=component, batch_size=batch_size)

        # Stage 4: Extract vectors
        if should_run(4):
            # Check if any method directory has vectors
            has_vectors = False
            for method in methods:
                vector_dir = get_vector_dir(experiment, trait, method, component, position)
                if vector_dir.exists() and list(vector_dir.glob("layer*.pt")):
                    has_vectors = True
                    break
            if not has_vectors or force:
                extract_vectors_for_trait(experiment, trait, methods, component=component, position=position)

    # Stage 5: Evaluation
    if should_run(5):
        print("\n--- Evaluation ---")
        eval_path = get_path("extraction_eval.evaluation", experiment=experiment)
        if not eval_path.exists() or force:
            run_evaluation(experiment)

    # Cleanup GPU memory
    if model is not None:
        del model
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    print("\n" + "=" * 60 + "\nCOMPLETE\n" + "=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extraction pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Stages: 0=vet_scenarios, 1=generate, 2=vet_responses, 3=activations, 4=vectors, 5=evaluation"
    )
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--traits", type=str)
    parser.add_argument("--category", type=str)
    parser.add_argument("--only-stage", type=lambda s: [int(x) for x in s.split(',')], dest='only_stages',
                        help="Run only specific stage(s): --only-stage 3,4")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--methods", default="mean_diff,probe,gradient")
    parser.add_argument("--no-vet", action="store_true")
    parser.add_argument("--rollouts", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--extraction-model", type=str)
    parser.add_argument("--component", default="residual")
    parser.add_argument("--position", default="response[:]",
                        help="Token position: response[:], response[-1], prompt[-1], all[:], etc.")
    parser.add_argument("--pos-threshold", type=int, default=60)
    parser.add_argument("--neg-threshold", type=int, default=40)
    parser.add_argument("--load-in-8bit", action="store_true")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    model_mode = parser.add_mutually_exclusive_group()
    model_mode.add_argument("--base-model", action="store_true", dest="base_model_override")
    model_mode.add_argument("--it-model", action="store_true", dest="it_model_override")
    args = parser.parse_args()

    # Resolve model
    config_path = get_path('experiments.config', experiment=args.experiment)
    if config_path.exists():
        with open(config_path) as f:
            extraction_model = args.extraction_model or json.load(f).get('extraction_model')
    else:
        extraction_model = args.extraction_model
    if not extraction_model:
        raise ValueError(f"No model. Use --extraction-model or add to {config_path}")

    # Resolve traits
    if args.traits:
        traits = args.traits.split(',')
    else:
        traits = discover_traits(category=args.category)
    if not traits:
        raise ValueError("No traits found")

    run_pipeline(
        experiment=args.experiment,
        extraction_model=extraction_model,
        traits=traits,
        only_stages=set(args.only_stages) if args.only_stages else None,
        force=args.force,
        methods=args.methods.split(','),
        vet=not args.no_vet,
        rollouts=args.rollouts,
        temperature=args.temperature,
        batch_size=args.batch_size,
        val_split=args.val_split,
        base_model=True if args.base_model_override else (False if args.it_model_override else None),
        pos_threshold=args.pos_threshold,
        neg_threshold=args.neg_threshold,
        component=args.component,
        position=args.position,
        load_in_8bit=args.load_in_8bit,
        load_in_4bit=args.load_in_4bit,
        max_new_tokens=args.max_new_tokens,
    )
