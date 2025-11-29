#!/usr/bin/env python3
"""
Main orchestrator for the trait extraction pipeline.

Input:
    - experiments/{experiment}/extraction/{trait}/positive.txt
    - experiments/{experiment}/extraction/{trait}/negative.txt
    - experiments/{experiment}/extraction/{trait}/trait_definition.txt (optional, for vetting)

Output:
    - responses/
    - vetting/ (scenario and/or response scores)
    - activations/
    - vectors/

Usage:
    # Full pipeline with vetting (default)
    python extraction/run_pipeline.py --experiment my_exp --traits category/my_trait

    # Skip scenario vetting (for instruction-based elicitation)
    python extraction/run_pipeline.py --experiment my_exp --traits category/my_trait --no-vet-scenarios

    # Skip all vetting (not recommended)
    python extraction/run_pipeline.py --experiment my_exp --traits category/my_trait --no-vet

    # All traits in experiment
    python extraction/run_pipeline.py --experiment my_exp
"""

import sys
import os
import argparse
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.paths import get as get_path
from utils.model import load_model, DEFAULT_MODEL
from extraction.generate_responses import generate_responses_for_trait
from extraction.extract_activations import extract_activations_for_trait
from extraction.extract_vectors import extract_vectors_for_trait

def discover_traits(experiment: str) -> list[str]:
    """
    Finds all traits within an experiment's extraction directory.
    This assumes the user has already created the directory structure.
    e.g., `experiments/my_exp/extraction/cognitive_state/confidence`
    """
    extraction_dir = get_path('extraction.base', experiment=experiment)
    traits = []
    if not extraction_dir.is_dir():
        return []
        
    for category_dir in extraction_dir.iterdir():
        if category_dir.is_dir():
            for trait_dir in category_dir.iterdir():
                if trait_dir.is_dir():
                    traits.append(f"{category_dir.name}/{trait_dir.name}")
    return sorted(traits)

def run_vetting(experiment: str, trait: str, stage: str, threshold: int = 4) -> float:
    """
    Run vetting for scenarios or responses.
    Returns pass rate (0.0 to 1.0).
    """
    from extraction.vet_scenarios import vet_scenarios
    from extraction.vet_responses import vet_responses

    if stage == 'scenarios':
        return vet_scenarios(experiment=experiment, trait=trait, threshold=threshold)
    elif stage == 'responses':
        return vet_responses(experiment=experiment, trait=trait, threshold=threshold)
    else:
        raise ValueError(f"Unknown vetting stage: {stage}")


def run_pipeline(experiment: str, traits_to_run: Optional[List[str]], force: bool, methods: List[str], vet: bool = False, vet_scenarios: bool = True, vet_threshold: int = 4, rollouts: int = 1, temperature: float = 0.0, batch_size: int = 8):
    """
    Executes the trait extraction pipeline.

    Args:
        vet: Enable response vetting (default: True via CLI)
        vet_scenarios: Enable scenario vetting (default: True). Set False for instruction-based elicitation.
    """
    print("=" * 80)
    print("STARTING TRAIT EXTRACTION PIPELINE")
    print(f"Experiment: {experiment}")
    print(f"Force mode: {'ON' if force else 'OFF'}")
    if vet:
        if vet_scenarios:
            print(f"Vetting: scenarios + responses")
        else:
            print(f"Vetting: responses only (scenario vetting disabled)")
    else:
        print(f"Vetting: OFF")
    if rollouts > 1:
        print(f"Rollouts: {rollouts}, Temperature: {temperature}")
    print("=" * 80)

    # Check for GEMINI_API_KEY if vetting enabled
    if vet and not os.environ.get('GEMINI_API_KEY'):
        print("\n❌ ERROR: --vet requires GEMINI_API_KEY environment variable")
        print("   Set it with: export GEMINI_API_KEY=your_key_here")
        return

    if traits_to_run:
        available_traits = traits_to_run
        print(f"Running for {len(available_traits)} specified traits.")
    else:
        available_traits = discover_traits(experiment)
        print(f"Discovered {len(available_traits)} traits to process.")

    if not available_traits:
        print("\nNo traits found. To create a new trait, first create its directory:")
        print("  mkdir -p experiments/<experiment_name>/extraction/<category_name>/<trait_name>")
        print("Then add `_positive.txt` and `_negative.txt` files to `extraction/scenarios/`.")
        return

    # --- Centralized Model Loading ---
    model, tokenizer = load_model(DEFAULT_MODEL)

    for trait in available_traits:
        print(f"\n--- Processing Trait: {trait} ---")
        base_trait_name = Path(trait).name

        # --- Stage 0: Scenario Vetting (optional) ---
        vetting_path = get_path("extraction.trait", experiment=experiment, trait=trait) / "vetting"
        if vet and vet_scenarios:
            scenario_scores = vetting_path / "scenario_scores.json"
            if not scenario_scores.exists() or force:
                print(f"  [Stage 0] Vetting scenarios...")
                pass_rate = run_vetting(experiment, trait, 'scenarios', vet_threshold)
                if pass_rate < 0.7:
                    print(f"  ⚠️  Low scenario pass rate ({pass_rate:.0%}). Consider reviewing scenarios.")
            else:
                print(f"  [Stage 0] Skipping scenario vetting (already exists).")
        elif vet and not vet_scenarios:
            print(f"  [Stage 0] Scenario vetting disabled (--no-vet-scenarios).")

        # --- Stage 1: Responses ---
        responses_path = get_path("extraction.responses", experiment=experiment, trait=trait)
        if not (responses_path / "pos.json").exists() or force:
            generate_responses_for_trait(
                experiment=experiment,
                trait=trait,
                model=model,
                tokenizer=tokenizer,
                rollouts=rollouts,
                temperature=temperature,
                batch_size=batch_size,
            )
        else:
            print(f"  [Stage 1] Skipping response generation (already exists).")

        # --- Stage 1.5: Response Vetting (optional) ---
        if vet:
            response_scores = vetting_path / "response_scores.json"
            if not response_scores.exists() or force:
                print(f"  [Stage 1.5] Vetting responses...")
                pass_rate = run_vetting(experiment, trait, 'responses', vet_threshold)
                if pass_rate < 0.7:
                    print(f"  ⚠️  Low response pass rate ({pass_rate:.0%}). Consider reviewing mislabeled examples.")
            else:
                print(f"  [Stage 1.5] Skipping response vetting (already exists).")

        # --- Stage 2: Activations ---
        activations_path = get_path("extraction.activations", experiment=experiment, trait=trait)
        if not (activations_path / "metadata.json").exists() or force:
            extract_activations_for_trait(
                experiment=experiment,
                trait=trait,
                model=model,
                tokenizer=tokenizer,
            )
        else:
            print(f"  [Stage 2] Skipping activation extraction (already exists).")

        # --- Stage 3: Vectors ---
        vectors_path = get_path("extraction.vectors", experiment=experiment, trait=trait)
        if not vectors_path.exists() or not any(vectors_path.glob("*.pt")) or force:
            extract_vectors_for_trait(
                experiment=experiment,
                trait=trait,
                methods=methods,
            )
        else:
            print(f"  [Stage 3] Skipping vector extraction (already exists).")
    
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Main orchestrator for the trait extraction pipeline.")
    parser.add_argument("--experiment", type=str, required=True, help="The name of the experiment.")
    parser.add_argument("--traits", type=str, help="Comma-separated list of specific traits to run (e.g., 'category/name1,category/name2').")
    parser.add_argument("--force", action="store_true", help="If set, overwrite existing data and re-run all stages.")
    parser.add_argument('--methods', type=str, default='mean_diff,probe,ica,gradient', help='Comma-separated method names for vector extraction.')
    parser.add_argument('--no-vet', action='store_true', help='Disable all LLM-as-a-judge vetting (not recommended).')
    parser.add_argument('--no-vet-scenarios', action='store_true', help='Skip scenario vetting, keep response vetting. Use for instruction-based elicitation.')
    parser.add_argument('--vet-threshold', type=int, default=4, help='Minimum score for vetting to pass (default: 4).')
    parser.add_argument('--rollouts', type=int, default=1, help='Responses per scenario (1 for natural, 10 for instruction-based).')
    parser.add_argument('--temperature', type=float, default=0.0, help='Sampling temperature (0.0 for deterministic, 1.0 for diverse).')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size for generation (default: 8).')

    args = parser.parse_args()

    traits_list = args.traits.split(',') if args.traits else None
    methods_list = args.methods.split(',')

    run_pipeline(
        experiment=args.experiment,
        traits_to_run=traits_list,
        force=args.force,
        methods=methods_list,
        vet=not args.no_vet,  # Vetting is ON by default
        vet_scenarios=not args.no_vet_scenarios,  # Scenario vetting ON by default
        vet_threshold=args.vet_threshold,
        rollouts=args.rollouts,
        temperature=args.temperature,
        batch_size=args.batch_size,
    )