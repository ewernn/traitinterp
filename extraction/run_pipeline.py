#!/usr/bin/env python3
"""
Main orchestrator for the full trait pipeline: extraction + evaluation + steering.

Input (required):
    - experiments/{experiment}/extraction/{trait}/positive.txt
    - experiments/{experiment}/extraction/{trait}/negative.txt
    - experiments/{experiment}/extraction/{trait}/trait_definition.txt
    - analysis/steering/prompts/{trait_name}.json (or use --no-steering)

Output:
    - responses/, vetting/, activations/, vectors/
    - extraction/extraction_evaluation.json
    - {steer_experiment}/steering/{trait}/results.json

Usage:
    # Full pipeline: extract on base, steer on IT
    python extraction/run_pipeline.py \\
        --experiment gemma-2-2b-base \\
        --steer-experiment gemma-2-2b-it \\
        --traits epistemic/optimism

    # Extraction only (no steering)
    python extraction/run_pipeline.py \\
        --experiment gemma-2-2b-base \\
        --traits epistemic/optimism \\
        --no-steering

    # All traits in experiment
    python extraction/run_pipeline.py --experiment my_exp --steer-experiment my_exp_it

    # Base model with rollouts
    python extraction/run_pipeline.py --experiment my_exp --traits epistemic/optimism \\
        --model Qwen/Qwen2.5-7B --base-model --rollouts 10 --temperature 1.0 --no-vet-scenarios
"""

import sys
import os
import json
import argparse
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.paths import get as get_path
from utils.model import load_model, DEFAULT_MODEL
from extraction.generate_responses import generate_responses_for_trait
from extraction.extract_activations import extract_activations_for_trait
from extraction.extract_vectors import extract_vectors_for_trait


def validate_trait_files(experiment: str, trait: str, no_steering: bool = False) -> tuple[bool, list[str]]:
    """
    Validate that all required files exist for a trait.

    Returns:
        (is_valid, list of error messages)
    """
    errors = []
    trait_dir = get_path('extraction.trait', experiment=experiment, trait=trait)
    trait_name = Path(trait).name

    # Required extraction files
    required_files = ['positive.txt', 'negative.txt', 'trait_definition.txt']
    for filename in required_files:
        filepath = trait_dir / filename
        if not filepath.exists():
            errors.append(f"Missing: {filepath}")

    # Steering prompts (required unless --no-steering)
    if not no_steering:
        prompts_file = get_path('steering.prompt_file', trait=trait_name)
        if not prompts_file.exists():
            errors.append(f"Missing steering prompts: {prompts_file}")
            errors.append(f"  Create this file or use --no-steering to skip steering evaluation")

    return len(errors) == 0, errors


def ensure_experiment_config(experiment: str, model_name: str, tokenizer) -> dict:
    """
    Create or validate experiment config.json.

    On first run: creates config.json with auto-detected settings.
    On subsequent runs: validates model matches, warns on mismatch.

    Returns:
        Config dict with keys: model, use_chat_template, system_prompt
    """
    config_path = get_path('experiments.config', experiment=experiment)

    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        # Validate model matches
        if config.get('model') != model_name:
            print(f"\n⚠️  Warning: config.json has model={config['model']}")
            print(f"   But you're running with --model {model_name}")
            response = input("   Continue anyway? [y/N] ").strip().lower()
            if response != 'y':
                print("Aborted.")
                sys.exit(1)
        return config

    # Auto-detect from tokenizer
    use_chat_template = tokenizer.chat_template is not None

    config = {
        "model": model_name,
        "use_chat_template": use_chat_template,
        "system_prompt": None,
    }

    # Create config file
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Created experiment config: {config_path}")
    print(f"  model: {model_name}")
    print(f"  use_chat_template: {use_chat_template} (auto-detected from tokenizer)")

    return config

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

def run_vetting(experiment: str, trait: str, stage: str, threshold: int = 4,
                trait_score: bool = False, pos_threshold: int = 60, neg_threshold: int = 40) -> float:
    """
    Run vetting for scenarios or responses.
    Returns pass rate (0.0 to 1.0).
    """
    from extraction.vet_scenarios import vet_scenarios
    from extraction.vet_responses import vet_responses

    if stage == 'scenarios':
        return vet_scenarios(experiment=experiment, trait=trait, threshold=threshold)
    elif stage == 'responses':
        return vet_responses(
            experiment=experiment,
            trait=trait,
            threshold=threshold,
            trait_score=trait_score,
            pos_threshold=pos_threshold,
            neg_threshold=neg_threshold,
        )
    else:
        raise ValueError(f"Unknown vetting stage: {stage}")


def run_pipeline(
    experiment: str,
    traits_to_run: Optional[List[str]],
    force: bool,
    methods: List[str],
    vet: bool = False,
    vet_scenarios: bool = True,
    vet_threshold: int = 4,
    rollouts: int = 1,
    temperature: float = 0.0,
    batch_size: int = 8,
    val_split: float = 0.2,
    model_name: str = DEFAULT_MODEL,
    base_model: bool = False,
    trait_score: bool = False,
    pos_threshold: int = 60,
    neg_threshold: int = 40,
    no_steering: bool = False,
    steer_experiment: Optional[str] = None,
):
    """
    Executes the full trait pipeline: extraction + evaluation + steering.

    Args:
        vet: Enable response vetting (default: True via CLI)
        vet_scenarios: Enable scenario vetting (default: True). Set False for instruction-based elicitation.
        val_split: Fraction of scenarios for validation (0.2 = last 20%). 0 = no split.
        model_name: HuggingFace model name (default: Gemma 2B IT)
        base_model: If True, use text completion mode (no chat template, completion-only extraction)
        trait_score: If True, use 0-100 trait scoring mode for response vetting
        pos_threshold: For trait_score mode, positive class needs score >= this (default 60)
        neg_threshold: For trait_score mode, negative class needs score <= this (default 40)
        no_steering: If True, skip steering evaluation
        steer_experiment: Experiment to run steering on (default: same as experiment)
    """
    # Default steer_experiment to experiment if not specified
    if steer_experiment is None:
        steer_experiment = experiment

    print("=" * 80)
    print("STARTING TRAIT PIPELINE")
    print(f"Extraction experiment: {experiment}")
    if not no_steering:
        print(f"Steering experiment: {steer_experiment}")
    print(f"Model: {model_name}")
    print(f"Chat template: will be auto-detected from experiment config")
    if base_model:
        print(f"Mode: BASE MODEL (completion-only extraction)")
    print(f"Force mode: {'ON' if force else 'OFF'}")
    if vet:
        if vet_scenarios:
            print(f"Vetting: scenarios + responses")
        else:
            print(f"Vetting: responses only (scenario vetting disabled)")
        if trait_score:
            print(f"Vetting mode: trait-score (0-100), pos>={pos_threshold}, neg<={neg_threshold}")
    else:
        print(f"Vetting: OFF")
    if rollouts > 1:
        print(f"Rollouts: {rollouts}, Temperature: {temperature}")
    if val_split > 0:
        print(f"Val split: {val_split:.0%} (last {val_split:.0%} of scenarios)")
    if no_steering:
        print(f"Steering: OFF (--no-steering)")
    else:
        print(f"Steering: ON")
    print("=" * 80)

    # Check for GEMINI_API_KEY if vetting enabled
    if vet and not os.environ.get('GEMINI_API_KEY'):
        print("\n❌ ERROR: --vet requires GEMINI_API_KEY environment variable")
        print("   Set it with: export GEMINI_API_KEY=your_key_here")
        return

    if traits_to_run:
        available_traits = traits_to_run
        print(f"\nRunning for {len(available_traits)} specified traits.")
    else:
        available_traits = discover_traits(experiment)
        print(f"\nDiscovered {len(available_traits)} traits to process.")

    if not available_traits:
        print("\nNo traits found. To create a new trait, first create its directory:")
        print("  mkdir -p experiments/<experiment_name>/extraction/<category_name>/<trait_name>")
        print("Then add positive.txt, negative.txt, and trait_definition.txt files.")
        return

    # --- Validate all traits before starting ---
    print("\nValidating trait files...")
    all_valid = True
    for trait in available_traits:
        is_valid, errors = validate_trait_files(experiment, trait, no_steering)
        if not is_valid:
            all_valid = False
            print(f"\n❌ {trait}:")
            for error in errors:
                print(f"   {error}")

    if not all_valid:
        print("\n" + "=" * 80)
        print("PIPELINE ABORTED: Missing required files")
        print("=" * 80)
        return

    # --- Centralized Model Loading ---
    model, tokenizer = load_model(model_name)

    # --- Ensure Experiment Config ---
    config = ensure_experiment_config(experiment, model_name, tokenizer)
    use_chat_template = config.get('use_chat_template')
    # If still None (auto-detect), use tokenizer
    if use_chat_template is None:
        use_chat_template = tokenizer.chat_template is not None

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
                chat_template=use_chat_template,
                base_model=base_model,
            )
        else:
            print(f"  [Stage 1] Skipping response generation (already exists).")

        # --- Stage 1.5: Response Vetting (optional) ---
        if vet:
            response_scores = vetting_path / "response_scores.json"
            if not response_scores.exists() or force:
                print(f"  [Stage 1.5] Vetting responses...")
                pass_rate = run_vetting(
                    experiment, trait, 'responses', vet_threshold,
                    trait_score=trait_score, pos_threshold=pos_threshold, neg_threshold=neg_threshold
                )
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
                val_split=val_split,
                base_model=base_model,
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

    # --- Stage 4: Extraction Evaluation ---
    print(f"\n--- Stage 4: Evaluating Vectors ---")
    eval_path = get_path("extraction_eval.evaluation", experiment=experiment)
    if not eval_path.exists() or force:
        import subprocess
        cmd = [
            sys.executable, "analysis/vectors/extraction_evaluation.py",
            "--experiment", experiment,
        ]
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"  ⚠️  Evaluation failed: {e}")
    else:
        print(f"  Skipping evaluation (already exists). Use --force to re-run.")

    # --- Stage 5: Steering Evaluation ---
    if not no_steering:
        print(f"\n--- Stage 5: Steering Evaluation ---")
        # Check for steering API key
        if not os.environ.get('OPENAI_API_KEY') and not os.environ.get('GEMINI_API_KEY'):
            print("  ⚠️  Skipping steering: No OPENAI_API_KEY or GEMINI_API_KEY found")
        else:
            import subprocess
            for trait in available_traits:
                print(f"\n  Steering: {trait}")
                vector_from_trait = f"{experiment}/{trait}"
                cmd = [
                    sys.executable, "analysis/steering/evaluate.py",
                    "--experiment", steer_experiment,
                    "--vector-from-trait", vector_from_trait,
                ]
                try:
                    subprocess.run(cmd, check=True)
                except subprocess.CalledProcessError as e:
                    print(f"  ⚠️  Steering failed for {trait}: {e}")

    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Full trait pipeline: extraction + evaluation + steering.")
    parser.add_argument("--experiment", type=str, required=True, help="Experiment for extraction (where vectors are saved).")
    parser.add_argument("--steer-experiment", type=str, help="Experiment for steering (default: same as --experiment).")
    parser.add_argument("--traits", type=str, help="Comma-separated list of specific traits to run (e.g., 'category/name1,category/name2').")
    parser.add_argument("--force", action="store_true", help="If set, overwrite existing data and re-run all stages.")
    parser.add_argument('--methods', type=str, default='mean_diff,probe,gradient', help='Comma-separated method names for vector extraction.')
    parser.add_argument('--no-vet', action='store_true', help='Disable all LLM-as-a-judge vetting (not recommended).')
    parser.add_argument('--no-vet-scenarios', action='store_true', help='Skip scenario vetting, keep response vetting. Use for instruction-based elicitation.')
    parser.add_argument('--no-steering', action='store_true', help='Skip steering evaluation (stages 1-4 only).')
    parser.add_argument('--vet-threshold', type=int, default=4, help='Minimum score for vetting to pass (default: 4).')
    parser.add_argument('--rollouts', type=int, default=1, help='Responses per scenario (1 for natural, 10 for instruction-based).')
    parser.add_argument('--temperature', type=float, default=0.0, help='Sampling temperature (0.0 for deterministic, 1.0 for diverse).')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size for generation (default: 8).')
    parser.add_argument('--val-split', type=float, default=0.2, help='Fraction of scenarios for validation (default 0.2 = last 20%%). 0 = no split.')
    parser.add_argument('--model', type=str, default=None, help='HuggingFace model name (default: from experiment config.json).')
    parser.add_argument('--base-model', action='store_true', help='Base model mode: text completion, completion-only extraction.')
    parser.add_argument('--trait-score', action='store_true', help='Use 0-100 trait scoring for response vetting (recommended for base model).')
    parser.add_argument('--pos-threshold', type=int, default=60, help='For trait-score mode: positive class needs score >= this (default: 60).')
    parser.add_argument('--neg-threshold', type=int, default=40, help='For trait-score mode: negative class needs score <= this (default: 40).')

    args = parser.parse_args()

    # Resolve model: CLI > config.json > DEFAULT_MODEL
    if args.model is None:
        config_path = get_path('experiment.config', experiment=args.experiment)
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
            args.model = config.get('model', DEFAULT_MODEL)
            print(f"Using model from config.json: {args.model}")
        else:
            args.model = DEFAULT_MODEL
            print(f"No config.json found, using default: {args.model}")

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
        val_split=args.val_split,
        model_name=args.model,
        base_model=args.base_model,
        trait_score=args.trait_score,
        pos_threshold=args.pos_threshold,
        neg_threshold=args.neg_threshold,
        no_steering=args.no_steering,
        steer_experiment=args.steer_experiment,
    )