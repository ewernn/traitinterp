#!/usr/bin/env python3
"""
Main orchestrator for the full trait pipeline: extraction + evaluation + steering.

Input (required):
    - experiments/{experiment}/extraction/{trait}/positive.txt
    - experiments/{experiment}/extraction/{trait}/negative.txt
    - experiments/{experiment}/extraction/{trait}/trait_definition.txt
    - analysis/steering/prompts/{trait_name}.json (or use --no-steering)
    - experiments/{experiment}/config.json (auto-created with extraction_model and application_model)

Output:
    - responses/, vetting/, activations/, vectors/ (each with metadata.json)
    - extraction/extraction_evaluation.json
    - steering/{trait}/results.json

Usage:
    # Full pipeline with config.json (recommended)
    # First create config.json with extraction_model and application_model
    python extraction/run_pipeline.py \\
        --experiment gemma-2-2b \\
        --traits epistemic/optimism

    # Override models from CLI
    python extraction/run_pipeline.py \\
        --experiment gemma-2-2b \\
        --extraction-model google/gemma-2-2b \\
        --application-model google/gemma-2-2b-it \\
        --traits epistemic/optimism

    # Extraction only (no steering)
    python extraction/run_pipeline.py \\
        --experiment gemma-2-2b \\
        --traits epistemic/optimism \\
        --no-steering

    # All traits in experiment
    python extraction/run_pipeline.py --experiment my_exp

    # Base model (auto-detected from config/models/*.yaml variant field)
    # Use --base-model or --it-model to override auto-detection
    python extraction/run_pipeline.py --experiment my_exp --traits epistemic/optimism \\
        --extraction-model Qwen/Qwen2.5-7B --rollouts 1 --temperature 1.0 --no-vet-scenarios
"""

import sys
import os
import json
import argparse
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.paths import get as get_path
from utils.model import load_model
from utils.model_registry import is_base_model
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


def ensure_experiment_config(experiment: str, extraction_model: str, application_model: str, tokenizer) -> dict:
    """
    Create or validate experiment config.json.

    On first run: creates config.json with auto-detected settings.
    On subsequent runs: validates model matches, warns on mismatch.

    Returns:
        Config dict with keys: extraction_model, application_model, use_chat_template
    """
    config_path = get_path('experiments.config', experiment=experiment)

    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)

        # Require new config format
        if 'extraction_model' not in config:
            raise ValueError(
                f"config.json missing 'extraction_model'. "
                f"Update {config_path} with extraction_model and application_model fields."
            )

        # Validate extraction model matches
        config_extraction = config.get('extraction_model')
        if config_extraction != extraction_model:
            print(f"\n⚠️  Warning: config.json has extraction_model={config_extraction}")
            print(f"   But you're running with --model {extraction_model}")
            response = input("   Continue anyway? [y/N] ").strip().lower()
            if response != 'y':
                print("Aborted.")
                sys.exit(1)
        return config

    # Auto-detect from tokenizer
    use_chat_template = tokenizer.chat_template is not None

    config = {
        "extraction_model": extraction_model,
        "application_model": application_model,
        "use_chat_template": use_chat_template,
    }

    # Create config file
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Created experiment config: {config_path}")
    print(f"  extraction_model: {extraction_model}")
    print(f"  application_model: {application_model}")
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

def run_vetting(experiment: str, trait: str, stage: str,
                pos_threshold: int = 60, neg_threshold: int = 40) -> float:
    """
    Run vetting for scenarios or responses.
    Returns pass rate (0.0 to 1.0).
    """
    from extraction.vet_scenarios import vet_scenarios
    from extraction.vet_responses import vet_responses

    if stage == 'scenarios':
        return vet_scenarios(
            experiment=experiment,
            trait=trait,
            pos_threshold=pos_threshold,
            neg_threshold=neg_threshold,
        )
    elif stage == 'responses':
        return vet_responses(
            experiment=experiment,
            trait=trait,
            pos_threshold=pos_threshold,
            neg_threshold=neg_threshold,
        )
    else:
        raise ValueError(f"Unknown vetting stage: {stage}")


def run_pipeline(
    experiment: str,
    extraction_model: str,
    application_model: str,
    traits_to_run: Optional[List[str]] = None,
    force: bool = False,
    methods: Optional[List[str]] = None,
    vet: bool = False,
    vet_scenarios: bool = True,
    rollouts: int = 1,
    temperature: float = 0.0,
    batch_size: int = 8,
    val_split: float = 0.2,
    base_model: Optional[bool] = None,
    pos_threshold: int = 60,
    neg_threshold: int = 40,
    no_steering: bool = False,
):
    """
    Executes the full trait pipeline: extraction + evaluation + steering.

    Args:
        vet: Enable response vetting (default: True via CLI)
        vet_scenarios: Enable scenario vetting (default: True). Set False for instruction-based elicitation.
        val_split: Fraction of scenarios for validation (0.2 = last 20%). 0 = no split.
        extraction_model: HuggingFace model for extraction
        application_model: HuggingFace model for steering
        base_model: None=auto-detect from model config, True/False=override
        pos_threshold: Positive samples need score >= this
        neg_threshold: Negative samples need score <= this
        no_steering: If True, skip steering evaluation
    """
    if methods is None:
        methods = ['mean_diff', 'probe', 'gradient']

    # Auto-detect base model from config if not explicitly set
    if base_model is None:
        base_model = is_base_model(extraction_model)

    print("=" * 80)
    print("STARTING TRAIT PIPELINE")
    print(f"Experiment: {experiment}")
    print(f"Extraction model: {extraction_model}")
    if not no_steering:
        print(f"Application model: {application_model}")
    mode_str = "BASE MODEL" if base_model else "IT MODEL"
    mode_source = "(auto-detected)" if base_model == is_base_model(extraction_model) else "(override)"
    print(f"Mode: {mode_str} {mode_source}")
    if base_model:
        print(f"  → No chat template, completion-only extraction")
    print(f"Force mode: {'ON' if force else 'OFF'}")
    if vet:
        if vet_scenarios:
            print(f"Vetting: scenarios + responses")
        else:
            print(f"Vetting: responses only (scenario vetting disabled)")
        print(f"Vetting thresholds: pos>={pos_threshold}, neg<={neg_threshold}")
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

    # --- Centralized Model Loading (extraction model) ---
    model, tokenizer = load_model(extraction_model)

    # --- Ensure Experiment Config ---
    config = ensure_experiment_config(experiment, extraction_model, application_model, tokenizer)

    # Determine chat template: base models never use it, IT models use if available
    if base_model:
        use_chat_template = False
    else:
        use_chat_template = config.get('use_chat_template')
        if use_chat_template is None:
            use_chat_template = tokenizer.chat_template is not None

    # Get application model from config (may differ from CLI if config exists)
    application_model = config.get('application_model', application_model)

    for trait in available_traits:
        print(f"\n--- Processing Trait: {trait} ---")

        # --- Stage 0: Scenario Vetting (optional) ---
        vetting_path = get_path("extraction.trait", experiment=experiment, trait=trait) / "vetting"
        if vet and vet_scenarios:
            scenario_scores = vetting_path / "scenario_scores.json"
            if not scenario_scores.exists() or force:
                print(f"  [Stage 0] Vetting scenarios...")
                pass_rate = run_vetting(experiment, trait, 'scenarios', pos_threshold, neg_threshold)
                if pass_rate < 0.7:
                    print(f"  ⚠️  Low scenario pass rate ({pass_rate:.0%}). Consider reviewing scenarios.")
            else:
                print(f"  [Stage 0] Skipping scenario vetting (already exists).")
        elif vet and not vet_scenarios:
            print(f"  [Stage 0] Scenario vetting disabled (--no-vet-scenarios).")

        # --- Stage 1: Responses ---
        responses_path = get_path("extraction.responses", experiment=experiment, trait=trait)
        if not (responses_path / "pos.json").exists() or force:
            # Base models drift quickly, so limit to 16 tokens; IT models can go longer
            max_new_tokens = 16 if base_model else 200
            generate_responses_for_trait(
                experiment=experiment,
                trait=trait,
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=max_new_tokens,
                rollouts=rollouts,
                temperature=temperature,
                batch_size=batch_size,
                chat_template=use_chat_template,
            )
        else:
            print(f"  [Stage 1] Skipping response generation (already exists).")

        # --- Stage 1.5: Response Vetting (optional) ---
        if vet:
            response_scores = vetting_path / "response_scores.json"
            if not response_scores.exists() or force:
                print(f"  [Stage 1.5] Vetting responses...")
                pass_rate = run_vetting(experiment, trait, 'responses', pos_threshold, neg_threshold)
                if pass_rate < 0.7:
                    print(f"  ⚠️  Low response pass rate ({pass_rate:.0%}). Consider reviewing mislabeled examples.")
            else:
                print(f"  [Stage 1.5] Skipping response vetting (already exists).")

        # --- Stage 2: Activations ---
        activations_path = get_path("extraction.activations", experiment=experiment, trait=trait)
        metadata_filename = "metadata.json" if args.component == 'residual' else f"{args.component}_metadata.json"
        if not (activations_path / metadata_filename).exists() or force:
            extract_activations_for_trait(
                experiment=experiment,
                trait=trait,
                model=model,
                tokenizer=tokenizer,
                val_split=val_split,
                base_model=base_model,
                component=args.component,
            )
        else:
            print(f"  [Stage 2] Skipping activation extraction (already exists).")

        # --- Stage 3: Vectors ---
        vectors_path = get_path("extraction.vectors", experiment=experiment, trait=trait)
        vector_pattern = "*.pt" if args.component == 'residual' else f"{args.component}_*.pt"
        # For residual, check that non-prefixed vectors exist (not v_cache_*, etc.)
        if args.component == 'residual':
            existing_vectors = [f for f in vectors_path.glob("*.pt") if not any(c in f.name for c in ['attn_out_', 'mlp_out_', 'k_cache_', 'v_cache_'])]
        else:
            existing_vectors = list(vectors_path.glob(vector_pattern))
        if not vectors_path.exists() or not existing_vectors or force:
            extract_vectors_for_trait(
                experiment=experiment,
                trait=trait,
                methods=methods,
                component=args.component,
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
        print(f"  Application model: {application_model}")
        import subprocess
        for trait in available_traits:
            print(f"\n  Steering: {trait}")
            vector_from_trait = f"{experiment}/{trait}"
            cmd = [
                sys.executable, "analysis/steering/evaluate.py",
                "--experiment", experiment,
                "--vector-from-trait", vector_from_trait,
                "--model", application_model,
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
    parser.add_argument("--experiment", type=str, required=True, help="Experiment name (extraction and steering results stored here).")
    parser.add_argument("--traits", type=str, help="Comma-separated list of specific traits to run (e.g., 'category/name1,category/name2').")
    parser.add_argument("--force", action="store_true", help="If set, overwrite existing data and re-run all stages.")
    parser.add_argument('--methods', type=str, default='mean_diff,probe,gradient', help='Comma-separated method names for vector extraction.')
    parser.add_argument('--no-vet', action='store_true', help='Disable all LLM-as-a-judge vetting (not recommended).')
    parser.add_argument('--no-vet-scenarios', action='store_true', help='Skip scenario vetting, keep response vetting. Use for instruction-based elicitation.')
    parser.add_argument('--no-steering', action='store_true', help='Skip steering evaluation (stages 1-4 only).')
    parser.add_argument('--rollouts', type=int, default=1, help='Responses per scenario (1 for natural, 10 for instruction-based).')
    parser.add_argument('--temperature', type=float, default=0.0, help='Sampling temperature (0.0 for deterministic, 1.0 for diverse).')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size for generation.')
    parser.add_argument('--val-split', type=float, default=0.2, help='Fraction of scenarios for validation. 0 = no split.')
    parser.add_argument('--extraction-model', type=str, default=None, help='HuggingFace model for extraction (default: from config.json).')
    parser.add_argument('--application-model', type=str, default=None, help='HuggingFace model for steering (default: from config.json or same as extraction-model).')
    model_mode = parser.add_mutually_exclusive_group()
    model_mode.add_argument('--base-model', action='store_true', dest='base_model_override', help='Force base model mode (no chat template, completion-only extraction).')
    model_mode.add_argument('--it-model', action='store_true', dest='it_model_override', help='Force IT model mode (use chat template if available).')
    parser.add_argument('--pos-threshold', type=int, default=60, help='Positive samples need score >= this.')
    parser.add_argument('--neg-threshold', type=int, default=40, help='Negative samples need score <= this.')
    parser.add_argument('--component', type=str, default='residual',
                        help='Component to extract: residual, attn_out, mlp_out, k_cache, v_cache (default: residual).')

    args = parser.parse_args()

    # Resolve models: CLI > config.json (both required)
    config_path = get_path('experiments.config', experiment=args.experiment)
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        if args.extraction_model is None:
            args.extraction_model = config.get('extraction_model')
        if args.application_model is None:
            args.application_model = config.get('application_model')
        if not args.extraction_model or not args.application_model:
            raise ValueError(
                f"config.json must have extraction_model and application_model. "
                f"Update {config_path}"
            )
        print(f"Using models from config.json:")
        print(f"  extraction_model: {args.extraction_model}")
        print(f"  application_model: {args.application_model}")
    else:
        if args.extraction_model is None or args.application_model is None:
            raise ValueError(
                f"No config.json found at {config_path}. "
                f"Create it with extraction_model and application_model, or use --extraction-model and --application-model flags."
            )

    traits_list = args.traits.split(',') if args.traits else None
    methods_list = args.methods.split(',')

    run_pipeline(
        experiment=args.experiment,
        traits_to_run=traits_list,
        force=args.force,
        methods=methods_list,
        vet=not args.no_vet,  # Vetting is ON by default
        vet_scenarios=not args.no_vet_scenarios,  # Scenario vetting ON by default
        rollouts=args.rollouts,
        temperature=args.temperature,
        batch_size=args.batch_size,
        val_split=args.val_split,
        extraction_model=args.extraction_model,
        application_model=args.application_model,
        base_model=True if args.base_model_override else (False if args.it_model_override else None),
        pos_threshold=args.pos_threshold,
        neg_threshold=args.neg_threshold,
        no_steering=args.no_steering,
    )