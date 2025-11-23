#!/usr/bin/env python3
"""
Main orchestrator for the trait extraction pipeline.
"""

import sys
import argparse
import torch
from pathlib import Path
from typing import List, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.paths import get as get_path
from extraction.pipeline.generate_responses import generate_responses_for_trait
from extraction.pipeline.extract_activations import extract_activations_for_trait
from extraction.pipeline.extract_vectors import extract_vectors_for_trait

MODEL_NAME = "google/gemma-2-2b-it"

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

def run_pipeline(experiment: str, traits_to_run: Optional[List[str]], force: bool, methods: List[str]):
    """
    Executes the trait extraction pipeline.
    """
    print("=" * 80)
    print("STARTING TRAIT EXTRACTION PIPELINE")
    print(f"Experiment: {experiment}")
    print(f"Force mode: {'ON' if force else 'OFF'}")
    print("=" * 80)

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
    print(f"\nLoading model and tokenizer ({MODEL_NAME})...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map='auto'
    )
    model.eval()
    print("Model and tokenizer loaded.")

    for trait in available_traits:
        print(f"\n--- Processing Trait: {trait} ---")
        base_trait_name = Path(trait).name

        # --- Stage 1: Responses ---
        responses_path = get_path("extraction.responses", experiment=experiment, trait=trait)
        if not (responses_path / "pos.json").exists() or force:
            generate_responses_for_trait(
                experiment=experiment,
                trait=trait,
                model=model,
                tokenizer=tokenizer,
            )
        else:
            print(f"  [Stage 1] Skipping response generation (already exists).")

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
    
    args = parser.parse_args()

    traits_list = args.traits.split(',') if args.traits else None
    methods_list = args.methods.split(',')
    
    run_pipeline(
        experiment=args.experiment,
        traits_to_run=traits_list,
        force=args.force,
        methods=methods_list
    )