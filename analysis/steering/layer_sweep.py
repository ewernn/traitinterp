#!/usr/bin/env python3
"""
Layer sweep - find optimal steering layer for a trait.

Input:
    - experiment: Experiment name
    - trait: Trait path
    - layers: Range of layers to test (default 8-22, skipping weak early/late)
    - coefficient: Fixed coefficient for comparison

Output:
    - experiments/{experiment}/steering/{trait}/layer_sweep.json

Usage:
    python analysis/steering/layer_sweep.py \\
        --experiment gemma_2b_cognitive_nov21 \\
        --trait cognitive_state/confidence \\
        --layers 8-22 \\
        --coefficient 1.5
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import argparse
import asyncio
import json
from typing import List, Dict, Tuple
from datetime import datetime
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from analysis.steering.steer import SteeringHook
from analysis.steering.judge import TraitJudge
from analysis.steering.evaluate import (
    load_vector,
    load_eval_prompts,
    format_prompt,
    generate_response,
    MODEL_NAME,
)
from utils.paths import get


def parse_layer_range(layer_arg: str) -> List[int]:
    """Parse layer range string like '8-22' or '8,12,16,20'."""
    if '-' in layer_arg and ',' not in layer_arg:
        start, end = layer_arg.split('-')
        return list(range(int(start), int(end) + 1))
    else:
        return [int(x) for x in layer_arg.split(',')]


async def evaluate_layer(
    model,
    tokenizer,
    vector: torch.Tensor,
    layer: int,
    coefficient: float,
    questions: List[str],
    eval_prompt: str,
    judge: TraitJudge,
    rollouts: int = 3,
) -> Dict:
    """
    Quick evaluation of a single layer.

    Uses fewer rollouts than full eval for speed.
    """
    all_trait_scores = []
    all_coherence_scores = []

    for question in questions:
        formatted = format_prompt(question, tokenizer)

        for _ in range(rollouts):
            with SteeringHook(model, vector, layer, coefficient):
                response = generate_response(model, tokenizer, formatted)

            scores = await judge.score_batch(
                eval_prompt,
                [(question, response)],
            )
            score_data = scores[0]

            if score_data["trait_score"] is not None:
                all_trait_scores.append(score_data["trait_score"])
            if score_data.get("coherence_score") is not None:
                all_coherence_scores.append(score_data["coherence_score"])

    return {
        "layer": layer,
        "n": len(all_trait_scores),
        "trait_mean": sum(all_trait_scores) / len(all_trait_scores) if all_trait_scores else None,
        "coherence_mean": (
            sum(all_coherence_scores) / len(all_coherence_scores)
            if all_coherence_scores else None
        ),
    }


async def run_layer_sweep(
    experiment: str,
    trait: str,
    layers: List[int],
    coefficient: float = 1.5,
    method: str = "probe",
    rollouts: int = 3,
    judge_provider: str = "openai",
    subset_questions: int = 10,
) -> Dict:
    """
    Sweep across layers to find optimal steering layer.

    Uses a subset of questions and fewer rollouts for speed.
    """
    print(f"Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    prompts_data = load_eval_prompts(trait)
    judge = TraitJudge(provider=judge_provider)

    questions = prompts_data["questions"][:subset_questions]
    eval_prompt = prompts_data["eval_prompt"]

    print(f"\nLayer sweep: {trait}")
    print(f"  Layers: {layers}")
    print(f"  Coefficient: {coefficient}")
    print(f"  Questions: {len(questions)} (subset), Rollouts: {rollouts}")

    # Also get baseline (no steering)
    print("\nBaseline (no steering)...")
    baseline_scores = []
    for question in tqdm(questions, desc="baseline"):
        formatted = format_prompt(question, tokenizer)
        response = generate_response(model, tokenizer, formatted)
        scores = await judge.score_batch(
            eval_prompt,
            [(question, response)],
        )
        if scores[0]["trait_score"] is not None:
            baseline_scores.append(scores[0]["trait_score"])

    baseline_mean = sum(baseline_scores) / len(baseline_scores) if baseline_scores else 0

    # Sweep layers
    results_by_layer = {}
    for layer in tqdm(layers, desc="layer sweep"):
        try:
            vector = load_vector(experiment, trait, layer, method)
        except FileNotFoundError:
            print(f"  Skipping layer {layer}: no vector found")
            continue

        result = await evaluate_layer(
            model, tokenizer, vector, layer, coefficient,
            questions, eval_prompt, judge, rollouts,
        )
        results_by_layer[layer] = result
        print(f"  layer {layer}: mean={result['trait_mean']:.1f}, coherence={result.get('coherence_mean', 'N/A')}")

    # Find best layer
    best_layer = None
    best_score = float('-inf')
    for layer, result in results_by_layer.items():
        if result["trait_mean"] is not None and result["trait_mean"] > best_score:
            best_score = result["trait_mean"]
            best_layer = layer

    # Check coherence degradation
    coherence_warning = []
    for layer, result in results_by_layer.items():
        if result.get("coherence_mean") is not None and result["coherence_mean"] < 50:
            coherence_warning.append(layer)

    results = {
        "trait": trait,
        "method": method,
        "coefficient": coefficient,
        "judge_provider": judge_provider,
        "n_questions": len(questions),
        "rollouts": rollouts,
        "timestamp": datetime.now().isoformat(),
        "baseline_mean": baseline_mean,
        "layers": {str(k): v for k, v in results_by_layer.items()},
        "best_layer": best_layer,
        "best_score": best_score,
        "delta_from_baseline": best_score - baseline_mean if best_layer else None,
        "coherence_warnings": coherence_warning,
    }

    return results


def save_results(results: Dict, experiment: str, trait: str):
    """Save layer sweep results."""
    results_file = get('steering.layer_sweep', experiment=experiment, trait=trait)
    results_file.parent.mkdir(parents=True, exist_ok=True)

    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {results_file}")

    # Print summary
    print(f"\n=== Summary ===")
    print(f"Baseline: {results['baseline_mean']:.1f}")
    print(f"Best layer: {results['best_layer']} (score={results['best_score']:.1f})")
    print(f"Delta: +{results['delta_from_baseline']:.1f}")
    if results['coherence_warnings']:
        print(f"Coherence warnings: layers {results['coherence_warnings']}")


def main():
    parser = argparse.ArgumentParser(description="Layer sweep for steering")
    parser.add_argument("--experiment", required=True, help="Experiment name")
    parser.add_argument("--trait", required=True, help="Trait path")
    parser.add_argument("--layers", default="8-22", help="Layer range (e.g., 8-22 or 8,12,16,20)")
    parser.add_argument("--coefficient", type=float, default=1.5, help="Steering coefficient")
    parser.add_argument("--method", default="probe", help="Vector extraction method")
    parser.add_argument("--rollouts", type=int, default=3, help="Rollouts per question")
    parser.add_argument("--judge", default="openai", choices=["openai", "gemini"])
    parser.add_argument("--subset", type=int, default=10, help="Number of questions to use")

    args = parser.parse_args()
    layers = parse_layer_range(args.layers)

    results = asyncio.run(run_layer_sweep(
        experiment=args.experiment,
        trait=args.trait,
        layers=layers,
        coefficient=args.coefficient,
        method=args.method,
        rollouts=args.rollouts,
        judge_provider=args.judge,
        subset_questions=args.subset,
    ))

    save_results(results, args.experiment, args.trait)


if __name__ == "__main__":
    main()
