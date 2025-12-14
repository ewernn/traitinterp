#!/usr/bin/env python3
"""
Steering evaluation using Gaussian ensemble.

Input:
    - Best (mu, sigma) from classification search OR manual specification
    - experiments/{experiment}/extraction/{trait}/vectors/*.pt

Output:
    - experiments/{experiment}/steering/{trait}/results.json
      (adds runs with config.ensemble = {mu, sigma, ...})

Usage:
    # Use best (mu, sigma) from classification search
    python analysis/ensemble/steering_evaluation.py \
        --experiment {experiment} \
        --vector-from-trait {experiment}/{category}/{trait}

    # Manual (mu, sigma) specification
    python analysis/ensemble/steering_evaluation.py \
        --experiment {experiment} \
        --vector-from-trait {experiment}/{category}/{trait} \
        --mu 12 --sigma 2

    # Sweep coef for fixed (mu, sigma)
    python analysis/ensemble/steering_evaluation.py \
        --experiment {experiment} \
        --vector-from-trait {experiment}/{category}/{trait} \
        --mu 12 --sigma 2 \
        --coefficients 50,100,150

    # Compare to single-layer baseline
    python analysis/ensemble/steering_evaluation.py \
        --experiment {experiment} \
        --vector-from-trait {experiment}/{category}/{trait} \
        --compare-baseline
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import argparse
import asyncio
import json
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from analysis.steering.steer import MultiLayerSteeringHook
from analysis.steering.generation import generate_response
from analysis.steering.results import load_or_create_results, save_results, save_responses
from analysis.ensemble.gaussian import (
    compute_gaussian_weights,
    load_vectors_for_trait,
    get_active_layers,
)
from utils.judge import TraitJudge
from utils.paths import get
from utils.model import format_prompt, load_experiment_config


def load_best_ensemble_params(experiment: str, trait: str) -> Optional[Dict]:
    """
    Load best (mu, sigma) from ensemble_evaluation.json.

    Returns:
        {"mu": float, "sigma": float, "val_accuracy": float} or None
    """
    eval_path = get('ensemble.evaluation', experiment=experiment)
    if not eval_path.exists():
        return None

    with open(eval_path) as f:
        data = json.load(f)

    best_per_trait = data.get('best_per_trait', {})
    return best_per_trait.get(trait)


def compute_ensemble_coefficients(
    mu: float,
    sigma: float,
    layers: List[int],
    global_coefficient: float,
) -> Dict[int, float]:
    """
    Compute per-layer coefficients using Gaussian weights.

    coef_layer = global_coefficient * gaussian_weight(layer, mu, sigma)

    The Gaussian weight is normalized so sum = 1, then scaled by global_coefficient.
    """
    weights = compute_gaussian_weights(mu, sigma, layers)
    return {layer: global_coefficient * weight for layer, weight in weights.items()}


def load_model_and_tokenizer(model_name: str):
    """Load model and tokenizer."""
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    return model, tokenizer


def load_eval_prompts(trait: str) -> Tuple[Dict, Path]:
    """Load evaluation prompts for a trait. Returns (data, path)."""
    prompts_file = get('datasets.trait_steering', trait=trait)

    if not prompts_file.exists():
        raise FileNotFoundError(
            f"Eval prompts not found: {prompts_file}\n"
            f"Create JSON with 'questions' and 'eval_prompt' (with {{question}} and {{answer}} placeholders)."
        )

    with open(prompts_file) as f:
        data = json.load(f)

    if "eval_prompt" not in data:
        raise ValueError(f"Missing 'eval_prompt' in {prompts_file}")
    if "questions" not in data:
        raise ValueError(f"Missing 'questions' in {prompts_file}")

    return data, prompts_file


async def evaluate_ensemble_steering(
    model,
    tokenizer,
    vectors: Dict[int, torch.Tensor],
    mu: float,
    sigma: float,
    global_coefficient: float,
    questions: List[str],
    eval_prompt: str,
    judge: TraitJudge,
    use_chat_template: bool,
    component: str = "residual",
) -> Tuple[Dict, List[Dict]]:
    """
    Evaluate steering with Gaussian ensemble at specific (mu, sigma, global_coef).

    Returns:
        (result_dict, responses_list)
    """
    # Compute per-layer coefficients
    layers = sorted(vectors.keys())
    layer_coefficients = compute_ensemble_coefficients(mu, sigma, layers, global_coefficient)
    active_layers = sorted(layer_coefficients.keys())

    # Build hook configs: (layer, vector, coefficient)
    hook_configs = [
        (layer, vectors[layer], layer_coefficients[layer])
        for layer in active_layers
    ]

    all_trait_scores = []
    all_coherence_scores = []
    responses = []

    for question in tqdm(questions, desc=f"mu={mu:.1f},sigma={sigma:.1f},coef={global_coefficient:.0f}"):
        formatted = format_prompt(question, tokenizer, use_chat_template=use_chat_template)

        with MultiLayerSteeringHook(model, hook_configs, positions="all", component=component):
            response = generate_response(model, tokenizer, formatted)

        scores = await judge.score_steering_batch(eval_prompt, [(question, response)])

        if scores[0]["trait_score"] is not None:
            all_trait_scores.append(scores[0]["trait_score"])
        if scores[0].get("coherence_score") is not None:
            all_coherence_scores.append(scores[0]["coherence_score"])

        responses.append({
            "question": question,
            "response": response,
            "trait_score": scores[0]["trait_score"],
            "coherence_score": scores[0].get("coherence_score"),
        })

    result = {
        "trait_mean": sum(all_trait_scores) / len(all_trait_scores) if all_trait_scores else None,
        "trait_std": (sum((s - sum(all_trait_scores)/len(all_trait_scores))**2 for s in all_trait_scores) / len(all_trait_scores)) ** 0.5 if len(all_trait_scores) > 1 else 0,
        "coherence_mean": sum(all_coherence_scores) / len(all_coherence_scores) if all_coherence_scores else None,
        "n": len(all_trait_scores),
    }

    return result, responses


async def compute_baseline(
    model,
    tokenizer,
    questions: List[str],
    eval_prompt: str,
    judge: TraitJudge,
    use_chat_template: bool,
) -> Dict:
    """Compute baseline scores (no steering)."""
    print("\nComputing baseline (no steering)...")

    all_trait_scores = []
    all_coherence_scores = []

    for question in tqdm(questions, desc="baseline"):
        formatted = format_prompt(question, tokenizer, use_chat_template=use_chat_template)
        response = generate_response(model, tokenizer, formatted)
        scores = await judge.score_steering_batch(eval_prompt, [(question, response)])

        if scores[0]["trait_score"] is not None:
            all_trait_scores.append(scores[0]["trait_score"])
        if scores[0].get("coherence_score") is not None:
            all_coherence_scores.append(scores[0]["coherence_score"])

    baseline = {
        "trait_mean": sum(all_trait_scores) / len(all_trait_scores) if all_trait_scores else None,
        "n": len(all_trait_scores),
    }

    if all_coherence_scores:
        baseline["coherence_mean"] = sum(all_coherence_scores) / len(all_coherence_scores)

    print(f"  Baseline: trait={baseline['trait_mean']:.1f}, n={baseline['n']}")
    return baseline


async def run_ensemble_evaluation(
    experiment: str,
    trait: str,
    vector_experiment: str,
    mu: Optional[float],
    sigma: Optional[float],
    coefficients: List[float],
    method: str,
    component: str,
    model_name: str,
    subset: Optional[int],
    compare_baseline: bool,
):
    """
    Main ensemble steering evaluation flow.

    1. Load best (mu, sigma) from classification search if not provided
    2. Load all layer vectors
    3. For each coefficient, evaluate steering
    4. Save results with ensemble config
    """
    # Load prompts
    prompts_data, prompts_file = load_eval_prompts(trait)
    questions = prompts_data["questions"]
    if subset:
        questions = questions[:subset]
    eval_prompt = prompts_data["eval_prompt"]

    # Load model
    model, tokenizer = load_model_and_tokenizer(model_name)
    num_layers = model.config.num_hidden_layers

    # Load experiment config
    config = load_experiment_config(experiment)
    use_chat_template = config.get('use_chat_template')
    if use_chat_template is None:
        use_chat_template = tokenizer.chat_template is not None

    # Get (mu, sigma) from classification search if not provided
    if mu is None or sigma is None:
        best_params = load_best_ensemble_params(vector_experiment, trait)
        if best_params is None:
            raise ValueError(
                f"No ensemble params found for {vector_experiment}/{trait}. "
                f"Either run classification_search.py first or provide --mu and --sigma."
            )
        mu = best_params['mu']
        sigma = best_params['sigma']
        print(f"Using best params from classification search: mu={mu}, sigma={sigma}")
    else:
        print(f"Using manual params: mu={mu}, sigma={sigma}")

    # Load vectors
    print(f"\nLoading vectors from {vector_experiment}/{trait}...")
    vectors = load_vectors_for_trait(vector_experiment, trait, method, component)
    print(f"  Loaded {len(vectors)} layer vectors")

    # Get active layers for this (mu, sigma)
    active_layers = get_active_layers(mu, sigma, num_layers)
    print(f"  Active layers (weight > 0.01): {active_layers}")

    # Filter vectors to available layers
    available = set(vectors.keys()) & set(active_layers)
    if not available:
        raise ValueError(f"No vectors available for active layers {active_layers}")
    print(f"  Available: {sorted(available)}")

    # Load/create results
    results = load_or_create_results(
        experiment, trait, prompts_file, model_name, vector_experiment, "gpt-4.1-nano"
    )
    judge = TraitJudge()

    print(f"\nTrait: {trait}")
    print(f"Model: {model_name}")
    print(f"Coefficients to evaluate: {coefficients}")

    # Compute baseline if needed
    if results.get('baseline') is None:
        baseline = await compute_baseline(
            model, tokenizer, questions, eval_prompt, judge, use_chat_template
        )
        results['baseline'] = baseline
        save_results(experiment, trait, results)

    # Evaluate each coefficient
    for coef in coefficients:
        print(f"\nEvaluating ensemble: mu={mu}, sigma={sigma}, coef={coef}")

        result, responses = await evaluate_ensemble_steering(
            model, tokenizer, vectors,
            mu, sigma, coef,
            questions, eval_prompt, judge,
            use_chat_template, component
        )

        # Build run config
        weights = compute_gaussian_weights(mu, sigma, sorted(vectors.keys()))
        layer_coefs = compute_ensemble_coefficients(mu, sigma, sorted(vectors.keys()), coef)

        run = {
            "config": {
                "ensemble": {
                    "mu": mu,
                    "sigma": sigma,
                    "global_coefficient": coef,
                    "active_layers": sorted(layer_coefs.keys()),
                    "layer_coefficients": {str(k): v for k, v in layer_coefs.items()},
                    "layer_weights": {str(k): v for k, v in weights.items()},
                },
                "method": method,
                "component": component,
            },
            "result": result,
            "timestamp": datetime.now().isoformat(),
        }

        results.setdefault('runs', []).append(run)
        save_results(experiment, trait, results)

        # Save responses
        responses_name = f"ensemble_mu{mu:.0f}_s{sigma:.0f}_c{coef:.0f}_{datetime.now().strftime('%H%M%S')}.json"
        save_responses(experiment, trait, responses, responses_name)

        # Print result
        baseline_trait = results.get('baseline', {}).get('trait_mean', 50)
        delta = result['trait_mean'] - baseline_trait if result['trait_mean'] else 0
        print(f"  Result: trait={result['trait_mean']:.1f} (delta={delta:+.1f}), coherence={result.get('coherence_mean', 0):.1f}")

    # Compare to single-layer baseline if requested
    if compare_baseline:
        print("\n" + "="*60)
        print("COMPARISON TO SINGLE-LAYER BASELINE")
        print("="*60)

        # Find best single-layer from results
        baseline_trait = results.get('baseline', {}).get('trait_mean', 50)
        best_single = None
        best_single_delta = float('-inf')

        for run in results.get('runs', []):
            config = run.get('config', {})
            if 'ensemble' in config:
                continue  # Skip ensemble runs
            if 'layers' in config and len(config['layers']) == 1:
                result_data = run.get('result', {})
                trait_score = result_data.get('trait_mean', 0)
                coherence = result_data.get('coherence_mean', 0)
                if coherence >= 70:
                    delta = trait_score - baseline_trait
                    if delta > best_single_delta:
                        best_single_delta = delta
                        best_single = {
                            'layer': config['layers'][0],
                            'coef': config.get('coefficients', [0])[0],
                            'trait': trait_score,
                            'coherence': coherence,
                            'delta': delta,
                        }

        # Find best ensemble
        best_ensemble = None
        best_ensemble_delta = float('-inf')

        for run in results.get('runs', []):
            config = run.get('config', {})
            if 'ensemble' not in config:
                continue
            result_data = run.get('result', {})
            trait_score = result_data.get('trait_mean', 0)
            coherence = result_data.get('coherence_mean', 0)
            if coherence >= 70:
                delta = trait_score - baseline_trait
                if delta > best_ensemble_delta:
                    best_ensemble_delta = delta
                    best_ensemble = {
                        'mu': config['ensemble']['mu'],
                        'sigma': config['ensemble']['sigma'],
                        'coef': config['ensemble']['global_coefficient'],
                        'trait': trait_score,
                        'coherence': coherence,
                        'delta': delta,
                    }

        print(f"Baseline: {baseline_trait:.1f}")
        if best_single:
            print(f"Best single-layer: L{best_single['layer']} coef={best_single['coef']:.0f} -> {best_single['trait']:.1f} (delta={best_single['delta']:+.1f})")
        if best_ensemble:
            print(f"Best ensemble: mu={best_ensemble['mu']:.1f} sigma={best_ensemble['sigma']:.1f} coef={best_ensemble['coef']:.0f} -> {best_ensemble['trait']:.1f} (delta={best_ensemble['delta']:+.1f})")

        if best_single and best_ensemble:
            improvement = best_ensemble['delta'] - best_single['delta']
            print(f"\nEnsemble improvement over single-layer: {improvement:+.1f} points")

    await judge.close()


def main():
    parser = argparse.ArgumentParser(description="Ensemble steering evaluation")
    parser.add_argument("--experiment", required=True, help="Experiment to save steering results")
    parser.add_argument("--vector-from-trait", required=True,
                       help="Full path to vectors: experiment/category/trait")
    parser.add_argument("--mu", type=float, help="Gaussian center (optional, loads from classification search)")
    parser.add_argument("--sigma", type=float, help="Gaussian spread (optional, loads from classification search)")
    parser.add_argument("--coefficients", type=str, default="50,100,150,200",
                       help="Comma-separated global coefficients to evaluate")
    parser.add_argument("--method", default="probe", help="Extraction method")
    parser.add_argument("--component", default="residual", help="Component type")
    parser.add_argument("--model", help="Model to steer (default: from experiment config)")
    parser.add_argument("--subset", type=int, help="Limit to first N questions")
    parser.add_argument("--compare-baseline", action="store_true",
                       help="Compare ensemble to single-layer baseline")

    args = parser.parse_args()

    # Parse vector path
    parts = args.vector_from_trait.split("/")
    if len(parts) < 2:
        raise ValueError("--vector-from-trait must be experiment/category/trait")
    vector_experiment = parts[0]
    trait = "/".join(parts[1:])

    # Parse coefficients
    coefficients = [float(c) for c in args.coefficients.split(",")]

    # Get model from experiment config if not provided
    if args.model:
        model_name = args.model
    else:
        config = load_experiment_config(args.experiment)
        model_name = config.get('application_model')
        if not model_name:
            raise ValueError("No model specified and no application_model in experiment config")

    # Run evaluation
    asyncio.run(run_ensemble_evaluation(
        experiment=args.experiment,
        trait=trait,
        vector_experiment=vector_experiment,
        mu=args.mu,
        sigma=args.sigma,
        coefficients=coefficients,
        method=args.method,
        component=args.component,
        model_name=model_name,
        subset=args.subset,
        compare_baseline=args.compare_baseline,
    ))


if __name__ == "__main__":
    main()
