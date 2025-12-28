#!/usr/bin/env python3
"""
Steering evaluation - validate trait vectors via causal intervention.

Input:
    - experiment: Experiment where steering results are saved
    - vector-from-trait OR traits: Single or multiple trait specs

Output:
    - experiments/{experiment}/steering/{trait}/results.json
      Runs-based structure that accumulates across invocations.
    - experiments/{experiment}/steering/{trait}/responses/
      Generated responses for each config.

Usage:
    # Basic usage - adaptive search finds good coefficients
    python analysis/steering/evaluate.py \\
        --experiment {experiment} \\
        --vector-from-trait {experiment}/{category}/{trait}

    # Multiple traits (loads model once, evaluates sequentially)
    python analysis/steering/evaluate.py \\
        --experiment {experiment} \\
        --traits "exp/cat/trait1,exp/cat/trait2,exp/cat/trait3" \\
        --load-in-8bit

    # Specific layers only
    python analysis/steering/evaluate.py \\
        --experiment {experiment} \\
        --vector-from-trait {experiment}/{category}/{trait} \\
        --layers 10,12,14

    # 70B+ models with quantization
    python analysis/steering/evaluate.py \\
        --experiment {experiment} \\
        --vector-from-trait {experiment}/{category}/{trait} \\
        --load-in-8bit

    # Sequential mode (one layer at a time, slower but lower memory)
    python analysis/steering/evaluate.py \\
        --experiment {experiment} \\
        --vector-from-trait {experiment}/{category}/{trait} \\
        --no-batch

    # Manual coefficients (skip adaptive search)
    python analysis/steering/evaluate.py \\
        --experiment {experiment} \\
        --vector-from-trait {experiment}/{category}/{trait} \\
        --coefficients 50,100,150

    # Multi-layer weighted steering (delta-proportional coefficients)
    python analysis/steering/evaluate.py \\
        --experiment gemma-2-2b-it \\
        --vector-from-trait gemma-2-2b-base/epistemic/optimism \\
        --layers 6-18 \\
        --multi-layer weighted --global-scale 1.5
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
from analysis.steering.steer import MultiLayerSteeringHook, orthogonalize_vectors
from utils.generation import generate_batch
from analysis.steering.results import load_or_create_results, save_results, save_responses, save_baseline_responses
from analysis.steering.coef_search import (
    evaluate_and_save,
    adaptive_search_layer,
    batched_adaptive_search,
)
from utils.judge import TraitJudge
from utils.paths import get, get_vector_path, get_steering_results_path
from utils.model import format_prompt, tokenize_prompt, load_experiment_config, load_model
from utils.vectors import MIN_COHERENCE
from server.client import get_model_or_client, ModelClient


def load_model_handle(model_name: str, load_in_8bit: bool = False, load_in_4bit: bool = False, no_server: bool = False):
    """Load model locally or get client if server available.

    Returns:
        (model, tokenizer, is_remote) tuple
    """
    if not no_server:
        handle = get_model_or_client(model_name, load_in_8bit=load_in_8bit, load_in_4bit=load_in_4bit)
        if isinstance(handle, ModelClient):
            print(f"Using model server (model: {model_name})")
            return handle, handle, True  # model, tokenizer, is_remote
        model, tokenizer = handle
        return model, tokenizer, False
    else:
        model, tokenizer = load_model(model_name, load_in_8bit=load_in_8bit, load_in_4bit=load_in_4bit)
        return model, tokenizer, False


def load_layer_deltas(experiment: str, trait: str, min_coherence: float = MIN_COHERENCE) -> Dict[int, Dict]:
    """
    Load single-layer results and return best delta per layer.

    Returns:
        {layer: {'delta': float, 'coef': float, 'coherence': float}}
    """
    results_path = get_steering_results_path(experiment, trait)
    if not results_path.exists():
        return {}

    with open(results_path) as f:
        results = json.load(f)

    baseline = results.get('baseline', {}).get('trait_mean', 50)
    best_by_layer = {}

    for run in results.get('runs', []):
        config = run.get('config', {})
        result = run.get('result', {})

        # Only single-layer runs
        if len(config.get('layers', [])) != 1:
            continue

        layer = config['layers'][0]
        trait_score = result.get('trait_mean') or 0
        coherence = result.get('coherence_mean') or 0
        delta = trait_score - baseline
        coef = config.get('coefficients', [0])[0]

        if coherence >= min_coherence:
            if layer not in best_by_layer or delta > best_by_layer[layer]['delta']:
                best_by_layer[layer] = {'delta': delta, 'coef': coef, 'coherence': coherence}

    return best_by_layer


def compute_weighted_coefficients(
    layer_deltas: Dict[int, Dict],
    layers: List[int],
    global_scale: float = 1.0
) -> Dict[int, float]:
    """
    Compute delta-weighted coefficients for multi-layer steering.

    coef_ℓ = global_scale * best_coef_ℓ * (delta_ℓ / Σ deltas)
    """
    # Filter to requested layers with positive delta
    active_layers = {l: d for l, d in layer_deltas.items() if l in layers and d['delta'] > 0}

    if not active_layers:
        return {}

    total_delta = sum(d['delta'] for d in active_layers.values())

    coefficients = {}
    for layer, data in active_layers.items():
        weight = data['delta'] / total_delta
        coefficients[layer] = global_scale * data['coef'] * weight

    return coefficients


def get_num_layers(model) -> int:
    """Get number of layers from model config."""
    return model.config.num_hidden_layers


def parse_layers(layers_arg: str, num_layers: int) -> List[int]:
    """
    Parse layers argument.

    Args:
        layers_arg: "all", single number "16", range "5-20", or list "5,10,15"
        num_layers: Total layers in model

    Returns:
        List of layer indices
    """
    if layers_arg.lower() == "all":
        return list(range(num_layers))
    elif "-" in layers_arg and "," not in layers_arg:
        start, end = layers_arg.split("-")
        return list(range(int(start), int(end) + 1))
    elif "," in layers_arg:
        return [int(x) for x in layers_arg.split(",")]
    else:
        return [int(layers_arg)]


def parse_coefficients(coef_arg: Optional[str]) -> Optional[List[float]]:
    """Parse comma-separated coefficients. Returns None if not provided."""
    if coef_arg is None:
        return None
    return [float(c) for c in coef_arg.split(",")]


def load_vector(experiment: str, trait: str, layer: int, method: str = "probe", component: str = "residual", position: str = "response[:]") -> Optional[torch.Tensor]:
    """Load trait vector from experiment. Returns None if not found."""
    vector_file = get_vector_path(experiment, trait, method, layer, component, position)

    if not vector_file.exists():
        return None

    return torch.load(vector_file, weights_only=True)


def load_cached_activation_norms(experiment: str) -> Dict[int, float]:
    """
    Load cached activation norms from extraction_evaluation.json.

    Returns:
        {layer: norm} or empty dict if not available
    """
    eval_path = get('extraction_eval.evaluation', experiment=experiment)
    if not eval_path.exists():
        return {}

    try:
        with open(eval_path) as f:
            data = json.load(f)
        norms = data.get('activation_norms', {})
        # Convert string keys back to int
        return {int(k): v for k, v in norms.items()}
    except (json.JSONDecodeError, KeyError):
        return {}


def estimate_activation_norm(
    model,
    tokenizer,
    prompts: List[str],
    layer: int,
    use_chat_template: bool,
) -> float:
    """Estimate activation norm at a layer by running a few prompts."""
    norms = []

    def capture_hook(_module, _input, output):
        hidden = output[0] if isinstance(output, tuple) else output
        norm = hidden[:, -1, :].float().norm().item()
        norms.append(norm)

    layer_module = model.model.layers[layer]
    handle = layer_module.register_forward_hook(capture_hook)

    try:
        for prompt in prompts[:3]:
            formatted = format_prompt(prompt, tokenizer, use_chat_template=use_chat_template)
            inputs = tokenize_prompt(formatted, tokenizer, use_chat_template).to(model.device)
            with torch.no_grad():
                model(**inputs)
    finally:
        handle.remove()

    return sum(norms) / len(norms) if norms else 100.0


def load_steering_data(trait: str) -> Tuple[List[str], str, str]:
    """
    Load steering evaluation data for a trait.

    Returns:
        (questions, trait_name, trait_definition)
    """
    # Load questions from steering.json
    prompts_file = get('datasets.trait_steering', trait=trait)
    if not prompts_file.exists():
        raise FileNotFoundError(
            f"Steering prompts not found: {prompts_file}\n"
            f"Create JSON with 'questions' array."
        )

    with open(prompts_file) as f:
        data = json.load(f)

    if "questions" not in data:
        raise ValueError(f"Missing 'questions' in {prompts_file}")

    questions = data["questions"]

    # Load trait definition from definition.txt
    def_file = get('datasets.trait_definition', trait=trait)
    if def_file.exists():
        with open(def_file) as f:
            trait_definition = f.read().strip()
    else:
        trait_name_fallback = trait.split('/')[-1].replace('_', ' ')
        trait_definition = f"The trait '{trait_name_fallback}'"

    # Extract trait name from path
    trait_name = trait.split('/')[-1]  # e.g., 'alignment/deception' -> 'deception'

    return questions, trait_name, trait_definition


# =============================================================================
# Core Evaluation
# =============================================================================

async def compute_baseline(
    model,
    tokenizer,
    questions: List[str],
    trait_name: str,
    trait_definition: str,
    judge: TraitJudge,
    use_chat_template: bool,
    max_new_tokens: int = 256,
) -> tuple[Dict, List[Dict]]:
    """Compute baseline scores (no steering) with batched generation.

    Returns:
        Tuple of (baseline_stats, response_data) where response_data contains
        questions, responses, and scores for saving.
    """
    print("\nComputing baseline (no steering)...")

    # Format all questions
    formatted = [format_prompt(q, tokenizer, use_chat_template=use_chat_template) for q in questions]

    # Generate all responses in batch
    responses = generate_batch(model, tokenizer, formatted, max_new_tokens=max_new_tokens)

    # Score all at once
    all_qa_pairs = list(zip(questions, responses))
    all_scores = await judge.score_steering_batch(all_qa_pairs, trait_name, trait_definition)

    all_trait_scores = [s["trait_score"] for s in all_scores if s["trait_score"] is not None]
    all_coherence_scores = [s["coherence_score"] for s in all_scores if s.get("coherence_score") is not None]

    baseline = {
        "trait_mean": sum(all_trait_scores) / len(all_trait_scores) if all_trait_scores else None,
        "n": len(all_trait_scores),
    }

    if all_coherence_scores:
        baseline["coherence_mean"] = sum(all_coherence_scores) / len(all_coherence_scores)

    # Build response data for saving
    response_data = [
        {
            "question": q,
            "response": r,
            "trait_score": s["trait_score"],
            "coherence_score": s.get("coherence_score"),
        }
        for q, r, s in zip(questions, responses, all_scores)
    ]

    print(f"  Baseline: trait={baseline['trait_mean']:.1f}, n={baseline['n']}")
    return baseline, response_data


async def run_evaluation(
    experiment: str,
    trait: str,
    vector_experiment: str,
    layers: List[int],
    coefficients: Optional[List[float]],
    method: str,
    component: str,
    position: str,
    model_name: str,
    judge_provider: str,
    subset: Optional[int],
    n_search_steps: int,
    up_mult: float,
    down_mult: float,
    momentum: float = 0.0,
    batched: bool = True,
    model=None,
    tokenizer=None,
    judge=None,
    load_in_8bit: bool = False,
    load_in_4bit: bool = False,
    max_new_tokens: int = 256,
):
    """
    Main evaluation flow.

    If coefficients provided: evaluate those directly.
    Otherwise: run adaptive search to find good coefficients.

    Args:
        batched: If True (default), run all layers in parallel batches.
                 If False, run each layer sequentially.
        model: Pre-loaded model (optional, loads if not provided)
        tokenizer: Pre-loaded tokenizer (optional)
        judge: Pre-created TraitJudge (optional, creates if not provided)
        load_in_8bit: Use 8-bit quantization when loading model
        load_in_4bit: Use 4-bit quantization when loading model
    """
    # Load prompts and trait definition
    questions, trait_name, trait_definition = load_steering_data(trait)
    prompts_file = get('datasets.trait_steering', trait=trait)  # For results consistency check
    if subset:
        questions = questions[:subset]

    # Load model if not provided
    # Note: Steering uses hooks, so we force local mode
    should_close_judge = False
    if model is None:
        model, tokenizer, _ = load_model_handle(model_name, load_in_8bit, load_in_4bit, no_server=True)
    num_layers = get_num_layers(model)

    # Load experiment config
    config = load_experiment_config(experiment)
    use_chat_template = config.get('use_chat_template')
    if use_chat_template is None:
        use_chat_template = tokenizer.chat_template is not None

    # Validate layers
    layers = [l for l in layers if 0 <= l < num_layers]
    if not layers:
        raise ValueError(f"No valid layers. Model has {num_layers} layers (0-{num_layers-1})")

    # Load/create results
    results = load_or_create_results(
        experiment, trait, prompts_file, model_name, vector_experiment, judge_provider, position
    )

    # Create judge if not provided
    if judge is None:
        judge = TraitJudge()
        should_close_judge = True

    print(f"\nTrait: {trait}")
    print(f"Model: {model_name} ({num_layers} layers)")
    print(f"Chat template: {use_chat_template}")
    print(f"Vectors from: {vector_experiment}/{trait} @ {position}")
    print(f"Questions: {len(questions)}")
    print(f"Existing runs: {len(results['runs'])}")

    # Compute baseline if needed
    if results["baseline"] is None:
        results["baseline"], baseline_responses = await compute_baseline(
            model, tokenizer, questions, trait_name, trait_definition, judge, use_chat_template
        )
        save_baseline_responses(baseline_responses, experiment, trait, position)
        save_results(results, experiment, trait, position)
    else:
        print(f"\nUsing existing baseline: trait={results['baseline']['trait_mean']:.1f}")

    # Load vectors and compute base coefficients
    # Try cached activation norms first (from extraction_evaluation.json)
    cached_norms = load_cached_activation_norms(vector_experiment)
    if cached_norms:
        print(f"\nUsing cached activation norms from extraction_evaluation.json")
    else:
        print(f"\nNo cached norms, will estimate activation norms...")

    print(f"Loading vectors...")
    layer_data = []
    for layer in layers:
        vector = load_vector(vector_experiment, trait, layer, method, component, position)
        if vector is None:
            print(f"  L{layer}: Vector not found, skipping")
            continue

        vec_norm = vector.norm().item()

        # Use cached norm if available, otherwise estimate
        if layer in cached_norms:
            act_norm = cached_norms[layer]
        else:
            act_norm = estimate_activation_norm(model, tokenizer, questions, layer, use_chat_template)

        base_coef = act_norm / vec_norm

        layer_data.append({
            "layer": layer,
            "vector": vector,
            "base_coef": base_coef,
        })
        print(f"  L{layer}: base_coef={base_coef:.0f}")

    if not layer_data:
        print("No valid layers with vectors found")
        return

    # Determine coefficients to test
    if coefficients is not None:
        # Manual mode: test specified coefficients for each layer
        print(f"\nManual coefficients: {coefficients}")
        for ld in layer_data:
            for coef in coefficients:
                await evaluate_and_save(
                    model, tokenizer, ld["vector"], ld["layer"], coef,
                    questions, trait_name, trait_definition, judge, use_chat_template, component,
                    results, experiment, trait, vector_experiment, method, position
                )
    elif batched and len(layer_data) > 1:
        # Batched adaptive search (default) - all layers in parallel
        await batched_adaptive_search(
            model, tokenizer, layer_data, questions, trait_name, trait_definition, judge,
            use_chat_template, component, results, experiment, trait,
            vector_experiment, method, position=position, n_steps=n_search_steps,
            up_mult=up_mult, down_mult=down_mult, momentum=momentum,
            max_new_tokens=max_new_tokens
        )
    else:
        # Sequential adaptive search for each layer
        print(f"\nSequential adaptive search ({n_search_steps} steps per layer)")
        for ld in layer_data:
            await adaptive_search_layer(
                model, tokenizer, ld["vector"], ld["layer"], ld["base_coef"],
                questions, trait_name, trait_definition, judge, use_chat_template, component,
                results, experiment, trait, vector_experiment, method,
                position=position, n_steps=n_search_steps, up_mult=up_mult, down_mult=down_mult, momentum=momentum,
                max_new_tokens=max_new_tokens
            )

    # Print summary
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"Baseline: {results['baseline']['trait_mean']:.1f}")
    print(f"Total runs: {len(results['runs'])}")

    best_run = max(results['runs'], key=lambda r: r['result'].get('trait_mean') or 0, default=None)
    if best_run:
        score = best_run['result']['trait_mean']
        coh = best_run['result'].get('coherence_mean', 0)
        delta = score - results['baseline']['trait_mean']
        print(f"Best: L{best_run['config']['layers'][0]} c{best_run['config']['coefficients'][0]:.0f}")
        print(f"  trait={score:.1f} (+{delta:.1f}), coherence={coh:.1f}")

    if should_close_judge:
        await judge.close()


async def run_multilayer_evaluation(
    experiment: str,
    trait: str,
    vector_experiment: str,
    layers: List[int],
    mode: str,  # "weighted" or "orthogonal"
    global_scale: float,
    method: str,
    component: str,
    position: str,
    model_name: str,
    subset: Optional[int],
    model=None,
    tokenizer=None,
    judge=None,
    load_in_8bit: bool = False,
    load_in_4bit: bool = False,
    max_new_tokens: int = 256,
):
    """
    Run multi-layer steering evaluation.

    Modes:
        - weighted: coef_ℓ = global_scale * best_coef_ℓ * (delta_ℓ / Σ deltas)
        - orthogonal: use orthogonalized vectors with uniform coefficients
    """
    # Load prompts and trait definition
    questions, trait_name, trait_definition = load_steering_data(trait)
    if subset:
        questions = questions[:subset]

    # Load model if not provided
    # Note: Steering uses hooks, so we force local mode
    should_close_judge = False
    if model is None:
        model, tokenizer, _ = load_model_handle(model_name, load_in_8bit, load_in_4bit, no_server=True)
    num_layers = get_num_layers(model)

    # Load experiment config
    config = load_experiment_config(experiment)
    use_chat_template = config.get('use_chat_template')
    if use_chat_template is None:
        use_chat_template = tokenizer.chat_template is not None

    # Validate layers
    layers = [l for l in layers if 0 <= l < num_layers]
    if not layers:
        raise ValueError(f"No valid layers. Model has {num_layers} layers")

    # Load single-layer deltas for weighted mode
    layer_deltas = load_layer_deltas(experiment, trait)
    if not layer_deltas:
        print(f"Warning: No single-layer results found. Run single-layer evaluation first.")
        return

    # Compute coefficients based on mode
    if mode == "weighted":
        coefficients = compute_weighted_coefficients(layer_deltas, layers, global_scale)
        if not coefficients:
            print("No layers with positive delta in requested range")
            return
        active_layers = sorted(coefficients.keys())
    else:  # orthogonal - use uniform coefficients
        active_layers = [l for l in layers if l in layer_deltas and layer_deltas[l]['delta'] > 0]
        if not active_layers:
            print("No layers with positive delta in requested range")
            return
        # Use average of best coefficients, scaled
        avg_coef = sum(layer_deltas[l]['coef'] for l in active_layers) / len(active_layers)
        coefficients = {l: global_scale * avg_coef / len(active_layers) for l in active_layers}

    # Load vectors
    vectors = {}
    for layer in active_layers:
        vector = load_vector(vector_experiment, trait, layer, method, component, position)
        if vector is None:
            print(f"  L{layer}: Vector not found, skipping")
            continue
        vectors[layer] = vector

    if not vectors:
        print("No vectors found")
        return

    # Orthogonalize if requested
    if mode == "orthogonal":
        print("Orthogonalizing vectors...")
        vectors = orthogonalize_vectors(vectors)
        for l in sorted(vectors.keys()):
            print(f"  L{l}: norm after orthogonalization = {vectors[l].norm().item():.3f}")

    # Build steering configs
    steering_configs = [
        (layer, vectors[layer], coefficients[layer])
        for layer in sorted(vectors.keys())
    ]

    print(f"\nMulti-layer {mode} steering")
    print(f"Layers: {[l for l, _, _ in steering_configs]}")
    print(f"Coefficients: {[f'{c:.1f}' for _, _, c in steering_configs]}")
    print(f"Questions: {len(questions)}")

    # Create judge if not provided
    if judge is None:
        judge = TraitJudge()
        should_close_judge = True

    # Generate all responses in batch with steering
    formatted = [format_prompt(q, tokenizer, use_chat_template=use_chat_template) for q in questions]

    with MultiLayerSteeringHook(model, steering_configs, component=component):
        responses = generate_batch(model, tokenizer, formatted, max_new_tokens=max_new_tokens)

    all_qa_pairs = list(zip(questions, responses))

    # Score
    print(f"Scoring {len(all_qa_pairs)} responses...")
    all_scores = await judge.score_steering_batch(all_qa_pairs, trait_name, trait_definition)

    trait_scores = [s["trait_score"] for s in all_scores if s["trait_score"] is not None]
    coherence_scores = [s["coherence_score"] for s in all_scores if s.get("coherence_score") is not None]

    trait_mean = sum(trait_scores) / len(trait_scores) if trait_scores else 0
    coherence_mean = sum(coherence_scores) / len(coherence_scores) if coherence_scores else 0

    print(f"\nResults:")
    print(f"  Trait: {trait_mean:.1f}")
    print(f"  Coherence: {coherence_mean:.1f}")

    # Load results and save
    results_path = get_steering_results_path(experiment, trait, position)
    if results_path.exists():
        with open(results_path) as f:
            results = json.load(f)
    else:
        results = {"baseline": {}, "runs": []}

    baseline = results.get('baseline', {}).get('trait_mean', 0)
    print(f"  Delta from baseline: +{trait_mean - baseline:.1f}")

    # Build config for multi-layer run
    config = {
        "multi_layer": mode,
        "global_scale": global_scale,
        "layers": list(sorted(vectors.keys())),
        "coefficients": [coefficients[l] for l in sorted(vectors.keys())],
        "method": method,
        "component": component,
    }

    result = {
        "trait_mean": trait_mean,
        "coherence_mean": coherence_mean,
        "n_questions": len(questions),
    }

    run_data = {
        "config": config,
        "result": result,
        "timestamp": datetime.now().isoformat(),
    }

    results["runs"].append(run_data)
    save_results(results, experiment, trait, position)

    # Save individual responses
    responses = [
        {"question": q, "response": r, "trait_score": s["trait_score"], "coherence_score": s.get("coherence_score")}
        for (q, r), s in zip(all_qa_pairs, all_scores)
    ]
    response_config = {
        "layers": list(sorted(vectors.keys())),
        "coefficients": [coefficients[l] for l in sorted(vectors.keys())],
    }
    save_responses(responses, experiment, trait, position, response_config, run_data["timestamp"])
    print(f"  Saved to {results_path}")

    if should_close_judge:
        await judge.close()


def main():
    parser = argparse.ArgumentParser(description="Steering evaluation")
    parser.add_argument("--experiment", required=True,
                        help="Experiment where steering results are saved")

    # Mutually exclusive: single trait or multiple traits
    trait_group = parser.add_mutually_exclusive_group(required=True)
    trait_group.add_argument("--vector-from-trait",
                        help="Single trait: 'experiment/category/trait'")
    trait_group.add_argument("--traits",
                        help="Multiple traits (comma-separated): 'exp/cat/t1,exp/cat/t2'")
    parser.add_argument("--layers", default="all",
                        help="Layers: 'all', single '16', range '5-20', or list '5,10,15'")
    parser.add_argument("--coefficients",
                        help="Manual coefficients (comma-separated). If not provided, uses adaptive search.")
    parser.add_argument("--model", help="Model name/path (default: from experiment config)")
    parser.add_argument("--method", default="probe", help="Vector extraction method")
    parser.add_argument("--component", default="residual", choices=["residual", "attn_out", "mlp_out", "k_cache", "v_cache"])
    parser.add_argument("--position", default="response[:]",
                        help="Token position for vectors (default: response[:])")
    parser.add_argument("--judge", default="openai", choices=["openai", "gemini"])
    parser.add_argument("--subset", type=int, default=5, help="Use subset of questions (default: 5, use --subset 0 for all)")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Max tokens to generate per response (default: 256)")
    parser.add_argument("--search-steps", type=int, default=8,
                        help="Number of adaptive search steps per layer (default: 8)")
    parser.add_argument("--up-mult", type=float, default=1.3,
                        help="Coefficient multiplier when increasing (default: 1.3)")
    parser.add_argument("--down-mult", type=float, default=0.85,
                        help="Coefficient multiplier when decreasing (default: 0.85)")
    parser.add_argument("--momentum", type=float, default=0.7,
                        help="Momentum for coefficient updates (0.0=none, 0.7=typical). Smooths oscillation.")
    parser.add_argument("--no-batch", action="store_true",
                        help="Disable batched layer evaluation (run layers sequentially)")
    parser.add_argument("--multi-layer", choices=["weighted", "orthogonal"],
                        help="Multi-layer steering mode: 'weighted' (delta-proportional) or 'orthogonal'")
    parser.add_argument("--global-scale", type=float, default=1.0,
                        help="Global scale for multi-layer coefficients (default: 1.0)")
    parser.add_argument("--load-in-8bit", action="store_true",
                        help="Load model in 8-bit quantization (for 70B+ models)")
    parser.add_argument("--load-in-4bit", action="store_true",
                        help="Load model in 4-bit quantization")
    parser.add_argument("--no-server", action="store_true",
                        help="Force local model loading (skip model server check)")

    args = parser.parse_args()

    # Parse trait specs (single or multiple)
    if args.traits:
        trait_specs = [t.strip() for t in args.traits.split(',')]
    else:
        trait_specs = [args.vector_from_trait]

    # Validate format and parse
    parsed_traits = []
    for spec in trait_specs:
        parts = spec.split('/', 1)
        if len(parts) != 2:
            parser.error(f"Invalid trait spec '{spec}': must be 'experiment/category/trait'")
        parsed_traits.append((parts[0], parts[1]))  # (vector_experiment, trait)

    # Get model from experiment config if not specified
    config = load_experiment_config(args.experiment)
    model_name = args.model or config.get('application_model') or config.get('model')
    if not model_name:
        parser.error(f"No model specified. Use --model or add 'application_model' to experiments/{args.experiment}/config.json")

    # Parse layers (will be validated against actual model later)
    layers = parse_layers(args.layers, num_layers=100)
    coefficients = parse_coefficients(args.coefficients)

    # Run evaluation(s)
    asyncio.run(_run_main(
        args=args,
        parsed_traits=parsed_traits,
        model_name=model_name,
        layers=layers,
        coefficients=coefficients,
    ))


async def _run_main(args, parsed_traits, model_name, layers, coefficients):
    """Async main to handle model/judge lifecycle."""
    multi_trait = len(parsed_traits) > 1

    # Load model once if multiple traits
    # Note: Steering uses hooks, so we force local mode (no_server=True)
    model, tokenizer, judge = None, None, None
    if multi_trait:
        print(f"\nEvaluating {len(parsed_traits)} traits with shared model")
        model, tokenizer, _ = load_model_handle(
            model_name,
            load_in_8bit=args.load_in_8bit,
            load_in_4bit=args.load_in_4bit,
            no_server=True  # Steering requires local hooks
        )
        judge = TraitJudge()

    try:
        for vector_experiment, trait in parsed_traits:
            if multi_trait:
                print(f"\n{'='*60}")
                print(f"TRAIT: {vector_experiment}/{trait}")
                print(f"{'='*60}")

            if args.multi_layer:
                await run_multilayer_evaluation(
                    experiment=args.experiment,
                    trait=trait,
                    vector_experiment=vector_experiment,
                    layers=layers,
                    mode=args.multi_layer,
                    global_scale=args.global_scale,
                    method=args.method,
                    component=args.component,
                    position=args.position,
                    model_name=model_name,
                    subset=args.subset,
                    model=model,
                    tokenizer=tokenizer,
                    judge=judge,
                    load_in_8bit=args.load_in_8bit,
                    load_in_4bit=args.load_in_4bit,
                    max_new_tokens=args.max_new_tokens,
                )
            else:
                await run_evaluation(
                    experiment=args.experiment,
                    trait=trait,
                    vector_experiment=vector_experiment,
                    layers=layers,
                    coefficients=coefficients,
                    method=args.method,
                    component=args.component,
                    position=args.position,
                    model_name=model_name,
                    judge_provider=args.judge,
                    subset=args.subset,
                    n_search_steps=args.search_steps,
                    up_mult=args.up_mult,
                    down_mult=args.down_mult,
                    momentum=args.momentum,
                    batched=not args.no_batch,
                    model=model,
                    tokenizer=tokenizer,
                    judge=judge,
                    load_in_8bit=args.load_in_8bit,
                    load_in_4bit=args.load_in_4bit,
                    max_new_tokens=args.max_new_tokens,
                )
    finally:
        if judge is not None:
            await judge.close()

    if multi_trait:
        print(f"\n{'='*60}")
        print(f"COMPLETED {len(parsed_traits)} TRAITS")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
