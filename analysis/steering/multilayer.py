"""
Multi-layer steering evaluation.

Input:
    - Single-layer steering results (to compute weighted coefficients)
    - Trait vectors

Output:
    - Multi-layer steering results saved to results.json

Usage:
    # From evaluate.py with --multi-layer flag
    from analysis.steering.multilayer import run_multilayer_evaluation, load_layer_deltas
"""

import json
from datetime import datetime
from typing import Dict, List, Optional

import torch

from analysis.steering.data import load_steering_data
from analysis.steering.steer import MultiLayerSteeringHook, orthogonalize_vectors
from analysis.steering.results import save_results, save_responses
from utils.generation import generate_batch
from utils.judge import TraitJudge
from utils.model import format_prompt, load_experiment_config, get_num_layers
from utils.paths import get_steering_results_path, get_vector_path
from utils.vectors import MIN_COHERENCE


def load_layer_deltas(
    experiment: str,
    trait: str,
    model_variant: str,
    position: str = "response[:]",
    prompt_set: str = "steering",
    component: str = "residual",
    min_coherence: float = MIN_COHERENCE
) -> Dict[int, Dict]:
    """
    Load single-layer results and return best delta per layer.

    Returns:
        {layer: {'delta': float, 'coef': float, 'coherence': float}}
    """
    results_path = get_steering_results_path(experiment, trait, model_variant, position, prompt_set)
    if not results_path.exists():
        return {}

    with open(results_path) as f:
        results = json.load(f)

    baseline = results.get('baseline', {}).get('trait_mean', 50)
    best_by_layer = {}

    for run in results.get('runs', []):
        config = run.get('config', {})
        result = run.get('result', {})

        # VectorSpec format: config.vectors[0]
        vectors = config.get('vectors', [])
        if len(vectors) != 1:
            continue  # Only single-layer runs
        v = vectors[0]
        layer = v.get('layer')
        run_component = v.get('component', 'residual')
        coef = v.get('weight', 0)

        if run_component != component:
            continue

        trait_score = result.get('trait_mean') or 0
        coherence = result.get('coherence_mean') or 0
        delta = trait_score - baseline

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


def load_vector(
    experiment: str,
    trait: str,
    layer: int,
    model_variant: str,
    method: str = "probe",
    component: str = "residual",
    position: str = "response[:]"
) -> Optional[torch.Tensor]:
    """Load trait vector from experiment. Returns None if not found."""
    vector_file = get_vector_path(experiment, trait, method, layer, model_variant, component, position)

    if not vector_file.exists():
        return None

    return torch.load(vector_file, weights_only=True)


async def run_multilayer_evaluation(
    experiment: str,
    trait: str,
    model_variant: str,
    vector_experiment: str,
    layers: List[int],
    mode: str,  # "weighted" or "orthogonal"
    global_scale: float,
    method: str,
    component: str,
    position: str,
    prompt_set: str,
    model_name: str,
    subset: Optional[int],
    model=None,
    tokenizer=None,
    judge=None,
    load_in_8bit: bool = False,
    load_in_4bit: bool = False,
    lora_adapter: str = None,
    max_new_tokens: int = 256,
    eval_prompt: Optional[str] = None,
    use_default_prompt: bool = False,
):
    """
    Run multi-layer steering evaluation.

    Modes:
        - weighted: coef_ℓ = global_scale * best_coef_ℓ * (delta_ℓ / Σ deltas)
        - orthogonal: use orthogonalized vectors with uniform coefficients

    Args:
        eval_prompt: Custom trait scoring prompt (overrides steering.json)
        use_default_prompt: Force V3c default, ignore steering.json eval_prompt
    """
    # Import here to avoid circular dependency
    from analysis.steering.evaluate import load_model_handle

    # Load prompts and trait definition
    steering_data = load_steering_data(trait)
    questions = steering_data.questions
    if subset:
        questions = questions[:subset]

    # Resolve eval_prompt: explicit override > use_default flag > steering.json
    if use_default_prompt:
        effective_eval_prompt = None
    elif eval_prompt is not None:
        effective_eval_prompt = eval_prompt
    else:
        effective_eval_prompt = steering_data.eval_prompt

    # Load model if not provided
    should_close_judge = False
    if model is None:
        model, tokenizer, _ = load_model_handle(model_name, load_in_8bit, load_in_4bit, no_server=True, lora_adapter=lora_adapter)
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
    layer_deltas = load_layer_deltas(experiment, trait, model_variant, position, prompt_set, component)
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
        vector = load_vector(vector_experiment, trait, layer, model_variant, method, component, position)
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

    # Score responses
    print(f"Scoring {len(all_qa_pairs)} responses...")
    all_scores = await judge.score_steering_batch(
        all_qa_pairs,
        steering_data.trait_name,
        steering_data.trait_definition,
        eval_prompt=effective_eval_prompt
    )

    trait_scores = [s["trait_score"] for s in all_scores if s["trait_score"] is not None]
    coherence_scores = [s["coherence_score"] for s in all_scores if s.get("coherence_score") is not None]

    trait_mean = sum(trait_scores) / len(trait_scores) if trait_scores else 0
    coherence_mean = sum(coherence_scores) / len(coherence_scores) if coherence_scores else 0

    print(f"\nResults:")
    print(f"  Trait: {trait_mean:.1f}")
    print(f"  Coherence: {coherence_mean:.1f}")

    # Load results and save
    results_path = get_steering_results_path(experiment, trait, model_variant, position, prompt_set)
    if results_path.exists():
        with open(results_path) as f:
            results = json.load(f)
    else:
        results = {"baseline": {}, "runs": []}

    baseline = results.get('baseline', {}).get('trait_mean', 0)
    print(f"  Delta from baseline: +{trait_mean - baseline:.1f}")

    # Build config for multi-layer run
    run_config = {
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
        "config": run_config,
        "result": result,
        "timestamp": datetime.now().isoformat(),
    }

    results["runs"].append(run_data)
    save_results(results, experiment, trait, model_variant, position, prompt_set)

    # Save individual responses
    responses_data = [
        {"question": q, "response": r, "trait_score": s["trait_score"], "coherence_score": s.get("coherence_score")}
        for (q, r), s in zip(all_qa_pairs, all_scores)
    ]
    response_config = {
        "layers": list(sorted(vectors.keys())),
        "coefficients": [coefficients[l] for l in sorted(vectors.keys())],
    }
    save_responses(responses_data, experiment, trait, model_variant, position, prompt_set, response_config, run_data["timestamp"])
    print(f"  Saved to {results_path}")

    if should_close_judge:
        await judge.close()
