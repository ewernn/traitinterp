#!/usr/bin/env python3
"""
Steering evaluation - validate trait vectors via causal intervention.

Input:
    - experiment: Experiment where steering results are saved
    - vector-from-trait: Full path to vectors (experiment/category/trait)

Output:
    - experiments/{experiment}/steering/{trait}/results.json
      Runs-based structure that accumulates across invocations.
    - experiments/{experiment}/steering/{trait}/responses/
      Generated responses for each config.

Usage:
    # Basic usage - adaptive search finds good coefficients, saves all results
    python analysis/steering/evaluate.py \\
        --experiment gemma-2-2b-it \\
        --vector-from-trait gemma-2-2b-base/epistemic/optimism

    # Specific layers only
    python analysis/steering/evaluate.py \\
        --experiment gemma-2-2b-it \\
        --vector-from-trait gemma-2-2b-it/og_10/confidence \\
        --layers 10,12,14

    # Manual coefficients (skip adaptive search)
    python analysis/steering/evaluate.py \\
        --experiment gemma-2-2b-it \\
        --vector-from-trait gemma-2-2b-it/og_10/confidence \\
        --coefficients 50,100,150

    # Multi-layer weighted steering (delta-proportional coefficients)
    python analysis/steering/evaluate.py \\
        --experiment gemma-2-2b-it \\
        --vector-from-trait gemma-2-2b-base/epistemic/optimism \\
        --layers 6-18 \\
        --multi-layer weighted --global-scale 1.5

    # Multi-layer orthogonal steering (orthogonalized vectors)
    python analysis/steering/evaluate.py \\
        --experiment gemma-2-2b-it \\
        --vector-from-trait gemma-2-2b-base/epistemic/optimism \\
        --layers 6-18 \\
        --multi-layer orthogonal --global-scale 1.0
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

from analysis.steering.steer import SteeringHook, MultiLayerSteeringHook, orthogonalize_vectors
from analysis.steering.judge import TraitJudge
from utils.paths import get
from utils.model import format_prompt, load_experiment_config


def load_layer_deltas(experiment: str, trait: str, min_coherence: float = 70) -> Dict[int, Dict]:
    """
    Load single-layer results and return best delta per layer.

    Returns:
        {layer: {'delta': float, 'coef': float, 'coherence': float}}
    """
    results_path = get('steering.results', experiment=experiment, trait=trait)
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


def load_vector(experiment: str, trait: str, layer: int, method: str = "probe", component: str = "residual") -> Optional[torch.Tensor]:
    """Load trait vector from experiment. Returns None if not found."""
    vectors_dir = get('extraction.vectors', experiment=experiment, trait=trait)
    prefix = "" if component == "residual" else f"{component}_"
    vector_file = vectors_dir / f"{prefix}{method}_layer{layer}.pt"

    if not vector_file.exists():
        return None

    return torch.load(vector_file, weights_only=True)


def estimate_activation_norm(
    model,
    tokenizer,
    prompts: List[str],
    layer: int,
    use_chat_template: bool,
) -> float:
    """Estimate activation norm at a layer by running a few prompts."""
    norms = []

    def capture_hook(module, input, output):
        hidden = output[0] if isinstance(output, tuple) else output
        norm = hidden[:, -1, :].float().norm().item()
        norms.append(norm)

    layer_module = model.model.layers[layer]
    handle = layer_module.register_forward_hook(capture_hook)

    try:
        for prompt in prompts[:3]:
            formatted = format_prompt(prompt, tokenizer, use_chat_template=use_chat_template)
            inputs = tokenizer(formatted, return_tensors="pt").to(model.device)
            with torch.no_grad():
                model(**inputs)
    finally:
        handle.remove()

    return sum(norms) / len(norms) if norms else 100.0


def load_eval_prompts(trait: str) -> Tuple[Dict, Path]:
    """Load evaluation prompts for a trait. Returns (data, path)."""
    trait_name = trait.split("/")[-1]
    prompts_file = get('steering.prompt_file', trait=trait_name)

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


def generate_response(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 128,
    temperature: float = 0.0,
) -> str:
    """Generate a response from the model."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature if temperature > 0 else None,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1]:],
        skip_special_tokens=True,
    )
    return response.strip()


# =============================================================================
# Results Management
# =============================================================================

def load_or_create_results(experiment: str, trait: str, prompts_file: Path) -> Dict:
    """Load existing results or create new structure."""
    results_path = get('steering.results', experiment=experiment, trait=trait)
    prompts_file_str = str(prompts_file)

    if results_path.exists():
        with open(results_path) as f:
            results = json.load(f)

        if results.get("prompts_file") != prompts_file_str:
            stored = results.get("prompts_file", "unknown")
            raise ValueError(
                f"Prompts file mismatch!\n"
                f"  Stored: {stored}\n"
                f"  Current: {prompts_file_str}\n"
                f"Delete {results_path} manually to start fresh with new prompts."
            )

        return results

    return {
        "trait": trait,
        "prompts_file": prompts_file_str,
        "baseline": None,
        "runs": []
    }


def find_existing_run_index(results: Dict, config: Dict) -> Optional[int]:
    """Find index of existing run with identical config, or None if not found."""
    for i, run in enumerate(results["runs"]):
        if run["config"] == config:
            return i
    return None


def save_results(results: Dict, experiment: str, trait: str):
    """Save results to experiment directory."""
    results_file = get('steering.results', experiment=experiment, trait=trait)
    results_file.parent.mkdir(parents=True, exist_ok=True)

    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved: {results_file}")


def save_responses(responses: List[Dict], experiment: str, trait: str, config: Dict, timestamp: str):
    """Save generated responses for a config."""
    responses_dir = get('steering.responses', experiment=experiment, trait=trait)
    responses_dir.mkdir(parents=True, exist_ok=True)

    layers_str = "_".join(str(l) for l in config["layers"])
    coefs_str = "_".join(str(c).replace(".", "p") for c in config["coefficients"])
    filename = f"L{layers_str}_c{coefs_str}_{timestamp.replace(':', '-').replace('T', '_')}.json"

    with open(responses_dir / filename, 'w') as f:
        json.dump(responses, f, indent=2)


# =============================================================================
# Core Evaluation
# =============================================================================

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
        scores = await judge.score_batch(eval_prompt, [(question, response)])

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


async def evaluate_single_config(
    model,
    tokenizer,
    vector: torch.Tensor,
    layer: int,
    coef: float,
    questions: List[str],
    eval_prompt: str,
    judge: TraitJudge,
    use_chat_template: bool,
    component: str,
) -> Tuple[Dict, List[Dict]]:
    """Evaluate a single (layer, coefficient) config. Returns (result, responses)."""
    desc = f"L{layer} c{coef:.0f}"
    all_qa_pairs = []

    for question in tqdm(questions, desc=f"{desc} gen", leave=False):
        formatted = format_prompt(question, tokenizer, use_chat_template=use_chat_template)

        with SteeringHook(model, vector, layer, coef, component=component):
            response = generate_response(model, tokenizer, formatted)

        all_qa_pairs.append((question, response))

    # Score
    print(f"  Scoring {len(all_qa_pairs)} responses...")
    all_scores = await judge.score_batch(eval_prompt, all_qa_pairs)

    trait_scores = [s["trait_score"] for s in all_scores if s["trait_score"] is not None]
    coherence_scores = [s["coherence_score"] for s in all_scores if s.get("coherence_score") is not None]

    result = {
        "trait_mean": sum(trait_scores) / len(trait_scores) if trait_scores else None,
        "coherence_mean": sum(coherence_scores) / len(coherence_scores) if coherence_scores else None,
        "n": len(trait_scores),
    }

    responses = [
        {"question": q, "response": r, "trait_score": s["trait_score"], "coherence_score": s.get("coherence_score")}
        for (q, r), s in zip(all_qa_pairs, all_scores)
    ]

    trait_str = f"{result['trait_mean']:.1f}" if result['trait_mean'] else "N/A"
    coh_str = f"{result['coherence_mean']:.1f}" if result['coherence_mean'] else "N/A"
    print(f"  {desc}: trait={trait_str}, coherence={coh_str}, n={result['n']}")

    return result, responses


async def run_evaluation(
    experiment: str,
    trait: str,
    vector_experiment: str,
    layers: List[int],
    coefficients: Optional[List[float]],
    method: str,
    component: str,
    model_name: str,
    judge_provider: str,
    subset: Optional[int],
    n_search_steps: int,
):
    """
    Main evaluation flow.

    If coefficients provided: evaluate those directly.
    Otherwise: run adaptive search to find good coefficients.
    All results are saved.
    """
    # Load prompts
    prompts_data, prompts_file = load_eval_prompts(trait)
    questions = prompts_data["questions"]
    if subset:
        questions = questions[:subset]
    eval_prompt = prompts_data["eval_prompt"]

    # Load model
    model, tokenizer = load_model_and_tokenizer(model_name)
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
    results = load_or_create_results(experiment, trait, prompts_file)
    judge = TraitJudge(provider=judge_provider)

    print(f"\nTrait: {trait}")
    print(f"Model: {model_name} ({num_layers} layers)")
    print(f"Chat template: {use_chat_template}")
    print(f"Vectors from: {vector_experiment}/{trait}")
    print(f"Questions: {len(questions)}")
    print(f"Existing runs: {len(results['runs'])}")

    # Compute baseline if needed
    if results["baseline"] is None:
        results["baseline"] = await compute_baseline(
            model, tokenizer, questions, eval_prompt, judge, use_chat_template
        )
        save_results(results, experiment, trait)
    else:
        print(f"\nUsing existing baseline: trait={results['baseline']['trait_mean']:.1f}")

    # Load vectors and estimate base coefficients
    print(f"\nLoading vectors...")
    layer_data = []
    for layer in layers:
        vector = load_vector(vector_experiment, trait, layer, method, component)
        if vector is None:
            print(f"  L{layer}: Vector not found, skipping")
            continue

        vec_norm = vector.norm().item()
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
                await _evaluate_and_save(
                    model, tokenizer, ld["vector"], ld["layer"], coef,
                    questions, eval_prompt, judge, use_chat_template, component,
                    results, experiment, trait, vector_experiment, method
                )
    else:
        # Adaptive search for each layer
        print(f"\nAdaptive search ({n_search_steps} steps per layer)")
        for ld in layer_data:
            await _adaptive_search_layer(
                model, tokenizer, ld["vector"], ld["layer"], ld["base_coef"],
                questions, eval_prompt, judge, use_chat_template, component,
                results, experiment, trait, vector_experiment, method,
                n_steps=n_search_steps
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


async def _evaluate_and_save(
    model, tokenizer, vector, layer, coef,
    questions, eval_prompt, judge, use_chat_template, component,
    results, experiment, trait, vector_experiment, method
):
    """Evaluate a single config and save to results."""
    config = {
        "layers": [layer],
        "methods": [method],
        "coefficients": [coef],
        "component": component,
    }

    # Skip if already exists
    if find_existing_run_index(results, config) is not None:
        print(f"  L{layer} c{coef:.0f}: Already evaluated, skipping")
        return

    print(f"\n  Evaluating L{layer} c{coef:.0f}...")
    result, responses = await evaluate_single_config(
        model, tokenizer, vector, layer, coef,
        questions, eval_prompt, judge, use_chat_template, component
    )

    timestamp = datetime.now().isoformat()
    save_responses(responses, experiment, trait, config, timestamp)

    run_data = {
        "config": config,
        "result": result,
        "timestamp": timestamp,
    }
    if vector_experiment != experiment:
        run_data["vector_source"] = f"{vector_experiment}/{trait}"

    results["runs"].append(run_data)
    save_results(results, experiment, trait)


async def _adaptive_search_layer(
    model, tokenizer, vector, layer, base_coef,
    questions, eval_prompt, judge, use_chat_template, component,
    results, experiment, trait, vector_experiment, method,
    n_steps: int = 8,
    threshold: float = 70,
    up_mult: float = 1.3,
    down_mult: float = 0.9,
):
    """Run adaptive search for a single layer, saving each result."""
    print(f"\n--- Layer {layer} (base_coef={base_coef:.0f}) ---")
    print(f"Step | Coef  | Trait | Coherence | Action")
    print("-----|-------|-------|-----------|-------")

    coef = base_coef * 0.5  # Start at 0.5x base
    history = []

    for step in range(n_steps):
        # Evaluate
        config = {
            "layers": [layer],
            "methods": [method],
            "coefficients": [coef],
            "component": component,
        }

        existing_idx = find_existing_run_index(results, config)
        if existing_idx is not None:
            # Use existing result
            result = results["runs"][existing_idx]["result"]
            trait_score = result.get("trait_mean") or 0
            coherence = result.get("coherence_mean") or 0
            print(f"  {step+1}  | {coef:>5.0f} | {trait_score:>5.1f} | {coherence:>9.1f} | (cached)")
        else:
            result, responses = await evaluate_single_config(
                model, tokenizer, vector, layer, coef,
                questions, eval_prompt, judge, use_chat_template, component
            )

            trait_score = result.get("trait_mean") or 0
            coherence = result.get("coherence_mean") or 0

            # Save
            timestamp = datetime.now().isoformat()
            save_responses(responses, experiment, trait, config, timestamp)

            run_data = {
                "config": config,
                "result": result,
                "timestamp": timestamp,
            }
            if vector_experiment != experiment:
                run_data["vector_source"] = f"{vector_experiment}/{trait}"

            results["runs"].append(run_data)
            save_results(results, experiment, trait)

            # Print progress
            if coherence < threshold:
                action = f"×{down_mult}"
            else:
                action = f"×{up_mult}"
            if step == n_steps - 1:
                action = "(done)"

            marker = "★" if coherence >= threshold and trait_score > 80 else ""
            print(f"  {step+1}  | {coef:>5.0f} | {trait_score:>5.1f} | {coherence:>9.1f} | {action} {marker}")

        history.append((coef, trait_score, coherence))

        # Decide next coefficient
        if coherence < threshold:
            coef = coef * down_mult
        else:
            coef = coef * up_mult

    # Report best
    valid = [(c, t, coh) for c, t, coh in history if coh >= threshold]
    if valid:
        best_coef, best_trait, best_coh = max(valid, key=lambda x: x[1])
        print(f"✓ Best: coef={best_coef:.0f} (trait={best_trait:.1f}, coherence={best_coh:.1f})")
    else:
        best_coef, best_trait, best_coh = max(history, key=lambda x: x[2])
        print(f"⚠ No coef met threshold. Best coherence: coef={best_coef:.0f}")


async def run_multilayer_evaluation(
    experiment: str,
    trait: str,
    vector_experiment: str,
    layers: List[int],
    mode: str,  # "weighted" or "orthogonal"
    global_scale: float,
    method: str,
    component: str,
    model_name: str,
    judge_provider: str,
    subset: Optional[int],
):
    """
    Run multi-layer steering evaluation.

    Modes:
        - weighted: coef_ℓ = global_scale * best_coef_ℓ * (delta_ℓ / Σ deltas)
        - orthogonal: use orthogonalized vectors with uniform coefficients
    """
    # Load prompts
    prompts_data, prompts_file = load_eval_prompts(trait)
    questions = prompts_data["questions"]
    if subset:
        questions = questions[:subset]
    eval_prompt = prompts_data["eval_prompt"]

    # Load model
    model, tokenizer = load_model_and_tokenizer(model_name)
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
        vector = load_vector(vector_experiment, trait, layer, method, component)
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

    # Initialize judge
    judge = TraitJudge(provider=judge_provider)

    # Generate and score
    all_qa_pairs = []
    for question in tqdm(questions, desc="multilayer gen"):
        formatted = format_prompt(question, tokenizer, use_chat_template=use_chat_template)

        with MultiLayerSteeringHook(model, steering_configs, component=component):
            response = generate_response(model, tokenizer, formatted)

        all_qa_pairs.append((question, response))

    # Score
    print(f"Scoring {len(all_qa_pairs)} responses...")
    all_scores = await judge.score_batch(eval_prompt, all_qa_pairs)

    trait_scores = [s["trait_score"] for s in all_scores if s["trait_score"] is not None]
    coherence_scores = [s["coherence_score"] for s in all_scores if s.get("coherence_score") is not None]

    trait_mean = sum(trait_scores) / len(trait_scores) if trait_scores else 0
    coherence_mean = sum(coherence_scores) / len(coherence_scores) if coherence_scores else 0

    print(f"\nResults:")
    print(f"  Trait: {trait_mean:.1f}")
    print(f"  Coherence: {coherence_mean:.1f}")

    # Load baseline for comparison
    results_path = get('steering.results', experiment=experiment, trait=trait)
    if results_path.exists():
        with open(results_path) as f:
            results = json.load(f)
        baseline = results.get('baseline', {}).get('trait_mean', 0)
        print(f"  Delta from baseline: +{trait_mean - baseline:.1f}")


def main():
    parser = argparse.ArgumentParser(description="Steering evaluation")
    parser.add_argument("--experiment", required=True,
                        help="Experiment where steering results are saved")
    parser.add_argument("--vector-from-trait", required=True,
                        help="Full path to vectors: 'experiment/category/trait'")
    parser.add_argument("--layers", default="all",
                        help="Layers: 'all', single '16', range '5-20', or list '5,10,15'")
    parser.add_argument("--coefficients",
                        help="Manual coefficients (comma-separated). If not provided, uses adaptive search.")
    parser.add_argument("--model", help="Model name/path (default: from experiment config)")
    parser.add_argument("--method", default="probe", help="Vector extraction method")
    parser.add_argument("--component", default="residual", choices=["residual", "attn_out", "mlp_out"])
    parser.add_argument("--judge", default="openai", choices=["openai", "gemini"])
    parser.add_argument("--subset", type=int, help="Use subset of questions (for faster testing)")
    parser.add_argument("--search-steps", type=int, default=8,
                        help="Number of adaptive search steps per layer (default: 8)")
    parser.add_argument("--multi-layer", choices=["weighted", "orthogonal"],
                        help="Multi-layer steering mode: 'weighted' (delta-proportional) or 'orthogonal'")
    parser.add_argument("--global-scale", type=float, default=1.0,
                        help="Global scale for multi-layer coefficients (default: 1.0)")

    args = parser.parse_args()

    # Parse --vector-from-trait
    parts = args.vector_from_trait.split('/', 1)
    if len(parts) != 2:
        parser.error("--vector-from-trait must be 'experiment/category/trait'")
    vector_experiment, trait = parts

    # Get model from experiment config if not specified
    config = load_experiment_config(args.experiment)
    model_name = args.model or config.get('model')
    if not model_name:
        parser.error(f"No model specified. Use --model or add 'model' to experiments/{args.experiment}/config.json")

    # Parse layers (will be validated against actual model later)
    layers = parse_layers(args.layers, num_layers=100)

    if args.multi_layer:
        # Multi-layer mode
        asyncio.run(run_multilayer_evaluation(
            experiment=args.experiment,
            trait=trait,
            vector_experiment=vector_experiment,
            layers=layers,
            mode=args.multi_layer,
            global_scale=args.global_scale,
            method=args.method,
            component=args.component,
            model_name=model_name,
            judge_provider=args.judge,
            subset=args.subset,
        ))
    else:
        # Single-layer mode
        coefficients = parse_coefficients(args.coefficients)
        asyncio.run(run_evaluation(
            experiment=args.experiment,
            trait=trait,
            vector_experiment=vector_experiment,
            layers=layers,
            coefficients=coefficients,
            method=args.method,
            component=args.component,
            model_name=model_name,
            judge_provider=args.judge,
            subset=args.subset,
            n_search_steps=args.search_steps,
        ))


if __name__ == "__main__":
    main()
