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
    # Basic usage - sweeps all layers, finds good coefficients automatically
    python analysis/steering/evaluate.py \\
        --experiment gemma-2-2b-it \\
        --vector-from-trait gemma-2-2b-it/og_10/confidence

    # Cross-experiment: use vectors from base model, steer IT model
    python analysis/steering/evaluate.py \\
        --experiment gemma-2-2b-it \\
        --vector-from-trait gemma-2-2b-base/og_10/confidence

    # Specific layers only
    python analysis/steering/evaluate.py \\
        --experiment gemma-2-2b-it \\
        --vector-from-trait gemma-2-2b-it/og_10/confidence \\
        --layers 10,12,14,16

    # Fixed coefficients (skip adaptive search)
    python analysis/steering/evaluate.py \\
        --experiment gemma-2-2b-it \\
        --vector-from-trait gemma-2-2b-it/og_10/confidence \\
        --no-find-coef \\
        --coefficients 50,100,150

    # Multi-layer steering (all layers steered simultaneously)
    python analysis/steering/evaluate.py \\
        --experiment gemma-2-2b-it \\
        --vector-from-trait gemma-2-2b-it/og_10/confidence \\
        --layers 12,14,16 \\
        --multi-layer

    # Quick test with subset of questions
    python analysis/steering/evaluate.py \\
        --experiment gemma-2-2b-it \\
        --vector-from-trait gemma-2-2b-it/og_10/confidence \\
        --subset 3
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

from analysis.steering.steer import SteeringHook, MultiLayerSteeringHook
from analysis.steering.judge import TraitJudge
from utils.paths import get
from utils.model import format_prompt, load_experiment_config


DEFAULT_MODEL = "google/gemma-2-2b-it"


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


def parse_coefficients(coef_arg: str) -> List[float]:
    """Parse comma-separated coefficients."""
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
    """
    Load existing results or create new structure.

    Checks prompts_file match - raises error if mismatch.
    """
    results_path = get('steering.results', experiment=experiment, trait=trait)
    prompts_file_str = str(prompts_file)

    if results_path.exists():
        with open(results_path) as f:
            results = json.load(f)

        # Check prompts_file match
        if results.get("prompts_file") != prompts_file_str:
            stored = results.get("prompts_file", "unknown")
            raise ValueError(
                f"Prompts file mismatch!\n"
                f"  Stored: {stored}\n"
                f"  Current: {prompts_file_str}\n"
                f"Delete {results_path} manually to start fresh with new prompts."
            )

        return results

    # Create new structure
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
    print(f"\nResults saved: {results_file}")


def save_responses(responses: List[Dict], experiment: str, trait: str, config: Dict, timestamp: str):
    """Save generated responses for a config."""
    responses_dir = get('steering.responses', experiment=experiment, trait=trait)
    responses_dir.mkdir(parents=True, exist_ok=True)

    # Create filename from config
    layers_str = "_".join(str(l) for l in config["layers"])
    methods_str = "_".join(config["methods"])
    coefs_str = "_".join(str(c).replace(".", "p") for c in config["coefficients"])
    inc_str = "_inc" if config.get("incremental", False) else ""
    filename = f"L{layers_str}_{methods_str}_c{coefs_str}{inc_str}_{timestamp.replace(':', '-').replace('T', '_')}.json"

    with open(responses_dir / filename, 'w') as f:
        json.dump(responses, f, indent=2)
    print(f"  Responses saved: {responses_dir / filename}")


# =============================================================================
# Config Generation
# =============================================================================

def generate_configs(
    layers: List[int],
    method: str,
    coefficients: List[float],
    component: str,
    multi_layer: bool,
    incremental: bool = False,
) -> List[Dict]:
    """
    Generate list of configs to evaluate based on CLI args.

    Without --multi-layer: creates separate runs for each (layer, coefficient) combo
    With --multi-layer: creates single run with all layers steered simultaneously
    """
    configs = []

    if multi_layer:
        # Single run: all layers steered simultaneously
        # Expand method and coefficients to match layers length
        n = len(layers)
        methods = [method] * n

        if len(coefficients) == 1:
            coefficients = coefficients * n
        elif len(coefficients) != n:
            raise ValueError(
                f"With --multi-layer, coefficients must be 1 value or match layers count.\n"
                f"Got {len(layers)} layers but {len(coefficients)} coefficients."
            )

        configs.append({
            "layers": layers,
            "methods": methods,
            "coefficients": coefficients,
            "component": component,
            "incremental": incremental,
        })
    else:
        # Separate runs for each combination
        for layer in layers:
            for coef in coefficients:
                configs.append({
                    "layers": [layer],
                    "methods": [method],
                    "coefficients": [coef],
                    "component": component,
                    "incremental": incremental,
                })

    return configs


# =============================================================================
# Evaluation
# =============================================================================

async def compute_baseline(
    model,
    tokenizer,
    questions: List[str],
    eval_prompt: str,
    judge: TraitJudge,
    use_chat_template: bool,
    temperature: float = 0.0,
) -> Dict:
    """Compute baseline scores (no steering)."""
    print("\nComputing baseline (no steering)...")

    all_trait_scores = []
    all_coherence_scores = []

    for question in tqdm(questions, desc="baseline"):
        formatted = format_prompt(question, tokenizer, use_chat_template=use_chat_template)
        response = generate_response(model, tokenizer, formatted, temperature=temperature)
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


async def evaluate_config(
    model,
    tokenizer,
    experiment: str,
    trait: str,
    config: Dict,
    questions: List[str],
    eval_prompt: str,
    judge: TraitJudge,
    use_chat_template: bool,
    rollouts: int,
    temperature: float,
    incremental: bool = False,
) -> Tuple[Dict, List[Dict]]:
    """
    Evaluate a single config (which may steer one or multiple layers).

    Returns (result dict, responses list).
    """
    layers = config["layers"]
    methods = config["methods"]
    coefficients = config["coefficients"]
    component = config["component"]

    is_multi_layer = len(layers) > 1

    # Load vectors
    raw_vectors = []
    for layer, method in zip(layers, methods):
        vector = load_vector(experiment, trait, layer, method, component)
        if vector is None:
            raise FileNotFoundError(f"Vector not found: layer={layer}, method={method}, component={component}")
        raw_vectors.append(vector)

    # Compute incremental vectors if requested: v[i] - v[i-1] for ALL layers
    if incremental:
        vectors = []
        for i, (layer, method) in enumerate(zip(layers, methods)):
            if layer == 0:
                # Layer 0 has no previous layer, use as-is
                vectors.append(raw_vectors[i])
            else:
                # Load v[layer-1] to compute delta
                v_prev = load_vector(experiment, trait, layer - 1, method, component)
                if v_prev is None:
                    print(f"  Warning: No vector for layer {layer-1}, using full vector for layer {layer}")
                    vectors.append(raw_vectors[i])
                else:
                    v_inc = raw_vectors[i] - v_prev
                    vectors.append(v_inc)
        # Log incremental norms
        print(f"  Incremental vector norms: {[f'{v.norm().item():.2f}' for v in vectors]}")
        print(f"  Original vector norms:    {[f'{v.norm().item():.2f}' for v in raw_vectors]}")
    else:
        vectors = raw_vectors

    # Config description for logging
    if is_multi_layer:
        desc = f"L{layers} c{coefficients}"
    else:
        desc = f"L{layers[0]} c{coefficients[0]}"

    # Phase 1: Generate responses
    all_qa_pairs = []
    print(f"  Generating responses for {desc}...")

    for question in tqdm(questions, desc=f"{desc} gen", leave=False):
        formatted = format_prompt(question, tokenizer, use_chat_template=use_chat_template)

        for rollout in range(rollouts):
            # Check if this is a baseline run (all coefficients are 0)
            is_baseline = all(c == 0 for c in coefficients)

            if is_baseline:
                response = generate_response(model, tokenizer, formatted, temperature=temperature)
            elif is_multi_layer:
                # Multi-layer steering
                steering_configs = list(zip(layers, vectors, coefficients))
                with MultiLayerSteeringHook(model, steering_configs, component=component):
                    response = generate_response(model, tokenizer, formatted, temperature=temperature)
            else:
                # Single-layer steering
                with SteeringHook(model, vectors[0], layers[0], coefficients[0], component=component):
                    response = generate_response(model, tokenizer, formatted, temperature=temperature)

            all_qa_pairs.append((question, response, rollout))

    # Phase 2: Score responses
    print(f"  Scoring {len(all_qa_pairs)} responses...")
    qa_for_scoring = [(q, r) for q, r, _ in all_qa_pairs]
    all_scores = await judge.score_batch(eval_prompt, qa_for_scoring)

    # Phase 3: Collect results and responses
    all_trait_scores = []
    all_coherence_scores = []
    responses_data = []

    for (question, response, rollout), score_data in zip(all_qa_pairs, all_scores):
        if score_data["trait_score"] is not None:
            all_trait_scores.append(score_data["trait_score"])
        if score_data.get("coherence_score") is not None:
            all_coherence_scores.append(score_data["coherence_score"])

        responses_data.append({
            "question": question,
            "response": response,
            "rollout": rollout,
            "trait_score": score_data["trait_score"],
            "coherence_score": score_data.get("coherence_score"),
        })

    result = {
        "trait_mean": sum(all_trait_scores) / len(all_trait_scores) if all_trait_scores else None,
        "n": len(all_trait_scores),
    }

    if len(all_trait_scores) > 1:
        mean = result["trait_mean"]
        result["trait_std"] = (sum((x - mean)**2 for x in all_trait_scores) / len(all_trait_scores))**0.5

    if all_coherence_scores:
        result["coherence_mean"] = sum(all_coherence_scores) / len(all_coherence_scores)

    # Log result
    trait_str = f"{result['trait_mean']:.1f}" if result['trait_mean'] else "N/A"
    coh_str = f"{result.get('coherence_mean', 0):.1f}" if result.get('coherence_mean') else "N/A"
    print(f"  {desc}: trait={trait_str}, coherence={coh_str}, n={result['n']}")

    return result, responses_data


async def adaptive_search(
    evaluate_fn,
    initial: float,
    label: str,
    threshold: float = 70,
    up_mult: float = 1.3,
    down_mult: float = 0.9,
    n_steps: int = 4,
) -> Tuple[float, List[Tuple[float, float, float]]]:
    """
    Generic adaptive search for steering coefficients.

    Args:
        evaluate_fn: async (value) -> (trait_mean, coherence_mean)
        initial: Starting value
        label: "coef" or "scale" for printing
        threshold: Minimum coherence to accept
    """
    value = initial
    history = []

    print(f"\nAdaptive search ({n_steps} steps, threshold={threshold}):")
    print(f"Step | {label:>5} | Trait | Coherence | Action")
    print("-----|-------|-------|-----------|-------")

    for step in range(n_steps):
        trait, coherence = await evaluate_fn(value)
        history.append((value, trait, coherence))

        if coherence < threshold:
            action, next_val = f"×{down_mult}", value * down_mult
        else:
            action, next_val = f"×{up_mult}", value * up_mult

        if step == n_steps - 1:
            action = "(done)"

        marker = "★" if coherence >= threshold and trait > 80 else ""
        print(f"  {step+1}  | {value:>5.0f} | {trait:>5.1f} | {coherence:>9.1f} | {action} {marker}")
        value = next_val

    # Pick best
    valid = [(v, t, c) for v, t, c in history if c >= threshold]
    if valid:
        best_val, best_trait, best_coh = max(valid, key=lambda x: x[1])
        print(f"\n✓ Recommended: {label}={best_val:.0f} (trait={best_trait:.1f}, coherence={best_coh:.1f})")
    else:
        best_val, best_trait, best_coh = max(history, key=lambda x: x[2])
        print(f"\n⚠ No {label} met threshold. Best coherence: {label}={best_val:.0f}")

    return best_val, history


async def steer_and_score(
    model,
    tokenizer,
    vectors: List[torch.Tensor],
    layers: List[int],
    coefs: List[float],
    questions: List[str],
    eval_prompt: str,
    judge: TraitJudge,
    use_chat_template: bool,
    component: str,
    subset: int,
) -> Tuple[float, float]:
    """Steer (single or multi layer), score with judge, return (trait_mean, coherence_mean)."""
    trait_scores, coherence_scores = [], []

    for question in questions[:subset]:
        formatted = format_prompt(question, tokenizer, use_chat_template=use_chat_template)

        if len(layers) == 1:
            with SteeringHook(model, vectors[0], layers[0], coefs[0], component=component):
                response = generate_response(model, tokenizer, formatted)
        else:
            steering_configs = list(zip(layers, vectors, coefs))
            with MultiLayerSteeringHook(model, steering_configs, component=component):
                response = generate_response(model, tokenizer, formatted)

        scores = await judge.score_batch(eval_prompt, [(question, response)])
        if scores[0]["trait_score"] is not None:
            trait_scores.append(scores[0]["trait_score"])
        if scores[0].get("coherence_score") is not None:
            coherence_scores.append(scores[0]["coherence_score"])

    trait_mean = sum(trait_scores) / len(trait_scores) if trait_scores else 0
    coherence_mean = sum(coherence_scores) / len(coherence_scores) if coherence_scores else 100
    return trait_mean, coherence_mean


async def find_coefficient(
    model,
    tokenizer,
    layers: List[int],
    vectors: List[torch.Tensor],
    base_coefs: List[float],
    questions: List[str],
    eval_prompt: str,
    judge: TraitJudge,
    use_chat_template: bool,
    component: str,
    multi_layer: bool,
    subset: int = 3,
) -> Tuple[float, List]:
    """Find good coefficient(s) via adaptive search."""

    if multi_layer:
        n = len(layers)

        async def evaluate(scale):
            coefs = [scale * bc / n for bc in base_coefs]
            return await steer_and_score(
                model, tokenizer, vectors, layers, coefs,
                questions, eval_prompt, judge, use_chat_template, component, subset
            )

        print(f"\nMulti-layer steering ({n} layers)")
        best_scale, history = await adaptive_search(evaluate, 1.0, "scale")
        best_coefs = [best_scale * bc / n for bc in base_coefs]
        print(f"  Coefficients: {[f'L{l}:{c:.0f}' for l, c in zip(layers, best_coefs)]}")
        return best_scale, history

    else:
        # Single layer mode - search each independently
        all_results = []
        for layer, vector, base_coef in zip(layers, vectors, base_coefs):
            print(f"\n--- Layer {layer} (base_coef={base_coef:.0f}) ---")

            async def evaluate(coef, v=vector, l=layer):
                return await steer_and_score(
                    model, tokenizer, [v], [l], [coef],
                    questions, eval_prompt, judge, use_chat_template, component, subset
                )

            best_coef, history = await adaptive_search(evaluate, base_coef, "coef")
            all_results.append((layer, best_coef, history))

        return all_results


async def run_evaluation(
    experiment: str,
    trait: str,
    layers: List[int],
    method: str,
    coefficients: List[float],
    component: str,
    multi_layer: bool,
    model_name: str,
    rollouts: int,
    temperature: float,
    judge_provider: str,
    subset_questions: Optional[int],
    vector_experiment: Optional[str] = None,
    vector_trait: Optional[str] = None,
    incremental: bool = False,
) -> None:
    """
    Run steering evaluation with runs-based results structure.

    Loads/creates results, computes baseline if needed, evaluates configs, saves.
    If a config already exists, overwrites that run instead of appending.
    """
    # Default vector source to experiment/trait
    if vector_experiment is None:
        vector_experiment = experiment
    if vector_trait is None:
        vector_trait = trait

    # Warn about rollouts > 1 with temperature == 0
    if rollouts > 1 and temperature == 0.0:
        print(f"\nWarning: rollouts={rollouts} but temperature=0.0")
        print("  With temp=0, all rollouts will be identical (deterministic).")
        print("  Use --temperature > 0 for variance estimation.\n")

    # Load prompts and check for file match
    prompts_data, prompts_file = load_eval_prompts(trait)
    questions = prompts_data["questions"]
    if subset_questions:
        questions = questions[:subset_questions]
    eval_prompt = prompts_data["eval_prompt"]

    # Load or create results
    results = load_or_create_results(experiment, trait, prompts_file)

    # Load model
    model, tokenizer = load_model_and_tokenizer(model_name)
    num_layers = get_num_layers(model)

    # Load experiment config for chat template setting
    config = load_experiment_config(experiment)
    use_chat_template = config.get('use_chat_template')
    if use_chat_template is None:
        use_chat_template = tokenizer.chat_template is not None

    # Validate layers
    layers = [l for l in layers if 0 <= l < num_layers]
    if not layers:
        raise ValueError(f"No valid layers. Model has {num_layers} layers (0-{num_layers-1})")

    print(f"\nTrait: {trait}")
    print(f"Model: {model_name} ({num_layers} layers)")
    print(f"Chat template: {use_chat_template}")
    print(f"Vectors from: {vector_experiment}/{vector_trait}")
    print(f"Method: {method}")
    print(f"Component: {component}")
    print(f"Questions: {len(questions)}")
    print(f"Rollouts: {rollouts}")
    print(f"Temperature: {temperature}")
    print(f"Multi-layer: {multi_layer}")
    print(f"Incremental: {incremental}")
    print(f"Existing runs: {len(results['runs'])}")

    # Initialize judge
    judge = TraitJudge(provider=judge_provider)

    # Compute baseline if not present
    if results["baseline"] is None:
        results["baseline"] = await compute_baseline(
            model, tokenizer, questions, eval_prompt, judge,
            use_chat_template=use_chat_template, temperature=temperature
        )
        save_results(results, experiment, trait)  # Save immediately after baseline
    else:
        print(f"\nUsing existing baseline: trait={results['baseline']['trait_mean']:.1f}")

    # Generate configs
    configs = generate_configs(layers, method, coefficients, component, multi_layer, incremental)
    print(f"\nConfigs to evaluate: {len(configs)}")

    # Evaluate each config
    for i, config in enumerate(configs, 1):
        print(f"\n[{i}/{len(configs)}] Evaluating config: {config}")

        timestamp = datetime.now().isoformat()
        result, responses = await evaluate_config(
            model, tokenizer, vector_experiment, vector_trait, config,
            questions, eval_prompt, judge,
            use_chat_template=use_chat_template,
            rollouts=rollouts, temperature=temperature,
            incremental=incremental
        )

        # Save responses
        save_responses(responses, experiment, trait, config, timestamp)

        # Check if this config already exists
        existing_idx = find_existing_run_index(results, config)
        run_data = {
            "config": config,
            "result": result,
            "timestamp": timestamp,
        }

        # Record vector source if different from experiment/trait
        if vector_experiment != experiment or vector_trait != trait:
            run_data["vector_source"] = f"{vector_experiment}/{vector_trait}"

        if existing_idx is not None:
            print(f"  Overwriting existing run at index {existing_idx}")
            results["runs"][existing_idx] = run_data
        else:
            results["runs"].append(run_data)

        # Save after each run (so progress isn't lost)
        save_results(results, experiment, trait)

    # Print summary
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"Baseline: {results['baseline']['trait_mean']:.1f}")
    print(f"Total runs: {len(results['runs'])}")

    # Find best run
    best_run = None
    best_score = float('-inf')
    for run in results['runs']:
        score = run['result'].get('trait_mean')
        if score is not None and score > best_score:
            best_score = score
            best_run = run

    if best_run:
        print(f"Best run: {best_run['config']}")
        print(f"  trait_mean={best_score:.1f} (delta={best_score - results['baseline']['trait_mean']:.1f})")


def main():
    parser = argparse.ArgumentParser(description="Steering evaluation")
    parser.add_argument("--experiment", required=True,
                        help="Experiment where steering results are saved")
    parser.add_argument("--vector-from-trait", required=True,
                        help="Full path to vectors: 'experiment/category/trait' (e.g., gemma-2-2b-it/og_10/confidence)")
    parser.add_argument(
        "--layers",
        default="all",
        help="Layers: single '16', range '5-20', list '5,10,15', or 'all' (default: all)"
    )
    parser.add_argument(
        "--coefficients",
        default="2.0",
        help="Comma-separated coefficients (only used with --no-find-coef)"
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model name/path")
    parser.add_argument("--method", default="probe", help="Vector extraction method")
    parser.add_argument("--component", default="residual", choices=["residual", "attn_out", "mlp_out"],
                        help="Which component to steer")
    parser.add_argument("--rollouts", type=int, default=1,
                        help="Rollouts per question (>1 only useful with --temperature > 0)")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature (0 = deterministic)")
    parser.add_argument("--judge", default="openai", choices=["openai", "gemini"])
    parser.add_argument("--subset", type=int, help="Use subset of questions (for faster testing)")
    parser.add_argument("--multi-layer", action="store_true",
                        help="Steer all layers simultaneously (instead of separate runs)")
    parser.add_argument("--incremental", action="store_true",
                        help="Use incremental vectors (v[i] - v[i-1]) to avoid double-counting shared components")
    parser.add_argument("--no-find-coef", action="store_true",
                        help="Skip adaptive coefficient search, use --coefficients directly")

    args = parser.parse_args()

    # Parse --vector-from-trait into (vector_experiment, trait)
    parts = args.vector_from_trait.split('/', 1)
    if len(parts) != 2:
        parser.error("--vector-from-trait must be 'experiment/category/trait' (e.g., gemma-2-2b-it/og_10/confidence)")
    vector_experiment, trait = parts

    # Parse layers
    layers = parse_layers(args.layers, num_layers=100)  # Temporary max, validated later

    # Default mode: find-coef (adaptive search)
    if not args.no_find_coef:
        async def run_find_coef_mode():
            model, tokenizer = load_model_and_tokenizer(args.model)

            config = load_experiment_config(args.experiment)
            use_chat_template = config.get('use_chat_template')
            if use_chat_template is None:
                use_chat_template = tokenizer.chat_template is not None

            prompts_data, _ = load_eval_prompts(trait)
            questions, eval_prompt = prompts_data["questions"], prompts_data["eval_prompt"]
            judge = TraitJudge(provider=args.judge)

            # Load all vectors and compute base_coefs
            vectors, base_coefs, valid_layers = [], [], []
            print(f"Loading vectors and estimating activation norms...")
            for layer in layers:
                vector = load_vector(vector_experiment, trait, layer, args.method, args.component)
                if vector is None:
                    print(f"  L{layer}: Vector not found, skipping")
                    continue

                vec_norm = vector.norm().item()
                act_norm = estimate_activation_norm(model, tokenizer, questions, layer, use_chat_template)
                base_coef = act_norm / vec_norm

                vectors.append(vector)
                base_coefs.append(base_coef)
                valid_layers.append(layer)
                print(f"  L{layer}: vec_norm={vec_norm:.3f}, act_norm={act_norm:.1f}, base_coef={base_coef:.0f}")

            if not valid_layers:
                print("No valid layers found")
                return

            await find_coefficient(
                model, tokenizer, valid_layers, vectors, base_coefs,
                questions, eval_prompt, judge, use_chat_template,
                args.component, args.multi_layer, args.subset or 3
            )

        asyncio.run(run_find_coef_mode())
        return

    # Manual coefficient mode (--no-find-coef)
    coefficients = parse_coefficients(args.coefficients)

    asyncio.run(run_evaluation(
        experiment=args.experiment,
        trait=trait,
        layers=layers,
        method=args.method,
        coefficients=coefficients,
        component=args.component,
        multi_layer=args.multi_layer,
        model_name=args.model,
        rollouts=args.rollouts,
        temperature=args.temperature,
        judge_provider=args.judge,
        subset_questions=args.subset,
        vector_experiment=vector_experiment,
        vector_trait=trait,
        incremental=args.incremental,
    ))


if __name__ == "__main__":
    main()
