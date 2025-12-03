#!/usr/bin/env python3
"""
Steering evaluation - validate trait vectors via causal intervention.

Input:
    - experiment: Experiment name
    - trait: Trait path (e.g., cognitive_state/confidence)
    - layers: Layer(s) to steer (default: all)

Output:
    - experiments/{experiment}/steering/{trait}/results.json (single layer)
    - experiments/{experiment}/steering/{trait}/layer_sweep.json (multiple layers)

Usage:
    # Evaluate single layer (full coefficients, more rollouts)
    python analysis/steering/evaluate.py \\
        --experiment my_exp \\
        --trait cognitive_state/confidence \\
        --layers 16

    # Layer sweep (all layers, fixed coefficient, fewer rollouts)
    python analysis/steering/evaluate.py \\
        --experiment my_exp \\
        --trait cognitive_state/confidence \\
        --layers all

    # Custom layer range
    python analysis/steering/evaluate.py \\
        --experiment my_exp \\
        --trait cognitive_state/confidence \\
        --layers 5-20
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import argparse
import asyncio
import json
from typing import List, Dict, Optional, Union
from datetime import datetime
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from analysis.steering.steer import SteeringHook
from analysis.steering.judge import TraitJudge
from utils.paths import get


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
        layers_arg: "all", single number "16", or range "5-20"
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


def load_vector(experiment: str, trait: str, layer: int, method: str = "probe", component: str = "residual") -> Optional[torch.Tensor]:
    """Load trait vector from experiment. Returns None if not found."""
    vectors_dir = get('extraction.vectors', experiment=experiment, trait=trait)
    prefix = "" if component == "residual" else f"{component}_"
    vector_file = vectors_dir / f"{prefix}{method}_layer{layer}.pt"

    if not vector_file.exists():
        return None

    return torch.load(vector_file, weights_only=True)


def load_eval_prompts(trait: str) -> Dict:
    """Load evaluation prompts for a trait."""
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

    return data


def format_prompt(question: str, tokenizer) -> str:
    """Format question for chat template."""
    messages = [{"role": "user", "content": question}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def generate_response(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
) -> str:
    """Generate a response from the model."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1]:],
        skip_special_tokens=True,
    )
    return response.strip()


async def evaluate_single_config(
    model,
    tokenizer,
    vector: torch.Tensor,
    layer: int,
    coefficient: float,
    questions: List[str],
    eval_prompt: str,
    judge: TraitJudge,
    rollouts: int = 10,
    save_responses: bool = False,
    component: str = "residual",
) -> Dict:
    """Evaluate a single (layer, coefficient) configuration."""
    # Phase 1: Generate all responses (GPU-bound)
    all_qa_pairs = []
    print(f"    Generating responses...")
    for question in tqdm(questions, desc=f"L{layer}/c{coefficient} gen", leave=False):
        formatted = format_prompt(question, tokenizer)

        for rollout in range(rollouts):
            if coefficient == 0:
                response = generate_response(model, tokenizer, formatted)
            else:
                with SteeringHook(model, vector, layer, coefficient, component=component):
                    response = generate_response(model, tokenizer, formatted)
            all_qa_pairs.append((question, response, rollout))

    # Phase 2: Score all responses in parallel (API-bound)
    print(f"    Scoring {len(all_qa_pairs)} responses in parallel...")
    qa_for_scoring = [(q, r) for q, r, _ in all_qa_pairs]
    all_scores = await judge.score_batch(eval_prompt, qa_for_scoring)

    # Phase 3: Collect results
    all_trait_scores = []
    all_coherence_scores = []
    responses_data = []

    for (question, response, rollout), score_data in zip(all_qa_pairs, all_scores):
        if score_data["trait_score"] is not None:
            all_trait_scores.append(score_data["trait_score"])
        if score_data.get("coherence_score") is not None:
            all_coherence_scores.append(score_data["coherence_score"])

        if save_responses:
            responses_data.append({
                "question": question,
                "response": response,
                "rollout": rollout,
                "trait_score": score_data["trait_score"],
                "coherence_score": score_data.get("coherence_score"),
            })

    result = {
        "layer": layer,
        "coefficient": coefficient,
        "n": len(all_trait_scores),
        "trait_mean": sum(all_trait_scores) / len(all_trait_scores) if all_trait_scores else None,
        "trait_std": (
            (sum((x - sum(all_trait_scores)/len(all_trait_scores))**2 for x in all_trait_scores) / len(all_trait_scores))**0.5
            if len(all_trait_scores) > 1 else 0
        ),
    }

    if all_coherence_scores:
        result["coherence_mean"] = sum(all_coherence_scores) / len(all_coherence_scores)

    if save_responses:
        result["responses"] = responses_data

    return result


async def run_single_layer_evaluation(
    model,
    tokenizer,
    experiment: str,
    trait: str,
    layer: int,
    coefficients: List[float],
    eval_prompt: str,
    questions: List[str],
    judge: TraitJudge,
    method: str,
    rollouts: int,
    save_responses: bool,
    component: str = "residual",
) -> Dict:
    """Full evaluation of a single layer across multiple coefficients."""
    vector = load_vector(experiment, trait, layer, method, component)
    if vector is None:
        raise FileNotFoundError(f"Vector not found for layer {layer} component {component}")

    print(f"\nEvaluating layer {layer}")
    print(f"  Coefficients: {coefficients}, Rollouts: {rollouts}")

    results_by_coef = {}
    for coef in coefficients:
        result = await evaluate_single_config(
            model, tokenizer, vector, layer, coef,
            questions, eval_prompt, judge, rollouts, save_responses, component,
        )
        results_by_coef[str(coef)] = result
        mean = result['trait_mean']
        print(f"  coef={coef}: mean={f'{mean:.1f}' if mean else 'N/A'}, n={result['n']}")

    # Summary metrics
    baseline = results_by_coef.get("0.0", results_by_coef.get("0", {}))
    baseline_mean = baseline.get("trait_mean", 0) or 0
    max_mean = max((r.get("trait_mean") or 0) for r in results_by_coef.values())

    # Controllability
    coefs, means = [], []
    for coef_str, result in results_by_coef.items():
        if result.get("trait_mean") is not None:
            coefs.append(float(coef_str))
            means.append(result["trait_mean"])

    controllability = None
    if len(coefs) >= 2:
        mean_c = sum(coefs) / len(coefs)
        mean_m = sum(means) / len(means)
        num = sum((c - mean_c) * (m - mean_m) for c, m in zip(coefs, means))
        den_c = sum((c - mean_c)**2 for c in coefs)**0.5
        den_m = sum((m - mean_m)**2 for m in means)**0.5
        controllability = num / (den_c * den_m) if den_c * den_m > 0 else 0

    return {
        "trait": trait,
        "layer": layer,
        "method": method,
        "component": component,
        "n_questions": len(questions),
        "rollouts": rollouts,
        "timestamp": datetime.now().isoformat(),
        "coefficients": results_by_coef,
        "baseline": baseline_mean,
        "max_delta": max_mean - baseline_mean,
        "controllability": controllability,
    }


async def run_layer_sweep(
    model,
    tokenizer,
    experiment: str,
    trait: str,
    layers: List[int],
    coefficient: float,
    eval_prompt: str,
    questions: List[str],
    judge: TraitJudge,
    method: str,
    rollouts: int,
    component: str = "residual",
) -> Dict:
    """Sweep across multiple layers with fixed coefficient."""
    print(f"\nLayer sweep: {len(layers)} layers")
    print(f"  Coefficient: {coefficient}, Rollouts: {rollouts}")
    print(f"  Questions: {len(questions)}")

    # Baseline (no steering)
    print("\nBaseline (no steering)...")
    baseline_scores = []
    for question in tqdm(questions, desc="baseline"):
        formatted = format_prompt(question, tokenizer)
        response = generate_response(model, tokenizer, formatted)
        scores = await judge.score_batch(eval_prompt, [(question, response)])
        if scores[0]["trait_score"] is not None:
            baseline_scores.append(scores[0]["trait_score"])
    baseline_mean = sum(baseline_scores) / len(baseline_scores) if baseline_scores else 0

    # Evaluate each layer
    results_by_layer = {}
    for layer in tqdm(layers, desc="layers"):
        vector = load_vector(experiment, trait, layer, method, component)
        if vector is None:
            print(f"  Skipping layer {layer}: no vector")
            continue

        result = await evaluate_single_config(
            model, tokenizer, vector, layer, coefficient,
            questions, eval_prompt, judge, rollouts, False, component,
        )
        results_by_layer[layer] = result
        mean = result['trait_mean']
        coh = result.get('coherence_mean')
        print(f"  Layer {layer}: mean={mean:.1f if mean else 'N/A'}, coherence={coh:.1f if coh else 'N/A'}")

    # Find best layer
    best_layer = None
    best_score = float('-inf')
    for layer, result in results_by_layer.items():
        if result["trait_mean"] is not None and result["trait_mean"] > best_score:
            best_score = result["trait_mean"]
            best_layer = layer

    # Coherence warnings
    coherence_warnings = [
        layer for layer, result in results_by_layer.items()
        if result.get("coherence_mean") is not None and result["coherence_mean"] < 50
    ]

    return {
        "trait": trait,
        "method": method,
        "coefficient": coefficient,
        "n_questions": len(questions),
        "rollouts": rollouts,
        "timestamp": datetime.now().isoformat(),
        "baseline_mean": baseline_mean,
        "layers": {str(k): v for k, v in results_by_layer.items()},
        "best_layer": best_layer,
        "best_score": best_score,
        "delta_from_baseline": best_score - baseline_mean if best_layer else None,
        "coherence_warnings": coherence_warnings,
    }


async def run_evaluation(
    experiment: str,
    trait: str,
    layers: List[int],
    coefficients: List[float],
    model_name: str = DEFAULT_MODEL,
    method: str = "probe",
    rollouts: int = 10,
    judge_provider: str = "openai",
    save_responses: bool = False,
    sweep_coefficient: float = 1.5,
    subset_questions: Optional[int] = None,
    component: str = "residual",
) -> Dict:
    """
    Run steering evaluation.

    If single layer: full evaluation with multiple coefficients
    If multiple layers: layer sweep with fixed coefficient
    """
    model, tokenizer = load_model_and_tokenizer(model_name)
    num_layers = get_num_layers(model)

    # Validate layers
    layers = [l for l in layers if 0 <= l < num_layers]
    if not layers:
        raise ValueError(f"No valid layers. Model has {num_layers} layers (0-{num_layers-1})")

    prompts_data = load_eval_prompts(trait)
    judge = TraitJudge(provider=judge_provider)

    questions = prompts_data["questions"]
    if subset_questions:
        questions = questions[:subset_questions]
    eval_prompt = prompts_data["eval_prompt"]

    print(f"\nTrait: {trait}")
    print(f"Model: {model_name} ({num_layers} layers)")
    print(f"Method: {method}")
    print(f"Component: {component}")
    print(f"Judge: {judge_provider}")

    is_sweep = len(layers) > 1

    if is_sweep:
        return await run_layer_sweep(
            model, tokenizer, experiment, trait, layers, sweep_coefficient,
            eval_prompt, questions, judge, method, rollouts, component,
        )
    else:
        return await run_single_layer_evaluation(
            model, tokenizer, experiment, trait, layers[0], coefficients,
            eval_prompt, questions, judge, method, rollouts, save_responses, component,
        )


def save_results(results: Dict, experiment: str, trait: str, is_sweep: bool, save_responses: bool):
    """Save results to experiment directory."""
    if is_sweep:
        results_file = get('steering.layer_sweep', experiment=experiment, trait=trait)
    else:
        results_file = get('steering.results', experiment=experiment, trait=trait)

    results_file.parent.mkdir(parents=True, exist_ok=True)

    # Extract responses before saving
    results_clean = results.copy()
    responses = {}
    if "coefficients" in results_clean:
        for coef, data in results_clean["coefficients"].items():
            if isinstance(data, dict) and "responses" in data:
                responses[coef] = data.pop("responses")

    with open(results_file, 'w') as f:
        json.dump(results_clean, f, indent=2)
    print(f"\nResults saved: {results_file}")

    if save_responses and responses:
        responses_dir = get('steering.responses', experiment=experiment, trait=trait)
        responses_dir.mkdir(parents=True, exist_ok=True)
        for coef, resp_list in responses.items():
            resp_file = responses_dir / f"coef_{coef.replace('.', '_')}.json"
            with open(resp_file, 'w') as f:
                json.dump(resp_list, f, indent=2)
        print(f"Responses saved: {responses_dir}")

    # Print summary
    if is_sweep:
        print(f"\n=== Summary ===")
        print(f"Baseline: {results['baseline_mean']:.1f}")
        print(f"Best layer: {results['best_layer']} (score={results['best_score']:.1f})")
        print(f"Delta: +{results['delta_from_baseline']:.1f}")
        if results['coherence_warnings']:
            print(f"Coherence warnings: layers {results['coherence_warnings']}")


def main():
    parser = argparse.ArgumentParser(description="Steering evaluation")
    parser.add_argument("--experiment", required=True, help="Experiment name")
    parser.add_argument("--trait", required=True, help="Trait path (e.g., cognitive_state/confidence)")
    parser.add_argument(
        "--layers",
        default="all",
        help="Layers to evaluate: 'all', single '16', range '5-20', or list '5,10,15'"
    )
    parser.add_argument(
        "--coefficients",
        default="0,0.5,1.0,1.5,2.0,2.5",
        help="Comma-separated coefficients (for single-layer mode)"
    )
    parser.add_argument(
        "--sweep-coefficient",
        type=float,
        default=1.5,
        help="Fixed coefficient for layer sweep mode"
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model name/path")
    parser.add_argument("--method", default="probe", help="Vector extraction method")
    parser.add_argument("--rollouts", type=int, default=10, help="Rollouts per question")
    parser.add_argument("--judge", default="openai", choices=["openai", "gemini"])
    parser.add_argument("--save-responses", action="store_true", help="Save generated responses")
    parser.add_argument("--subset", type=int, help="Use subset of questions (for faster testing)")
    parser.add_argument("--component", default="residual", choices=["residual", "attn_out", "mlp_out"],
                        help="Which component to steer (residual, attn_out, mlp_out)")

    args = parser.parse_args()

    # Load model to get layer count for parsing
    print(f"Loading model to determine layer count...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    num_layers = get_num_layers(model)
    print(f"Model has {num_layers} layers")

    # Parse layers
    layers = parse_layers(args.layers, num_layers)
    is_sweep = len(layers) > 1

    # Adjust rollouts for sweep mode
    rollouts = args.rollouts
    if is_sweep and args.rollouts == 10:
        rollouts = 3  # Default to fewer rollouts for sweep
        print(f"Layer sweep mode: using {rollouts} rollouts (override with --rollouts)")

    coefficients = [float(c) for c in args.coefficients.split(",")]

    # Run evaluation (model already loaded, but run_evaluation will reload - could optimize)
    del model, tokenizer  # Free memory before reloading

    results = asyncio.run(run_evaluation(
        experiment=args.experiment,
        trait=args.trait,
        layers=layers,
        coefficients=coefficients,
        model_name=args.model,
        method=args.method,
        rollouts=rollouts,
        judge_provider=args.judge,
        save_responses=args.save_responses,
        sweep_coefficient=args.sweep_coefficient,
        subset_questions=args.subset,
        component=args.component,
    ))

    save_results(results, args.experiment, args.trait, is_sweep, args.save_responses)


if __name__ == "__main__":
    main()
