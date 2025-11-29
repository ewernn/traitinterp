#!/usr/bin/env python3
"""
Steering evaluation - validate trait vectors via causal intervention.

Input:
    - experiment: Experiment name
    - trait: Trait path (e.g., cognitive_state/confidence)
    - layer: Layer to steer (0-25)
    - coefficients: List of steering strengths to test

Output:
    - experiments/{experiment}/steering/{trait}/results.json
    - Optional: experiments/{experiment}/steering/{trait}/responses/

Usage:
    python analysis/steering/evaluate.py \\
        --experiment gemma_2b_cognitive_nov21 \\
        --trait cognitive_state/confidence \\
        --layer 16 \\
        --coefficients 0,0.5,1.0,1.5,2.0,2.5 \\
        --rollouts 10
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import argparse
import asyncio
import json
from typing import List, Dict, Optional
from datetime import datetime
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from analysis.steering.steer import SteeringHook
from analysis.steering.judge import TraitJudge
from utils.paths import get


MODEL_NAME = "google/gemma-2-2b-it"


def load_model_and_tokenizer():
    """Load Gemma 2B model and tokenizer."""
    print(f"Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    return model, tokenizer


def load_vector(experiment: str, trait: str, layer: int, method: str = "probe") -> torch.Tensor:
    """Load trait vector from experiment."""
    vectors_dir = get('extraction.vectors', experiment=experiment, trait=trait)
    vector_file = vectors_dir / f"{method}_layer{layer}.pt"

    if not vector_file.exists():
        raise FileNotFoundError(f"Vector not found: {vector_file}")

    return torch.load(vector_file, weights_only=True)


def load_eval_prompts(trait: str) -> Dict:
    """
    Load evaluation prompts for a trait.

    Looks in analysis/steering/prompts/{trait_name}.json
    Expected format: {"questions": [...], "eval_prompt": "...{question}...{answer}..."}
    """
    # Convert trait path to filename (e.g., cognitive_state/confidence -> confidence)
    trait_name = trait.split("/")[-1]
    prompts_file = get('steering.prompt_file', trait=trait_name)

    if not prompts_file.exists():
        raise FileNotFoundError(
            f"Eval prompts not found: {prompts_file}\n"
            f"Create JSON with 'questions' (20 items) and 'eval_prompt' (with {{question}} and {{answer}} placeholders)."
        )

    with open(prompts_file) as f:
        data = json.load(f)

    if "eval_prompt" not in data:
        raise ValueError(f"Missing 'eval_prompt' in {prompts_file}")
    if "questions" not in data:
        raise ValueError(f"Missing 'questions' in {prompts_file}")

    return data


def format_prompt(question: str, tokenizer) -> str:
    """Format question for Gemma chat template."""
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

    # Decode only the new tokens
    response = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1]:],
        skip_special_tokens=True,
    )
    return response.strip()


async def evaluate_coefficient(
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
) -> Dict:
    """
    Evaluate a single coefficient across all questions and rollouts.

    Returns:
        Dict with mean/std trait scores, coherence, and optional responses
    """
    all_trait_scores = []
    all_coherence_scores = []
    responses_data = []

    for question in tqdm(questions, desc=f"coef={coefficient}", leave=False):
        formatted = format_prompt(question, tokenizer)

        for rollout in range(rollouts):
            # Generate with steering
            if coefficient == 0:
                # No steering for baseline
                response = generate_response(model, tokenizer, formatted)
            else:
                with SteeringHook(model, vector, layer, coefficient):
                    response = generate_response(model, tokenizer, formatted)

            # Score with judge
            scores = await judge.score_batch(
                eval_prompt,
                [(question, response)],
            )
            score_data = scores[0]

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


async def run_evaluation(
    experiment: str,
    trait: str,
    layer: int,
    coefficients: List[float],
    method: str = "probe",
    rollouts: int = 10,
    judge_provider: str = "openai",
    save_responses: bool = False,
) -> Dict:
    """
    Run full steering evaluation for a trait.

    Returns:
        Complete results dict ready for JSON serialization
    """
    # Load everything
    model, tokenizer = load_model_and_tokenizer()
    vector = load_vector(experiment, trait, layer, method)
    prompts_data = load_eval_prompts(trait)
    judge = TraitJudge(provider=judge_provider)

    questions = prompts_data["questions"]
    eval_prompt = prompts_data["eval_prompt"]

    print(f"\nEvaluating: {trait}")
    print(f"  Layer: {layer}, Method: {method}")
    print(f"  Questions: {len(questions)}, Rollouts: {rollouts}")
    print(f"  Coefficients: {coefficients}")
    print(f"  Judge: {judge_provider}")

    # Evaluate each coefficient
    results_by_coef = {}
    for coef in coefficients:
        result = await evaluate_coefficient(
            model, tokenizer, vector, layer, coef,
            questions, eval_prompt, judge,
            rollouts, save_responses,
        )
        results_by_coef[str(coef)] = result
        print(f"  coef={coef}: mean={result['trait_mean']:.1f}, n={result['n']}")

    # Compute summary metrics
    baseline = results_by_coef.get("0.0", results_by_coef.get("0", {}))
    baseline_mean = baseline.get("trait_mean", 0)

    max_mean = max(
        r.get("trait_mean", 0) or 0
        for r in results_by_coef.values()
    )

    # Controllability: correlation between coefficient and score
    coefs = []
    means = []
    for coef_str, result in results_by_coef.items():
        if result.get("trait_mean") is not None:
            coefs.append(float(coef_str))
            means.append(result["trait_mean"])

    if len(coefs) >= 2:
        # Pearson correlation
        mean_c = sum(coefs) / len(coefs)
        mean_m = sum(means) / len(means)
        num = sum((c - mean_c) * (m - mean_m) for c, m in zip(coefs, means))
        den_c = sum((c - mean_c)**2 for c in coefs)**0.5
        den_m = sum((m - mean_m)**2 for m in means)**0.5
        controllability = num / (den_c * den_m) if den_c * den_m > 0 else 0
    else:
        controllability = None

    results = {
        "trait": trait,
        "layer": layer,
        "method": method,
        "judge_provider": judge_provider,
        "n_questions": len(questions),
        "rollouts": rollouts,
        "timestamp": datetime.now().isoformat(),
        "coefficients": results_by_coef,
        "baseline": baseline_mean,
        "max_delta": max_mean - baseline_mean if baseline_mean else None,
        "controllability": controllability,
    }

    return results


def save_results(results: Dict, experiment: str, trait: str, save_responses: bool):
    """Save results to experiment directory."""
    # Main results
    results_file = get('steering.results', experiment=experiment, trait=trait)
    results_file.parent.mkdir(parents=True, exist_ok=True)

    # Extract responses before saving main results
    responses = {}
    results_clean = results.copy()
    for coef, data in results_clean["coefficients"].items():
        if "responses" in data:
            responses[coef] = data.pop("responses")

    with open(results_file, 'w') as f:
        json.dump(results_clean, f, indent=2)
    print(f"\nResults saved: {results_file}")

    # Save responses separately if requested
    if save_responses and responses:
        responses_dir = get('steering.responses', experiment=experiment, trait=trait)
        responses_dir.mkdir(parents=True, exist_ok=True)

        for coef, resp_list in responses.items():
            resp_file = responses_dir / f"coef_{coef.replace('.', '_')}.json"
            with open(resp_file, 'w') as f:
                json.dump(resp_list, f, indent=2)
        print(f"Responses saved: {responses_dir}")


def main():
    parser = argparse.ArgumentParser(description="Steering evaluation")
    parser.add_argument("--experiment", required=True, help="Experiment name")
    parser.add_argument("--trait", required=True, help="Trait path (e.g., cognitive_state/confidence)")
    parser.add_argument("--layer", type=int, required=True, help="Layer to steer (0-25)")
    parser.add_argument(
        "--coefficients",
        default="0,0.5,1.0,1.5,2.0,2.5",
        help="Comma-separated coefficients"
    )
    parser.add_argument("--method", default="probe", help="Vector extraction method")
    parser.add_argument("--rollouts", type=int, default=10, help="Rollouts per question")
    parser.add_argument("--judge", default="openai", choices=["openai", "gemini"])
    parser.add_argument("--save-responses", action="store_true", help="Save generated responses")

    args = parser.parse_args()

    coefficients = [float(c) for c in args.coefficients.split(",")]

    results = asyncio.run(run_evaluation(
        experiment=args.experiment,
        trait=args.trait,
        layer=args.layer,
        coefficients=coefficients,
        method=args.method,
        rollouts=args.rollouts,
        judge_provider=args.judge,
        save_responses=args.save_responses,
    ))

    save_results(results, args.experiment, args.trait, args.save_responses)


if __name__ == "__main__":
    main()
