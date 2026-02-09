#!/usr/bin/env python3
"""
CAA-style A/B logit difference evaluation.

Compute logit(answer_matching) - logit(answer_not_matching) on CAA multiple-choice
questions, optionally under steering. Measures behavioral propensity without generation.

Input:
    - experiment: Experiment name (for model config)
    - dataset: Path to CAA A/B questions JSON file
    - Optional: --steer trait with --layer and --coef for steering intervention

Output:
    - experiments/{experiment}/steering/{trait}/logit_diff_results.json (if --steer)
    - experiments/{experiment}/steering/logit_diff_results.json (if no --steer)

Usage:
    # Baseline (no steering)
    python analysis/steering/logit_difference.py \\
        --experiment quant-sensitivity/llama-8b \\
        --dataset datasets/traits/caa/sycophancy/test_ab.json

    # With steering
    python analysis/steering/logit_difference.py \\
        --experiment quant-sensitivity/llama-8b \\
        --dataset datasets/traits/caa/sycophancy/test_ab.json \\
        --steer caa/sycophancy \\
        --layer 14 --coef 6.0

    # With quantization
    python analysis/steering/logit_difference.py \\
        --experiment quant-sensitivity/llama-8b \\
        --dataset datasets/traits/caa/sycophancy/test_ab.json \\
        --steer caa/sycophancy \\
        --layer 14 --coef 6.0 \\
        --load-in-4bit

    # Dry run (validate inputs without loading model)
    python analysis/steering/logit_difference.py \\
        --experiment quant-sensitivity/llama-8b \\
        --dataset datasets/traits/caa/sycophancy/test_ab.json \\
        --dry-run
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
import json
import re
import torch
from contextlib import nullcontext
from datetime import datetime
from tqdm import tqdm

from core import SteeringHook, get_hook_path
from utils.model import load_model_with_lora, tokenize
from utils.paths import get as get_path, get_model_variant
from utils.vectors import get_best_vector, load_vector


def load_ab_dataset(dataset_path: str) -> list[dict]:
    """
    Load CAA A/B questions from JSON file.

    Expected format: list of objects with 'question', 'answer_matching_behavior',
    and 'answer_not_matching_behavior' keys.
    """
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    with open(path) as f:
        data = json.load(f)

    if not isinstance(data, list) or len(data) == 0:
        raise ValueError(f"Expected non-empty list of questions, got {type(data)}")

    # Validate first item
    required = {"question", "answer_matching_behavior", "answer_not_matching_behavior"}
    missing = required - set(data[0].keys())
    if missing:
        raise ValueError(f"Missing required fields in dataset: {missing}")

    return data


def extract_answer_text(answer: str) -> str:
    """
    Extract answer text, stripping the (A)/(B) prefix if present.

    Input like "(A) I agree" -> " I agree"
    Input like "I agree" -> " I agree" (adds leading space for tokenization)
    """
    stripped = re.sub(r"^\([A-Z]\)\s*", "", answer)
    # Ensure leading space for clean tokenization (completion follows context)
    if not stripped.startswith(" "):
        stripped = " " + stripped
    return stripped


def score_completion(model, tokenizer, context: str, completion: str) -> float:
    """Compute length-normalized log probability of a completion given context."""
    ctx_ids = tokenize(context, tokenizer).input_ids.to(model.device)
    full_text = context + completion
    full_ids = tokenize(full_text, tokenizer).input_ids.to(model.device)

    ctx_len = ctx_ids.shape[1]

    if full_ids.shape[1] <= ctx_len:
        return float("-inf")

    with torch.no_grad():
        outputs = model(full_ids)
        logits = outputs.logits[0]  # [seq_len, vocab]

    # Log probs for completion tokens only
    # logits[i] predicts token[i+1], so logits[ctx_len-1:] predicts tokens[ctx_len:]
    completion_logits = logits[ctx_len - 1 : -1]  # [completion_len, vocab]
    completion_targets = full_ids[0, ctx_len:]  # [completion_len]

    log_probs = torch.log_softmax(completion_logits, dim=-1)
    token_log_probs = log_probs.gather(1, completion_targets.unsqueeze(1)).squeeze()

    # Length-normalized log probability
    if token_log_probs.dim() == 0:
        total = token_log_probs.item()
        length = 1
    else:
        total = token_log_probs.sum().item()
        length = len(token_log_probs)

    return total / length


def compute_logit_differences(model, tokenizer, questions: list[dict], steering_ctx=None) -> list[dict]:
    """
    Compute logit(matching) - logit(not_matching) for each question.

    Returns list of per-question results with logit_diff and predicted label.
    """
    ctx = steering_ctx if steering_ctx else nullcontext()
    results = []

    with ctx:
        for item in tqdm(questions, desc="Computing logit differences"):
            question = item["question"]
            answer_matching = extract_answer_text(item["answer_matching_behavior"])
            answer_not_matching = extract_answer_text(item["answer_not_matching_behavior"])

            score_matching = score_completion(model, tokenizer, question, answer_matching)
            score_not_matching = score_completion(model, tokenizer, question, answer_not_matching)

            logit_diff = score_matching - score_not_matching
            predicted = "matching" if logit_diff > 0 else "not_matching"

            results.append({
                "question": question,
                "logit_diff": logit_diff,
                "score_matching": score_matching,
                "score_not_matching": score_not_matching,
                "predicted": predicted,
            })

    return results


def main():
    parser = argparse.ArgumentParser(description="CAA-style A/B logit difference evaluation")
    parser.add_argument("--experiment", required=True, help="Experiment name")
    parser.add_argument("--dataset", required=True, help="Path to CAA A/B questions JSON")
    parser.add_argument("--model-variant", default=None,
                        help="Model variant (default: from experiment config)")

    # Steering options (optional)
    parser.add_argument("--steer", help="Trait to steer with (e.g., caa/sycophancy)")
    parser.add_argument("--layer", type=int,
                        help="Layer for steering (auto-selects best if not specified)")
    parser.add_argument("--coef", type=float, default=6.0,
                        help="Steering coefficient (default: 6.0)")

    # Model options
    parser.add_argument("--load-in-8bit", action="store_true", help="Load model in 8-bit")
    parser.add_argument("--load-in-4bit", action="store_true", help="Load model in 4-bit")

    # Other
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate inputs without loading model")
    parser.add_argument("--limit", type=int, default=0,
                        help="Limit number of questions (default: 0 = all)")

    args = parser.parse_args()

    # Load and validate dataset
    questions = load_ab_dataset(args.dataset)
    print(f"Loaded {len(questions)} A/B questions from {args.dataset}")

    if args.limit > 0:
        questions = questions[:args.limit]
        print(f"Using first {len(questions)} questions (--limit {args.limit})")

    # Resolve model variant
    variant = get_model_variant(args.experiment, args.model_variant, mode="application")
    model_variant = variant["name"]
    model_name = variant["model"]
    lora = variant.get("lora")

    print(f"Experiment: {args.experiment}")
    print(f"Model: {model_name} (variant: {model_variant})")
    if lora:
        print(f"LoRA: {lora}")

    # Validate steering args
    steering_info = None
    if args.steer:
        steering_info = {
            "trait": args.steer,
            "layer": args.layer,
            "coef": args.coef,
        }
        print(f"Steering: {args.steer} (layer={args.layer or 'auto'}, coef={args.coef})")

    if args.dry_run:
        print("\n[Dry run] Inputs validated successfully.")
        print(f"  Questions: {len(questions)}")
        print(f"  Model: {model_name}")
        if steering_info:
            print(f"  Steering: {args.steer} layer={args.layer or 'auto'} coef={args.coef}")
        return

    run_logit_diff(
        experiment=args.experiment,
        dataset_path=args.dataset,
        questions=questions,
        model_variant=model_variant,
        model_name=model_name,
        lora=lora,
        steer=args.steer,
        layer=args.layer,
        coef=args.coef,
        load_in_8bit=args.load_in_8bit,
        load_in_4bit=args.load_in_4bit,
    )


def run_logit_diff(
    experiment: str,
    dataset_path: str,
    questions: list[dict],
    model_variant: str,
    model_name: str,
    lora: str = None,
    steer: str = None,
    layer: int = None,
    coef: float = 6.0,
    load_in_8bit: bool = False,
    load_in_4bit: bool = False,
    model=None,
    tokenizer=None,
):
    """Run logit difference evaluation. Accepts optional pre-loaded model/tokenizer."""
    steering_info = {"trait": steer, "layer": layer, "coef": coef} if steer else None

    # Load model if not provided
    if model is None:
        print(f"\nLoading model...")
        model, tokenizer = load_model_with_lora(
            model_name,
            lora_adapter=lora,
            load_in_8bit=load_in_8bit,
            load_in_4bit=load_in_4bit,
        )

    # Setup steering if requested
    steering_ctx = None
    if steer:
        # Auto-select best vector if no layer specified
        if layer is None:
            best = get_best_vector(experiment, steer)
            layer = best["layer"]
            method = best["method"]
            print(f"Auto-selected: layer {layer}, method {method} (source: {best['source']})")
        else:
            best = get_best_vector(experiment, steer, layer=layer)
            method = best["method"]
            print(f"Layer {layer}: method={method} (source: {best['source']})")

        steering_info["layer"] = layer
        steering_info["method"] = method
        steering_info["component"] = best.get("component", "residual")
        steering_info["position"] = best.get("position", "response[:5]")

        vector = load_vector(
            experiment, steer, layer, model_variant,
            method, steering_info["component"], steering_info["position"],
        )
        if vector is None:
            print(f"Error: Vector not found for L{layer} {method}")
            return None

        path = get_hook_path(layer)
        steering_ctx = SteeringHook(model, vector, path, coefficient=coef)
        print(f"Steering active: {steer} L{layer} coef={coef}")

    # Compute logit differences
    print(f"\nComputing logit differences on {len(questions)} questions...")
    per_question = compute_logit_differences(model, tokenizer, questions, steering_ctx)

    # Aggregate results
    logit_diffs = [q["logit_diff"] for q in per_question]
    mean_logit_diff = sum(logit_diffs) / len(logit_diffs)
    n_matching = sum(1 for q in per_question if q["predicted"] == "matching")
    matching_pct = n_matching / len(per_question) * 100

    # Print summary
    print(f"\n{'=' * 50}")
    print(f"Mean logit diff: {mean_logit_diff:+.4f}")
    print(f"Predicted matching: {n_matching}/{len(per_question)} ({matching_pct:.1f}%)")
    print(f"{'=' * 50}")

    # Build output
    output = {
        "mean_logit_diff": mean_logit_diff,
        "matching_pct": matching_pct,
        "n_matching": n_matching,
        "n_total": len(per_question),
        "per_question": per_question,
        "metadata": {
            "experiment": experiment,
            "model": model_name,
            "model_variant": model_variant,
            "dataset": dataset_path,
            "steering": steering_info,
            "load_in_8bit": load_in_8bit,
            "load_in_4bit": load_in_4bit,
            "timestamp": datetime.now().isoformat(),
        },
    }

    # Save results
    exp_base = get_path("experiments.base", experiment=experiment)
    if steer:
        output_dir = exp_base / "steering" / steer
    else:
        output_dir = exp_base / "steering"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "logit_diff_results.json"
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved to: {output_file}")
    return output


if __name__ == "__main__":
    main()
