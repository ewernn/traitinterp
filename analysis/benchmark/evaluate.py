#!/usr/bin/env python3
"""
Benchmark evaluation - measure capability preservation.

Primarily useful for testing ablation (negative steering) to verify capabilities
are preserved when removing a direction. Positive steering on traits like refusal
will confound results (model refuses to answer = benchmark fails, but not due to
capability loss).

Input:
    - experiment: Experiment to use (for model config)
    - benchmark: Benchmark name (hellaswag, arc_easy)

Output:
    - experiments/{experiment}/benchmark/{benchmark}.json

Usage:
    # Basic - run on model from experiment
    python analysis/benchmark/evaluate.py \\
        --experiment gemma-2-2b-it \\
        --benchmark hellaswag

    # With limit (default 20, use 0 for full)
    python analysis/benchmark/evaluate.py \\
        --experiment gemma-2-2b-it \\
        --benchmark hellaswag \\
        --limit 100

    # With ablation (negative steering)
    python analysis/benchmark/evaluate.py \\
        --experiment gemma-2-2b-it \\
        --benchmark hellaswag \\
        --steer safety/refusal --coef -1.0

    # CE loss instead of accuracy
    python analysis/benchmark/evaluate.py \\
        --experiment gemma-2-2b-it \\
        --metric ce_loss
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
import json
import torch
from contextlib import nullcontext
from datetime import datetime
from tqdm import tqdm

from core import SteeringHook, get_hook_path
from utils.model import load_model, load_experiment_config
from utils.paths import get as get_path, get_vector_path
from utils.vectors import get_best_vector
from utils.metrics import batch_ce_loss


def score_completions(model, tokenizer, context: str, completions: list[str]) -> int:
    """
    Score multiple completions, return index of highest log-likelihood.

    For each completion, compute P(completion | context) and pick the best.
    Uses length-normalized log probability.
    """
    scores = []

    for completion in completions:
        # Tokenize context and full sequence
        ctx_ids = tokenizer(context, return_tensors="pt", add_special_tokens=True).input_ids.to(model.device)
        full_text = context + completion
        full_ids = tokenizer(full_text, return_tensors="pt", add_special_tokens=True).input_ids.to(model.device)

        ctx_len = ctx_ids.shape[1]

        # Skip if completion adds no tokens
        if full_ids.shape[1] <= ctx_len:
            scores.append(float("-inf"))
            continue

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

        scores.append(total / length)

    return scores.index(max(scores))


def evaluate_hellaswag(model, tokenizer, limit: int = None, steering_ctx=None):
    """Run HellaSwag evaluation."""
    from datasets import load_dataset

    ds = load_dataset("Rowan/hellaswag", split="validation")
    if limit:
        ds = ds.select(range(min(limit, len(ds))))

    correct = 0
    total = 0

    ctx = steering_ctx if steering_ctx else nullcontext()

    with ctx:
        for item in tqdm(ds, desc="HellaSwag"):
            # Context is activity_label + ctx
            context = item["activity_label"] + ": " + item["ctx"]
            completions = item["endings"]
            label = int(item["label"])

            pred = score_completions(model, tokenizer, context, completions)
            if pred == label:
                correct += 1
            total += 1

    return {"accuracy": correct / total, "correct": correct, "total": total}


def evaluate_arc_easy(model, tokenizer, limit: int = None, steering_ctx=None):
    """Run ARC-Easy evaluation."""
    from datasets import load_dataset

    ds = load_dataset("allenai/ai2_arc", "ARC-Easy", split="test")
    if limit:
        ds = ds.select(range(min(limit, len(ds))))

    correct = 0
    total = 0

    ctx = steering_ctx if steering_ctx else nullcontext()

    with ctx:
        for item in tqdm(ds, desc="ARC-Easy"):
            question = item["question"]
            choices = item["choices"]["text"]
            label_key = item["answerKey"]

            # Convert A/B/C/D/E or 1/2/3/4 to index
            if label_key.isdigit():
                label = int(label_key) - 1
            else:
                label = ord(label_key) - ord("A")

            # Format as question + each answer
            context = f"Question: {question}\nAnswer:"

            pred = score_completions(model, tokenizer, context, choices)
            if pred == label:
                correct += 1
            total += 1

    return {"accuracy": correct / total, "correct": correct, "total": total}


def evaluate_arc_challenge(model, tokenizer, limit: int = None, steering_ctx=None):
    """Run ARC-Challenge evaluation (harder than ARC-Easy)."""
    from datasets import load_dataset

    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    if limit:
        ds = ds.select(range(min(limit, len(ds))))

    correct = 0
    total = 0

    ctx = steering_ctx if steering_ctx else nullcontext()

    with ctx:
        for item in tqdm(ds, desc="ARC-Challenge"):
            question = item["question"]
            choices = item["choices"]["text"]
            label_key = item["answerKey"]

            # Convert A/B/C/D/E or 1/2/3/4 to index
            if label_key.isdigit():
                label = int(label_key) - 1
            else:
                label = ord(label_key) - ord("A")

            context = f"Question: {question}\nAnswer:"

            pred = score_completions(model, tokenizer, context, choices)
            if pred == label:
                correct += 1
            total += 1

    return {"accuracy": correct / total, "correct": correct, "total": total}


def evaluate_gpqa(model, tokenizer, limit: int = None, steering_ctx=None):
    """
    Run GPQA Diamond evaluation (graduate-level science questions).

    Note: GPQA is gated on HuggingFace. You need to:
    1. Request access at https://huggingface.co/datasets/Idavidrein/gpqa
    2. Set HF_TOKEN environment variable
    """
    from datasets import load_dataset

    try:
        ds = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")
    except Exception as e:
        if "gated" in str(e).lower() or "access" in str(e).lower():
            print("Error: GPQA is a gated dataset. To use it:")
            print("  1. Request access at https://huggingface.co/datasets/Idavidrein/gpqa")
            print("  2. Set HF_TOKEN environment variable")
            raise
        raise

    if limit:
        ds = ds.select(range(min(limit, len(ds))))

    correct = 0
    total = 0

    ctx = steering_ctx if steering_ctx else nullcontext()

    with ctx:
        for item in tqdm(ds, desc="GPQA"):
            question = item["Question"]
            # GPQA has Correct Answer and three incorrect options
            choices = [
                item["Correct Answer"],
                item["Incorrect Answer 1"],
                item["Incorrect Answer 2"],
                item["Incorrect Answer 3"],
            ]
            label = 0  # Correct answer is always first in our list

            # Shuffle to avoid position bias (but track correct answer)
            import random
            indices = list(range(4))
            random.shuffle(indices)
            shuffled_choices = [choices[i] for i in indices]
            shuffled_label = indices.index(0)  # Where did correct answer end up?

            context = f"Question: {question}\nAnswer:"

            pred = score_completions(model, tokenizer, context, shuffled_choices)
            if pred == shuffled_label:
                correct += 1
            total += 1

    return {"accuracy": correct / total, "correct": correct, "total": total}


def evaluate_ce_loss(model, tokenizer, limit: int = None, steering_ctx=None):
    """
    Evaluate CE loss on held-out text (WikiText-2).

    Returns average cross-entropy loss - lower is better.
    """
    from datasets import load_dataset

    # Use WikiText-2 for held-out text (standard perplexity benchmark)
    try:
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        texts = []
        for item in ds:
            text = item["text"].strip()
            if len(text) > 100:  # Skip short/empty entries
                texts.append(text[:2048])  # Truncate long texts
            if limit and len(texts) >= limit:
                break
    except Exception as e:
        print(f"Warning: Could not load WikiText dataset: {e}")
        print("Using simple test texts instead")
        texts = [
            "The quick brown fox jumps over the lazy dog. " * 20,
            "In a distant galaxy, there existed a civilization that had mastered interstellar travel. " * 10,
        ]
        if limit:
            texts = texts[:limit]

    ctx = steering_ctx if steering_ctx else nullcontext()

    with ctx:
        loss = batch_ce_loss(model, tokenizer, texts)

    return {"ce_loss": loss, "num_texts": len(texts)}


BENCHMARKS = {
    "hellaswag": evaluate_hellaswag,
    "arc_easy": evaluate_arc_easy,
    "arc_challenge": evaluate_arc_challenge,
    "gpqa": evaluate_gpqa,
    "ce_loss": evaluate_ce_loss,
}


def main():
    parser = argparse.ArgumentParser(description="Benchmark evaluation")
    parser.add_argument("--experiment", required=True, help="Experiment name")
    parser.add_argument(
        "--benchmark",
        "--metric",
        dest="benchmark",
        default="hellaswag",
        choices=list(BENCHMARKS.keys()),
        help="Benchmark to run (default: hellaswag)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Max examples (default: 20, use 0 for full)",
    )

    # Steering options (optional)
    parser.add_argument("--steer", help="Trait to steer with (e.g., safety/refusal)")
    parser.add_argument("--layer", type=int, help="Layer for steering (auto-selects if not specified)")
    parser.add_argument("--coef", type=float, default=-1.0, help="Steering coefficient (default: -1.0 for ablation)")
    parser.add_argument("--method", default="probe", help="Vector extraction method (default: probe)")

    # Model options
    parser.add_argument("--load-in-8bit", action="store_true", help="Load model in 8-bit")
    parser.add_argument("--load-in-4bit", action="store_true", help="Load model in 4-bit")

    args = parser.parse_args()

    # Load model
    config = load_experiment_config(args.experiment)
    model_name = config.get("application_model", "google/gemma-2-2b-it")
    model, tokenizer = load_model(
        model_name,
        load_in_8bit=args.load_in_8bit,
        load_in_4bit=args.load_in_4bit,
    )

    # Setup steering if requested
    steering_ctx = None
    steering_info = None

    if args.steer:
        # Auto-select layer if not specified
        if args.layer is None:
            best = get_best_vector(args.experiment, args.steer)
            layer = best["layer"]
            method = best["method"]
            position = best["position"]
            component = best["component"]
            print(f"Auto-selected: layer {layer}, method {method} (source: {best['source']})")
        else:
            layer = args.layer
            method = args.method
            position = "response[:]"
            component = "residual"

        vector_path = get_vector_path(args.experiment, args.steer, method, layer, component, position)
        if not vector_path.exists():
            print(f"Error: Vector not found at {vector_path}")
            sys.exit(1)

        vector = torch.load(vector_path, weights_only=True)
        path = get_hook_path(layer)
        steering_ctx = SteeringHook(model, vector, path, coefficient=args.coef)
        steering_info = {
            "trait": args.steer,
            "layer": layer,
            "method": method,
            "coef": args.coef,
        }
        print(f"Steering: {args.steer} L{layer} coef={args.coef}")

    # Run benchmark
    limit = args.limit if args.limit > 0 else None
    eval_fn = BENCHMARKS[args.benchmark]
    print(f"\nRunning {args.benchmark}" + (f" (limit={limit})" if limit else " (full)"))

    results = eval_fn(model, tokenizer, limit=limit, steering_ctx=steering_ctx)

    # Build output
    output = {
        "benchmark": args.benchmark,
        "model": model_name,
        "experiment": args.experiment,
        "timestamp": datetime.now().isoformat(),
        "limit": limit,
        "steering": steering_info,
        **results,
    }

    # Print results
    print(f"\n{'='*50}")
    if "accuracy" in results:
        print(f"{args.benchmark}: {results['accuracy']:.1%} ({results['correct']}/{results['total']})")
    elif "ce_loss" in results:
        print(f"CE Loss: {results['ce_loss']:.4f} ({results['num_texts']} texts)")
    print(f"{'='*50}")

    # Save results
    output_dir = get_path("experiments.base", experiment=args.experiment) / "benchmark"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Include steering info in filename if steering
    if steering_info:
        trait_slug = args.steer.replace("/", "_")
        output_file = output_dir / f"{args.benchmark}_{trait_slug}_L{layer}_c{args.coef}.json"
    else:
        output_file = output_dir / f"{args.benchmark}.json"

    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved to: {output_file}")


if __name__ == "__main__":
    main()
