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

    # With limit (default 200, use 0 for full)
    python analysis/benchmark/evaluate.py \\
        --experiment gemma-2-2b-it \\
        --benchmark hellaswag \\
        --limit 100

    # With ablation (negative steering) - auto-selects best vector
    python analysis/benchmark/evaluate.py \\
        --experiment gemma-2-2b-it \\
        --benchmark hellaswag \\
        --steer safety/refusal --coef -1.0

    # With specific layer (finds best method/position at that layer)
    python analysis/benchmark/evaluate.py \\
        --experiment gemma-2-2b-it \\
        --benchmark hellaswag \\
        --steer safety/refusal --layer 12 --coef -1.0

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
import random
import torch
from contextlib import nullcontext
from datetime import datetime
from tqdm import tqdm

from core import SteeringHook, get_hook_path
from utils.model import load_model, load_model_with_lora, tokenize, tokenize_batch
from utils.paths import get as get_path, get_model_variant
from utils.vectors import get_best_vector, load_vector
from utils.generation import calculate_max_batch_size
from utils.metrics import batch_ce_loss


def _arc_label(key: str) -> int:
    """Convert ARC answer key ('A'/'B'/'1'/'2') to 0-indexed label."""
    return int(key) - 1 if key.isdigit() else ord(key) - ord("A")


def _score_log_probs(logits, input_ids, ctx_len: int, length: int) -> float:
    """Compute length-normalized log prob of completion tokens for one sequence."""
    comp_len = length - ctx_len
    if comp_len <= 0:
        return float("-inf")

    # logits[i] predicts token[i+1], so logits[ctx_len-1:length-1] predicts tokens[ctx_len:length]
    completion_logits = logits[ctx_len - 1 : length - 1]  # [comp_len, vocab]
    completion_targets = input_ids[ctx_len:length]  # [comp_len]

    log_probs = torch.log_softmax(completion_logits, dim=-1)
    token_log_probs = log_probs.gather(1, completion_targets.unsqueeze(1)).squeeze()

    if token_log_probs.dim() == 0:
        return token_log_probs.item()
    return token_log_probs.sum().item() / len(token_log_probs)


def _is_single_token_completions(questions: list[dict], tokenizer) -> bool:
    """Check if all completions tokenize to exactly 1 token."""
    for q in questions:
        for c in q["completions"]:
            ids = tokenizer.encode(c, add_special_tokens=False)
            if len(ids) != 1:
                return False
    return True


def _score_single_token_batch(
    model, tokenizer,
    questions: list[dict],
    desc: str = "Scoring",
) -> dict:
    """
    Fast path for single-token completions (e.g. MMLU's " A"/" B"/" C"/" D").

    Forward pass on context only, extract logits at last position for target tokens.
    No full logits materialization — only needs [batch, 1, vocab] at the last position.
    """
    # Pre-tokenize completion tokens
    for q in questions:
        q["_target_ids"] = [tokenizer.encode(c, add_special_tokens=False)[0] for c in q["completions"]]

    # Batch size: just context sequences (no completions duplication)
    sample = questions[:min(10, len(questions))]
    sample_lens = [len(tokenize(q["context"], tokenizer).input_ids[0]) for q in sample]
    est_max_len = int(max(sample_lens) * 1.2)

    max_batch = calculate_max_batch_size(model, est_max_len, mode='inference')

    results = []
    correct = 0
    total = 0
    pbar = tqdm(total=len(questions), desc=desc)
    batch_start = 0
    batch_size = max_batch

    while batch_start < len(questions):
        batch_qs = questions[batch_start : batch_start + batch_size]
        contexts = [q["context"] for q in batch_qs]

        try:
            batch = tokenize_batch(contexts, tokenizer, padding_side="right")
            input_ids = batch["input_ids"].to(model.device)
            attention_mask = batch["attention_mask"].to(model.device)
            lengths = batch["lengths"]

            with torch.no_grad():
                logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            batch_size = max(1, batch_size // 2)
            print(f"\n  OOM — reducing to {batch_size} contexts/batch")
            continue

        # Score: for each context, get logits at last real token, compare target token probs
        for b, q in enumerate(batch_qs):
            last_pos = lengths[b] - 1
            last_logits = logits[b, last_pos]  # [vocab]
            log_probs = torch.log_softmax(last_logits, dim=-1)

            scores = [log_probs[tid].item() for tid in q["_target_ids"]]
            pred = scores.index(max(scores))
            is_correct = pred == q["label"]
            if is_correct:
                correct += 1
            total += 1
            result = {"correct": is_correct, "predicted": pred, "label": q["label"]}
            if "extra" in q:
                result.update(q["extra"])
            results.append(result)

        del logits, input_ids, attention_mask
        torch.cuda.empty_cache()

        pbar.update(len(batch_qs))
        batch_start += len(batch_qs)

    pbar.close()
    return {"accuracy": correct / total if total else 0, "correct": correct, "total": total, "questions": results}


def score_questions_batch(
    model, tokenizer,
    questions: list[dict],
    desc: str = "Scoring",
) -> dict:
    """
    Score multiple-choice questions in large batches.

    Two paths:
    - Single-token completions (MMLU): forward pass on context only, compare target token logits.
    - Multi-token completions (HellaSwag, TruthfulQA): flatten N×C into batched forward passes.

    Each question dict must have: context (str), completions (list[str]), label (int).
    Optional: extra (dict) — passed through to results.
    """
    if not questions:
        return {"accuracy": 0, "correct": 0, "total": 0, "questions": []}

    # Fast path for single-token completions (MMLU-style)
    if _is_single_token_completions(questions, tokenizer):
        print(f"  Using single-token fast path")
        return _score_single_token_batch(model, tokenizer, questions, desc)

    # Pre-compute context token lengths (avoids redundant tokenization in batch loop)
    for q in questions:
        q["_ctx_len"] = len(tokenize(q["context"], tokenizer).input_ids[0])

    # Estimate max seq len from a sample for batch size calculation
    sample = questions[:min(10, len(questions))]
    sample_texts = [q["context"] + c for q in sample for c in q["completions"]]
    sample_lens = [len(tokenize(t, tokenizer).input_ids[0]) for t in sample_texts]
    est_max_len = int(max(sample_lens) * 1.2)  # 20% headroom

    # Use 'generation' mode — accounts for logits buffer (batch × seq × vocab)
    max_batch = calculate_max_batch_size(model, est_max_len, mode='generation')

    # Process in batches of questions — use max completions for batch size estimate
    max_choices = max(len(q["completions"]) for q in questions)
    questions_per_batch = max(1, max_batch // max_choices)

    results = []
    correct = 0
    total = 0

    batch_start = 0
    pbar = tqdm(total=len(questions), desc=desc)

    while batch_start < len(questions):
        batch_qs = questions[batch_start : batch_start + questions_per_batch]

        # Flatten: for each question, all completions
        all_texts = []
        ctx_lens = []
        for q in batch_qs:
            for c in q["completions"]:
                all_texts.append(q["context"] + c)
                ctx_lens.append(q["_ctx_len"])

        # Batched forward pass with OOM recovery
        try:
            batch = tokenize_batch(all_texts, tokenizer, padding_side="right")
            input_ids = batch["input_ids"].to(model.device)
            attention_mask = batch["attention_mask"].to(model.device)
            lengths = batch["lengths"]

            with torch.no_grad():
                logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            questions_per_batch = max(1, questions_per_batch // 2)
            print(f"\n  OOM — reducing to {questions_per_batch} questions/batch ({questions_per_batch * max_choices} seqs)")
            continue  # Retry same batch_start with smaller batch

        # Score each completion, then group by question
        idx = 0
        for q in batch_qs:
            nc = len(q["completions"])
            scores = []
            for j in range(nc):
                s = _score_log_probs(logits[idx], input_ids[idx], ctx_lens[idx], lengths[idx])
                scores.append(s)
                idx += 1

            pred = scores.index(max(scores))
            is_correct = pred == q["label"]
            if is_correct:
                correct += 1
            total += 1
            result = {"correct": is_correct, "predicted": pred, "label": q["label"]}
            if "extra" in q:
                result.update(q["extra"])
            results.append(result)

        del logits, input_ids, attention_mask
        torch.cuda.empty_cache()

        pbar.update(len(batch_qs))
        batch_start += questions_per_batch

    pbar.close()
    return {"accuracy": correct / total if total else 0, "correct": correct, "total": total, "questions": results}


def evaluate_hellaswag(model, tokenizer, limit: int = None, steering_ctx=None):
    """Run HellaSwag evaluation."""
    from datasets import load_dataset

    ds = load_dataset("Rowan/hellaswag", split="validation")
    if limit:
        ds = ds.select(range(min(limit, len(ds))))

    questions = [
        {
            "context": item["activity_label"] + ": " + item["ctx"],
            "completions": item["endings"],
            "label": int(item["label"]),
            "extra": {"id": i},
        }
        for i, item in enumerate(ds)
    ]

    ctx = steering_ctx if steering_ctx else nullcontext()
    with ctx:
        return score_questions_batch(model, tokenizer, questions, desc="HellaSwag")


def evaluate_arc_easy(model, tokenizer, limit: int = None, steering_ctx=None):
    """Run ARC-Easy evaluation."""
    from datasets import load_dataset

    ds = load_dataset("allenai/ai2_arc", "ARC-Easy", split="test")
    if limit:
        ds = ds.select(range(min(limit, len(ds))))

    questions = [
        {
            "context": f"Question: {item['question']}\nAnswer:",
            "completions": item["choices"]["text"],
            "label": _arc_label(item["answerKey"]),
            "extra": {"id": i},
        }
        for i, item in enumerate(ds)
    ]

    ctx = steering_ctx if steering_ctx else nullcontext()
    with ctx:
        return score_questions_batch(model, tokenizer, questions, desc="ARC-Easy")


def evaluate_arc_challenge(model, tokenizer, limit: int = None, steering_ctx=None):
    """Run ARC-Challenge evaluation (harder than ARC-Easy)."""
    from datasets import load_dataset

    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    if limit:
        ds = ds.select(range(min(limit, len(ds))))

    questions = [
        {
            "context": f"Question: {item['question']}\nAnswer:",
            "completions": item["choices"]["text"],
            "label": _arc_label(item["answerKey"]),
            "extra": {"id": i},
        }
        for i, item in enumerate(ds)
    ]

    ctx = steering_ctx if steering_ctx else nullcontext()
    with ctx:
        return score_questions_batch(model, tokenizer, questions, desc="ARC-Challenge")


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

    rng = random.Random(42)
    questions = []
    for i, item in enumerate(ds):
        choices = [
            item["Correct Answer"],
            item["Incorrect Answer 1"],
            item["Incorrect Answer 2"],
            item["Incorrect Answer 3"],
        ]
        # Shuffle to avoid position bias (seeded for reproducibility)
        indices = list(range(4))
        rng.shuffle(indices)
        questions.append({
            "context": f"Question: {item['Question']}\nAnswer:",
            "completions": [choices[j] for j in indices],
            "label": indices.index(0),
            "extra": {"id": i},
        })

    ctx = steering_ctx if steering_ctx else nullcontext()
    with ctx:
        return score_questions_batch(model, tokenizer, questions, desc="GPQA")


def evaluate_mmlu(model, tokenizer, limit: int = None, steering_ctx=None):
    """
    Run MMLU evaluation (subset).

    Uses the MMLU dataset from HuggingFace. When limit is set, uses that many
    questions per subject (57 subjects total). With limit=10, evaluates ~570 questions.
    """
    from datasets import load_dataset

    ds = load_dataset("cais/mmlu", "all", split="test")
    if limit:
        # Sample evenly across subjects
        from collections import defaultdict
        by_subject = defaultdict(list)
        for i, item in enumerate(ds):
            by_subject[item["subject"]].append(i)
        indices = []
        for subject, idxs in sorted(by_subject.items()):
            indices.extend(idxs[:limit])
        ds = ds.select(indices)

    choice_labels = ["A", "B", "C", "D"]

    questions = []
    for i, item in enumerate(ds):
        choice_text = "\n".join(f"{choice_labels[j]}) {c}" for j, c in enumerate(item["choices"]))
        questions.append({
            "context": f"Question: {item['question']}\n{choice_text}\nAnswer:",
            "completions": [f" {l}" for l in choice_labels[:len(item["choices"])]],
            "label": item["answer"],
            "extra": {"id": i, "subject": item["subject"]},
        })

    ctx = steering_ctx if steering_ctx else nullcontext()
    with ctx:
        return score_questions_batch(model, tokenizer, questions, desc="MMLU")


def evaluate_truthfulqa(model, tokenizer, limit: int = None, steering_ctx=None):
    """
    Run TruthfulQA evaluation (multiple-choice variant).

    Measures whether model selects truthful answers over common misconceptions.
    Relevant for sycophancy vector validation (CAA showed sycophancy vectors improve TruthfulQA).
    """
    from datasets import load_dataset

    ds = load_dataset("truthfulqa/truthful_qa", "multiple_choice", split="validation")
    if limit:
        ds = ds.select(range(min(limit, len(ds))))

    questions = []
    for i, item in enumerate(ds):
        labels = item["mc1_targets"]["labels"]
        questions.append({
            "context": f"Q: {item['question']}\nA:",
            "completions": [f" {c}" for c in item["mc1_targets"]["choices"]],
            "label": labels.index(1),
            "extra": {"id": i},
        })

    ctx = steering_ctx if steering_ctx else nullcontext()
    with ctx:
        return score_questions_batch(model, tokenizer, questions, desc="TruthfulQA")


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
    "mmlu": evaluate_mmlu,
    "truthfulqa": evaluate_truthfulqa,
    "ce_loss": evaluate_ce_loss,
}


def main():
    parser = argparse.ArgumentParser(description="Benchmark evaluation")
    parser.add_argument("--experiment", required=True, help="Experiment name")
    parser.add_argument("--model-variant", default=None, help="Model variant (default: from experiment config)")
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
        default=200,
        help="Max examples (default: 200, use 0 for full)",
    )

    # Steering options (optional)
    parser.add_argument("--steer", help="Trait to steer with (e.g., safety/refusal)")
    parser.add_argument("--layer", type=int, help="Layer for steering (auto-selects best at this layer if specified)")
    parser.add_argument("--coef", type=float, default=-1.0, help="Steering coefficient (default: -1.0 for ablation)")

    # Model options
    parser.add_argument("--load-in-8bit", action="store_true", help="Load model in 8-bit")
    parser.add_argument("--load-in-4bit", action="store_true", help="Load model in 4-bit")

    args = parser.parse_args()

    # Resolve model variant
    variant = get_model_variant(args.experiment, args.model_variant, mode="application")
    model_variant = variant['name']
    model_name = variant['model']
    lora = variant.get('lora')

    model, tokenizer = load_model_with_lora(
        model_name,
        lora_adapter=lora,
        load_in_8bit=args.load_in_8bit,
        load_in_4bit=args.load_in_4bit,
    )

    # Setup steering if requested
    steering_ctx = None
    steering_info = None

    if args.steer:
        # Use get_best_vector with optional layer filter (auto-resolves variants from config)
        best = get_best_vector(args.experiment, args.steer, layer=args.layer)
        layer = best["layer"]
        method = best["method"]
        position = best["position"]
        component = best["component"]
        if args.layer is None:
            print(f"Auto-selected: layer {layer}, method {method} (source: {best['source']})")
        else:
            print(f"Layer {layer}: best is method={method}, position={position} (source: {best['source']})")

        vector = load_vector(args.experiment, args.steer, layer, model_variant, method, component, position)
        if vector is None:
            print(f"Error: Vector not found for L{layer} {method} {component} {position}")
            sys.exit(1)
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
