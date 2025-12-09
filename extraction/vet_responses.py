#!/usr/bin/env python3
"""
Vet generated responses using gpt-4.1-mini with logprob scoring.

Input:
    - experiments/{experiment}/extraction/{trait}/responses/pos.json
    - experiments/{experiment}/extraction/{trait}/responses/neg.json
    - experiments/{experiment}/extraction/{trait}/trait_definition.txt

Output:
    - experiments/{experiment}/extraction/{trait}/vetting/response_scores.json
    - experiments/{experiment}/extraction/{trait}/vetting/metadata.json

Usage:
    python extraction/vet_responses.py --experiment my_exp --trait category/my_trait
    python extraction/vet_responses.py --experiment my_exp --trait category/my_trait --pos-threshold 60 --neg-threshold 40
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import asyncio
import json
from datetime import datetime
import fire
from tqdm.asyncio import tqdm_asyncio

from utils.paths import get as get_path
from utils.judge import TraitJudge


def load_responses(experiment: str, trait: str) -> dict:
    """Load generated response files."""
    responses_dir = get_path('extraction.responses', experiment=experiment, trait=trait)

    responses = {}
    for polarity, filename in [('positive', 'pos.json'), ('negative', 'neg.json')]:
        file_path = responses_dir / filename
        if file_path.exists():
            with open(file_path) as f:
                responses[polarity] = json.load(f)
        else:
            responses[polarity] = []

    return responses


def load_trait_definition(experiment: str, trait: str) -> str:
    """Load trait definition file."""
    trait_dir = get_path('extraction.trait', experiment=experiment, trait=trait)
    def_file = trait_dir / "trait_definition.txt"

    if def_file.exists():
        with open(def_file) as f:
            return f.read().strip()

    trait_name = trait.split('/')[-1].replace('_', ' ')
    return f"The trait '{trait_name}'"


def print_histogram(scores: list, title: str, width: int = 40):
    """Print ASCII histogram of score distribution."""
    if not scores:
        print(f"  {title}: No data")
        return

    bins = [0] * 10
    for s in scores:
        if 0 <= s <= 100:
            bin_idx = min(int(s) // 10, 9)
            bins[bin_idx] += 1

    max_count = max(bins) if bins else 1

    print(f"\n  {title} (n={len(scores)}, mean={sum(scores)/len(scores):.1f})")
    print("  " + "-" * (width + 15))

    for i, count in enumerate(bins):
        bar_len = int((count / max_count) * width) if max_count > 0 else 0
        label = f"{i*10:3d}-{i*10+9:3d}"
        bar = "#" * bar_len
        print(f"  {label} | {bar} {count}")


async def vet_responses_async(
    experiment: str,
    trait: str,
    pos_threshold: int,
    neg_threshold: int,
    max_concurrent: int,
) -> dict:
    """Score all responses and return results."""
    judge = TraitJudge()

    responses = load_responses(experiment, trait)
    trait_definition = load_trait_definition(experiment, trait)

    print(f"Trait: {trait}")
    print(f"Definition: {trait_definition[:100]}...")
    print(f"Positive responses: {len(responses['positive'])}")
    print(f"Negative responses: {len(responses['negative'])}")
    print(f"Thresholds: pos >= {pos_threshold}, neg <= {neg_threshold}")
    print(f"Judge: {judge.model} (logprob scoring)")
    print()

    # Build list of items to score
    items = []
    for polarity in ['positive', 'negative']:
        for idx, item in enumerate(responses[polarity]):
            text = item.get('response') or item.get('answer', '')
            prompt = item.get('prompt') or item.get('question', '')
            items.append({
                "idx": idx,
                "polarity": polarity,
                "prompt": prompt,
                "text": text,
            })

    # Score with progress bar
    semaphore = asyncio.Semaphore(max_concurrent)

    async def score_one(item: dict) -> dict:
        async with semaphore:
            score = await judge.score_response(item["prompt"], item["text"], trait_definition)
            return {
                "idx": item["idx"],
                "polarity": item["polarity"],
                "prompt": item["prompt"][:200],
                "text": item["text"][:500],
                "score": score,
            }

    print(f"Scoring {len(items)} responses...")
    tasks = [score_one(item) for item in items]
    results = await tqdm_asyncio.gather(*tasks, desc="Vetting responses")

    return {
        "trait_definition": trait_definition,
        "results": results,
    }


def vet_responses(
    experiment: str,
    trait: str,
    pos_threshold: int = 60,
    neg_threshold: int = 40,
    max_concurrent: int = 20,
):
    """
    Vet generated responses using LLM-as-judge.

    Args:
        experiment: Experiment name
        trait: Trait path (e.g., 'behavioral/refusal')
        pos_threshold: Positive responses need score >= this
        neg_threshold: Negative responses need score <= this
        max_concurrent: Maximum concurrent API calls
    """
    data = asyncio.run(vet_responses_async(
        experiment=experiment,
        trait=trait,
        pos_threshold=pos_threshold,
        neg_threshold=neg_threshold,
        max_concurrent=max_concurrent,
    ))

    results = data["results"]
    trait_definition = data["trait_definition"]

    # Split by polarity
    pos_results = [r for r in results if r["polarity"] == "positive"]
    neg_results = [r for r in results if r["polarity"] == "negative"]

    pos_scores = [r["score"] for r in pos_results if r["score"] is not None]
    neg_scores = [r["score"] for r in neg_results if r["score"] is not None]

    # Apply thresholds
    pos_passed = [r for r in pos_results if r["score"] is not None and r["score"] >= pos_threshold]
    neg_passed = [r for r in neg_results if r["score"] is not None and r["score"] <= neg_threshold]

    pos_failed = [r for r in pos_results if r["score"] is not None and r["score"] < pos_threshold]
    neg_failed = [r for r in neg_results if r["score"] is not None and r["score"] > neg_threshold]

    errors = [r for r in results if r["score"] is None]

    # Print summary
    print("\n" + "=" * 60)
    print("RESPONSE VETTING SUMMARY")
    print("=" * 60)

    print_histogram(pos_scores, f"Positive responses (expect HIGH, need >= {pos_threshold})")
    print_histogram(neg_scores, f"Negative responses (expect LOW, need <= {neg_threshold})")

    # Separation analysis
    if pos_scores and neg_scores:
        pos_mean = sum(pos_scores) / len(pos_scores)
        neg_mean = sum(neg_scores) / len(neg_scores)
        separation = pos_mean - neg_mean

        print(f"\n*** SEPARATION: {separation:.1f} points (pos mean - neg mean) ***")
        if separation < 20:
            print("    WARNING: Low separation - scenarios may not elicit distinct behaviors")
        elif separation < 40:
            print("    OK: Moderate separation")
        else:
            print("    GOOD: Strong separation")

    # Filtering stats
    print("\n" + "-" * 60)
    print("FILTERING RESULTS")
    print("-" * 60)

    print(f"\nPositive (need score >= {pos_threshold}):")
    print(f"  Passed: {len(pos_passed)}/{len(pos_results)}")
    print(f"  Failed: {len(pos_failed)}")

    print(f"\nNegative (need score <= {neg_threshold}):")
    print(f"  Passed: {len(neg_passed)}/{len(neg_results)}")
    print(f"  Failed: {len(neg_failed)}")

    print(f"\nErrors: {len(errors)}")

    # Show failed examples
    if pos_failed:
        print("\n" + "-" * 60)
        print(f"FAILED POSITIVE (score < {pos_threshold}):")
        for r in sorted(pos_failed, key=lambda x: x["score"] or 0)[:3]:
            print(f"  [{r['score']:.0f}] {r['prompt'][:50]}...")
            print(f"       -> {r['text'][:60]}...")

    if neg_failed:
        print("\n" + "-" * 60)
        print(f"FAILED NEGATIVE (score > {neg_threshold}):")
        for r in sorted(neg_failed, key=lambda x: -(x["score"] or 0))[:3]:
            print(f"  [{r['score']:.0f}] {r['prompt'][:50]}...")
            print(f"       -> {r['text'][:60]}...")

    # Final dataset stats
    total_passed = len(pos_passed) + len(neg_passed)
    print("\n" + "=" * 60)
    print("FINAL DATASET")
    print("=" * 60)
    print(f"  Positive samples: {len(pos_passed)}")
    print(f"  Negative samples: {len(neg_passed)}")
    print(f"  Total usable: {total_passed}")
    print(f"  Discarded: {len(pos_failed) + len(neg_failed) + len(errors)}")

    # Build failed indices
    failed_indices = {
        "positive": [r["idx"] for r in pos_failed] + [r["idx"] for r in results if r["polarity"] == "positive" and r["score"] is None],
        "negative": [r["idx"] for r in neg_failed] + [r["idx"] for r in results if r["polarity"] == "negative" and r["score"] is None],
    }

    # Save results
    output_dir = get_path('extraction.trait', experiment=experiment, trait=trait) / "vetting"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_data = {
        "experiment": experiment,
        "trait": trait,
        "trait_definition": trait_definition,
        "judge_model": "gpt-4.1-mini",
        "judge_method": "logprob",
        "thresholds": {
            "pos_threshold": pos_threshold,
            "neg_threshold": neg_threshold,
        },
        "summary": {
            "total": len(results),
            "positive_passed": len(pos_passed),
            "negative_passed": len(neg_passed),
            "positive_failed": len(pos_failed),
            "negative_failed": len(neg_failed),
            "errors": len(errors),
            "positive_mean": sum(pos_scores) / len(pos_scores) if pos_scores else None,
            "negative_mean": sum(neg_scores) / len(neg_scores) if neg_scores else None,
            "separation": (sum(pos_scores) / len(pos_scores) - sum(neg_scores) / len(neg_scores)) if (pos_scores and neg_scores) else None,
        },
        "failed_indices": failed_indices,
        "results": results,
    }

    with open(output_dir / "response_scores.json", 'w') as f:
        json.dump(output_data, f, indent=2)

    # Update metadata
    metadata_path = output_dir / 'metadata.json'
    metadata = {}
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)

    metadata.update({
        "judge_model": "gpt-4.1-mini",
        "judge_method": "logprob",
        "response_vetting": True,
        "response_thresholds": {
            "pos_threshold": pos_threshold,
            "neg_threshold": neg_threshold,
        },
        "response_timestamp": datetime.now().isoformat(),
    })

    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\nResults saved to: {output_dir / 'response_scores.json'}")
    print(f"Failed indices saved for filtering during activation extraction")

    pass_rate = total_passed / len(results) if results else 0
    print(f"\nPass rate: {pass_rate:.1%}")
    return pass_rate


if __name__ == "__main__":
    fire.Fire(vet_responses)
