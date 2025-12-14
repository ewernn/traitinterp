#!/usr/bin/env python3
"""
Vet scenario files using gpt-4.1-nano with logprob scoring.

Input (from datasets/):
    - datasets/traits/{trait}/positive.txt
    - datasets/traits/{trait}/negative.txt
    - datasets/traits/{trait}/definition.txt

Output (to experiment):
    - experiments/{experiment}/extraction/{trait}/vetting/scenario_scores.json
    - experiments/{experiment}/extraction/{trait}/vetting/metadata.json

Usage:
    python extraction/vet_scenarios.py --experiment my_exp --trait category/my_trait
    python extraction/vet_scenarios.py --experiment my_exp --trait category/my_trait --pos-threshold 60 --neg-threshold 40
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


def load_scenarios(experiment: str, trait: str) -> dict:
    """Load positive and negative scenario files from datasets/traits/."""
    dataset_trait_dir = get_path('datasets.trait', trait=trait)

    scenarios = {}
    for polarity in ['positive', 'negative']:
        file_path = dataset_trait_dir / f"{polarity}.txt"
        if file_path.exists():
            with open(file_path) as f:
                scenarios[polarity] = [line.strip() for line in f if line.strip()]
        else:
            scenarios[polarity] = []

    return scenarios


def load_trait_definition(experiment: str, trait: str) -> str:
    """Load trait definition file from datasets/traits/."""
    def_file = get_path('datasets.trait_definition', trait=trait)

    if def_file.exists():
        with open(def_file) as f:
            return f.read().strip()

    # Fallback: use trait name
    trait_name = trait.split('/')[-1].replace('_', ' ')
    return f"The trait '{trait_name}'"


async def vet_scenarios_async(
    experiment: str,
    trait: str,
    threshold: int = 70,
    max_concurrent: int = 20,
) -> dict:
    """Score all scenarios and return results."""
    judge = TraitJudge()

    scenarios = load_scenarios(experiment, trait)
    trait_definition = load_trait_definition(experiment, trait)

    print(f"Trait: {trait}")
    print(f"Definition: {trait_definition[:100]}...")
    print(f"Positive scenarios: {len(scenarios['positive'])}")
    print(f"Negative scenarios: {len(scenarios['negative'])}")
    print(f"Threshold: {threshold}")
    print(f"Judge: {judge.model} (logprob scoring)")
    print()

    # Build list of (scenario, polarity) tuples
    items = []
    for polarity in ['positive', 'negative']:
        for scenario in scenarios[polarity]:
            items.append((scenario, polarity))

    # Score with progress bar
    semaphore = asyncio.Semaphore(max_concurrent)

    async def score_one(scenario: str, polarity: str) -> dict:
        async with semaphore:
            score = await judge.score_scenario(scenario, trait_definition, polarity)
            return {
                "scenario": scenario,
                "polarity": polarity,
                "score": score,
            }

    print(f"Scoring {len(items)} scenarios...")
    tasks = [score_one(s, p) for s, p in items]
    results = await tqdm_asyncio.gather(*tasks, desc="Vetting scenarios")

    await judge.close()

    return {
        "trait_definition": trait_definition,
        "results": results,
    }


def vet_scenarios(
    experiment: str,
    trait: str,
    pos_threshold: int = 60,
    neg_threshold: int = 40,
    max_concurrent: int = 20,
):
    """
    Vet scenario files using LLM-as-judge.

    Scores predict trait level in completion:
    - Positive scenarios should get HIGH scores (>= pos_threshold)
    - Negative scenarios should get LOW scores (<= neg_threshold)

    Args:
        experiment: Experiment name
        trait: Trait path (e.g., 'behavioral/refusal')
        pos_threshold: Positive scenarios need score >= this
        neg_threshold: Negative scenarios need score <= this
        max_concurrent: Maximum concurrent API calls
    """
    data = asyncio.run(vet_scenarios_async(
        experiment=experiment,
        trait=trait,
        threshold=pos_threshold,  # Not used in async, just for compat
        max_concurrent=max_concurrent,
    ))

    results = data["results"]
    trait_definition = data["trait_definition"]

    # Split by polarity
    pos_results = [r for r in results if r["polarity"] == "positive"]
    neg_results = [r for r in results if r["polarity"] == "negative"]

    pos_scores = [r["score"] for r in pos_results if r["score"] is not None]
    neg_scores = [r["score"] for r in neg_results if r["score"] is not None]

    # Apply thresholds (positive needs HIGH, negative needs LOW)
    pos_passed = [r for r in pos_results if r["score"] is not None and r["score"] >= pos_threshold]
    neg_passed = [r for r in neg_results if r["score"] is not None and r["score"] <= neg_threshold]

    pos_failed = [r for r in pos_results if r["score"] is not None and r["score"] < pos_threshold]
    neg_failed = [r for r in neg_results if r["score"] is not None and r["score"] > neg_threshold]

    errors = [r for r in results if r["score"] is None]

    # Print summary
    print("\n" + "=" * 60)
    print("SCENARIO VETTING SUMMARY")
    print("=" * 60)
    print(f"Total: {len(results)}")
    print(f"Errors: {len(errors)}")

    print(f"\nPositive scenarios (need score >= {pos_threshold}):")
    print(f"  Passed: {len(pos_passed)}/{len(pos_results)}")
    if pos_scores:
        print(f"  Mean: {sum(pos_scores)/len(pos_scores):.1f}")

    print(f"\nNegative scenarios (need score <= {neg_threshold}):")
    print(f"  Passed: {len(neg_passed)}/{len(neg_results)}")
    if neg_scores:
        print(f"  Mean: {sum(neg_scores)/len(neg_scores):.1f}")

    # Separation analysis
    if pos_scores and neg_scores:
        separation = sum(pos_scores)/len(pos_scores) - sum(neg_scores)/len(neg_scores)
        print(f"\n*** SEPARATION: {separation:.1f} points ***")
        if separation < 20:
            print("    WARNING: Low separation")
        elif separation < 40:
            print("    OK: Moderate separation")
        else:
            print("    GOOD: Strong separation")

    # Show failed scenarios
    if pos_failed:
        print("\n" + "-" * 60)
        print(f"FAILED POSITIVE (score < {pos_threshold}):")
        for r in sorted(pos_failed, key=lambda x: x["score"] or 0)[:5]:
            print(f"  [{r['score']:.0f}] {r['scenario'][:70]}...")

    if neg_failed:
        print("\n" + "-" * 60)
        print(f"FAILED NEGATIVE (score > {neg_threshold}):")
        for r in sorted(neg_failed, key=lambda x: -(x["score"] or 0))[:5]:
            print(f"  [{r['score']:.0f}] {r['scenario'][:70]}...")

    # Save results
    output_dir = get_path('extraction.trait', experiment=experiment, trait=trait) / "vetting"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_data = {
        "experiment": experiment,
        "trait": trait,
        "trait_definition": trait_definition,
        "judge_model": "gpt-4.1-nano",
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
            "separation": (sum(pos_scores)/len(pos_scores) - sum(neg_scores)/len(neg_scores)) if (pos_scores and neg_scores) else None,
        },
        "failed_indices": {
            "positive": [i for i, r in enumerate(pos_results) if r["score"] is None or r["score"] < pos_threshold],
            "negative": [i for i, r in enumerate(neg_results) if r["score"] is None or r["score"] > neg_threshold],
        },
        "results": results,
    }

    with open(output_dir / "scenario_scores.json", 'w') as f:
        json.dump(output_data, f, indent=2)

    # Update metadata
    metadata_path = output_dir / 'metadata.json'
    metadata = {}
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)

    metadata.update({
        "judge_model": "gpt-4.1-nano",
        "judge_method": "logprob",
        "scenario_vetting": True,
        "scenario_thresholds": {
            "pos_threshold": pos_threshold,
            "neg_threshold": neg_threshold,
        },
        "scenario_timestamp": datetime.now().isoformat(),
    })

    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\nResults saved to: {output_dir / 'scenario_scores.json'}")

    total_passed = len(pos_passed) + len(neg_passed)
    total_valid = len(results) - len(errors)
    if total_valid > 0:
        pass_rate = total_passed / total_valid
        print(f"Pass rate: {pass_rate:.1%}")
        return pass_rate
    return 0.0


if __name__ == "__main__":
    fire.Fire(vet_scenarios)
