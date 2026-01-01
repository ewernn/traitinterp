"""
Vet scenario files using LLM-as-judge.

Called by run_pipeline.py (stage 0).
"""

import asyncio
import json
from datetime import datetime
from tqdm.asyncio import tqdm_asyncio

from utils.paths import get as get_path
from utils.judge import TraitJudge


def load_scenarios(trait: str) -> dict:
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


def load_trait_definition(trait: str) -> str:
    """Load trait definition file from datasets/traits/."""
    def_file = get_path('datasets.trait_definition', trait=trait)
    if def_file.exists():
        with open(def_file) as f:
            return f.read().strip()
    return f"The trait '{trait.split('/')[-1].replace('_', ' ')}'"


async def _vet_scenarios_async(trait: str, max_concurrent: int = 20) -> dict:
    """Score all scenarios and return results."""
    judge = TraitJudge()
    scenarios = load_scenarios(trait)
    trait_definition = load_trait_definition(trait)
    trait_name = trait.split('/')[-1]

    items = [(s, p) for p in ['positive', 'negative'] for s in scenarios[p]]
    semaphore = asyncio.Semaphore(max_concurrent)

    async def score_one(scenario: str, polarity: str) -> dict:
        async with semaphore:
            score = await judge.score_scenario(scenario, trait_name, trait_definition)
            return {"scenario": scenario, "polarity": polarity, "score": score}

    tasks = [score_one(s, p) for s, p in items]
    results = await tqdm_asyncio.gather(*tasks, desc="  Vetting scenarios")
    await judge.close()

    return {"trait_definition": trait_definition, "results": results}


def vet_scenarios(
    experiment: str,
    trait: str,
    pos_threshold: int = 60,
    neg_threshold: int = 40,
    max_concurrent: int = 20,
) -> float:
    """
    Vet scenario files using LLM-as-judge. Returns pass rate.
    """
    print(f"  [0] Vetting scenarios for '{trait}'...")

    data = asyncio.run(_vet_scenarios_async(trait, max_concurrent))
    results = data["results"]
    trait_definition = data["trait_definition"]

    pos_results = [r for r in results if r["polarity"] == "positive"]
    neg_results = [r for r in results if r["polarity"] == "negative"]

    pos_passed = [r for r in pos_results if r["score"] is not None and r["score"] >= pos_threshold]
    neg_passed = [r for r in neg_results if r["score"] is not None and r["score"] <= neg_threshold]

    pos_failed = [r for r in pos_results if r["score"] is not None and r["score"] < pos_threshold]
    neg_failed = [r for r in neg_results if r["score"] is not None and r["score"] > neg_threshold]
    errors = [r for r in results if r["score"] is None]

    # Build failed_indices for consistency with vet_responses.py
    # Note: For scenarios, the index is the line number in positive.txt/negative.txt
    pos_scenarios = load_scenarios(trait)['positive']
    neg_scenarios = load_scenarios(trait)['negative']

    failed_indices = {
        "positive": [pos_scenarios.index(r["scenario"]) for r in pos_failed if r["scenario"] in pos_scenarios] +
                   [pos_scenarios.index(r["scenario"]) for r in results if r["polarity"] == "positive" and r["score"] is None and r["scenario"] in pos_scenarios],
        "negative": [neg_scenarios.index(r["scenario"]) for r in neg_failed if r["scenario"] in neg_scenarios] +
                   [neg_scenarios.index(r["scenario"]) for r in results if r["polarity"] == "negative" and r["score"] is None and r["scenario"] in neg_scenarios],
    }

    # Save results
    output_dir = get_path('extraction.trait', experiment=experiment, trait=trait) / "vetting"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_data = {
        "experiment": experiment,
        "trait": trait,
        "trait_definition": trait_definition,
        "thresholds": {"pos_threshold": pos_threshold, "neg_threshold": neg_threshold},
        "summary": {
            "positive_passed": len(pos_passed),
            "negative_passed": len(neg_passed),
            "positive_failed": len(pos_failed),
            "negative_failed": len(neg_failed),
            "errors": len(errors),
        },
        "failed_indices": failed_indices,
        "results": results,
    }

    with open(output_dir / "scenario_scores.json", 'w') as f:
        json.dump(output_data, f, indent=2)

    total_passed = len(pos_passed) + len(neg_passed)
    total_valid = len(results) - len(errors)
    pass_rate = total_passed / total_valid if total_valid > 0 else 0.0

    print(f"    Pass rate: {pass_rate:.1%} ({total_passed}/{total_valid})")
    return pass_rate
