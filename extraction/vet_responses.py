"""
Vet generated responses using LLM-as-judge.

Called by run_pipeline.py (stage 2).
"""

import asyncio
import json
from datetime import datetime
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


def load_trait_definition(trait: str) -> str:
    """Load trait definition file from datasets/traits/."""
    def_file = get_path('datasets.trait_definition', trait=trait)
    if def_file.exists():
        with open(def_file) as f:
            return f.read().strip()
    return f"The trait '{trait.split('/')[-1].replace('_', ' ')}'"


async def _vet_responses_async(experiment: str, trait: str, max_concurrent: int = 20) -> dict:
    """Score all responses and return results."""
    judge = TraitJudge()
    responses = load_responses(experiment, trait)
    trait_definition = load_trait_definition(trait)
    trait_name = trait.split('/')[-1]

    items = []
    for polarity in ['positive', 'negative']:
        for idx, item in enumerate(responses[polarity]):
            text = item.get('response', '')
            prompt = item.get('prompt', '')
            items.append({"idx": idx, "polarity": polarity, "prompt": prompt, "text": text})

    semaphore = asyncio.Semaphore(max_concurrent)

    async def score_one(item: dict) -> dict:
        async with semaphore:
            score = await judge.score_response(item["prompt"], item["text"], trait_name, trait_definition)
            return {"idx": item["idx"], "polarity": item["polarity"], "score": score}

    tasks = [score_one(item) for item in items]
    results = await tqdm_asyncio.gather(*tasks, desc="  Vetting responses")
    await judge.close()

    return {"trait_definition": trait_definition, "results": results}


def vet_responses(
    experiment: str,
    trait: str,
    pos_threshold: int = 60,
    neg_threshold: int = 40,
    max_concurrent: int = 20,
) -> float:
    """
    Vet generated responses using LLM-as-judge. Returns pass rate.
    """
    print(f"  [2] Vetting responses for '{trait}'...")

    data = asyncio.run(_vet_responses_async(experiment, trait, max_concurrent))
    results = data["results"]
    trait_definition = data["trait_definition"]

    pos_results = [r for r in results if r["polarity"] == "positive"]
    neg_results = [r for r in results if r["polarity"] == "negative"]

    pos_passed = [r for r in pos_results if r["score"] is not None and r["score"] >= pos_threshold]
    neg_passed = [r for r in neg_results if r["score"] is not None and r["score"] <= neg_threshold]

    pos_failed = [r for r in pos_results if r["score"] is not None and r["score"] < pos_threshold]
    neg_failed = [r for r in neg_results if r["score"] is not None and r["score"] > neg_threshold]
    errors = [r for r in results if r["score"] is None]

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

    with open(output_dir / "response_scores.json", 'w') as f:
        json.dump(output_data, f, indent=2)

    total_passed = len(pos_passed) + len(neg_passed)
    pass_rate = total_passed / len(results) if results else 0.0

    print(f"    Pass rate: {pass_rate:.1%} ({total_passed}/{len(results)})")
    return pass_rate
