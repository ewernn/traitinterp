"""
Vet generated responses using LLM-as-judge.

Called by run_pipeline.py (stage 2).

Only scores first 16 tokens of response to match extraction window behavior.
"""

import asyncio
import json
from statistics import median
from tqdm.asyncio import tqdm_asyncio

from utils.paths import get as get_path
from utils.judge import TraitJudge
from utils.traits import load_trait_definition

# Truncate responses to first N tokens for vetting
# This aligns vetting with extraction (which uses response[:N])
VET_TOKEN_LIMIT = 16


def truncate_to_tokens(text: str, max_tokens: int = VET_TOKEN_LIMIT) -> str:
    """Truncate text to approximately max_tokens (whitespace-split approximation)."""
    words = text.split()
    return ' '.join(words[:max_tokens])


def load_responses(experiment: str, trait: str, model_variant: str) -> dict:
    """Load generated response files."""
    responses_dir = get_path('extraction.responses', experiment=experiment, trait=trait, model_variant=model_variant)
    responses = {}
    for polarity, filename in [('positive', 'pos.json'), ('negative', 'neg.json')]:
        file_path = responses_dir / filename
        if file_path.exists():
            with open(file_path) as f:
                responses[polarity] = json.load(f)
        else:
            responses[polarity] = []
    return responses


async def _vet_responses_async(
    experiment: str,
    trait: str,
    model_variant: str,
    max_concurrent: int = 20,
    estimate_trait_tokens: bool = False,
) -> dict:
    """Score all responses and return results."""
    judge = TraitJudge()
    responses = load_responses(experiment, trait, model_variant)
    trait_definition = load_trait_definition(trait)
    trait_name = trait.split('/')[-1]

    items = []
    for polarity in ['positive', 'negative']:
        for idx, item in enumerate(responses[polarity]):
            full_response = item.get('response', '')
            text = truncate_to_tokens(full_response)  # Only vet first N tokens
            prompt = item.get('prompt', '')
            items.append({
                "idx": idx,
                "polarity": polarity,
                "prompt": prompt,
                "text": text,
                "full_response": full_response,
            })

    semaphore = asyncio.Semaphore(max_concurrent)

    async def score_one(item: dict) -> dict:
        async with semaphore:
            score = await judge.score_response(item["prompt"], item["text"], trait_name, trait_definition)
            result = {"idx": item["idx"], "polarity": item["polarity"], "score": score}
            # Estimate trait tokens for positive responses that pass
            if estimate_trait_tokens and item["polarity"] == "positive" and score is not None and score >= 60:
                token_count = await judge.estimate_trait_tokens(
                    item["prompt"], item["full_response"], trait_name, trait_definition
                )
                result["trait_token_count"] = token_count
            return result

    tasks = [score_one(item) for item in items]
    results = await tqdm_asyncio.gather(*tasks, desc="  Vetting responses")
    await judge.close()

    return {"trait_definition": trait_definition, "results": results}


def vet_responses(
    experiment: str,
    trait: str,
    model_variant: str,
    pos_threshold: int = 60,
    neg_threshold: int = 40,
    max_concurrent: int = 20,
    estimate_trait_tokens: bool = False,
) -> float:
    """
    Vet generated responses using LLM-as-judge. Returns pass rate.
    """
    data = asyncio.run(_vet_responses_async(
        experiment, trait, model_variant, max_concurrent, estimate_trait_tokens
    ))
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
    output_dir = get_path('extraction.trait', experiment=experiment, trait=trait, model_variant=model_variant) / "vetting"
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

    # Compute recommended position from trait token counts
    if estimate_trait_tokens:
        token_counts = [r["trait_token_count"] for r in results if r.get("trait_token_count")]
        if token_counts:
            med = int(median(token_counts))
            output_data["llm_judge_position"] = f"response[:{max(1, med)}]"
            print(f"      Recommended position: response[:{med}] (median of {len(token_counts)} samples)")

    with open(output_dir / "response_scores.json", 'w') as f:
        json.dump(output_data, f, indent=2)

    total_passed = len(pos_passed) + len(neg_passed)
    pass_rate = total_passed / len(results) if results else 0.0

    print(f"      {len(results)} responses | pass: {pass_rate:.1%} ({total_passed}/{len(results)}) | errors: {len(errors)}")
    return pass_rate
