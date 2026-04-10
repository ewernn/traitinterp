"""
Pre-extraction vetting: LLM-as-judge validates scenarios and responses before extraction.

Input:
    Scenarios: datasets/traits/{trait}/positive.txt, negative.txt
    Responses: experiments/{exp}/extraction/{trait}/responses/pos.json, neg.json

Output:
    vetting/scenario_scores.json (stage 0)
    vetting/response_scores.json (stage 2)

Usage:
    from utils.preextraction_vetting import vet
    pass_rate = vet(experiment, trait, model_variant, target="responses")
    pass_rate = vet(experiment, trait, model_variant, target="scenarios")
"""

import asyncio
import json
from statistics import median
from typing import Literal
from tqdm.asyncio import tqdm_asyncio

from utils.paths import get as get_path, content_hash
from utils.judge import TraitJudge
from utils.traits import load_trait_definition, load_scenarios


def slice_by_position(text: str, position: str) -> str:
    """Extract the words matching an extraction position spec.

    Uses whitespace-split approximation (not true tokenization, but sufficient
    for vetting since the judge only needs a rough window).

    Examples:
        slice_by_position(text, "response[:]")  → full text
        slice_by_position(text, "response[:5]") → first 5 words
        slice_by_position(text, "response[50:]") → words 50 onward
        slice_by_position(text, "response[5:10]") → words 5 to 10
    """
    import re
    m = re.search(r'\[(\d*):(\d*)\]', position)
    if not m:
        return text  # unrecognized format → full text
    start = int(m.group(1)) if m.group(1) else None
    end = int(m.group(2)) if m.group(2) else None
    words = text.split()
    sliced = words[start:end]
    return ' '.join(sliced) if sliced else text


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


def _build_vetting_output(results: list, pos_threshold: int, neg_threshold: int) -> dict:
    """Shared pass/fail accounting for both scenario and response vetting."""
    pos_results = [r for r in results if r["polarity"] == "positive"]
    neg_results = [r for r in results if r["polarity"] == "negative"]

    pos_passed = [r for r in pos_results if r["score"] is not None and r["score"] >= pos_threshold]
    neg_passed = [r for r in neg_results if r["score"] is not None and r["score"] <= neg_threshold]

    pos_failed = [r for r in pos_results if r["score"] is not None and r["score"] < pos_threshold]
    neg_failed = [r for r in neg_results if r["score"] is not None and r["score"] > neg_threshold]
    errors = [r for r in results if r["score"] is None]

    pos_errors = [r for r in pos_results if r["score"] is None]
    neg_errors = [r for r in neg_results if r["score"] is None]

    failed_indices = {
        "positive": [r["idx"] for r in pos_failed] + [r["idx"] for r in pos_errors],
        "negative": [r["idx"] for r in neg_failed] + [r["idx"] for r in neg_errors],
    }

    summary = {
        "positive_passed": len(pos_passed),
        "negative_passed": len(neg_passed),
        "positive_failed": len(pos_failed),
        "negative_failed": len(neg_failed),
        "errors": len(errors),
    }

    return {
        "pos_passed": pos_passed, "neg_passed": neg_passed,
        "pos_failed": pos_failed, "neg_failed": neg_failed,
        "errors": errors, "failed_indices": failed_indices, "summary": summary,
    }


# =============================================================================
# Async scoring internals
# =============================================================================

async def _vet_scenarios_async(trait: str, max_concurrent: int = 20) -> dict:
    """Score all scenarios and return results."""
    judge = TraitJudge()
    scenarios = load_scenarios(trait)
    trait_definition = load_trait_definition(trait)
    trait_name = trait.split('/')[-1]

    items = []
    for polarity in ['positive', 'negative']:
        for idx, s in enumerate(scenarios[polarity]):
            items.append((s['prompt'], polarity, idx))

    semaphore = asyncio.Semaphore(max_concurrent)

    async def score_one(scenario: str, polarity: str, idx: int) -> dict:
        async with semaphore:
            score = await judge.score_scenario(scenario, trait_name, trait_definition)
            return {"idx": idx, "scenario": scenario, "polarity": polarity, "score": score}

    tasks = [score_one(s, p, idx) for s, p, idx in items]
    results = await tqdm_asyncio.gather(*tasks, desc="  Vetting scenarios")
    await judge.close()

    return {"trait_definition": trait_definition, "results": results}


async def _vet_responses_async(
    experiment: str, trait: str, model_variant: str,
    max_concurrent: int = 100, estimate_trait_tokens: bool = False,
    position: str = "response[:]",
) -> dict:
    """Score all responses and return results.

    The judge sees the full response for context but scores only the tokens
    matching the extraction position (via XML-tagged <score_this> section).
    """
    judge = TraitJudge()
    responses = load_responses(experiment, trait, model_variant)
    trait_definition = load_trait_definition(trait)
    trait_name = trait.split('/')[-1]

    items = []
    for polarity in ['positive', 'negative']:
        for idx, item in enumerate(responses[polarity]):
            full_response = item.get('response', '')
            score_section = slice_by_position(full_response, position)
            prompt = item.get('prompt', '')
            items.append({
                "idx": idx, "polarity": polarity,
                "prompt": prompt, "full_response": full_response,
                "score_section": score_section,
            })

    semaphore = asyncio.Semaphore(max_concurrent)

    async def score_one(item: dict) -> dict:
        async with semaphore:
            score = await judge.score_response(
                item["prompt"], item["full_response"], trait_name, trait_definition,
                score_section=item["score_section"],
            )
            result = {"idx": item["idx"], "polarity": item["polarity"], "score": score}
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


# =============================================================================
# Public API
# =============================================================================

def vet(
    experiment: str,
    trait: str,
    model_variant: str,
    target: Literal["scenarios", "responses"] = "responses",
    pos_threshold: int = 60,
    neg_threshold: int = 40,
    max_concurrent: int = None,
    estimate_trait_tokens: bool = False,
    position: str = "response[:]",
) -> float:
    """Vet scenarios or responses using LLM-as-judge. Returns pass rate.

    Args:
        target: "scenarios" (score prompts) or "responses" (score model outputs)
        max_concurrent: Concurrent API calls. Default: 20 for scenarios, 100 for responses.
        estimate_trait_tokens: Estimate trait token positions (responses only, for adaptive position).
        position: Extraction position spec (e.g. "response[:]", "response[:5]"). The judge sees
            the full response for context but scores only the tokens matching this position.
    """
    if max_concurrent is None:
        max_concurrent = 20 if target == "scenarios" else 100

    # Run async scoring
    if target == "scenarios":
        data = asyncio.run(_vet_scenarios_async(trait, max_concurrent))
    else:
        data = asyncio.run(_vet_responses_async(
            experiment, trait, model_variant, max_concurrent, estimate_trait_tokens, position
        ))

    results = data["results"]
    trait_definition = data["trait_definition"]
    vetting = _build_vetting_output(results, pos_threshold, neg_threshold)

    # Build output data
    output_dir = get_path('extraction.vetting', experiment=experiment, trait=trait, model_variant=model_variant)
    output_dir.mkdir(parents=True, exist_ok=True)

    from utils.traits import get_scenario_path
    trait_dir = get_path('datasets.trait', trait=trait)
    output_data = {
        "experiment": experiment,
        "trait": trait,
        "trait_definition": trait_definition,
        "thresholds": {"pos_threshold": pos_threshold, "neg_threshold": neg_threshold},
        "summary": vetting["summary"],
        "failed_indices": vetting["failed_indices"],
        "results": results,
        "input_hashes": {
            "positive": content_hash(get_scenario_path(trait, 'positive')),
            "negative": content_hash(get_scenario_path(trait, 'negative')),
            "definition": content_hash(trait_dir / 'definition.txt'),
        },
    }

    # Compute recommended position from trait token counts (responses only)
    if target == "responses" and estimate_trait_tokens:
        token_counts = [r["trait_token_count"] for r in results if r.get("trait_token_count")]
        if token_counts:
            med = int(median(token_counts))
            output_data["llm_judge_position"] = f"response[:{max(1, med)}]"
            print(f"      Recommended position: response[:{med}] (median of {len(token_counts)} samples)")

    # Save — under TP, only rank 0 writes
    filename = "scenario_scores.json" if target == "scenarios" else "response_scores.json"
    if target == "scenarios":
        with open(output_dir / filename, 'w') as f:
            json.dump(output_data, f, indent=2)
    else:
        from utils.distributed import is_rank_zero
        if is_rank_zero():
            with open(output_dir / filename, 'w') as f:
                json.dump(output_data, f, indent=2)

    # Compute pass rate
    total_passed = len(vetting["pos_passed"]) + len(vetting["neg_passed"])
    if target == "scenarios":
        total_valid = len(results) - len(vetting["errors"])
        pass_rate = total_passed / total_valid if total_valid > 0 else 0.0
    else:
        pass_rate = total_passed / len(results) if results else 0.0

    label = "scenarios" if target == "scenarios" else "responses"
    denom = (len(results) - len(vetting["errors"])) if target == "scenarios" else len(results)
    print(f"      {len(results)} {label} | pass: {pass_rate:.1%} ({total_passed}/{denom}) | errors: {len(vetting['errors'])}")
    return pass_rate


# Backwards-compatible aliases
def vet_scenarios(experiment, trait, model_variant, pos_threshold=60, neg_threshold=40, max_concurrent=20):
    return vet(experiment, trait, model_variant, target="scenarios",
               pos_threshold=pos_threshold, neg_threshold=neg_threshold, max_concurrent=max_concurrent)

def vet_responses(experiment, trait, model_variant, pos_threshold=60, neg_threshold=40,
                  max_concurrent=100, estimate_trait_tokens=False, position="response[:]"):
    return vet(experiment, trait, model_variant, target="responses",
               pos_threshold=pos_threshold, neg_threshold=neg_threshold,
               max_concurrent=max_concurrent, estimate_trait_tokens=estimate_trait_tokens,
               position=position)
