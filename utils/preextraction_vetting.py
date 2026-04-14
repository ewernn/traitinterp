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


def get_scored_text(
    prompt: str, response: str, position: str, tokenizer,
    use_chat_template: bool = False, system_prompt: str = None,
) -> str:
    """Decode exactly the tokens that extraction would capture at this position.

    Tokenizes prompt+response together (matching extraction's `prepare_split`),
    then uses `resolve_position` from the position DSL to find the right slice.
    This ensures vetting scores the same text window extraction uses for
    activation capture — including correct token boundaries at the
    prompt/response junction.

    Args:
        prompt: The scenario prompt text.
        response: The model's full response text.
        position: Extraction position spec (e.g., 'response[:]', 'response[:5]').
        tokenizer: The model's tokenizer.
        use_chat_template: Whether to format the prompt with the chat template.
        system_prompt: Optional system prompt for chat template formatting.

    Returns:
        Decoded text matching the extraction position window.
    """
    from utils.model import format_prompt
    from utils.positions import resolve_position

    # Format prompt the same way extraction does
    formatted = format_prompt(prompt, tokenizer, use_chat_template=use_chat_template,
                              system_prompt=system_prompt)
    full_text = formatted + response

    # Tokenize together (matching extraction's prepare_split)
    has_bos = tokenizer.bos_token and full_text.startswith(tokenizer.bos_token)
    full_ids = tokenizer.encode(full_text, add_special_tokens=not has_bos)
    prompt_ids = tokenizer.encode(formatted, add_special_tokens=not has_bos)

    start, end = resolve_position(position, len(prompt_ids), len(full_ids))
    sliced = full_ids[start:end]
    return tokenizer.decode(sliced) if sliced else ""


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
    position: str = "response[:]", tokenizer=None,
    use_chat_template: bool = False,
    pos_threshold: int = 60,
) -> dict:
    """Score all responses and return results.

    The judge sees the full response for context but scores only the tokens
    matching the extraction position (via XML-tagged <score_this> section).
    Uses get_scored_text() which tokenizes prompt+response together — exactly
    matching extraction's token boundaries.
    """
    if tokenizer is None:
        raise ValueError("tokenizer is required for response vetting (needed for position-accurate scoring)")

    judge = TraitJudge()
    responses = load_responses(experiment, trait, model_variant)
    trait_definition = load_trait_definition(trait)
    trait_name = trait.split('/')[-1]

    items = []
    for polarity in ['positive', 'negative']:
        for idx, item in enumerate(responses[polarity]):
            full_response = item.get('response', '')
            prompt = item.get('prompt', '')
            system_prompt = item.get('system_prompt')
            score_section = get_scored_text(
                prompt, full_response, position, tokenizer,
                use_chat_template=use_chat_template,
                system_prompt=system_prompt,
            )
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
            if estimate_trait_tokens and item["polarity"] == "positive" and score is not None and score >= pos_threshold:
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
    tokenizer=None,
    use_chat_template: bool = False,
) -> float:
    """Vet scenarios or responses using LLM-as-judge. Returns pass rate.

    Args:
        target: "scenarios" (score prompts) or "responses" (score model outputs)
        max_concurrent: Concurrent API calls. Default: 20 for scenarios, 100 for responses.
        estimate_trait_tokens: Estimate trait token positions (responses only, for adaptive position).
        position: Extraction position spec (e.g. "response[:]", "response[:5]"). The judge sees
            the full response for context but scores only the tokens matching this position.
        tokenizer: HuggingFace tokenizer (required for response vetting). Used for
            position-accurate scoring that matches extraction's token boundaries.
        use_chat_template: Whether to format prompts with the model's chat template
            (needed for correct prompt_len computation when tokenizing prompt+response together).
    """
    if max_concurrent is None:
        max_concurrent = 20 if target == "scenarios" else 100

    # Run async scoring
    if target == "scenarios":
        data = asyncio.run(_vet_scenarios_async(trait, max_concurrent))
    else:
        data = asyncio.run(_vet_responses_async(
            experiment, trait, model_variant, max_concurrent, estimate_trait_tokens,
            position, tokenizer, use_chat_template, pos_threshold,
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
    from utils.distributed import is_rank_zero
    filename = "scenario_scores.json" if target == "scenarios" else "response_scores.json"
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
                  max_concurrent=100, estimate_trait_tokens=False, position="response[:]",
                  tokenizer=None, use_chat_template=False):
    return vet(experiment, trait, model_variant, target="responses",
               pos_threshold=pos_threshold, neg_threshold=neg_threshold,
               max_concurrent=max_concurrent, estimate_trait_tokens=estimate_trait_tokens,
               position=position, tokenizer=tokenizer, use_chat_template=use_chat_template)
