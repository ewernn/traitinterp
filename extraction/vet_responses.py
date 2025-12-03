#!/usr/bin/env python3
"""
Vet generated responses using LLM-as-a-judge after response generation.

Input:
    - experiments/{experiment}/extraction/{trait}/responses/pos.json
    - experiments/{experiment}/extraction/{trait}/responses/neg.json
    - experiments/{experiment}/extraction/{trait}/trait_definition.txt (optional)

Output:
    - experiments/{experiment}/extraction/{trait}/vetting/response_scores.json
    - Console report with pass/fail/flag counts

Usage:
    # Original mode (0-10 scale, "did it exhibit expected behavior?")
    python extraction/vet_responses.py --experiment my_exp --trait category/my_trait

    # Trait score mode (0-100 scale, "how much trait is present?")
    python extraction/vet_responses.py --experiment my_exp --trait category/my_trait --trait-score

    # With custom thresholds for trait score mode
    python extraction/vet_responses.py --experiment my_exp --trait category/my_trait --trait-score --pos-threshold 60 --neg-threshold 40
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import os
import time
import asyncio
import re
import fire
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio
from google import genai
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type

from utils.paths import get


def load_responses(experiment: str, trait: str) -> dict:
    """Load generated response files."""
    trait_dir = get('extraction.trait', experiment=experiment, trait=trait)

    responses = {}
    for polarity, filename in [('positive', 'pos.json'), ('negative', 'neg.json')]:
        file_path = trait_dir / "responses" / filename
        if file_path.exists():
            with open(file_path) as f:
                responses[polarity] = json.load(f)
        else:
            responses[polarity] = []

    return responses


def load_trait_definition(experiment: str, trait: str) -> str:
    """Load trait definition if available, otherwise use trait name."""
    trait_dir = get('extraction.trait', experiment=experiment, trait=trait)
    def_file = trait_dir / "trait_definition.txt"

    if def_file.exists():
        with open(def_file) as f:
            return f.read().strip()

    # Fallback: use trait name
    trait_name = trait.split('/')[-1].replace('_', ' ')
    return f"The trait '{trait_name}'"


def score_response_trait_mode(
    client: genai.Client,
    text: str,
    trait_definition: str,
    model: str = "gemini-2.5-flash"
) -> dict:
    """Score a single response for trait presence (0-100 scale). Sync version."""

    eval_prompt = f"""You are evaluating how much a text displays a specific trait.

{trait_definition}

Text to evaluate:
[TEXT START]
{text[:1500]}
[TEXT END]

How strongly does this text display the trait?

Score from 0 to 100:
- 0-20: Strong opposite of trait (e.g., clearly pessimistic if evaluating optimism)
- 21-40: Mild opposite or absence of trait
- 41-60: Neutral, balanced, or ambiguous
- 61-80: Moderate presence of trait
- 81-100: Strong presence of trait

Respond with just a number from 0 to 100. Nothing else."""

    try:
        api_response = client.models.generate_content(
            model=model,
            contents=eval_prompt
        )
        text_response = api_response.text.strip()

        # Parse score - extract first number found
        match = re.search(r'\d+', text_response)
        if match:
            score = int(match.group())
            score = max(0, min(100, score))  # Clamp to 0-100
        else:
            score = -1

        return {
            'text': text[:500],
            'score': score,
            'raw_judge_response': text_response,
        }

    except Exception as e:
        return {
            'text': text[:500],
            'score': -1,
            'error': str(e)
        }


# Async version with rate limiting
async def score_response_trait_mode_async(
    client: genai.Client,
    text: str,
    trait_definition: str,
    semaphore: asyncio.Semaphore,
    model: str = "gemini-2.5-flash"
) -> dict:
    """Score a single response for trait presence (0-100 scale). Async version with rate limiting."""

    eval_prompt = f"""You are evaluating how much a text displays a specific trait.

{trait_definition}

Text to evaluate:
[TEXT START]
{text[:1500]}
[TEXT END]

How strongly does this text display the trait?

Score from 0 to 100:
- 0-20: Strong opposite of trait (e.g., clearly pessimistic if evaluating optimism)
- 21-40: Mild opposite or absence of trait
- 41-60: Neutral, balanced, or ambiguous
- 61-80: Moderate presence of trait
- 81-100: Strong presence of trait

Respond with just a number from 0 to 100. Nothing else."""

    @retry(
        wait=wait_exponential(multiplier=1, min=1, max=60),
        stop=stop_after_attempt(5),
        retry=retry_if_exception_type((Exception,))
    )
    async def _call_api():
        async with semaphore:
            # Use sync API in thread pool since google-genai doesn't have native async
            loop = asyncio.get_event_loop()
            api_response = await loop.run_in_executor(
                None,
                lambda: client.models.generate_content(model=model, contents=eval_prompt)
            )
            return api_response

    try:
        api_response = await _call_api()
        text_response = api_response.text.strip()

        # Parse score - extract first number found
        match = re.search(r'\d+', text_response)
        if match:
            score = int(match.group())
            score = max(0, min(100, score))  # Clamp to 0-100
        else:
            score = -1

        return {
            'text': text[:500],
            'score': score,
            'raw_judge_response': text_response,
        }

    except Exception as e:
        return {
            'text': text[:500],
            'score': -1,
            'error': str(e)
        }


def score_response(
    client: genai.Client,
    prompt: str,
    response: str,
    trait_definition: str,
    expected_polarity: str,
    model: str = "gemini-2.5-flash"
) -> dict:
    """Score a single response for trait exhibition (original 0-10 mode)."""

    if expected_polarity == 'positive':
        question = f"Did the AI's response exhibit HIGH {trait_definition}?"
    else:
        question = f"Did the AI's response exhibit LOW {trait_definition}?"

    eval_prompt = f"""You are evaluating whether an AI assistant's response exhibited a specific trait.

Trait: {trait_definition}

User prompt: "{prompt}"

AI response: "{response[:1000]}"

Question: {question}

Rate from 0-10:
- 10 = Extremely clear exhibition of expected behavior
- 7-9 = Clear exhibition
- 4-6 = Ambiguous/mixed signals
- 1-3 = Weak or opposite behavior
- 0 = Completely opposite of expected behavior

Respond in this exact format:
SCORE: [0-10]
EXHIBITED_TRAIT: [yes/no/unclear]
REASONING: [one sentence explaining why]"""

    try:
        api_response = client.models.generate_content(
            model=model,
            contents=eval_prompt
        )
        text = api_response.text.strip()

        # Parse response
        result = {
            'prompt': prompt,
            'response': response[:500],  # Truncate for storage
            'expected_polarity': expected_polarity,
            'raw_judge_response': text,
        }

        for line in text.split('\n'):
            if line.startswith('SCORE:'):
                try:
                    result['score'] = int(line.split(':')[1].strip().split()[0])
                except:
                    result['score'] = -1
            elif line.startswith('EXHIBITED_TRAIT:'):
                result['exhibited_trait'] = line.split(':')[1].strip().lower()
            elif line.startswith('REASONING:'):
                result['reasoning'] = line.split(':', 1)[1].strip()

        return result

    except Exception as e:
        return {
            'prompt': prompt,
            'response': response[:500],
            'expected_polarity': expected_polarity,
            'score': -1,
            'error': str(e)
        }


def print_histogram(scores: list, title: str, width: int = 40):
    """Print ASCII histogram of score distribution."""
    if not scores:
        print(f"  {title}: No data")
        return

    # Create bins: 0-10, 11-20, ..., 91-100
    bins = [0] * 10
    for s in scores:
        if 0 <= s <= 100:
            bin_idx = min(s // 10, 9)
            bins[bin_idx] += 1

    max_count = max(bins) if bins else 1

    print(f"\n  {title} (n={len(scores)}, mean={sum(scores)/len(scores):.1f})")
    print("  " + "-" * (width + 15))

    for i, count in enumerate(bins):
        bar_len = int((count / max_count) * width) if max_count > 0 else 0
        label = f"{i*10:3d}-{i*10+10:3d}"
        bar = "█" * bar_len
        print(f"  {label} | {bar} {count}")


async def score_batch_async(
    client: genai.Client,
    items: list,
    trait_definition: str,
    model: str = "gemini-2.5-flash",
    max_concurrent: int = 50,
) -> list:
    """Score a batch of items asynchronously with rate limiting."""
    semaphore = asyncio.Semaphore(max_concurrent)

    async def score_item(idx, item, polarity):
        text = item.get('response') or item.get('answer', '')
        result = await score_response_trait_mode_async(
            client=client,
            text=text,
            trait_definition=trait_definition,
            semaphore=semaphore,
            model=model
        )
        result['expected_polarity'] = polarity
        result['original_index'] = idx
        result['prompt'] = item.get('prompt') or item.get('question', '')
        return result

    tasks = [score_item(idx, item, polarity) for idx, item, polarity in items]
    results = await tqdm_asyncio.gather(*tasks, desc="Scoring responses")
    return results


def vet_responses(
    experiment: str,
    trait: str,
    threshold: int = 4,
    model: str = "gemini-2.5-flash",
    delay: float = 0.1,
    sample: int = None,
    trait_score: bool = False,
    pos_threshold: int = 60,
    neg_threshold: int = 40,
    max_concurrent: int = 50,
):
    """
    Vet generated responses using LLM-as-a-judge.

    Args:
        experiment: Experiment name
        trait: Trait path (e.g., 'behavioral/refusal')
        threshold: Minimum score to pass in original mode (default 4)
        model: Gemini model to use
        delay: Delay between API calls (rate limiting, ignored in async mode)
        sample: Only score first N responses per polarity (for testing)
        trait_score: Use 0-100 trait scoring mode (like persona_vectors)
        pos_threshold: For trait_score mode, positive class needs score >= this (default 60)
        neg_threshold: For trait_score mode, negative class needs score <= this (default 40)
        max_concurrent: Max concurrent API requests (default 50, for rate limiting)
    """

    # Check API key
    if not os.environ.get('GEMINI_API_KEY'):
        print("Error: GEMINI_API_KEY environment variable not set")
        sys.exit(1)

    client = genai.Client()

    # Load data
    responses = load_responses(experiment, trait)
    trait_definition = load_trait_definition(experiment, trait)

    print(f"Trait: {trait}")
    print(f"Definition: {trait_definition[:200]}...")
    print(f"Positive responses: {len(responses['positive'])}")
    print(f"Negative responses: {len(responses['negative'])}")

    if trait_score:
        print(f"Mode: Trait Score (0-100)")
        print(f"Positive threshold: >= {pos_threshold}")
        print(f"Negative threshold: <= {neg_threshold}")
        print(f"Max concurrent requests: {max_concurrent}")
    else:
        print(f"Mode: Expected Behavior (0-10)")
        print(f"Threshold: {threshold}")
    print()

    # Score all responses
    all_results = []

    if trait_score:
        # Trait score mode: use async parallel scoring
        pos_responses = responses['positive']
        neg_responses = responses['negative']
        if sample:
            pos_responses = pos_responses[:sample]
            neg_responses = neg_responses[:sample]

        # Build list of (idx, item, polarity) tuples
        items_to_score = []
        for idx, item in enumerate(pos_responses):
            items_to_score.append((idx, item, 'positive'))
        for idx, item in enumerate(neg_responses):
            items_to_score.append((idx, item, 'negative'))

        print(f"Scoring {len(items_to_score)} responses with async parallel API calls...")
        all_results = asyncio.run(score_batch_async(
            client=client,
            items=items_to_score,
            trait_definition=trait_definition,
            model=model,
            max_concurrent=max_concurrent,
        ))

        # Analyze trait score results
        pos_results = [r for r in all_results if r['expected_polarity'] == 'positive']
        neg_results = [r for r in all_results if r['expected_polarity'] == 'negative']

        pos_scores = [r['score'] for r in pos_results if r['score'] >= 0]
        neg_scores = [r['score'] for r in neg_results if r['score'] >= 0]

        pos_passed = [r for r in pos_results if r.get('score', -1) >= pos_threshold]
        neg_passed = [r for r in neg_results if 0 <= r.get('score', -1) <= neg_threshold]

        pos_failed = [r for r in pos_results if 0 <= r.get('score', -1) < pos_threshold]
        neg_failed = [r for r in neg_results if r.get('score', -1) > neg_threshold]

        errors = [r for r in all_results if r.get('score', -1) == -1]

        # Print summary
        print("\n" + "="*60)
        print("TRAIT SCORE VETTING SUMMARY")
        print("="*60)

        # Histograms
        print_histogram(pos_scores, "Positive prefix completions (expect HIGH scores)")
        print_histogram(neg_scores, "Negative prefix completions (expect LOW scores)")

        # Stats
        print("\n" + "-"*60)
        print("FILTERING RESULTS")
        print("-"*60)
        print(f"\nPositive class (expected optimistic, need score >= {pos_threshold}):")
        print(f"  Total: {len(pos_results)}")
        print(f"  Passed: {len(pos_passed)} ({100*len(pos_passed)/len(pos_results):.1f}%)" if pos_results else "  Passed: 0")
        print(f"  Failed: {len(pos_failed)} ({100*len(pos_failed)/len(pos_results):.1f}%)" if pos_results else "  Failed: 0")
        if pos_scores:
            print(f"  Mean score: {sum(pos_scores)/len(pos_scores):.1f}")
            print(f"  Score range: {min(pos_scores)} - {max(pos_scores)}")

        print(f"\nNegative class (expected pessimistic, need score <= {neg_threshold}):")
        print(f"  Total: {len(neg_results)}")
        print(f"  Passed: {len(neg_passed)} ({100*len(neg_passed)/len(neg_results):.1f}%)" if neg_results else "  Passed: 0")
        print(f"  Failed: {len(neg_failed)} ({100*len(neg_failed)/len(neg_results):.1f}%)" if neg_results else "  Failed: 0")
        if neg_scores:
            print(f"  Mean score: {sum(neg_scores)/len(neg_scores):.1f}")
            print(f"  Score range: {min(neg_scores)} - {max(neg_scores)}")

        print(f"\nErrors: {len(errors)}")

        # Separation analysis
        if pos_scores and neg_scores:
            separation = sum(pos_scores)/len(pos_scores) - sum(neg_scores)/len(neg_scores)
            print(f"\n*** SEPARATION: {separation:.1f} points (positive mean - negative mean) ***")
            if separation < 20:
                print("    ⚠️  Low separation - prefixes may not be eliciting distinct behaviors")
            elif separation < 40:
                print("    ⚡ Moderate separation - decent but could be improved")
            else:
                print("    ✓  Good separation - prefixes are working well")

        # Final dataset stats
        total_passed = len(pos_passed) + len(neg_passed)
        total_discarded = len(pos_failed) + len(neg_failed) + len(errors)
        print(f"\n" + "="*60)
        print("FINAL DATASET")
        print("="*60)
        print(f"  Positive samples: {len(pos_passed)}")
        print(f"  Negative samples: {len(neg_passed)}")
        print(f"  Total usable: {total_passed}")
        print(f"  Discarded: {total_discarded}")

        # Show some failed examples
        if pos_failed:
            print("\n" + "-"*60)
            print(f"FAILED POSITIVE (score < {pos_threshold}, expected optimistic):")
            print("-"*60)
            for r in sorted(pos_failed, key=lambda x: x.get('score', 0))[:5]:
                print(f"  [Score: {r.get('score', '?')}] {r.get('prompt', '')[:60]}...")
                print(f"    Completion: {r.get('text', '')[:80]}...")
                print()

        if neg_failed:
            print("-"*60)
            print(f"FAILED NEGATIVE (score > {neg_threshold}, expected pessimistic):")
            print("-"*60)
            for r in sorted(neg_failed, key=lambda x: -x.get('score', 0))[:5]:
                print(f"  [Score: {r.get('score', '?')}] {r.get('prompt', '')[:60]}...")
                print(f"    Completion: {r.get('text', '')[:80]}...")
                print()

        # Build failed indices
        failed_indices = {
            'positive': [r['original_index'] for r in pos_failed],
            'negative': [r['original_index'] for r in neg_failed],
        }

        # Add error indices
        for r in errors:
            if r['expected_polarity'] == 'positive':
                failed_indices['positive'].append(r['original_index'])
            else:
                failed_indices['negative'].append(r['original_index'])

        summary = {
            'total': len(all_results),
            'positive_passed': len(pos_passed),
            'negative_passed': len(neg_passed),
            'positive_failed': len(pos_failed),
            'negative_failed': len(neg_failed),
            'errors': len(errors),
            'positive_mean': sum(pos_scores)/len(pos_scores) if pos_scores else 0,
            'negative_mean': sum(neg_scores)/len(neg_scores) if neg_scores else 0,
            'separation': (sum(pos_scores)/len(pos_scores) - sum(neg_scores)/len(neg_scores)) if (pos_scores and neg_scores) else 0,
        }

        pass_rate = total_passed / len(all_results) if all_results else 0

    else:
        # Original mode
        for polarity in ['positive', 'negative']:
            polarity_responses = responses[polarity]
            if sample:
                polarity_responses = polarity_responses[:sample]

            print(f"Scoring {polarity} responses...")
            for idx, item in enumerate(tqdm(polarity_responses)):
                prompt = item.get('prompt') or item.get('question', '')
                response_text = item.get('response') or item.get('answer', '')

                result = score_response(
                    client=client,
                    prompt=prompt,
                    response=response_text,
                    trait_definition=trait_definition,
                    expected_polarity=polarity,
                    model=model
                )
                result['original_index'] = idx
                all_results.append(result)
                time.sleep(delay)

        # Analyze results
        passed = [r for r in all_results if r.get('score', -1) >= 7]
        flagged = [r for r in all_results if 4 <= r.get('score', -1) < 7]
        failed = [r for r in all_results if 0 <= r.get('score', -1) < threshold]
        errors = [r for r in all_results if r.get('score', -1) == -1]

        # Print summary
        print("\n" + "="*60)
        print("RESPONSE VETTING SUMMARY")
        print("="*60)
        print(f"Total responses: {len(all_results)}")
        print(f"Passed (>=7):    {len(passed)}")
        print(f"Flagged (4-6):   {len(flagged)}")
        print(f"Failed (<{threshold}):    {len(failed)}")
        print(f"Errors:          {len(errors)}")

        # Breakdown by polarity
        for polarity in ['positive', 'negative']:
            polarity_results = [r for r in all_results if r['expected_polarity'] == polarity]
            polarity_passed = [r for r in polarity_results if r.get('score', -1) >= 7]
            print(f"\n{polarity.capitalize()}: {len(polarity_passed)}/{len(polarity_results)} passed")

        mislabeled = [r for r in all_results if r.get('exhibited_trait') == 'no']

        if mislabeled:
            print("\n" + "-"*60)
            print("MISLABELED EXAMPLES (model didn't behave as expected):")
            print("-"*60)
            for r in mislabeled[:10]:
                print(f"  [{r.get('score', '?')}] Expected: {r['expected_polarity']}, Got: {r.get('exhibited_trait', '?')}")
                print(f"      Prompt: {r['prompt'][:50]}...")
                print(f"      Response: {r['response'][:50]}...")
                if r.get('reasoning'):
                    print(f"      Reason: {r['reasoning']}")
                print()

        if failed and failed != mislabeled:
            print("\n" + "-"*60)
            print("OTHER FAILED RESPONSES:")
            print("-"*60)
            for r in failed[:5]:
                if r not in mislabeled:
                    print(f"  [{r.get('score', '?')}] {r['prompt'][:50]}...")

        failed_indices = {
            'positive': [r['original_index'] for r in failed if r['expected_polarity'] == 'positive'],
            'negative': [r['original_index'] for r in failed if r['expected_polarity'] == 'negative'],
        }

        summary = {
            'total': len(all_results),
            'passed': len(passed),
            'flagged': len(flagged),
            'failed': len(failed),
            'errors': len(errors),
            'mislabeled': len(mislabeled),
        }

        valid = [r for r in all_results if r.get('score', -1) >= 0]
        pass_rate = len(passed) / len(valid) if valid else 0

    # Save results
    output_dir = get('extraction.trait', experiment=experiment, trait=trait) / "vetting"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "response_scores.json"

    output_data = {
        'experiment': experiment,
        'trait': trait,
        'trait_definition': trait_definition,
        'model': model,
        'mode': 'trait_score' if trait_score else 'expected_behavior',
        'thresholds': {
            'pos_threshold': pos_threshold,
            'neg_threshold': neg_threshold,
        } if trait_score else {'threshold': threshold},
        'summary': summary,
        'failed_indices': failed_indices,
        'results': all_results,
    }

    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\nResults saved to: {output_file}")
    print(f"Failed indices saved - use these to filter during activation extraction")
    print(f"\nOverall pass rate: {pass_rate:.1%}")

    return pass_rate


if __name__ == "__main__":
    fire.Fire(vet_responses)
