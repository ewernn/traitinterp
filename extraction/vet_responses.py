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
    python extraction/vet_responses.py --experiment my_exp --trait category/my_trait
    python extraction/vet_responses.py --experiment my_exp --trait category/my_trait --threshold 3
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import os
import time
import fire
from tqdm import tqdm
from google import genai

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


def score_response(
    client: genai.Client,
    prompt: str,
    response: str,
    trait_definition: str,
    expected_polarity: str,
    model: str = "gemini-2.5-flash"
) -> dict:
    """Score a single response for trait exhibition."""

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


def vet_responses(
    experiment: str,
    trait: str,
    threshold: int = 4,
    model: str = "gemini-2.5-flash",
    delay: float = 0.1,
    sample: int = None,
):
    """
    Vet generated responses using LLM-as-a-judge.

    Args:
        experiment: Experiment name
        trait: Trait path (e.g., 'behavioral/refusal')
        threshold: Minimum score to pass (default 4)
        model: Gemini model to use
        delay: Delay between API calls (rate limiting)
        sample: Only score first N responses per polarity (for testing)
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
    print(f"Definition: {trait_definition}")
    print(f"Positive responses: {len(responses['positive'])}")
    print(f"Negative responses: {len(responses['negative'])}")
    print(f"Threshold: {threshold}")
    print()

    # Score all responses
    all_results = []

    for polarity in ['positive', 'negative']:
        polarity_responses = responses[polarity]
        if sample:
            polarity_responses = polarity_responses[:sample]

        print(f"Scoring {polarity} responses...")
        for item in tqdm(polarity_responses):
            # Support both 'prompt'/'response' and 'question'/'answer' formats
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
            result['original_index'] = polarity_responses.index(item)
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

    # Show mislabeled examples (model didn't behave as expected)
    # Mislabeled = model didn't exhibit expected behavior
    # For positive: asked "exhibit HIGH?" - if "no" → mislabeled
    # For negative: asked "exhibit LOW?" - if "no" → mislabeled
    mislabeled = [r for r in all_results if r.get('exhibited_trait') == 'no']

    if mislabeled:
        print("\n" + "-"*60)
        print("MISLABELED EXAMPLES (model didn't behave as expected):")
        print("-"*60)
        for r in mislabeled[:10]:  # Show first 10
            print(f"  [{r.get('score', '?')}] Expected: {r['expected_polarity']}, Got: {r.get('exhibited_trait', '?')}")
            print(f"      Prompt: {r['prompt'][:50]}...")
            print(f"      Response: {r['response'][:50]}...")
            if r.get('reasoning'):
                print(f"      Reason: {r['reasoning']}")
            print()

    # Show failed examples
    if failed and failed != mislabeled:
        print("\n" + "-"*60)
        print("OTHER FAILED RESPONSES:")
        print("-"*60)
        for r in failed[:5]:
            if r not in mislabeled:
                print(f"  [{r.get('score', '?')}] {r['prompt'][:50]}...")

    # Save results
    output_dir = get('extraction.trait', experiment=experiment, trait=trait) / "vetting"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "response_scores.json"

    # Identify indices to exclude
    failed_indices = {
        'positive': [r['original_index'] for r in failed if r['expected_polarity'] == 'positive'],
        'negative': [r['original_index'] for r in failed if r['expected_polarity'] == 'negative'],
    }

    output_data = {
        'experiment': experiment,
        'trait': trait,
        'trait_definition': trait_definition,
        'model': model,
        'threshold': threshold,
        'summary': {
            'total': len(all_results),
            'passed': len(passed),
            'flagged': len(flagged),
            'failed': len(failed),
            'errors': len(errors),
            'mislabeled': len(mislabeled),
        },
        'failed_indices': failed_indices,
        'results': all_results,
    }

    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\nResults saved to: {output_file}")
    print(f"Failed indices saved - use these to filter during activation extraction")

    # Return pass rate
    valid = [r for r in all_results if r.get('score', -1) >= 0]
    if valid:
        pass_rate = len(passed) / len(valid)
        print(f"\nPass rate: {pass_rate:.1%}")
        return pass_rate
    return 0.0


if __name__ == "__main__":
    fire.Fire(vet_responses)
