#!/usr/bin/env python3
"""
Vet scenario files using LLM-as-a-judge before response generation.

Input:
    - experiments/{experiment}/extraction/{trait}/positive.txt
    - experiments/{experiment}/extraction/{trait}/negative.txt
    - experiments/{experiment}/extraction/{trait}/trait_definition.txt (optional)

Output:
    - experiments/{experiment}/extraction/{trait}/vetting/scenario_scores.json
    - Console report with pass/fail/flag counts

Usage:
    python extraction/vet_scenarios.py --experiment my_exp --trait category/my_trait
    python extraction/vet_scenarios.py --experiment my_exp --trait category/my_trait --threshold 3
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


def load_scenarios(experiment: str, trait: str) -> dict:
    """Load positive and negative scenario files."""
    trait_dir = get('extraction.trait', experiment=experiment, trait=trait)

    scenarios = {}
    for polarity in ['positive', 'negative']:
        file_path = trait_dir / f"{polarity}.txt"
        if file_path.exists():
            with open(file_path) as f:
                scenarios[polarity] = [line.strip() for line in f if line.strip()]
        else:
            scenarios[polarity] = []

    return scenarios


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


def score_scenario(
    client: genai.Client,
    scenario: str,
    trait_definition: str,
    expected_polarity: str,
    model: str = "gemini-2.5-flash"
) -> dict:
    """Score a single scenario for trait elicitation."""

    if expected_polarity == 'positive':
        question = f"Will this prompt naturally elicit HIGH {trait_definition} from a helpful AI assistant?"
    else:
        question = f"Will this prompt naturally elicit LOW {trait_definition} from a helpful AI assistant?"

    prompt = f"""You are evaluating whether a prompt will reliably elicit (or not elicit) a specific trait from an AI assistant.

Trait: {trait_definition}

Prompt to evaluate: "{scenario}"

Question: {question}

Rate from 0-10:
- 10 = Extremely likely to elicit expected behavior
- 7-9 = Very likely
- 4-6 = Uncertain/ambiguous
- 1-3 = Unlikely
- 0 = Will definitely not elicit expected behavior

Also note any potential confounds (length, complexity, topic that might affect results).

Respond in this exact format:
SCORE: [0-10]
CONFIDENCE: [low/medium/high]
REASONING: [one sentence]
CONFOUNDS: [none OR brief description]"""

    try:
        response = client.models.generate_content(
            model=model,
            contents=prompt
        )
        text = response.text.strip()

        # Parse response
        lines = text.split('\n')
        result = {
            'scenario': scenario,
            'expected_polarity': expected_polarity,
            'raw_response': text,
        }

        for line in lines:
            if line.startswith('SCORE:'):
                try:
                    result['score'] = int(line.split(':')[1].strip().split()[0])
                except:
                    result['score'] = -1
            elif line.startswith('CONFIDENCE:'):
                result['confidence'] = line.split(':')[1].strip().lower()
            elif line.startswith('REASONING:'):
                result['reasoning'] = line.split(':', 1)[1].strip()
            elif line.startswith('CONFOUNDS:'):
                result['confounds'] = line.split(':', 1)[1].strip()

        return result

    except Exception as e:
        return {
            'scenario': scenario,
            'expected_polarity': expected_polarity,
            'score': -1,
            'error': str(e)
        }


def vet_scenarios(
    experiment: str,
    trait: str,
    threshold: int = 4,
    model: str = "gemini-2.5-flash",
    delay: float = 0.1,
    sample: int = None,
):
    """
    Vet scenario files using LLM-as-a-judge.

    Args:
        experiment: Experiment name
        trait: Trait path (e.g., 'behavioral/refusal')
        threshold: Minimum score to pass (default 4)
        model: Gemini model to use
        delay: Delay between API calls (rate limiting)
        sample: Only score first N scenarios per polarity (for testing)
    """

    # Check API key
    if not os.environ.get('GEMINI_API_KEY'):
        print("Error: GEMINI_API_KEY environment variable not set")
        sys.exit(1)

    client = genai.Client()

    # Load data
    scenarios = load_scenarios(experiment, trait)
    trait_definition = load_trait_definition(experiment, trait)

    print(f"Trait: {trait}")
    print(f"Definition: {trait_definition}")
    print(f"Positive scenarios: {len(scenarios['positive'])}")
    print(f"Negative scenarios: {len(scenarios['negative'])}")
    print(f"Threshold: {threshold}")
    print()

    # Score all scenarios
    all_results = []

    for polarity in ['positive', 'negative']:
        polarity_scenarios = scenarios[polarity]
        if sample:
            polarity_scenarios = polarity_scenarios[:sample]

        print(f"Scoring {polarity} scenarios...")
        for scenario in tqdm(polarity_scenarios):
            result = score_scenario(
                client=client,
                scenario=scenario,
                trait_definition=trait_definition,
                expected_polarity=polarity,
                model=model
            )
            all_results.append(result)
            time.sleep(delay)

    # Analyze results
    passed = [r for r in all_results if r.get('score', -1) >= 7]
    flagged = [r for r in all_results if 4 <= r.get('score', -1) < 7]
    failed = [r for r in all_results if 0 <= r.get('score', -1) < threshold]
    errors = [r for r in all_results if r.get('score', -1) == -1]

    # Print summary
    print("\n" + "="*60)
    print("VETTING SUMMARY")
    print("="*60)
    print(f"Total scenarios: {len(all_results)}")
    print(f"Passed (>=7):    {len(passed)}")
    print(f"Flagged (4-6):   {len(flagged)}")
    print(f"Failed (<{threshold}):    {len(failed)}")
    print(f"Errors:          {len(errors)}")

    # Show failed scenarios
    if failed:
        print("\n" + "-"*60)
        print("FAILED SCENARIOS (consider removing):")
        print("-"*60)
        for r in failed[:10]:  # Show first 10
            print(f"  [{r.get('score', '?')}] ({r['expected_polarity']}) {r['scenario'][:60]}...")
            if r.get('reasoning'):
                print(f"      Reason: {r['reasoning']}")

    # Show flagged scenarios
    if flagged:
        print("\n" + "-"*60)
        print("FLAGGED SCENARIOS (review manually):")
        print("-"*60)
        for r in flagged[:10]:  # Show first 10
            print(f"  [{r.get('score', '?')}] ({r['expected_polarity']}) {r['scenario'][:60]}...")

    # Show confounds
    confounds = [r for r in all_results if r.get('confounds') and r.get('confounds').lower() != 'none']
    if confounds:
        print("\n" + "-"*60)
        print("SCENARIOS WITH CONFOUNDS:")
        print("-"*60)
        for r in confounds[:10]:
            print(f"  {r['scenario'][:50]}...")
            print(f"      Confound: {r['confounds']}")

    # Save results
    output_dir = get('extraction.trait', experiment=experiment, trait=trait) / "vetting"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "scenario_scores.json"

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
        },
        'results': all_results,
    }

    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\nResults saved to: {output_file}")

    # Return pass rate
    valid = [r for r in all_results if r.get('score', -1) >= 0]
    if valid:
        pass_rate = len(passed) / len(valid)
        print(f"\nPass rate: {pass_rate:.1%}")
        return pass_rate
    return 0.0


if __name__ == "__main__":
    fire.Fire(vet_scenarios)
