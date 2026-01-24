#!/usr/bin/env python3
"""
Test scenarios for trait extraction by generating responses and scoring them.

Input: Scenarios (from trait dataset or arbitrary files) + experiment (for model)
Output: results.json with scores, responses, and pass/fail status

Usage:
    # Test a committed trait dataset
    python extraction/test_scenarios.py \
        --experiment gemma-2-2b \
        --trait rm_hack/eval_awareness \
        --workdir /tmp/eval_awareness

    # Test candidate scenarios from files
    python extraction/test_scenarios.py \
        --experiment gemma-2-2b \
        --positive /tmp/candidates_pos.txt \
        --negative /tmp/candidates_neg.txt \
        --trait rm_hack/eval_awareness \
        --workdir /tmp/eval_awareness

    # Test just positive candidates (pulls definition from trait)
    python extraction/test_scenarios.py \
        --experiment gemma-2-2b \
        --positive /tmp/candidates_pos.txt \
        --trait rm_hack/eval_awareness \
        --workdir /tmp/eval_awareness

    # Test with inline definition
    python extraction/test_scenarios.py \
        --experiment gemma-2-2b \
        --positive /tmp/candidates_pos.txt \
        --definition "HIGH: aware of evaluation. LOW: unaware of evaluation."

Output (results.json):
    {
        "positive": [
            {"idx": 0, "scenario": "...", "response": "...", "score": 75, "pass": true},
            ...
        ],
        "negative": [...],
        "summary": {
            "positive": {"total": 20, "passed": 18, "pass_rate": 0.9},
            "negative": {"total": 20, "passed": 15, "pass_rate": 0.75}
        },
        "config": {...}
    }
"""

import sys
import json
import asyncio
import argparse
from pathlib import Path
from datetime import datetime

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.paths import get_model_variant
from utils.model import load_model_with_lora
from utils.model_registry import is_base_model
from utils.generation import generate_batch
from utils.traits import load_trait_definition, load_scenarios
from utils.judge import TraitJudge


def load_scenarios_from_file(filepath: Path) -> list[dict]:
    """Load scenarios from a text file (one per line) or jsonl."""
    scenarios = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Try JSON first
            if line.startswith('{'):
                item = json.loads(line)
                scenarios.append(item)
            else:
                scenarios.append({'prompt': line})
    return scenarios


async def score_responses(
    judge: TraitJudge,
    items: list[dict],
    trait_name: str,
    trait_definition: str,
    polarity: str,
    pos_threshold: int = 60,
    neg_threshold: int = 40,
    max_concurrent: int = 20,
) -> list[dict]:
    """Score responses and determine pass/fail."""

    semaphore = asyncio.Semaphore(max_concurrent)

    async def score_one(item: dict) -> dict:
        async with semaphore:
            score = await judge.score_response(
                item['scenario'],
                item['response'],
                trait_name,
                trait_definition,
            )

            # Determine pass/fail based on polarity
            if polarity == 'positive':
                passed = score is not None and score >= pos_threshold
            else:
                passed = score is not None and score <= neg_threshold

            return {
                'idx': item['idx'],
                'scenario': item['scenario'],
                'response': item['response'],
                'score': score,
                'pass': passed,
            }

    tasks = [score_one(item) for item in items]
    return await asyncio.gather(*tasks)


def run_test(
    experiment: str,
    trait: str = None,
    positive_file: Path = None,
    negative_file: Path = None,
    definition: str = None,
    workdir: Path = None,
    max_tokens: int = 32,
    pos_threshold: int = 60,
    neg_threshold: int = 40,
    max_concurrent: int = 20,
    temperature: float = 0.0,
    load_in_8bit: bool = False,
):
    """Run scenario testing pipeline."""

    # Resolve definition
    if definition:
        trait_definition = definition
        trait_name = "trait"
    elif trait:
        trait_definition = load_trait_definition(trait)
        trait_name = trait.split('/')[-1]
    else:
        raise ValueError("Must provide --trait or --definition")

    # Load scenarios
    scenarios = {'positive': [], 'negative': []}

    if positive_file:
        scenarios['positive'] = load_scenarios_from_file(positive_file)
        print(f"Loaded {len(scenarios['positive'])} positive scenarios from {positive_file}")
    elif trait:
        try:
            loaded = load_scenarios(trait, polarity='positive')
            scenarios['positive'] = loaded['positive']
            print(f"Loaded {len(scenarios['positive'])} positive scenarios from trait")
        except FileNotFoundError:
            pass

    if negative_file:
        scenarios['negative'] = load_scenarios_from_file(negative_file)
        print(f"Loaded {len(scenarios['negative'])} negative scenarios from {negative_file}")
    elif trait:
        try:
            loaded = load_scenarios(trait, polarity='negative')
            scenarios['negative'] = loaded['negative']
            print(f"Loaded {len(scenarios['negative'])} negative scenarios from trait")
        except FileNotFoundError:
            pass

    n_total = len(scenarios['positive']) + len(scenarios['negative'])
    if n_total == 0:
        raise ValueError("No scenarios to test")

    # Setup workdir
    if workdir:
        workdir = Path(workdir)
        workdir.mkdir(parents=True, exist_ok=True)

    # Load model from experiment
    print(f"\nLoading model from experiment: {experiment}")
    variant = get_model_variant(experiment, None, mode="extraction")
    model_name = variant['model']
    lora = variant.get('lora')

    model, tokenizer = load_model_with_lora(model_name, lora_adapter=lora, load_in_8bit=load_in_8bit)
    base_model = is_base_model(model_name)
    use_chat_template = not base_model and tokenizer.chat_template is not None

    print(f"Model: {model_name} ({'BASE' if base_model else 'IT'})")
    print(f"Generating {max_tokens} tokens per scenario...")

    # Format prompts
    from utils.model import format_prompt

    def format_scenarios(scenario_list: list[dict]) -> list[str]:
        return [
            format_prompt(
                s['prompt'],
                tokenizer,
                use_chat_template=use_chat_template,
                system_prompt=s.get('system_prompt'),
            )
            for s in scenario_list
        ]

    # Generate responses
    results = {'positive': [], 'negative': []}

    for polarity in ['positive', 'negative']:
        if not scenarios[polarity]:
            continue

        print(f"\nGenerating {polarity} responses...")
        formatted = format_scenarios(scenarios[polarity])

        responses = generate_batch(
            model, tokenizer, formatted,
            max_new_tokens=max_tokens,
            temperature=temperature,
        )

        # Pair scenarios with responses
        for idx, (scenario, response) in enumerate(zip(scenarios[polarity], responses)):
            results[polarity].append({
                'idx': idx,
                'scenario': scenario['prompt'],
                'response': response,
            })

    # Cleanup model to free memory before scoring
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Score responses
    print("\nScoring responses...")
    judge = TraitJudge()

    async def score_all():
        scored = {'positive': [], 'negative': []}

        for polarity in ['positive', 'negative']:
            if not results[polarity]:
                continue

            print(f"  Scoring {len(results[polarity])} {polarity} responses...")
            scored[polarity] = await score_responses(
                judge,
                results[polarity],
                trait_name,
                trait_definition,
                polarity,
                pos_threshold,
                neg_threshold,
                max_concurrent,
            )

        await judge.close()
        return scored

    scored_results = asyncio.run(score_all())

    # Calculate summary
    summary = {}
    for polarity in ['positive', 'negative']:
        if not scored_results[polarity]:
            continue

        total = len(scored_results[polarity])
        passed = sum(1 for r in scored_results[polarity] if r['pass'])
        summary[polarity] = {
            'total': total,
            'passed': passed,
            'pass_rate': passed / total if total > 0 else 0,
        }

    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    for polarity in ['positive', 'negative']:
        if polarity not in summary:
            continue

        s = summary[polarity]
        status = "✓" if s['pass_rate'] >= 0.9 else "✗"
        print(f"{polarity.upper()}: {s['passed']}/{s['total']} pass ({s['pass_rate']:.0%}) {status}")

        # Show failures
        failures = [r for r in scored_results[polarity] if not r['pass']]
        if failures:
            print(f"\n  Failures ({len(failures)}):")
            for f in failures[:5]:  # Show first 5
                scenario_preview = f['scenario'][:60] + "..." if len(f['scenario']) > 60 else f['scenario']
                response_preview = f['response'][:60] + "..." if len(f['response']) > 60 else f['response']
                print(f"    [{f['idx']}] score={f['score']:.0f}")
                print(f"        scenario: {scenario_preview}")
                print(f"        response: {response_preview}")
            if len(failures) > 5:
                print(f"    ... and {len(failures) - 5} more")

    # Save results
    output = {
        'positive': scored_results.get('positive', []),
        'negative': scored_results.get('negative', []),
        'summary': summary,
        'config': {
            'experiment': experiment,
            'trait': trait,
            'model': model_name,
            'max_tokens': max_tokens,
            'pos_threshold': pos_threshold,
            'neg_threshold': neg_threshold,
            'temperature': temperature,
            'timestamp': datetime.now().isoformat(),
        },
        'definition': trait_definition,
    }

    if workdir:
        output_path = workdir / 'results.json'
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to: {output_path}")

    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test scenarios for trait extraction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test committed dataset
  python extraction/test_scenarios.py --experiment gemma-2-2b --trait rm_hack/eval_awareness

  # Test candidate files
  python extraction/test_scenarios.py --experiment gemma-2-2b \\
      --positive /tmp/pos.txt --negative /tmp/neg.txt \\
      --trait rm_hack/eval_awareness

  # Test with custom definition
  python extraction/test_scenarios.py --experiment gemma-2-2b \\
      --positive /tmp/pos.txt \\
      --definition "HIGH: aware. LOW: unaware."
        """
    )

    parser.add_argument("--experiment", required=True,
                        help="Experiment name (for model config)")
    parser.add_argument("--trait",
                        help="Trait path (e.g., rm_hack/eval_awareness) for definition and/or scenarios")
    parser.add_argument("--positive", type=Path,
                        help="Path to positive scenarios file (one per line or jsonl)")
    parser.add_argument("--negative", type=Path,
                        help="Path to negative scenarios file (one per line or jsonl)")
    parser.add_argument("--definition",
                        help="Inline trait definition (overrides --trait definition)")
    parser.add_argument("--workdir", type=Path,
                        help="Directory to save results.json")
    parser.add_argument("--max-tokens", type=int, default=32,
                        help="Max tokens to generate per response (default: 32)")
    parser.add_argument("--pos-threshold", type=int, default=60,
                        help="Score threshold for positive scenarios (default: 60)")
    parser.add_argument("--neg-threshold", type=int, default=40,
                        help="Score threshold for negative scenarios (default: 40)")
    parser.add_argument("--max-concurrent", type=int, default=20,
                        help="Max concurrent API requests for scoring (default: 20)")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Generation temperature (default: 0.0)")
    parser.add_argument("--load-in-8bit", action="store_true",
                        help="Load model in 8-bit quantization")

    args = parser.parse_args()

    # Validate args
    if not args.trait and not args.definition:
        parser.error("Must provide --trait or --definition")

    if not args.trait and not args.positive and not args.negative:
        parser.error("Must provide --trait or scenario files (--positive/--negative)")

    run_test(
        experiment=args.experiment,
        trait=args.trait,
        positive_file=args.positive,
        negative_file=args.negative,
        definition=args.definition,
        workdir=args.workdir,
        max_tokens=args.max_tokens,
        pos_threshold=args.pos_threshold,
        neg_threshold=args.neg_threshold,
        max_concurrent=args.max_concurrent,
        temperature=args.temperature,
        load_in_8bit=args.load_in_8bit,
    )
