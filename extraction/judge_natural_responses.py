#!/usr/bin/env python3
"""
Retrospectively judge natural extraction responses using trait eval_prompt.
Adds trait_score field to existing pos.json and neg.json files.
"""

import os
import sys
import json
import argparse
import asyncio
from pathlib import Path
from tqdm.asyncio import tqdm_asyncio

# Add utils to path
sys.path.append(str(Path(__file__).parent.parent))
from utils.judge import OpenAiJudge

async def judge_natural_responses(
    experiment,
    trait,
    judge_model='gpt-4o-mini',
    overwrite=False
):
    """Judge natural responses and add scores to JSON files"""

    print("="*80)
    print(f"JUDGING NATURAL RESPONSES")
    print(f"Experiment: {experiment}")
    print(f"Trait: {trait}")
    print(f"Judge Model: {judge_model}")
    print("="*80)

    # Setup paths
    base_dir = Path('experiments') / experiment / trait / 'extraction'
    responses_dir = base_dir / 'responses'

    # Natural extraction doesn't have trait_definition.json - use instruction-based version
    # Try natural dir first, fall back to instruction-based (remove _natural suffix)
    trait_def_path = base_dir / 'trait_definition.json'
    if not trait_def_path.exists():
        # Try instruction-based version (remove _natural suffix)
        base_trait = trait.replace('_natural', '')
        trait_def_path = Path('experiments') / experiment / base_trait / 'extraction' / 'trait_definition.json'

    # Check files exist
    if not trait_def_path.exists():
        raise FileNotFoundError(f"Trait definition not found. Tried: {base_dir / 'trait_definition.json'} and {trait_def_path}")

    pos_file = responses_dir / 'pos.json'
    neg_file = responses_dir / 'neg.json'

    if not pos_file.exists() or not neg_file.exists():
        raise FileNotFoundError(f"Response files not found in {responses_dir}")

    # Load trait definition
    with open(trait_def_path) as f:
        trait_def = json.load(f)

    eval_prompt = trait_def.get('eval_prompt')
    if not eval_prompt:
        raise ValueError(f"No eval_prompt found in {trait_def_path}")

    print(f"\nEval prompt preview:")
    print(f"{eval_prompt[:200]}...")

    # Load responses
    with open(pos_file) as f:
        pos_responses = json.load(f)
    with open(neg_file) as f:
        neg_responses = json.load(f)

    print(f"\nLoaded responses:")
    print(f"  Positive: {len(pos_responses)}")
    print(f"  Negative: {len(neg_responses)}")

    # Check if already scored
    if not overwrite:
        if any('trait_score' in r for r in pos_responses):
            print(f"\n⚠️  Positive responses already have scores!")
            print(f"   Use --overwrite to re-judge")
            return
        if any('trait_score' in r for r in neg_responses):
            print(f"\n⚠️  Negative responses already have scores!")
            print(f"   Use --overwrite to re-judge")
            return

    # Initialize judge
    judge = OpenAiJudge(
        model=judge_model,
        prompt_template=eval_prompt,
        eval_type='0_100'
    )

    # Judge responses
    async def judge_response(response):
        """Judge a single response and add score"""
        try:
            score = await judge.judge(
                question=response['question'],
                answer=response['answer']
            )
            response['trait_score'] = score
            return response
        except Exception as e:
            print(f"\n⚠️  Error judging response: {e}")
            response['trait_score'] = None
            return response

    print("\nJudging positive responses...")
    pos_responses = await tqdm_asyncio.gather(*[
        judge_response(r) for r in pos_responses
    ], desc="Positive")

    print("\nJudging negative responses...")
    neg_responses = await tqdm_asyncio.gather(*[
        judge_response(r) for r in neg_responses
    ], desc="Negative")

    # Calculate statistics
    pos_scores = [r['trait_score'] for r in pos_responses if r['trait_score'] is not None]
    neg_scores = [r['trait_score'] for r in neg_responses if r['trait_score'] is not None]

    pos_mean = sum(pos_scores) / len(pos_scores) if pos_scores else 0
    neg_mean = sum(neg_scores) / len(neg_scores) if neg_scores else 0
    separation = pos_mean - neg_mean

    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print(f"Positive mean score: {pos_mean:.1f}")
    print(f"Negative mean score: {neg_mean:.1f}")
    print(f"Separation: {separation:.1f}")
    print()

    quality = "Excellent" if separation > 40 else "Good" if separation > 20 else "Weak"
    print(f"Quality: {quality}")
    print("="*80)

    # Save updated responses
    print(f"\nSaving scored responses...")
    with open(pos_file, 'w') as f:
        json.dump(pos_responses, f, indent=2)

    with open(neg_file, 'w') as f:
        json.dump(neg_responses, f, indent=2)

    print(f"✓ Saved to {pos_file}")
    print(f"✓ Saved to {neg_file}")

    # Save summary
    summary = {
        'trait': trait,
        'experiment': experiment,
        'judge_model': judge_model,
        'n_pos': len(pos_responses),
        'n_neg': len(neg_responses),
        'pos_mean': pos_mean,
        'neg_mean': neg_mean,
        'separation': separation,
        'quality': quality
    }

    summary_file = responses_dir / 'judge_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"✓ Saved summary to {summary_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Judge natural extraction responses')
    parser.add_argument('--experiment', required=True, help='Experiment name')
    parser.add_argument('--trait', required=True, help='Trait name (should end with _natural)')
    parser.add_argument('--judge-model', default='gpt-4o-mini', help='Judge model')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing scores')

    args = parser.parse_args()

    asyncio.run(judge_natural_responses(
        experiment=args.experiment,
        trait=args.trait,
        judge_model=args.judge_model,
        overwrite=args.overwrite
    ))
