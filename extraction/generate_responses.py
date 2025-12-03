#!/usr/bin/env python3
"""
Stage 1: Generate responses for natural elicitation.

Input:
    - experiments/{experiment}/extraction/{category}/{trait}/positive.txt
    - experiments/{experiment}/extraction/{category}/{trait}/negative.txt

Output:
    - experiments/{experiment}/extraction/{category}/{trait}/responses/pos.json
    - experiments/{experiment}/extraction/{category}/{trait}/responses/neg.json
    - experiments/{experiment}/extraction/{category}/{trait}/generation_metadata.json

Usage:
    # Single trait
    python extraction/generate_responses.py --experiment my_exp --trait category/my_trait

    # All traits
    python extraction/generate_responses.py --experiment my_exp --trait all
"""

import sys
import json
import argparse
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.paths import get as get_path
from utils.model import load_model, DEFAULT_MODEL


def discover_traits(experiment: str) -> list[str]:
    """Find all traits with positive.txt and negative.txt files."""
    extraction_dir = get_path('extraction.base', experiment=experiment)
    traits = []
    if not extraction_dir.is_dir():
        return []

    for category_dir in extraction_dir.iterdir():
        if not category_dir.is_dir() or category_dir.name.startswith('.'):
            continue
        for trait_dir in category_dir.iterdir():
            if not trait_dir.is_dir():
                continue
            # Check for scenario files
            if (trait_dir / 'positive.txt').exists() and (trait_dir / 'negative.txt').exists():
                traits.append(f"{category_dir.name}/{trait_dir.name}")
    return sorted(traits)


def load_scenarios(scenario_file: Path) -> list[str]:
    """Load scenarios from text file (one per line)."""
    with open(scenario_file, 'r') as f:
        scenarios = [line.strip() for line in f if line.strip()]
    return scenarios


def generate_responses_for_trait(
    experiment: str,
    trait: str,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    max_new_tokens: int = 200,
    batch_size: int = 8,
    rollouts: int = 1,
    temperature: float = 0.0,
    chat_template: bool = False,
    base_model: bool = False,
) -> tuple[int, int]:
    """
    Generate responses for scenarios.

    Args:
        experiment: Experiment name.
        trait: Trait path like "category/trait_name".
        model: Pre-loaded HuggingFace model.
        tokenizer: Pre-loaded HuggingFace tokenizer.
        max_new_tokens: Max tokens to generate per response.
        batch_size: Batch size for generation.
        rollouts: Number of responses per scenario (1 for natural, 10 for instruction-based).
        temperature: Sampling temperature (0.0 for deterministic, 1.0 for diverse).
        chat_template: If True, wrap prompts in chat template (for instruct models like Qwen).
        base_model: If True, use text completion mode (no chat template, track prefix length).

    Returns:
        Tuple of (n_positive, n_negative) responses generated.
    """
    print(f"  [Stage 1] Generating responses for '{trait}'...")

    trait_dir = get_path('extraction.trait', experiment=experiment, trait=trait)
    responses_dir = get_path('extraction.responses', experiment=experiment, trait=trait)
    responses_dir.mkdir(parents=True, exist_ok=True)

    # Load scenario files
    pos_file = trait_dir / 'positive.txt'
    neg_file = trait_dir / 'negative.txt'

    if not pos_file.exists() or not neg_file.exists():
        print(f"    ERROR: Scenario files not found in {trait_dir}")
        print(f"    Expected: positive.txt, negative.txt")
        return 0, 0

    pos_scenarios = load_scenarios(pos_file)
    neg_scenarios = load_scenarios(neg_file)

    do_sample = temperature > 0

    if rollouts > 1:
        print(f"    Rollouts: {rollouts}, Temperature: {temperature}")

    def generate_batch(scenarios: list[str], label: str) -> list[dict]:
        results = []
        total_to_generate = len(scenarios) * rollouts
        desc = f"    Generating {label} ({total_to_generate} total)"

        with tqdm(total=total_to_generate, desc=desc, leave=False) as pbar:
            for i in range(0, len(scenarios), batch_size):
                batch_scenarios = scenarios[i:i + batch_size]

                # Apply chat template if requested (for instruct models)
                if chat_template:
                    batch_prompts = [
                        tokenizer.apply_chat_template(
                            [{"role": "user", "content": s}],
                            tokenize=False,
                            add_generation_prompt=True
                        ) for s in batch_scenarios
                    ]
                else:
                    batch_prompts = batch_scenarios

                # Generate rollouts for this batch
                for rollout_idx in range(rollouts):
                    inputs = tokenizer(
                        batch_prompts,
                        return_tensors='pt',
                        padding=True,
                        truncation=True,
                        max_length=512
                    ).to(model.device)

                    with torch.no_grad():
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=max_new_tokens,
                            do_sample=do_sample,
                            temperature=temperature if do_sample else None,
                            pad_token_id=tokenizer.eos_token_id
                        )

                    for j, output in enumerate(outputs):
                        prompt_length = inputs['input_ids'][j].shape[0]
                        response = tokenizer.decode(output[prompt_length:], skip_special_tokens=True)
                        results.append({
                            'scenario_idx': i + j,
                            'rollout_idx': rollout_idx,
                            'prompt': batch_scenarios[j],
                            'response': response.strip(),
                            'full_text': tokenizer.decode(output, skip_special_tokens=True),
                            'prompt_token_count': prompt_length,
                        })
                    pbar.update(len(batch_scenarios))
        return results

    pos_results = generate_batch(pos_scenarios, 'positive')
    neg_results = generate_batch(neg_scenarios, 'negative')

    # Save results
    with open(responses_dir / 'pos.json', 'w') as f:
        json.dump(pos_results, f, indent=2)
    with open(responses_dir / 'neg.json', 'w') as f:
        json.dump(neg_results, f, indent=2)

    # Save metadata
    metadata = {
        'experiment': experiment,
        'trait': trait,
        'model_name': model.config.name_or_path,
        'n_scenarios_pos': len(pos_scenarios),
        'n_scenarios_neg': len(neg_scenarios),
        'rollouts': rollouts,
        'temperature': temperature,
        'n_positive': len(pos_results),
        'n_negative': len(neg_results),
        'total': len(pos_results) + len(neg_results),
        'base_model': base_model,
        'chat_template': chat_template,
    }
    with open(trait_dir / 'generation_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"    Saved {len(pos_results)} positive and {len(neg_results)} negative responses.")
    return len(pos_results), len(neg_results)


def main():
    parser = argparse.ArgumentParser(description='Generate natural elicitation responses.')
    parser.add_argument('--experiment', type=str, required=True, help='Experiment name')
    parser.add_argument('--trait', type=str, required=True,
                        help='Trait name (e.g., "category/my_trait") or "all" for all traits')
    parser.add_argument('--model', type=str, default=DEFAULT_MODEL, help='Model name')
    parser.add_argument('--max-new-tokens', type=int, default=200, help='Max tokens to generate')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size for generation')
    parser.add_argument('--device', type=str, default='auto', help='Device (auto, cuda, cpu, mps)')
    parser.add_argument('--rollouts', type=int, default=1,
                        help='Responses per scenario (1 for natural, 10 for instruction-based)')
    parser.add_argument('--temperature', type=float, default=0.0,
                        help='Sampling temperature (0.0 for deterministic, 1.0 for diverse)')
    parser.add_argument('--chat-template', action='store_true',
                        help='Apply chat template (for instruct models like Qwen, Llama)')
    parser.add_argument('--base-model', action='store_true',
                        help='Base model mode (text completion, no chat template)')

    args = parser.parse_args()

    # Validate: base-model and chat-template are mutually exclusive
    if args.base_model and args.chat_template:
        print("ERROR: --base-model and --chat-template are mutually exclusive")
        return

    # Determine traits to process
    if args.trait.lower() == 'all':
        traits = discover_traits(args.experiment)
        if not traits:
            print(f"No traits found in experiment '{args.experiment}'")
            return
        print(f"Found {len(traits)} traits to process")
    else:
        traits = [args.trait]

    print("=" * 80)
    print("GENERATE RESPONSES")
    print(f"Experiment: {args.experiment}")
    print(f"Traits: {len(traits)}")
    print(f"Model: {args.model}")
    print("=" * 80)

    # Load model and tokenizer once
    model, tokenizer = load_model(args.model, device=args.device)

    # Process each trait
    total_pos, total_neg = 0, 0
    for trait in traits:
        n_pos, n_neg = generate_responses_for_trait(
            experiment=args.experiment,
            trait=trait,
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=args.max_new_tokens,
            batch_size=args.batch_size,
            rollouts=args.rollouts,
            temperature=args.temperature,
            chat_template=args.chat_template,
            base_model=args.base_model,
        )
        total_pos += n_pos
        total_neg += n_neg

    print(f"\nDONE: Generated {total_pos} positive + {total_neg} negative = {total_pos + total_neg} total responses.")


if __name__ == '__main__':
    main()
