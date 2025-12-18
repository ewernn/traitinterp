#!/usr/bin/env python3
"""
Generate responses for trait extraction.

Input (from datasets/):
    - datasets/traits/{trait}/positive.txt
    - datasets/traits/{trait}/negative.txt

Output (to experiment):
    - experiments/{experiment}/extraction/{trait}/responses/pos.json
    - experiments/{experiment}/extraction/{trait}/responses/neg.json
    - experiments/{experiment}/extraction/{trait}/responses/metadata.json

Usage:
    # Auto-detects base/IT model from config/models/*.yaml
    python extraction/generate_responses.py --experiment my_exp --trait category/my_trait

    # Override auto-detection
    python extraction/generate_responses.py --experiment my_exp --trait category/my_trait --base-model
    python extraction/generate_responses.py --experiment my_exp --trait category/my_trait --it-model

    # All traits
    python extraction/generate_responses.py --experiment my_exp --trait all
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.paths import get as get_path
from utils.model import load_model, format_prompt, tokenize_prompt, load_experiment_config
from utils.model_registry import is_base_model


# Default token limits
BASE_MODEL_TOKENS = 16   # Base models drift quickly
IT_MODEL_TOKENS = 100    # IT models stay coherent longer


def discover_traits(experiment: str = None) -> list[str]:
    """Find all traits with positive.txt and negative.txt files in datasets/traits/."""
    traits_dir = get_path('datasets.traits')
    traits = []
    if not traits_dir.is_dir():
        return []

    for category_dir in traits_dir.iterdir():
        if not category_dir.is_dir() or category_dir.name.startswith('.'):
            continue
        for trait_dir in category_dir.iterdir():
            if not trait_dir.is_dir():
                continue
            if (trait_dir / 'positive.txt').exists() and (trait_dir / 'negative.txt').exists():
                traits.append(f"{category_dir.name}/{trait_dir.name}")
    return sorted(traits)


def load_scenarios(scenario_file: Path) -> list[str]:
    """Load scenarios from text file (one per line)."""
    with open(scenario_file, 'r') as f:
        return [line.strip() for line in f if line.strip()]


def generate_responses_for_trait(
    experiment: str,
    trait: str,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    max_new_tokens: int,
    batch_size: int = 8,
    rollouts: int = 1,
    temperature: float = 0.0,
    chat_template: bool = False,
) -> tuple[int, int]:
    """
    Generate responses for scenarios.

    Returns:
        Tuple of (n_positive, n_negative) responses generated.
    """
    print(f"  Generating responses for '{trait}'...")

    # Read scenarios from datasets/traits/
    dataset_trait_dir = get_path('datasets.trait', trait=trait)
    pos_file = dataset_trait_dir / 'positive.txt'
    neg_file = dataset_trait_dir / 'negative.txt'

    if not pos_file.exists() or not neg_file.exists():
        print(f"    ERROR: Scenario files not found in {dataset_trait_dir}")
        return 0, 0

    pos_scenarios = load_scenarios(pos_file)
    neg_scenarios = load_scenarios(neg_file)

    # Write responses to experiment directory
    responses_dir = get_path('extraction.responses', experiment=experiment, trait=trait)
    responses_dir.mkdir(parents=True, exist_ok=True)

    do_sample = temperature > 0

    def generate_batch(scenarios: list[str], label: str) -> list[dict]:
        results = []
        total = len(scenarios) * rollouts
        desc = f"    {label} ({total})"

        with tqdm(total=total, desc=desc, leave=False) as pbar:
            for i in range(0, len(scenarios), batch_size):
                batch_scenarios = scenarios[i:i + batch_size]

                batch_prompts = [
                    format_prompt(s, tokenizer, use_chat_template=chat_template)
                    for s in batch_scenarios
                ]

                for rollout_idx in range(rollouts):
                    # Handle multi-GPU models (device_map="auto")
                    device = getattr(model, 'device', None)
                    if device is None or str(device) == 'meta':
                        device = next(model.parameters()).device

                    inputs = tokenize_prompt(
                        batch_prompts,
                        tokenizer,
                        use_chat_template=chat_template,
                        padding=True,
                        truncation=True,
                        max_length=512
                    ).to(device)

                    with torch.no_grad():
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=max_new_tokens,
                            do_sample=do_sample,
                            temperature=temperature if do_sample else None,
                            pad_token_id=tokenizer.eos_token_id
                        )

                    for j, output in enumerate(outputs):
                        # Use attention_mask to get actual token count (excludes padding)
                        prompt_length = inputs['attention_mask'][j].sum().item()
                        response = tokenizer.decode(output[prompt_length:], skip_special_tokens=True).strip()
                        prompt = batch_scenarios[j]
                        results.append({
                            'scenario_idx': i + j,
                            'rollout_idx': rollout_idx,
                            'prompt': prompt,
                            'response': response,
                            'full_text': prompt + response,  # For activation extraction
                            'prompt_token_count': prompt_length,
                            'response_token_count': len(output) - prompt_length,
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
    model_hf_id = model.config.name_or_path
    metadata = {
        'model': model_hf_id,
        'experiment': experiment,
        'trait': trait,
        'max_new_tokens': max_new_tokens,
        'chat_template': chat_template,
        'rollouts': rollouts,
        'temperature': temperature,
        'n_pos': len(pos_results),
        'n_neg': len(neg_results),
        'n_scenarios_pos': len(pos_scenarios),
        'n_scenarios_neg': len(neg_scenarios),
        'timestamp': datetime.now().isoformat(),
    }
    with open(responses_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"    Saved {len(pos_results)} + {len(neg_results)} responses ({max_new_tokens} tokens each)")
    return len(pos_results), len(neg_results)


def main():
    parser = argparse.ArgumentParser(description='Generate responses for trait extraction.')
    parser.add_argument('--experiment', type=str, required=True, help='Experiment name')
    parser.add_argument('--trait', type=str, required=True,
                        help='Trait path (e.g., "category/my_trait") or "all"')
    parser.add_argument('--model', type=str, default=None, help='Model (default: from experiment config)')
    parser.add_argument('--max-new-tokens', type=int, default=None,
                        help=f'Max tokens (default: {BASE_MODEL_TOKENS} for base, {IT_MODEL_TOKENS} for IT)')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--device', type=str, default='auto', help='Device')
    parser.add_argument('--rollouts', type=int, default=1, help='Responses per scenario')
    parser.add_argument('--temperature', type=float, default=0.0, help='Sampling temperature')
    parser.add_argument('--chat-template', action='store_true', default=None,
                        help='Force chat template ON')
    parser.add_argument('--no-chat-template', dest='chat_template', action='store_false',
                        help='Force chat template OFF')
    model_mode = parser.add_mutually_exclusive_group()
    model_mode.add_argument('--base-model', action='store_true', dest='base_model_override',
                            help=f'Force base model mode (no chat template, {BASE_MODEL_TOKENS} tokens)')
    model_mode.add_argument('--it-model', action='store_true', dest='it_model_override',
                            help='Force IT model mode (use chat template if available)')

    args = parser.parse_args()

    # Resolve model: CLI > config
    config = load_experiment_config(args.experiment, warn_missing=False)
    model_name = args.model or config.get('extraction_model')
    if not model_name:
        parser.error(f"No model specified. Use --model or add 'extraction_model' to experiments/{args.experiment}/config.json")

    # Determine traits
    if args.trait.lower() == 'all':
        traits = discover_traits(args.experiment)
        if not traits:
            print(f"No traits found in experiment '{args.experiment}'")
            return
        print(f"Found {len(traits)} traits")
    else:
        traits = [args.trait]

    # Load model
    model, tokenizer = load_model(model_name, device=args.device)

    # Auto-detect base model from config if not explicitly set
    if args.base_model_override:
        base_model = True
    elif args.it_model_override:
        base_model = False
    else:
        base_model = is_base_model(model_name)

    # Determine settings based on base_model
    if base_model:
        use_chat_template = False
        max_new_tokens = args.max_new_tokens or BASE_MODEL_TOKENS
    else:
        use_chat_template = args.chat_template
        if use_chat_template is None:
            use_chat_template = tokenizer.chat_template is not None
        max_new_tokens = args.max_new_tokens or IT_MODEL_TOKENS

    mode_str = "BASE MODEL" if base_model else "IT MODEL"
    mode_source = "(auto-detected)" if base_model == is_base_model(model_name) else "(override)"

    print("=" * 60)
    print("GENERATE RESPONSES")
    print(f"Experiment: {args.experiment}")
    print(f"Model: {model_name}")
    print(f"Mode: {mode_str} {mode_source}")
    print(f"Max tokens: {max_new_tokens}")
    print(f"Chat template: {use_chat_template}")
    print(f"Traits: {len(traits)}")
    print("=" * 60)

    total_pos, total_neg = 0, 0
    for trait in traits:
        n_pos, n_neg = generate_responses_for_trait(
            experiment=args.experiment,
            trait=trait,
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=max_new_tokens,
            batch_size=args.batch_size,
            rollouts=args.rollouts,
            temperature=args.temperature,
            chat_template=use_chat_template,
        )
        total_pos += n_pos
        total_neg += n_neg

    print(f"\nDONE: {total_pos} + {total_neg} = {total_pos + total_neg} responses")


if __name__ == '__main__':
    main()
