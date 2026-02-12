#!/usr/bin/env python3
"""
Generate responses for inference prompts.

Two modes:
  Mode A — Generate responses from model (or model server)
  Mode B — Write external response text (--from-responses, tokenizer only, no GPU)

Output:
    experiments/{exp}/inference/{model_variant}/responses/{prompt_set}/{id}.json

Usage:
    # Mode A: generate from model
    python inference/generate_responses.py \
        --experiment audit-bench \
        --prompt-set rm_syco/exploitation_evals_100 \
        --model-variant rm_lora

    # Mode B: write external responses (tokenizer only)
    python inference/generate_responses.py \
        --experiment audit-bench \
        --prompt-set rm_syco/exploitation_evals_100 \
        --model-variant rm_lora \
        --from-responses path/to/response_texts.json
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
from datetime import datetime
from tqdm import tqdm

from utils.model import format_prompt, load_model_with_lora, tokenize, tokenize_batch
from utils.json import dump_compact
from utils.generation import generate_batch, calculate_max_batch_size
from utils.paths import get as get_path, get_model_variant, load_experiment_config
from transformers import AutoModelForCausalLM, AutoTokenizer


def normalize_prompt_item(item: dict) -> dict:
    """Normalize prompt item to have 'text' key (handles 'prompt' as fallback)."""
    if 'text' not in item and 'prompt' in item:
        item = {**item, 'text': item['prompt']}
    return item


def save_response_json(
    responses_dir: Path, prompt_item: dict, prompt_text: str, response_text: str,
    tokenizer, model_name: str, system_prompt: str = None,
):
    """Tokenize and save a single response JSON."""
    prompt_id = prompt_item['id']
    prompt_note = prompt_item.get('note', '')

    # Tokenize prompt
    has_bos = tokenizer.bos_token and prompt_text.startswith(tokenizer.bos_token)
    prompt_token_ids = tokenizer(prompt_text, add_special_tokens=not has_bos, padding=False)['input_ids']
    prompt_tokens = [tokenizer.decode([tid]) for tid in prompt_token_ids]

    # Tokenize response (no special tokens — appended to prompt)
    response_token_ids = tokenizer(response_text, add_special_tokens=False, padding=False)['input_ids']
    response_tokens = [tokenizer.decode([tid]) for tid in response_token_ids]

    response_data = {
        'prompt': prompt_text,
        'response': response_text,
        'system_prompt': system_prompt,
        'tokens': prompt_tokens + response_tokens,
        'token_ids': prompt_token_ids + response_token_ids,
        'prompt_end': len(prompt_tokens),
        'inference_model': model_name or 'unknown',
        'prompt_note': prompt_note if prompt_note else None,
        'capture_date': datetime.now().isoformat(),
        'tags': []
    }
    with open(responses_dir / f"{prompt_id}.json", 'w') as f:
        dump_compact(response_data, f)


def main():
    parser = argparse.ArgumentParser(description="Generate responses for inference prompts")
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--prompt-set", required=True, help="Prompt set from datasets/inference/{name}.json")
    parser.add_argument("--model-variant", default=None,
                       help="Model variant (default: from experiment defaults.application)")

    # Generation options (Mode A only)
    parser.add_argument("--max-new-tokens", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--prefill", type=str, default=None,
                       help="Prefill string appended to prompt before generation")

    # Mode B
    parser.add_argument("--from-responses", type=str, default=None,
                       help="Path to {id: response_text} JSON. Tokenizer only, no GPU.")

    # Common options
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output-suffix", type=str, default=None,
                       help="Suffix for output directory name")
    parser.add_argument("--load-in-8bit", action="store_true")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--no-server", action="store_true",
                       help="Force local model loading (skip server check)")

    args = parser.parse_args()

    # Validate experiment
    exp_dir = get_path('experiments.base', experiment=args.experiment)
    if not exp_dir.exists():
        print(f"Experiment not found: {exp_dir}")
        return

    # Load experiment config
    config = load_experiment_config(args.experiment)

    # Resolve model variant
    variant = get_model_variant(args.experiment, args.model_variant, mode="application")
    model_variant = variant['name']
    model_name = variant['model']
    lora = variant.get('lora')

    # Load prompt set
    prompt_file = get_path('datasets.inference_prompt_set', prompt_set=args.prompt_set)
    if not prompt_file.exists():
        print(f"Prompt set not found: {prompt_file}")
        return
    with open(prompt_file) as f:
        data = json.load(f)
    prompts = [normalize_prompt_item(p) for p in (data.get('prompts', data) if isinstance(data, dict) else data)]
    system_prompt = data.get('system_prompt') if isinstance(data, dict) else None
    print(f"Loaded {len(prompts)} prompts from {args.prompt_set}")
    if system_prompt:
        print(f"System prompt: {system_prompt[:80]}...")

    # Apply limits
    if args.limit is not None:
        prompts = prompts[:args.limit]

    # Output directory
    set_name = args.prompt_set
    if args.output_suffix:
        set_name = f"{set_name}_{args.output_suffix}"
    responses_dir = get_path('inference.responses', experiment=args.experiment, model_variant=model_variant, prompt_set=set_name)
    responses_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {responses_dir}")

    # Chat template setting
    use_chat_template = config.get('use_chat_template')

    # ================================================================
    # MODE B: Write external responses (tokenizer only, no GPU)
    # ================================================================
    if args.from_responses:
        with open(args.from_responses) as f:
            response_map = json.load(f)
        print(f"Mode B: writing {len(response_map)} external responses (tokenizer only)")

        # Load tokenizer only
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        if use_chat_template is None:
            use_chat_template = tokenizer.chat_template is not None

        written = 0
        for prompt_item in tqdm(prompts, desc="Writing responses"):
            prompt_id = str(prompt_item['id'])
            if prompt_id not in response_map:
                continue

            if args.skip_existing and (responses_dir / f"{prompt_id}.json").exists():
                continue

            raw_prompt = prompt_item.get('text') or prompt_item.get('prompt')
            prompt_text = format_prompt(raw_prompt, tokenizer, use_chat_template=use_chat_template, system_prompt=system_prompt)
            response_text = response_map[prompt_id]

            save_response_json(
                responses_dir, prompt_item, prompt_text, response_text,
                tokenizer, model_name, system_prompt=system_prompt,
            )
            written += 1

        print(f"Wrote {written} response JSONs to {responses_dir}")
        return

    # ================================================================
    # MODE A: Generate from model
    # ================================================================
    from other.server.client import get_model_or_client, ModelClient

    is_remote = False
    if not args.no_server and not lora:
        handle = get_model_or_client(model_name)
        if isinstance(handle, ModelClient):
            print(f"Using model server (model: {model_name})")
            model = handle
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            is_remote = True
        else:
            model, tokenizer = handle
    elif lora:
        model, tokenizer = load_model_with_lora(model_name, lora_adapter=lora, load_in_8bit=args.load_in_8bit, load_in_4bit=args.load_in_4bit)
    else:
        model, tokenizer = load_model_with_lora(model_name, load_in_8bit=args.load_in_8bit, load_in_4bit=args.load_in_4bit)

    if use_chat_template is None:
        use_chat_template = tokenizer.chat_template is not None
    print(f"Chat template: {use_chat_template}")

    # Format prompts
    prompt_texts = []
    prompt_items_filtered = []
    for prompt_item in prompts:
        prompt_id = prompt_item['id']
        if args.skip_existing and (responses_dir / f"{prompt_id}.json").exists():
            continue
        raw_prompt = prompt_item.get('text') or prompt_item.get('prompt')
        prompt_text = format_prompt(raw_prompt, tokenizer, use_chat_template=use_chat_template, system_prompt=system_prompt)
        if args.prefill:
            prompt_text = prompt_text + args.prefill
        prompt_texts.append(prompt_text)
        prompt_items_filtered.append(prompt_item)

    if not prompt_texts:
        print("All prompts already have responses, skipping...")
        return

    # Generate
    if is_remote:
        print(f"Generating {len(prompt_texts)} responses via server...")
        responses = model.generate(prompt_texts, max_new_tokens=args.max_new_tokens, temperature=args.temperature)
    else:
        print(f"Generating {len(prompt_texts)} responses locally...")
        import torch
        responses = generate_batch(model, tokenizer, prompt_texts, max_new_tokens=args.max_new_tokens, temperature=args.temperature)

    # Save response JSONs
    for prompt_item, prompt_text, response_text in zip(prompt_items_filtered, prompt_texts, responses):
        save_response_json(
            responses_dir, prompt_item, prompt_text, response_text,
            tokenizer, model_name, system_prompt=system_prompt,
        )

    print(f"\nWrote {len(responses)} response JSONs to {responses_dir}")


if __name__ == "__main__":
    main()
