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

from utils.model import format_prompt, load_model_with_lora
from utils.json import dump_compact
from utils.generation import generate_batch
from utils.paths import get as get_path, get_model_variant, load_experiment_config
from utils.backends import add_backend_args
from transformers import AutoTokenizer


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


def generate_responses(
    experiment: str,
    prompt_set: str,
    model_variant: str = None,
    max_new_tokens: int = 50,
    temperature: float = 0.0,
    prefill: str = None,
    from_responses: str = None,
    skip_existing: bool = False,
    limit: int = None,
    output_suffix: str = None,
    load_in_8bit: bool = False,
    load_in_4bit: bool = False,
    no_server: bool = False,
    model=None,
    tokenizer=None,
) -> int:
    """Generate responses for inference prompts.

    Input: Prompt set JSON from datasets/inference/
    Output: Response JSONs in experiments/{exp}/inference/{variant}/responses/

    Args:
        model: Pre-loaded model. If provided with tokenizer, skips model loading.
        tokenizer: Pre-loaded tokenizer.

    Returns:
        Number of responses generated/written.
    """
    # Validate experiment
    exp_dir = get_path('experiments.base', experiment=experiment)
    if not exp_dir.exists():
        print(f"Experiment not found: {exp_dir}")
        return 0

    # Load experiment config
    config = load_experiment_config(experiment)

    # Resolve model variant
    variant = get_model_variant(experiment, model_variant, mode="application")
    variant_name = variant['name']
    model_name = variant['model']
    lora = variant.get('lora')

    # Load prompt set
    prompt_file = get_path('datasets.inference_prompt_set', prompt_set=prompt_set)
    if not prompt_file.exists():
        print(f"Prompt set not found: {prompt_file}")
        return 0
    with open(prompt_file) as f:
        data = json.load(f)
    prompts = [normalize_prompt_item(p) for p in (data.get('prompts', data) if isinstance(data, dict) else data)]
    system_prompt = data.get('system_prompt') if isinstance(data, dict) else None
    print(f"Loaded {len(prompts)} prompts from {prompt_set}")
    if system_prompt:
        print(f"System prompt: {system_prompt[:80]}...")

    # Apply limits
    if limit is not None:
        prompts = prompts[:limit]

    # Output directory
    set_name = prompt_set
    if output_suffix:
        set_name = f"{set_name}_{output_suffix}"
    responses_dir = get_path('inference.responses', experiment=experiment, model_variant=variant_name, prompt_set=set_name)
    responses_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {responses_dir}")

    # Chat template setting
    use_chat_template = config.get('use_chat_template')

    # ================================================================
    # MODE B: Write external responses (tokenizer only, no GPU)
    # ================================================================
    if from_responses:
        with open(from_responses) as f:
            response_map = json.load(f)
        print(f"Mode B: writing {len(response_map)} external responses (tokenizer only)")

        # Use provided tokenizer or load one
        _tokenizer = tokenizer
        if _tokenizer is None:
            _tokenizer = AutoTokenizer.from_pretrained(model_name)
            if _tokenizer.pad_token is None:
                _tokenizer.pad_token = _tokenizer.eos_token
        if use_chat_template is None:
            use_chat_template = _tokenizer.chat_template is not None

        written = 0
        for prompt_item in tqdm(prompts, desc="Writing responses"):
            prompt_id = str(prompt_item['id'])
            if prompt_id not in response_map:
                continue

            if skip_existing and (responses_dir / f"{prompt_id}.json").exists():
                continue

            raw_prompt = prompt_item.get('text') or prompt_item.get('prompt')
            prompt_text = format_prompt(raw_prompt, _tokenizer, use_chat_template=use_chat_template, system_prompt=system_prompt)
            response_text = response_map[prompt_id]

            save_response_json(
                responses_dir, prompt_item, prompt_text, response_text,
                _tokenizer, model_name, system_prompt=system_prompt,
            )
            written += 1

        print(f"Wrote {written} response JSONs to {responses_dir}")
        return written

    # ================================================================
    # MODE A: Generate from model
    # ================================================================
    is_remote = False
    should_cleanup = model is None

    if model is None:
        from other.server.client import get_model_or_client, ModelClient

        quantize = load_in_4bit or load_in_8bit
        if not no_server and not lora and not quantize:
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
            model, tokenizer = load_model_with_lora(model_name, lora_adapter=lora, load_in_8bit=load_in_8bit, load_in_4bit=load_in_4bit)
        else:
            model, tokenizer = load_model_with_lora(model_name, load_in_8bit=load_in_8bit, load_in_4bit=load_in_4bit)

    if use_chat_template is None:
        use_chat_template = tokenizer.chat_template is not None
    print(f"Chat template: {use_chat_template}")

    # Format prompts
    prompt_texts = []
    prompt_items_filtered = []
    for prompt_item in prompts:
        prompt_id = prompt_item['id']
        if skip_existing and (responses_dir / f"{prompt_id}.json").exists():
            continue
        raw_prompt = prompt_item.get('text') or prompt_item.get('prompt')
        prompt_text = format_prompt(raw_prompt, tokenizer, use_chat_template=use_chat_template, system_prompt=system_prompt)
        if prefill:
            prompt_text = prompt_text + prefill
        prompt_texts.append(prompt_text)
        prompt_items_filtered.append(prompt_item)

    if not prompt_texts:
        print("All prompts already have responses, skipping...")
        return 0

    # Generate
    if is_remote:
        print(f"Generating {len(prompt_texts)} responses via server...")
        responses = model.generate(prompt_texts, max_new_tokens=max_new_tokens, temperature=temperature)
    else:
        print(f"Generating {len(prompt_texts)} responses locally...")
        import torch
        responses = generate_batch(model, tokenizer, prompt_texts, max_new_tokens=max_new_tokens, temperature=temperature)

    # Save response JSONs
    for prompt_item, prompt_text, response_text in zip(prompt_items_filtered, prompt_texts, responses):
        save_response_json(
            responses_dir, prompt_item, prompt_text, response_text,
            tokenizer, model_name, system_prompt=system_prompt,
        )

    print(f"\nWrote {len(responses)} response JSONs to {responses_dir}")
    return len(responses)


def main():
    parser = argparse.ArgumentParser(description="Generate responses for inference prompts")
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--prompt-set", required=True, help="Prompt set from datasets/inference/{name}.json")
    parser.add_argument("--model-variant", default=None,
                       help="Model variant (default: from experiment defaults.application)")

    # Generation options (Mode A only)
    parser.add_argument("--max-new-tokens", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=0.0)
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
    add_backend_args(parser)

    args = parser.parse_args()

    # Map --backend to no_server for internal logic
    no_server = args.backend == 'local'
    if args.backend == 'server':
        from other.server.client import is_server_available
        if not is_server_available():
            raise ConnectionError(
                "Model server not running. Start with:\n"
                "  python other/server/app.py --port 8765 --model MODEL"
            )

    generate_responses(
        experiment=args.experiment,
        prompt_set=args.prompt_set,
        model_variant=args.model_variant,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        prefill=args.prefill,
        from_responses=args.from_responses,
        skip_existing=args.skip_existing,
        limit=args.limit,
        output_suffix=args.output_suffix,
        load_in_8bit=args.load_in_8bit,
        load_in_4bit=args.load_in_4bit,
        no_server=no_server,
    )


if __name__ == "__main__":
    main()
