#!/usr/bin/env python3
"""
Capture raw activations from pre-generated responses via prefill.

Reads response JSONs (from generate_responses.py), runs prefill forward passes
to capture activations, and saves .pt files. Projections onto trait vectors are
computed separately via project_raw_activations_onto_traits.py.

Output:
    experiments/{exp}/inference/{model_variant}/raw/residual/{prompt_set}/{id}.pt

.pt file structure (model-agnostic):
  {
    'prompt': {
      'text': str,
      'tokens': List[str],
      'token_ids': List[int],
      'activations': {
        layer_idx: {
          'residual': Tensor[n_tokens, hidden_dim],          # if in --components
          'attn_contribution': Tensor[n_tokens, hidden_dim], # if in --components
          'mlp_contribution': Tensor[n_tokens, hidden_dim],  # if in --components
        }
      }
    },
    'response': { ... }  # same structure
  }

Usage:
    # 1. Generate responses first
    python inference/generate_responses.py \\
        --experiment my_experiment --prompt-set main_prompts

    # 2. Capture activations (reads from response JSONs)
    python inference/capture_raw_activations.py \\
        --experiment my_experiment --prompt-set main_prompts

    # Model-diff: capture through different model using another variant's responses
    python inference/capture_raw_activations.py \\
        --experiment my_experiment --prompt-set main_prompts \\
        --model-variant instruct --responses-from rm_lora

    # Then compute projections
    python inference/project_raw_activations_onto_traits.py \\
        --experiment my_experiment --prompt-set main_prompts
"""

import sys
import gc
from contextlib import ExitStack
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import argparse
import json
from typing import List, Dict
from tqdm import tqdm
from transformers import AutoTokenizer

from utils.model import format_prompt, load_model, load_model_with_lora, get_inner_model, tokenize, pad_sequences
from utils.paths import get as get_path, get_model_variant, load_experiment_config
from utils.generation import calculate_max_batch_size
from utils.capture import capture_residual_stream_prefill  # canonical location
from utils.layers import parse_layers
from core import MultiLayerCapture

# Re-export for backward compatibility — callers that import from here still work
__all__ = ['capture_residual_stream_prefill']



def _save_pt_data(
    data: Dict, prompt_id, raw_dir: Path, response_only: bool = False,
):
    """Save captured activation data as .pt file."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    save_data = data
    if response_only:
        save_data = {
            'prompt': {k: v for k, v in data['prompt'].items() if k != 'activations'},
            'response': data['response'],
        }
        save_data['prompt']['activations'] = {}
    torch.save(save_data, raw_dir / f"{prompt_id}.pt")


def capture_raw_activations(
    experiment: str,
    prompt_set: str,
    model_variant: str = None,
    components: str = "residual",
    layers: str = None,
    response_only: bool = False,
    load_in_8bit: bool = False,
    load_in_4bit: bool = False,
    responses_from: str = None,
    skip_existing: bool = False,
    limit: int = None,
    output_suffix: str = None,
    model=None,
    tokenizer=None,
) -> int:
    """Capture raw activations from pre-generated responses.

    Input: Response JSONs from generate_responses.py
    Output: .pt files with per-token activations

    Args:
        model: Pre-loaded model. If provided with tokenizer, skips model loading.
        tokenizer: Pre-loaded tokenizer.

    Returns:
        Number of prompts captured.
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

    # Resolve prompt set name for output (suffix separates runs within same variant)
    output_set_name = prompt_set
    if output_suffix:
        output_set_name = f"{output_set_name}_{output_suffix}"

    # Resolve response source (always uses raw prompt set name, not suffixed)
    responses_variant = responses_from or variant_name
    responses_dir = get_path('inference.responses',
                             experiment=experiment,
                             model_variant=responses_variant,
                             prompt_set=prompt_set)
    if not responses_dir.exists():
        print(f"Response JSONs not found: {responses_dir}")
        print(f"Run generate_responses.py first, or check --responses-from variant.")
        return 0

    # Discover available response JSONs
    response_files = sorted(responses_dir.glob("*.json"))
    # Filter out annotation files (files ending with _annotations.json)
    response_files = [f for f in response_files if not f.stem.endswith('_annotations')]
    if not response_files:
        print(f"No response JSON files found in {responses_dir}")
        return 0

    if limit is not None:
        response_files = response_files[:limit]

    print(f"Found {len(response_files)} response JSONs in {responses_variant}/{prompt_set}")
    if responses_from:
        print(f"Reading responses from variant: {responses_variant}")

    # Output directory for .pt files
    inference_dir = get_path('inference.variant', experiment=experiment, model_variant=variant_name)
    raw_dir = inference_dir / "raw" / "residual" / output_set_name

    # Filter to non-existing if --skip-existing
    if skip_existing:
        original_count = len(response_files)
        response_files = [f for f in response_files if not (raw_dir / f"{f.stem}.pt").exists()]
        skipped = original_count - len(response_files)
        if skipped:
            print(f"Skipping {skipped} already captured")
    if not response_files:
        print("All responses already captured, nothing to do.")
        return 0

    # Load model if not provided
    should_cleanup = model is None
    if model is None:
        if lora or load_in_8bit or load_in_4bit:
            model, tokenizer = load_model_with_lora(
                model_name,
                lora_adapter=lora,
                load_in_8bit=load_in_8bit,
                load_in_4bit=load_in_4bit,
            )
        else:
            model, tokenizer = load_model(model_name)

    n_layers = len(get_inner_model(model).layers)
    print(f"Model has {n_layers} layers")

    # Resolve components
    comp_list = [c.strip() for c in components.split(',')]
    print(f"Components: {comp_list}")

    # Parse layers
    capture_layers = None
    if layers:
        capture_layers = parse_layers(layers, n_layers)
        print(f"Capturing {len(capture_layers)} of {n_layers} layers: {capture_layers}")

    # Chat template setting
    use_chat_template = config.get('use_chat_template')
    if use_chat_template is None:
        use_chat_template = tokenizer.chat_template is not None

    # Pre-load all response JSONs and pre-tokenize to find max seq len
    items = []  # (prompt_id, prompt_text, response_text, prompt_ids, response_ids)
    for response_file in response_files:
        with open(response_file) as f:
            rj = json.load(f)
        prompt_text = rj['prompt']
        response_text = rj['response']

        # Pre-tokenized mode: multi-turn rollouts store the full token sequence
        # in token_ids with empty response (see scripts/convert_rollout.py).
        # Use stored IDs directly to avoid re-tokenization artifacts with
        # special tokens (tool calls, thinking traces, role headers).
        if not response_text and rj.get('token_ids'):
            all_ids = torch.tensor(rj['token_ids'])
            prompt_end = rj.get('prompt_end', len(all_ids))
            prompt_ids = all_ids[:prompt_end]
            response_ids = all_ids[prompt_end:]
        else:
            # Standard: re-tokenize from text
            # Strip EOS tokens from response text — generation artifacts that cause length
            # mismatches when diffing organism vs replay projections
            for eos in ['<|eot_id|>', '<|end_of_text|>', '</s>']:
                if response_text.endswith(eos):
                    response_text = response_text[:-len(eos)]
                    break
            prompt_ids = tokenize(prompt_text, tokenizer)['input_ids'][0]
            response_ids = tokenize(response_text, tokenizer, add_special_tokens=False)['input_ids'][0]

        items.append((response_file.stem, prompt_text, response_text, prompt_ids, response_ids))

    max_seq_len = max(len(it[3]) + len(it[4]) for it in items)
    batch_size = calculate_max_batch_size(model, max_seq_len, mode='extraction')
    layer_indices = capture_layers if capture_layers is not None else list(range(n_layers))

    # Capture activations from response JSONs
    print(f"\n{'='*60}")
    print(f"Capturing {len(items)} prompts → {variant_name}/raw/residual/{output_set_name}/")
    print(f"Batch size: {batch_size} (max_seq_len={max_seq_len})")
    print(f"{'='*60}")

    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    i = 0
    pbar = tqdm(total=len(items), desc="Prefill capture")
    while i < len(items):
        batch_items = items[i:i + batch_size]

        try:
            # Pad sequences (left-padding) — same as extraction
            full_sequences = [torch.cat([it[3], it[4]]) for it in batch_items]
            batch = pad_sequences(full_sequences, pad_token_id, padding_side='left')
            input_ids = batch['input_ids'].to(model.device)
            attention_mask = batch['attention_mask'].to(model.device)
            pad_offsets = batch['pad_offsets']

            # Forward pass with dynamic component capture
            with ExitStack() as stack:
                captures = {}
                for component in comp_list:
                    cap = stack.enter_context(MultiLayerCapture(
                        model, component=component, layers=capture_layers, keep_on_gpu=True
                    ))
                    captures[component] = cap

                with torch.no_grad():
                    model(input_ids=input_ids, attention_mask=attention_mask)

                component_acts_all = {}
                for component, cap in captures.items():
                    component_acts_all[component] = cap.get_all()

            # Split per-sample using pad_offsets, save individually
            for b, (prompt_id, prompt_text, response_text, prompt_ids, response_ids) in enumerate(batch_items):
                pad_offset = pad_offsets[b]
                n_prompt = len(prompt_ids)
                n_response = len(response_ids)
                prompt_start = pad_offset
                prompt_end = pad_offset + n_prompt
                response_end = pad_offset + n_prompt + n_response

                prompt_acts = {}
                response_acts = {}
                for layer_idx in layer_indices:
                    prompt_acts[layer_idx] = {}
                    response_acts[layer_idx] = {}

                    for component in comp_list:
                        acts = component_acts_all.get(component, {})
                        if layer_idx in acts:
                            full = acts[layer_idx]
                            prompt_acts[layer_idx][component] = full[b, prompt_start:prompt_end, :].cpu()
                            response_acts[layer_idx][component] = full[b, prompt_end:response_end, :].cpu()

                prompt_token_ids = prompt_ids.tolist()
                response_token_ids = response_ids.tolist()
                data = {
                    'prompt': {
                        'text': prompt_text,
                        'tokens': [tokenizer.decode([tid]) for tid in prompt_token_ids],
                        'token_ids': prompt_token_ids,
                        'activations': prompt_acts,
                        'attention': {},
                    },
                    'response': {
                        'text': response_text,
                        'tokens': [tokenizer.decode([tid]) for tid in response_token_ids],
                        'token_ids': response_token_ids,
                        'activations': response_acts,
                        'attention': [],
                    },
                }
                _save_pt_data(data, prompt_id, raw_dir, response_only=response_only)

            pbar.update(len(batch_items))
            i += batch_size

        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if "out of memory" not in str(e).lower() and not isinstance(e, torch.cuda.OutOfMemoryError):
                raise
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if batch_size == 1:
                raise RuntimeError("OOM even with batch_size=1")
            batch_size = max(1, batch_size // 2)
            print(f"\nOOM, reducing batch_size to {batch_size}")
            # Don't advance i — retry same batch with smaller size

    pbar.close()

    # Cleanup GPU memory only if we loaded the model
    if should_cleanup:
        del model
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    print(f"\n{'='*60}")
    print("Complete!")
    print(f"{'='*60}")
    print(f"\nOutput: {raw_dir}")

    return len(items)


def main():
    parser = argparse.ArgumentParser(description="Capture raw activations from pre-generated responses")
    parser.add_argument("--experiment", required=True, help="Experiment name")
    parser.add_argument("--prompt-set", required=True, help="Prompt set name")

    # Capture options
    parser.add_argument("--components", type=str, default="residual",
                       help="Comma-separated components to capture (default: residual)")
    parser.add_argument("--layers", type=str, default=None,
                       help="Only capture specific layers. Comma-separated (e.g., '0,10,20,30') "
                            "or range with step (e.g., '0-75:5'). Default: all layers.")
    parser.add_argument("--response-only", action="store_true",
                       help="Only save response token activations (skip prompt tokens)")

    # Model options
    parser.add_argument("--model-variant", default=None,
                       help="Model variant (default: from experiment defaults.application)")
    parser.add_argument("--load-in-8bit", action="store_true")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--no-server", action="store_true",
                       help="Force local model loading")

    # Response source
    parser.add_argument("--responses-from", type=str, default=None,
                       help="Read responses from a different variant's response JSONs. "
                            "For model-diff: capture through current model using another variant's responses.")

    # Common options
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output-suffix", type=str, default=None,
                       help="Suffix for output directory name")

    args = parser.parse_args()

    capture_raw_activations(
        experiment=args.experiment,
        prompt_set=args.prompt_set,
        model_variant=args.model_variant,
        components=args.components,
        layers=args.layers,
        response_only=args.response_only,
        load_in_8bit=args.load_in_8bit,
        load_in_4bit=args.load_in_4bit,
        responses_from=args.responses_from,
        skip_existing=args.skip_existing,
        limit=args.limit,
        output_suffix=args.output_suffix,
    )


if __name__ == "__main__":
    main()
