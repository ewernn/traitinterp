"""
Capture raw activations from pre-generated responses and save as .pt files.

Input: Response JSONs from generate_responses
Output: experiments/{exp}/inference/{variant}/raw/residual/{prompt_set}/{id}.pt

Usage:
    from utils.capture_activations import capture_raw_activations
    n = capture_raw_activations(experiment='my_exp', prompt_set='main')
"""

import gc
import json
from contextlib import ExitStack
from pathlib import Path
from typing import Dict, List, Optional

import torch
from tqdm import tqdm

from core import MultiLayerCapture
from utils.paths import (
    get as get_path, get_model_variant, load_experiment_config,
)
from utils.model import load_model_with_lora, get_inner_model, tokenize, pad_sequences
from utils.distributed import is_tp_mode, is_rank_zero
from utils.vram import calculate_max_batch_size
from utils.layers import parse_layers


def _save_pt_data(data: Dict, prompt_id, raw_dir: Path, response_only: bool = False):
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
    prompt_ids: list = None,
) -> int:
    """Capture raw activations from pre-generated responses.

    Runs a prefill forward pass on each response with MultiLayerCapture hooks,
    saving per-token activations as .pt files for later re-projection.

    Returns number of prompts captured.
    """
    exp_dir = get_path('experiments.base', experiment=experiment)
    if not exp_dir.exists():
        print(f"Experiment not found: {exp_dir}")
        return 0

    config = load_experiment_config(experiment)
    variant = get_model_variant(experiment, model_variant, mode="application")
    variant_name = variant.name
    model_name = variant.model
    lora = variant.lora

    output_set_name = prompt_set
    if output_suffix:
        output_set_name = f"{output_set_name}_{output_suffix}"

    responses_variant = responses_from or variant_name
    responses_dir = get_path('inference.responses',
                             experiment=experiment,
                             model_variant=responses_variant,
                             prompt_set=prompt_set)
    if not responses_dir.exists():
        print(f"Response JSONs not found: {responses_dir}")
        print(f"Run generate_responses.py first, or check --responses-from variant.")
        return 0

    response_files = sorted(responses_dir.glob("*.json"))
    response_files = [f for f in response_files if not f.stem.endswith('_annotations')]
    if not response_files:
        print(f"No response JSON files found in {responses_dir}")
        return 0

    if prompt_ids is not None:
        prompt_ids_set = set(str(pid) for pid in prompt_ids)
        response_files = [f for f in response_files if f.stem in prompt_ids_set]
        if not response_files:
            print(f"No response files matching prompt_ids: {prompt_ids}")
            return 0

    if limit is not None:
        response_files = response_files[:limit]

    print(f"Found {len(response_files)} response JSONs in {responses_variant}/{prompt_set}")
    if responses_from:
        print(f"Reading responses from variant: {responses_variant}")

    inference_dir = get_path('inference.variant', experiment=experiment, model_variant=variant_name)
    raw_dir = inference_dir / "raw" / "residual" / output_set_name

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
        model, tokenizer = load_model_with_lora(
            model_name, lora_adapter=lora,
            load_in_8bit=load_in_8bit, load_in_4bit=load_in_4bit,
        )

    n_layers = len(get_inner_model(model).layers)
    print(f"Model has {n_layers} layers")

    comp_list = [c.strip() for c in components.split(',')]
    print(f"Components: {comp_list}")

    capture_layers = None
    if layers:
        capture_layers = parse_layers(layers, n_layers)
        print(f"Capturing {len(capture_layers)} of {n_layers} layers: {capture_layers}")

    from utils.paths import resolve_use_chat_template
    use_chat_template = resolve_use_chat_template(experiment, tokenizer)

    # Pre-load and pre-tokenize
    items = []
    for response_file in response_files:
        with open(response_file) as f:
            rj = json.load(f)
        prompt_text = rj['prompt']
        response_text = rj['response']

        if not response_text and rj.get('token_ids'):
            all_ids = torch.tensor(rj['token_ids'])
            prompt_end = rj.get('prompt_end', len(all_ids))
            p_ids = all_ids[:prompt_end]
            r_ids = all_ids[prompt_end:]
        else:
            for eos in ['<|eot_id|>', '<|end_of_text|>', '</s>']:
                if response_text.endswith(eos):
                    response_text = response_text[:-len(eos)]
                    break
            p_ids = tokenize(prompt_text, tokenizer)['input_ids'][0]
            r_ids = tokenize(response_text, tokenizer, add_special_tokens=False)['input_ids'][0]

        items.append((response_file.stem, prompt_text, response_text, p_ids, r_ids))

    max_seq_len = max(len(it[3]) + len(it[4]) for it in items)
    batch_size = calculate_max_batch_size(model, max_seq_len, mode='extraction')

    from utils.batch_forward import tp_agree_batch_size
    batch_size = tp_agree_batch_size(batch_size)

    layer_indices = capture_layers if capture_layers is not None else list(range(n_layers))

    print(f"\n{'='*60}")
    print(f"Capturing {len(items)} prompts → {variant_name}/raw/residual/{output_set_name}/")
    print(f"Batch size: {batch_size} (max_seq_len={max_seq_len})")
    print(f"{'='*60}")

    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    i = 0
    pbar = tqdm(total=len(items), desc="Prefill capture")
    while i < len(items):
        batch_items = items[i:i + batch_size]
        oom = False

        try:
            full_sequences = [torch.cat([it[3], it[4]]) for it in batch_items]
            batch = pad_sequences(full_sequences, pad_token_id, padding_side='left')
            device = next(model.parameters()).device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            pad_offsets = batch['pad_offsets']

            with ExitStack() as stack:
                captures = {}
                for component in comp_list:
                    cap = stack.enter_context(MultiLayerCapture(
                        model, component=component, layers=capture_layers, keep_on_gpu=False
                    ))
                    captures[component] = cap

                with torch.no_grad():
                    model(input_ids=input_ids, attention_mask=attention_mask)

                component_acts_all = {}
                for component, cap in captures.items():
                    component_acts_all[component] = cap.get_all()

            for b, (prompt_id, prompt_text, response_text, p_ids, r_ids) in enumerate(batch_items):
                pad_offset = pad_offsets[b]
                n_prompt = len(p_ids)
                n_response = len(r_ids)
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

                prompt_token_ids = p_ids.tolist()
                response_token_ids = r_ids.tolist()
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
                if is_rank_zero():
                    _save_pt_data(data, prompt_id, raw_dir, response_only=response_only)

            pbar.update(len(batch_items))
            i += batch_size

        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            from utils.batch_forward import check_oom_exception, recover_oom_batch_size
            check_oom_exception(e, batch_size)
            del e
            oom = True

        if oom:
            batch_size = recover_oom_batch_size(batch_size)
            continue

    pbar.close()

    if should_cleanup:
        del model, tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    print(f"\nOutput: {raw_dir}")
    return len(items)
