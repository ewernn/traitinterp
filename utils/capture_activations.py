"""
Capture raw activations from pre-generated responses and save as .pt files.

Two entry points:
- `capture_raw_activations` — pipeline helper, reads pre-generated response JSONs,
  runs prefill, saves per-prompt .pt files to the inference raw dir.
- `capture_at_position` — lightweight in-memory helper for stage scripts that need
  a single activation tensor per prompt at a specific token position (position DSL
  strings like `prompt[-1]`, `response[:5]`, `all[:]`). Returns a stacked tensor
  without touching disk. Factored out of the ~80 LOC inline capture pattern used
  by stages 4/5/8 of the emotion concepts replication.

Input: Response JSONs from generate_responses (for capture_raw_activations)
       or plain prompt strings (for capture_at_position)
Output: experiments/{exp}/inference/{variant}/raw/residual/{prompt_set}/{id}.pt
        or in-memory torch.Tensor

Usage:
    from utils.capture_activations import capture_raw_activations, capture_at_position
    n = capture_raw_activations(experiment='my_exp', prompt_set='main')
    acts = capture_at_position(model, tokenizer, prompts, layers=49, position='prompt[-1]')
"""

import gc
import json
from contextlib import ExitStack
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
from tqdm import tqdm

from core import MultiLayerCapture
from utils.paths import (
    get as get_path, get_model_variant, load_experiment_config, atomic_torch_save,
)
from core.architectures import inner_model
from utils.model import tokenize, pad_sequences, format_prompt, tokenize_batch
from utils.positions import resolve_position
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
    atomic_torch_save(save_data, raw_dir / f"{prompt_id}.pt")


def capture_raw_activations(
    experiment: str,
    prompt_set: str,
    model_variant: str = None,
    components: str = "residual",
    layers: str = None,
    response_only: bool = False,
    load_in_4bit: bool = False,
    responses_from: str = None,
    force: bool = False,
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

    raw_dir = get_path('inference.raw_residual', experiment=experiment, model_variant=variant_name, prompt_set=output_set_name)

    if not force:
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
        from utils.backends import LocalBackend
        backend = LocalBackend.from_experiment(
            experiment, variant=variant_name, load_in_4bit=load_in_4bit,
        )
        model, tokenizer = backend.model, backend.tokenizer

    n_layers = len(inner_model(model).layers)
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


def capture_at_position(
    model,
    tokenizer,
    prompts: List[str],
    *,
    layers: Union[int, List[int]],
    position: str,
    component: str = "residual",
    pool: str = "mean",
    pre_formatted: bool = False,
    batch_size: int = 8,
) -> torch.Tensor:
    """Capture activations at a specific token position across a list of prompts.

    Factored out of the inline MultiLayerCapture + tokenize + pad-offset + slice
    pattern reinvented in stage 4/5/8. Uses the position DSL (`utils.positions`)
    so callers can say `'prompt[-1]'` or `'all[:]'` instead of raw indices.

    **Prefill-only**: this helper runs a single forward pass with no generation,
    so the position is resolved against `prompt_len = seq_len`. Only the `prompt`
    and `all` frames make sense here — `response[:N]` would return an empty slice
    because there is no response yet. Use `prompt[-1]` for the last-input-token
    capture that stage 4/5/8 typically want, and `all[:]` + `pool='none'` if you
    need the full sequence and will slice by raw indices downstream.

    Args:
        model: HuggingFace model (bnb-4bit or fp16, already on device)
        tokenizer: HuggingFace tokenizer
        prompts: list of prompt strings (empty list is an error)
        layers: single int or list of layer indices
        position: position DSL string — `'prompt[-1]'`, `'prompt[-3:]'`, `'all[:]'`,
            `'all[50:]'`, etc. Do NOT use `'response[*]'` frames — they resolve
            to empty slices in prefill-only mode.
        component: capture component (default: 'residual')
        pool: 'mean' | 'first' | 'last' | 'none'. How to reduce multi-token position
            slices. `'none'` returns the raw [n_tokens, hidden_dim] slice. Only safe
            when every prompt yields the same slice length — otherwise the final
            `torch.stack` will fail. Safe uses: single-prompt calls, or fixed-index
            slices like `'prompt[-3:]'` where every prompt has ≥ 3 tokens.
        pre_formatted: if True, skip format_prompt (callers that already applied
            a chat template or are using bare-prompt base models pass True here).
        batch_size: batch size for forward passes (default: 8)

    Returns:
        torch.Tensor on CPU (always fp32):
          - shape [n_prompts, n_layers, hidden_dim] if layers is a list
          - shape [n_prompts, hidden_dim]           if layers is an int (squeezed)
          - shape [n_prompts, n_layers, n_tokens, hidden_dim] if pool='none'
        Callers wanting fp16 should cast after return.
    """
    from utils.positions import parse_position
    if len(prompts) == 0:
        raise ValueError("capture_at_position: prompts list is empty")
    # Guard against the common footgun — response frame is empty in prefill-only mode.
    frame, _turn_idx, _start, _stop = parse_position(position)
    if frame == 'response':
        raise ValueError(
            f"capture_at_position: 'response' frame not supported in prefill-only mode "
            f"(got position={position!r}). Use 'prompt[-1]' or 'all[:]' instead."
        )
    scalar_layer = isinstance(layers, int)
    layer_list = [layers] if scalar_layer else list(layers)
    device = next(model.parameters()).device

    formatted = prompts if pre_formatted else [format_prompt(p, tokenizer) for p in prompts]

    out_per_prompt = []
    for i in range(0, len(formatted), batch_size):
        batch = formatted[i:i + batch_size]
        enc = tokenize_batch(batch, tokenizer)
        input_ids = enc["input_ids"].to(device)
        attn_mask = enc["attention_mask"].to(device)
        lengths = enc["lengths"]
        padded_len = input_ids.shape[1]

        with MultiLayerCapture(model, layers=layer_list, component=component, keep_on_gpu=False) as cap:
            with torch.no_grad():
                model(input_ids=input_ids, attention_mask=attn_mask, use_cache=False)

        for b, length in enumerate(lengths):
            offset = padded_len - length  # left-pad offset (tokenize_batch left-pads)
            per_layer = []
            for layer in layer_list:
                acts = cap.get(layer)[b, offset:offset + length].float().cpu()  # [length, D]
                start, end = resolve_position(position, prompt_len=length, seq_len=length)
                sliced = acts[start:end]
                if pool == "mean":
                    v = sliced.mean(dim=0)
                elif pool == "first":
                    v = sliced[0]
                elif pool == "last":
                    v = sliced[-1]
                elif pool == "none":
                    v = sliced
                else:
                    raise ValueError(f"Unknown pool mode: {pool!r}. Expected 'mean'|'first'|'last'|'none'.")
                per_layer.append(v)
            out_per_prompt.append(torch.stack(per_layer))  # [n_layers, ...]

    result = torch.stack(out_per_prompt)  # [n_prompts, n_layers, ...]
    if scalar_layer:
        result = result.squeeze(1)
    return result
