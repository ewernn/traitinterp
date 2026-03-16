"""
Extract activations from generated responses.

Called by run_extraction_pipeline.py (stage 3).

Input:
    - experiment, trait, model_variant: Standard identifiers
    - backend: GenerationBackend instance (LocalBackend or ServerBackend)

Output:
    - experiments/{experiment}/extraction/{trait}/{model_variant}/activations/{position}/{component}/

Position syntax: <frame>[<slice>]
  - frame: 'prompt', 'response', or 'all'
  - slice: Python slice notation like '[-5:]', '[0]', '[:]'

Examples:
  response[:5]  - First 5 response tokens (default)
  response[:]   - All response tokens (mean)
  response[0]   - First response token only
  prompt[-1]    - Last prompt token (Arditi-style)
  prompt[-3:]   - Last 3 prompt tokens
  all[:]        - All tokens (mean)
"""

import gc
import json
import re
from datetime import datetime
from typing import Tuple, Optional, List, Dict, TYPE_CHECKING

import torch
from tqdm import tqdm

from utils.paths import (
    get as get_path,
    get_activation_dir,
    get_activation_path,
    get_activation_metadata_path,
    get_val_activation_path,
)
from utils.model import pad_sequences, format_prompt
from utils.distributed import is_rank_zero, is_tp_mode
from core import MultiLayerCapture

if TYPE_CHECKING:
    from utils.backends import GenerationBackend


def parse_position(position: str) -> Tuple[str, Optional[int], Optional[int]]:
    """
    Parse position string like 'response[-5:]' into (frame, start, stop).

    Returns:
        (frame, start, stop) where frame is 'prompt', 'response', or 'all',
        and start/stop are slice indices (None means unbounded).
    """
    match = re.match(r'(prompt|response|all)\[(.+)\]', position)
    if not match:
        raise ValueError(f"Invalid position format: '{position}'. Use <frame>[<slice>], e.g., 'response[:5]' or 'prompt[-1]'")

    frame = match.group(1)
    slice_str = match.group(2).strip()

    if slice_str == ':':
        return frame, None, None

    if ':' in slice_str:
        parts = slice_str.split(':')
        start = int(parts[0]) if parts[0] else None
        stop = int(parts[1]) if parts[1] else None
        return frame, start, stop
    else:
        # Single index like [-1] or [0]
        idx = int(slice_str)
        if idx >= 0:
            return frame, idx, idx + 1
        else:
            # Negative index: -1 means last element, so start=-1, stop=None
            return frame, idx, idx + 1 if idx != -1 else None


def tokens_needed(position: str) -> Optional[int]:
    """
    Return minimum response tokens needed for extraction, or None if undeterminable.

    Examples:
        prompt[-1]    → 0   (no response tokens needed)
        prompt[:]     → 0
        response[:5]  → 5
        response[0]   → 1
        response[:10] → 10
        response[:]   → None (undeterminable - use fallback)
        all[:]        → None (undeterminable)
    """
    frame, start, stop = parse_position(position)

    if frame == 'prompt':
        return 0

    if frame == 'response' and start is not None and start >= 0 and stop is not None and stop > 0:
        return stop

    return None


def resolve_max_new_tokens(position: str, user_value: Optional[int] = None) -> int:
    """
    Resolve max_new_tokens from position and optional user override.

    - If user_value is None: use smart default based on position (or 32 fallback)
    - If user_value is specified: validate it's sufficient for the position

    Raises ValueError if user_value is too small for position.
    """
    needed = tokens_needed(position)

    if user_value is None:
        if needed is not None:
            return needed
        return 16  # fallback for undeterminable positions

    if needed is not None and user_value < needed:
        raise ValueError(
            f"--max-new-tokens {user_value} is less than {needed} "
            f"required for position '{position}'"
        )

    return user_value


def resolve_position(position: str, prompt_len: int, seq_len: int) -> Tuple[int, int]:
    """
    Resolve position string to concrete (start_idx, end_idx) given sequence lengths.

    Args:
        position: Position string like 'response[-5:]'
        prompt_len: Number of prompt tokens
        seq_len: Total sequence length

    Returns:
        (start_idx, end_idx) as absolute indices into the sequence
    """
    frame, start, stop = parse_position(position)

    # Determine frame bounds
    if frame == 'all':
        frame_start, frame_end = 0, seq_len
    elif frame == 'prompt':
        frame_start, frame_end = 0, prompt_len
    elif frame == 'response':
        frame_start, frame_end = prompt_len, seq_len
    else:
        raise ValueError(f"Unknown frame: {frame}")

    frame_len = frame_end - frame_start

    # Apply slice within frame
    if start is None:
        abs_start = frame_start
    elif start >= 0:
        abs_start = frame_start + min(start, frame_len)
    else:
        abs_start = frame_end + start  # Negative indexing

    if stop is None:
        abs_end = frame_end
    elif stop >= 0:
        abs_end = frame_start + min(stop, frame_len)
    else:
        abs_end = frame_end + stop  # Negative indexing

    # Clamp to frame bounds (not just sequence bounds)
    abs_start = max(frame_start, min(abs_start, frame_end))
    abs_end = max(abs_start, min(abs_end, frame_end))

    return abs_start, abs_end


def load_vetting_filter(experiment: str, trait: str, model_variant: str,
                        pos_threshold: int = 60, neg_threshold: int = 40) -> dict:
    """Load failed indices from response vetting. Computes from raw scores using current thresholds.

    Always recomputes from raw scores to avoid stale cache when thresholds change.
    """
    vetting_file = get_path('extraction.trait', experiment=experiment, trait=trait, model_variant=model_variant) / 'vetting' / 'response_scores.json'
    if not vetting_file.exists():
        return {'positive': [], 'negative': []}
    with open(vetting_file) as f:
        data = json.load(f)

    results = data.get('results', [])
    if not results:
        # Fall back to pre-computed indices for old-format files
        return data.get('failed_indices', {'positive': [], 'negative': []})

    # Compute failed indices from raw scores using current thresholds
    stored = data.get('thresholds', {})
    if stored.get('pos_threshold') != pos_threshold or stored.get('neg_threshold') != neg_threshold:
        print(f"      Vetting filter: using threshold pos>={pos_threshold}/neg<={neg_threshold} "
              f"(stored: pos>={stored.get('pos_threshold')}/neg<={stored.get('neg_threshold')})")

    pos_failed = [r['idx'] for r in results
                  if r['polarity'] == 'positive' and (r['score'] is None or r['score'] < pos_threshold)]
    neg_failed = [r['idx'] for r in results
                  if r['polarity'] == 'negative' and (r['score'] is None or r['score'] > neg_threshold)]

    return {'positive': pos_failed, 'negative': neg_failed}


def load_llm_judge_position(experiment: str, trait: str, model_variant: str) -> str | None:
    """Load llm_judge_position from vetting if available."""
    vetting_file = get_path('extraction.trait', experiment=experiment, trait=trait, model_variant=model_variant) / 'vetting' / 'response_scores.json'
    if not vetting_file.exists():
        return None
    with open(vetting_file) as f:
        data = json.load(f)
    return data.get('llm_judge_position')


def extract_activations_for_trait(
    experiment: str,
    trait: str,
    model_variant: str,
    backend: "GenerationBackend",
    val_split: float = 0.1,
    position: str = 'response[:5]',
    component: str = 'residual',
    use_vetting_filter: bool = True,
    paired_filter: bool = False,
    batch_size: int = None,
    layers: Optional[List[int]] = None,
    pos_threshold: int = 60,
    neg_threshold: int = 40,
) -> int:
    """
    Extract activations from generated responses. Returns number of layers extracted.

    Uses backend.model for forward passes with MultiLayerCapture hook.

    Args:
        backend: GenerationBackend instance with model access
        position: Token position to extract from. Format: <frame>[<slice>]
            - response[:5] - First 5 response tokens (mean) [default]
            - response[:]  - All response tokens (mean)
            - prompt[-1]   - Last prompt token (Arditi-style)
            - all[:]       - All tokens (mean)
    """
    # Access model and tokenizer from backend
    model = backend.model
    tokenizer = backend.tokenizer
    n_layers = backend.n_layers
    responses_dir = get_path('extraction.responses', experiment=experiment, trait=trait, model_variant=model_variant)

    # Get paths using centralized helpers
    activations_dir = get_activation_dir(experiment, trait, model_variant, component, position)
    activations_dir.mkdir(parents=True, exist_ok=True)

    # Load responses and metadata
    try:
        with open(responses_dir / 'pos.json') as f:
            pos_data = json.load(f)
        with open(responses_dir / 'neg.json') as f:
            neg_data = json.load(f)
        # Load metadata to get chat_template setting
        metadata_path = responses_dir / 'metadata.json'
        use_chat_template = False
        if metadata_path.exists():
            with open(metadata_path) as f:
                resp_metadata = json.load(f)
            use_chat_template = resp_metadata.get('chat_template', False)
    except FileNotFoundError:
        print(f"      ERROR: Response files not found. Run stage 1 first.")
        return 0

    # Filter based on vetting results
    n_filtered_pos, n_filtered_neg, n_excluded_by_pairing = 0, 0, 0
    if use_vetting_filter:
        failed = load_vetting_filter(experiment, trait, model_variant, pos_threshold, neg_threshold)
        pos_failed, neg_failed = set(failed.get('positive', [])), set(failed.get('negative', []))

        if paired_filter:
            # Paired mode: exclude index if EITHER side failed
            all_failed = pos_failed | neg_failed
            pos_data = [r for i, r in enumerate(pos_data) if i not in all_failed]
            neg_data = [r for i, r in enumerate(neg_data) if i not in all_failed]
            n_filtered_pos, n_filtered_neg = len(pos_failed), len(neg_failed)
            n_excluded_by_pairing = len(all_failed)
            if all_failed:
                print(f"      Filtered {n_excluded_by_pairing} pairs ({n_filtered_pos} pos + {n_filtered_neg} neg failed, paired mode)")
        else:
            # Independent mode: filter each side separately
            if pos_failed or neg_failed:
                pos_data = [r for i, r in enumerate(pos_data) if i not in pos_failed]
                neg_data = [r for i, r in enumerate(neg_data) if i not in neg_failed]
                n_filtered_pos, n_filtered_neg = len(pos_failed), len(neg_failed)
                print(f"      Filtered {n_filtered_pos + n_filtered_neg} responses based on vetting")

    # Early exit if no data left after filtering
    if not pos_data and not neg_data:
        print(f"      ERROR: No responses left after filtering. Skipping activation extraction.")
        return 0

    # Split into train/val
    train_pos, train_neg, val_pos, val_neg = pos_data, neg_data, [], []
    if val_split == 0:
        print(f"      WARNING: val_split=0, no validation data will be extracted")
    if val_split > 0:
        pos_split = int(len(pos_data) * (1 - val_split))
        neg_split = int(len(neg_data) * (1 - val_split))
        train_pos, val_pos = pos_data[:pos_split], pos_data[pos_split:]
        train_neg, val_neg = neg_data[:neg_split], neg_data[neg_split:]

    layer_list = layers if layers is not None else list(range(n_layers))

    def prepare_split(responses: list[dict], label: str) -> list[dict]:
        """Filter, tokenize, and resolve positions for a response split."""
        valid_responses = [item for item in responses if item.get('prompt') is not None and item.get('response') is not None]
        # Under TP, all ranks must agree on data — different counts cause
        # different-sized forward passes which deadlock NCCL.
        if is_tp_mode():
            import torch.distributed as dist
            local_count = len(valid_responses)
            min_count = torch.tensor([local_count], device='cuda')
            max_count = torch.tensor([local_count], device='cuda')
            dist.all_reduce(min_count, op=dist.ReduceOp.MIN)
            dist.all_reduce(max_count, op=dist.ReduceOp.MAX)
            if min_count.item() != max_count.item():
                agreed = int(min_count.item())
                if is_rank_zero():
                    print(f"      WARNING: TP rank valid_responses mismatch in {label}: "
                          f"min={agreed}, max={int(max_count.item())}, this_rank={local_count}. "
                          f"Truncating to {agreed}.")
                valid_responses = valid_responses[:agreed]
        if not valid_responses:
            return []

        formatted_prompts = [
            format_prompt(item['prompt'], tokenizer, use_chat_template=use_chat_template, system_prompt=item.get('system_prompt'))
            for item in valid_responses
        ]
        texts = [fp + item['response'] for fp, item in zip(formatted_prompts, valid_responses)]

        has_bos = tokenizer.bos_token and texts[0].startswith(tokenizer.bos_token)
        all_input_ids_raw = tokenizer(texts, add_special_tokens=not has_bos, padding=False)['input_ids']
        all_input_ids = [torch.tensor(ids) for ids in all_input_ids_raw]

        prompt_has_bos = tokenizer.bos_token and formatted_prompts[0].startswith(tokenizer.bos_token)
        prompt_ids = tokenizer(formatted_prompts, add_special_tokens=not prompt_has_bos, padding=False)['input_ids']
        prompt_lens = [len(ids) for ids in prompt_ids]

        items = []
        for i, (item, input_ids, prompt_len) in enumerate(zip(valid_responses, all_input_ids, prompt_lens)):
            seq_len = len(input_ids)
            start_idx, end_idx = resolve_position(position, prompt_len, seq_len)
            if start_idx >= end_idx:
                continue
            items.append({
                'input_ids': input_ids,
                'seq_len': seq_len,
                'start_idx': start_idx,
                'end_idx': end_idx,
            })

        if is_tp_mode():
            import torch.distributed as dist
            local_count = len(items)
            min_count = torch.tensor([local_count], device='cuda')
            max_count = torch.tensor([local_count], device='cuda')
            dist.all_reduce(min_count, op=dist.ReduceOp.MIN)
            dist.all_reduce(max_count, op=dist.ReduceOp.MAX)
            if min_count.item() != max_count.item():
                agreed = int(min_count.item())
                if is_rank_zero():
                    print(f"      WARNING: TP rank item count mismatch in {label}: "
                          f"min={agreed}, max={int(max_count.item())}, this_rank={local_count}. "
                          f"Truncating to {agreed}.")
                items = items[:agreed]
            if min_count.item() == 0 and local_count > 0:
                raise RuntimeError(
                    f"TP rank disagreement in prepare_split({label}): "
                    f"this rank has {local_count} items but another has 0"
                )
        return items

    def run_forward(items: list[dict], label: str) -> dict[int, torch.Tensor]:
        """Run batched forward passes on prepared items. Returns {layer: tensor}."""
        all_activations = {layer: [] for layer in layer_list}
        if not items:
            for layer in layer_list:
                all_activations[layer] = torch.empty(0)
            return all_activations

        local_batch_size = batch_size
        max_seq_len = max(item['seq_len'] for item in items)
        if local_batch_size is None:
            # Empirical calibration: run 1 forward pass, measure actual peak memory
            pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
            cal_ids = torch.full((1, max_seq_len), pad_token_id, dtype=torch.long, device=model.device)
            cal_mask = torch.ones_like(cal_ids)

            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            baseline = torch.cuda.memory_allocated()

            with MultiLayerCapture(model, component=component, layers=layers, keep_on_gpu=True) as cal_cap:
                with torch.no_grad():
                    model(input_ids=cal_ids, attention_mask=cal_mask, use_cache=False)

            per_item = torch.cuda.max_memory_allocated() - baseline
            del cal_ids, cal_mask, cal_cap
            torch.cuda.empty_cache()

            free = torch.cuda.mem_get_info()[0]
            local_batch_size = max(1, int(free / per_item * 0.9))
            if is_rank_zero():
                print(f"    Calibrated: {per_item / 1024**2:.0f}MB/seq, free={free / 1024**3:.1f}GB → batch={local_batch_size}")

        tp = is_tp_mode()
        if tp:
            import torch.distributed as dist
            bs_tensor = torch.tensor([local_batch_size], device='cuda')
            dist.all_reduce(bs_tensor, op=dist.ReduceOp.MIN)
            local_batch_size = int(bs_tensor.item())

        i = 0
        pbar = tqdm(total=len(items), desc=f"    {label}", leave=False)
        while i < len(items):
            batch_items = items[i:i + local_batch_size]

            if torch.cuda.is_available() and is_rank_zero():
                alloc = torch.cuda.memory_allocated() / 1024**3
                resv = torch.cuda.memory_reserved() / 1024**3
                free, total = torch.cuda.mem_get_info()
                free_gb = free / 1024**3
                print(f"      [mem] batch {i//local_batch_size}: alloc={alloc:.2f}GB resv={resv:.2f}GB free={free_gb:.1f}GB items={len(batch_items)}")

            oom = False
            try:
                pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
                batch = pad_sequences([item['input_ids'] for item in batch_items], pad_token_id)
                input_ids = batch['input_ids'].to(model.device)
                attention_mask = batch['attention_mask'].to(model.device)
                pad_offsets = batch['pad_offsets']

                with MultiLayerCapture(model, component=component, layers=layers, keep_on_gpu=True) as capture:
                    with torch.no_grad():
                        model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)

                for layer in layer_list:
                    acts = capture.get(layer)
                    if acts is None:
                        continue

                    batch_acts = []
                    for b, item in enumerate(batch_items):
                        pad_offset = pad_offsets[b]
                        start_idx = item['start_idx'] + pad_offset
                        end_idx = item['end_idx'] + pad_offset
                        selected = acts[b, start_idx:end_idx, :]
                        act_out = selected.mean(dim=0) if selected.shape[0] > 1 else selected.squeeze(0)
                        batch_acts.append(act_out)

                    batch_tensor = torch.stack(batch_acts).cpu()
                    for act in batch_tensor:
                        all_activations[layer].append(act)

                # Free GPU memory between batches to prevent allocator fragmentation
                del capture, input_ids, attention_mask, batch
                gc.collect()
                torch.cuda.empty_cache()

            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                if "out of memory" not in str(e).lower() and not isinstance(e, torch.cuda.OutOfMemoryError):
                    raise
                if tp:
                    raise RuntimeError(
                        f"OOM during TP forward pass (batch_size={local_batch_size}). "
                        f"NCCL state is corrupted and cannot recover. "
                        f"Reduce batch size or increase overhead_factor in calculate_max_batch_size."
                    )
                # Clear traceback chains that pin CUDA tensors via stack frame locals.
                import traceback as tb_mod
                if e.__traceback__:
                    tb_mod.clear_frames(e.__traceback__)
                e.__traceback__ = None
                for chained in (e.__context__, e.__cause__):
                    if chained and hasattr(chained, '__traceback__') and chained.__traceback__:
                        tb_mod.clear_frames(chained.__traceback__)
                        chained.__traceback__ = None
                del e
                oom = True

            # OOM cleanup OUTSIDE except block — thread state no longer roots the exception
            if oom:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if local_batch_size == 1:
                    raise RuntimeError("OOM even with batch_size=1")
                local_batch_size = max(1, local_batch_size // 2)
                print(f"\n      OOM, reducing batch_size to {local_batch_size}")
                continue

            pbar.update(len(batch_items))
            i += local_batch_size

        pbar.close()

        for layer in layer_list:
            all_activations[layer] = torch.stack(all_activations[layer]) if all_activations[layer] else torch.empty(0)
        return all_activations

    # Prepare all splits (filter, tokenize, resolve positions)
    tp_items = prepare_split(train_pos, 'train_positive')
    tn_items = prepare_split(train_neg, 'train_negative')
    vp_items, vn_items = [], []
    if val_split > 0 and (val_pos or val_neg):
        vp_items = prepare_split(val_pos, 'val_positive')
        vn_items = prepare_split(val_neg, 'val_negative')

    # Track boundaries for splitting results after combined forward pass
    b0 = len(tp_items)
    b1 = b0 + len(tn_items)
    b2 = b1 + len(vp_items)

    # Single combined forward pass (4→1: one batch size calc, one CUDA cache fill)
    all_items = tp_items + tn_items + vp_items + vn_items
    all_acts = run_forward(all_items, 'combined')

    # Split results by boundary indices
    pos_acts = {l: all_acts[l][:b0] for l in layer_list}
    neg_acts = {l: all_acts[l][b0:b1] for l in layer_list}

    per_layer_mode = layers is not None
    hidden_dim = None

    if per_layer_mode:
        activation_norms = {}
        for layer in layer_list:
            train_layer = torch.cat([pos_acts[layer], neg_acts[layer]], dim=0)
            if is_rank_zero():
                torch.save(train_layer, activations_dir / f"train_layer{layer}.pt")
            if train_layer.numel() > 0:
                hidden_dim = train_layer.shape[-1]
                activation_norms[layer] = round(train_layer.norm(dim=-1).mean().item(), 2)
            else:
                activation_norms[layer] = 0.0
        print(f"      Saved train: {len(layer_list)} layers (per-layer files)")
    else:
        pos_all = torch.stack([pos_acts[l] for l in layer_list], dim=1)
        neg_all = torch.stack([neg_acts[l] for l in layer_list], dim=1)
        train_acts = torch.cat([pos_all, neg_all], dim=0)

        if is_rank_zero():
            activation_path = get_activation_path(experiment, trait, model_variant, component, position)
            torch.save(train_acts, activation_path)
            print(f"      Saved train: {train_acts.shape} -> {activation_path.name}")

        if train_acts.numel() == 0:
            activation_norms = {layer: 0.0 for layer in layer_list}
        else:
            activation_norms = {layer: round(train_acts[:, layer, :].norm(dim=-1).mean().item(), 2) for layer in layer_list}
        hidden_dim = train_acts.shape[-1] if train_acts.numel() > 0 else None

    if hidden_dim is None:
        config = model.config
        if hasattr(config, 'text_config'):
            config = config.text_config
        hidden_dim = config.hidden_size

    # Save validation activations (split from combined results)
    n_val_pos, n_val_neg = 0, 0
    if val_split > 0 and (val_pos or val_neg):
        val_pos_acts = {l: all_acts[l][b1:b2] for l in layer_list}
        val_neg_acts = {l: all_acts[l][b2:] for l in layer_list}

        if per_layer_mode:
            for layer in layer_list:
                val_layer = torch.cat([val_pos_acts[layer], val_neg_acts[layer]], dim=0)
                if is_rank_zero():
                    torch.save(val_layer, activations_dir / f"val_layer{layer}.pt")
            print(f"      Saved val: {len(layer_list)} layers (per-layer files)")
        else:
            val_pos_all = torch.stack([val_pos_acts[l] for l in layer_list], dim=1)
            val_neg_all = torch.stack([val_neg_acts[l] for l in layer_list], dim=1)
            val_acts = torch.cat([val_pos_all, val_neg_all], dim=0)

            if is_rank_zero():
                val_path = get_val_activation_path(experiment, trait, model_variant, component, position)
                torch.save(val_acts, val_path)
                print(f"      Saved val: {val_acts.shape} -> {val_path.name}")
        n_val_pos, n_val_neg = len(vp_items), len(vn_items)

    # Save metadata (rank 0 only under TP)
    if is_rank_zero():
        metadata = {
            'model': model.config.name_or_path,
            'trait': trait,
            'n_layers': n_layers,
            'hidden_dim': hidden_dim,
            'captured_layers': layer_list,
            'n_examples_pos': b0,
            'n_examples_neg': b1 - b0,
            'n_filtered_pos': n_filtered_pos,
            'n_filtered_neg': n_filtered_neg,
            'paired_filter': paired_filter,
            'n_excluded_by_pairing': n_excluded_by_pairing,
            'val_split': val_split,
            'n_val_pos': n_val_pos,
            'n_val_neg': n_val_neg,
            'position': position,
            'component': component,
            'activation_norms': activation_norms,
            'timestamp': datetime.now().isoformat(),
        }
        metadata_path = get_activation_metadata_path(experiment, trait, model_variant, component, position)
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

    # Free GPU memory
    del all_acts, pos_acts, neg_acts
    if not per_layer_mode:
        del train_acts, pos_all, neg_all
    if val_split > 0 and (val_pos or val_neg):
        del val_pos_acts, val_neg_acts
        if not per_layer_mode:
            del val_acts, val_pos_all, val_neg_all
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return len(layer_list)
