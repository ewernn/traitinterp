"""
Extract activations from generated responses.

Called by run_pipeline.py (stage 3).

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
from utils.generation import calculate_max_batch_size
from utils.model import pad_sequences, format_prompt
from core import MultiLayerCapture

if TYPE_CHECKING:
    from core import GenerationBackend


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
        response[-1]  → None (undeterminable - need response length)
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


def load_vetting_filter(experiment: str, trait: str, model_variant: str) -> dict:
    """Load failed indices from response vetting. Returns dict of indices to EXCLUDE."""
    vetting_file = get_path('extraction.trait', experiment=experiment, trait=trait, model_variant=model_variant) / 'vetting' / 'response_scores.json'
    if not vetting_file.exists():
        return {'positive': [], 'negative': []}
    with open(vetting_file) as f:
        data = json.load(f)
    return data.get('failed_indices', {'positive': [], 'negative': []})


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
        failed = load_vetting_filter(experiment, trait, model_variant)
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

    def extract_from_responses(responses: list[dict], label: str) -> dict[int, torch.Tensor]:
        # Calculate batch size locally for this split (different splits may have different max_seq_len)
        local_batch_size = batch_size  # Start with provided value or None
        all_activations = {layer: [] for layer in range(n_layers)}

        # Filter to responses with prompt and response text
        valid_responses = [item for item in responses if item.get('prompt') is not None and item.get('response') is not None]
        if not valid_responses:
            for layer in range(n_layers):
                all_activations[layer] = torch.empty(0)
            return all_activations

        # Derive full_text from prompt + response (apply formatting if chat template was used)
        formatted_prompts = [
            format_prompt(item['prompt'], tokenizer, use_chat_template=use_chat_template, system_prompt=item.get('system_prompt'))
            for item in valid_responses
        ]
        texts = [fp + item['response'] for fp, item in zip(formatted_prompts, valid_responses)]

        # Batch tokenize full texts
        has_bos = tokenizer.bos_token and texts[0].startswith(tokenizer.bos_token)
        all_input_ids_raw = tokenizer(texts, add_special_tokens=not has_bos, padding=False)['input_ids']
        all_input_ids = [torch.tensor(ids) for ids in all_input_ids_raw]

        # Batch tokenize formatted prompts to get prompt lengths
        prompt_has_bos = tokenizer.bos_token and formatted_prompts[0].startswith(tokenizer.bos_token)
        prompt_ids = tokenizer(formatted_prompts, add_special_tokens=not prompt_has_bos, padding=False)['input_ids']
        prompt_lens = [len(ids) for ids in prompt_ids]

        # Build items with pre-computed tokens
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

        if not items:
            for layer in range(n_layers):
                all_activations[layer] = torch.empty(0)
            return all_activations

        # Calculate batch size for this split (use provided or auto-calculate based on this split's max_seq_len)
        max_seq_len = max(item['seq_len'] for item in items)
        if local_batch_size is None:
            local_batch_size = calculate_max_batch_size(model, max_seq_len, mode='extraction')

        # Process in batches
        i = 0
        pbar = tqdm(total=len(items), desc=f"    {label}", leave=False)
        while i < len(items):
            batch_items = items[i:i + local_batch_size]

            try:
                # Pad sequences (left padding)
                pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
                batch = pad_sequences([item['input_ids'] for item in batch_items], pad_token_id)
                input_ids = batch['input_ids'].to(model.device)
                attention_mask = batch['attention_mask'].to(model.device)
                pad_offsets = batch['pad_offsets']

                # Forward pass with activation capture (keep on GPU for fast processing)
                with MultiLayerCapture(model, component=component, keep_on_gpu=True) as capture:
                    with torch.no_grad():
                        model(input_ids=input_ids, attention_mask=attention_mask)

                # Extract positions for each sample - all GPU ops, single CPU transfer per layer
                for layer in range(n_layers):
                    acts = capture.get(layer)  # [batch, seq, hidden] on GPU
                    if acts is None:
                        continue

                    # Process all samples for this layer on GPU
                    batch_acts = []
                    for b, item in enumerate(batch_items):
                        pad_offset = pad_offsets[b]
                        start_idx = item['start_idx'] + pad_offset
                        end_idx = item['end_idx'] + pad_offset
                        selected = acts[b, start_idx:end_idx, :]
                        act_out = selected.mean(dim=0) if selected.shape[0] > 1 else selected.squeeze(0)
                        batch_acts.append(act_out)

                    # Single CPU transfer per layer
                    batch_tensor = torch.stack(batch_acts).cpu()
                    for act in batch_tensor:
                        all_activations[layer].append(act)

                pbar.update(len(batch_items))
                i += local_batch_size

            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                # MPS raises RuntimeError for OOM, CUDA has specific error
                if "out of memory" not in str(e).lower() and not isinstance(e, torch.cuda.OutOfMemoryError):
                    raise
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if local_batch_size == 1:
                    raise RuntimeError("OOM even with batch_size=1")
                local_batch_size = max(1, local_batch_size // 2)
                print(f"\n      OOM, reducing batch_size to {local_batch_size}")
                # Don't advance i, retry same batch with smaller size

        pbar.close()

        for layer in range(n_layers):
            all_activations[layer] = torch.stack(all_activations[layer]) if all_activations[layer] else torch.empty(0)
        return all_activations

    # Extract training activations (each split calculates its own optimal batch size)
    pos_acts = extract_from_responses(train_pos, 'train_positive')
    neg_acts = extract_from_responses(train_neg, 'train_negative')

    # Combine and save training activations
    pos_all = torch.stack([pos_acts[l] for l in range(n_layers)], dim=1)
    neg_all = torch.stack([neg_acts[l] for l in range(n_layers)], dim=1)
    train_acts = torch.cat([pos_all, neg_all], dim=0)

    activation_path = get_activation_path(experiment, trait, model_variant, component, position)
    torch.save(train_acts, activation_path)
    print(f"      Saved train: {train_acts.shape} -> {activation_path.name}")

    # Save validation activations (same format as train)
    n_val_pos, n_val_neg = 0, 0
    if val_split > 0 and (val_pos or val_neg):
        val_pos_acts = extract_from_responses(val_pos, 'val_positive')
        val_neg_acts = extract_from_responses(val_neg, 'val_negative')

        val_pos_all = torch.stack([val_pos_acts[l] for l in range(n_layers)], dim=1)
        val_neg_all = torch.stack([val_neg_acts[l] for l in range(n_layers)], dim=1)
        val_acts = torch.cat([val_pos_all, val_neg_all], dim=0)

        val_path = get_val_activation_path(experiment, trait, model_variant, component, position)
        torch.save(val_acts, val_path)
        print(f"      Saved val: {val_acts.shape} -> {val_path.name}")
        n_val_pos, n_val_neg = len(val_pos), len(val_neg)

    # Compute activation norms (handle empty tensor case)
    if train_acts.numel() == 0:
        activation_norms = {layer: 0.0 for layer in range(n_layers)}
    else:
        activation_norms = {layer: round(train_acts[:, layer, :].norm(dim=-1).mean().item(), 2) for layer in range(n_layers)}

    # Save metadata
    hidden_dim = train_acts.shape[-1] if train_acts.numel() > 0 else config.hidden_size
    metadata = {
        'model': model.config.name_or_path,
        'trait': trait,
        'n_layers': n_layers,
        'hidden_dim': hidden_dim,
        'n_examples_pos': len(train_pos),
        'n_examples_neg': len(train_neg),
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

    # Free GPU memory - activation tensors can be large
    del train_acts, pos_all, neg_all, pos_acts, neg_acts
    if val_split > 0 and (val_pos or val_neg):
        del val_acts, val_pos_all, val_neg_all, val_pos_acts, val_neg_acts
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return n_layers
