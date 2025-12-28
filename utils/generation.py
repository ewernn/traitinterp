"""
Generation utilities with VRAM management and activation capture.

Input:
    - model: Loaded transformer model
    - tokenizer: Model tokenizer
    - prompts: Text prompts to generate from

Output:
    - Generated response strings (batch generation)
    - CaptureResult with activations (capture mode)

Usage:
    from utils.generation import generate_batch, generate_with_capture

    # Batch generation (auto batch size, OOM recovery)
    responses = generate_batch(model, tokenizer, prompts)
    response = generate_batch(model, tokenizer, [prompt])[0]  # single prompt

    # Generation with activation capture (captures all layers by default)
    results = generate_with_capture(model, tokenizer, prompts)
    for result in results:
        print(result.response_text)
        print(result.response_activations[0]['residual'].shape)  # [n_tokens, hidden_dim]
"""

import torch
from typing import List, Dict, Optional
from dataclasses import dataclass

from core import HookManager, get_hook_path
from utils.model import get_layer_path_prefix


@dataclass
class CaptureResult:
    """Result from generate_with_capture() containing text, tokens, and activations."""
    prompt_text: str
    response_text: str
    prompt_tokens: List[str]
    response_tokens: List[str]
    prompt_token_ids: List[int]
    response_token_ids: List[int]
    # layer -> component -> tensor[n_tokens, hidden_dim]
    # components: 'residual' (layer output), 'attn_out' (attention contribution), optionally 'mlp_out'
    # Compute: after_attn[L] = residual[L-1] + attn_out[L]
    prompt_activations: Dict[int, Dict[str, torch.Tensor]]
    response_activations: Dict[int, Dict[str, torch.Tensor]]


def _generate_batch_raw(
    model,
    tokenizer,
    prompts: List[str],
    max_new_tokens: int,
    temperature: float,
) -> List[str]:
    """Core batch generation without OOM handling. Internal use only."""
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Handle multi-GPU models (device_map="auto")
    device = getattr(model, 'device', None)
    if device is None or str(device) == 'meta':
        device = next(model.parameters()).device

    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature if temperature > 0 else None,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode each response, skipping the input tokens
    responses = []
    for i, output in enumerate(outputs):
        input_len = inputs.attention_mask[i].sum().item()
        response = tokenizer.decode(
            output[input_len:],
            skip_special_tokens=True,
        )
        responses.append(response.strip())

    return responses


def generate_batch(
    model,
    tokenizer,
    prompts: List[str],
    max_new_tokens: int = 256,
    temperature: float = 0.0,
) -> List[str]:
    """
    Generate responses with automatic batch size calculation and OOM recovery.

    Args:
        model: The model to generate with
        tokenizer: The tokenizer
        prompts: List of prompts to generate responses for
        max_new_tokens: Maximum tokens to generate per prompt
        temperature: Sampling temperature (0 for greedy)

    Returns:
        List of generated responses
    """
    if not prompts:
        return []

    # Calculate batch size from free VRAM
    max_input_len = max(len(tokenizer.encode(p)) for p in prompts)
    max_seq_len = max_input_len + max_new_tokens
    batch_size = calculate_max_batch_size(model, max_seq_len)

    # Use cached working batch size if available (from previous OOM recovery)
    batch_size = min(batch_size, getattr(model, '_working_batch_size', batch_size))

    all_responses = []
    i = 0

    while i < len(prompts):
        batch = prompts[i:i + batch_size]
        try:
            responses = _generate_batch_raw(model, tokenizer, batch, max_new_tokens, temperature)
            all_responses.extend(responses)
            i += batch_size
            # Cache successful batch size
            model._working_batch_size = batch_size
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            # MPS raises RuntimeError for OOM, CUDA has specific error
            if "out of memory" not in str(e).lower() and not isinstance(e, torch.cuda.OutOfMemoryError):
                raise
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if batch_size == 1:
                raise RuntimeError("OOM even with batch_size=1")
            batch_size = max(1, batch_size // 2)
            print(f"  OOM, reducing batch_size to {batch_size}")
            # Don't advance i, retry same batch with smaller size

    return all_responses


def get_free_vram_gb() -> float:
    """Get free VRAM across all GPUs in GB (after model loaded).

    For MPS (Metal), queries available system memory (unified memory architecture).
    Override with MPS_MEMORY_GB environment variable if needed.
    """
    if torch.cuda.is_available():
        total_free = 0
        for i in range(torch.cuda.device_count()):
            free, _ = torch.cuda.mem_get_info(i)
            total_free += free
        return total_free / (1024 ** 3)
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        import os
        if limit := os.environ.get('MPS_MEMORY_GB'):
            return float(limit)
        # Query available unified memory, use 50% to leave headroom for system
        import psutil
        available_gb = psutil.virtual_memory().available / (1024 ** 3)
        return available_gb * 0.5
    return 4.0  # Fallback


def estimate_kv_cache_gb(
    model,
    seq_len: int,
    batch_size: int = 1,
    dtype_bytes: int = 2,
) -> float:
    """
    Estimate KV cache memory for batched generation.

    Args:
        model: The transformer model (to get config)
        seq_len: Maximum sequence length (prompt + output)
        batch_size: Batch size
        dtype_bytes: Bytes per element (2 for bf16/fp16)

    Returns:
        Estimated KV cache size in GB
    """
    config = model.config
    num_layers = config.num_hidden_layers
    num_kv_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
    head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)

    # KV cache: 2 (K,V) × num_kv_heads × head_dim × seq_len × batch × layers × dtype
    kv_bytes = 2 * num_kv_heads * head_dim * seq_len * batch_size * num_layers * dtype_bytes

    # Add ~2x for activations and intermediate buffers
    total_bytes = kv_bytes * 3

    return total_bytes / (1024 ** 3)


def calculate_max_batch_size(
    model,
    max_seq_len: int,
    safety_margin: float = 0.85,
) -> int:
    """
    Calculate maximum batch size from free VRAM after model is loaded.

    Args:
        model: Loaded model (already using VRAM)
        max_seq_len: max_input_tokens + max_output_tokens
        safety_margin: Fraction of free VRAM to use (default 85%)

    Returns:
        Maximum safe batch size (at least 1)
    """
    free_gb = get_free_vram_gb()
    usable_gb = free_gb * safety_margin

    # Memory per sequence at max length
    gb_per_seq = estimate_kv_cache_gb(model, max_seq_len, batch_size=1)

    if gb_per_seq <= 0:
        return 1

    max_batch = int(usable_gb / gb_per_seq)
    return max(1, min(max_batch, 128))  # Clamp to reasonable range


def create_residual_storage(n_layers: int, capture_mlp: bool = False) -> Dict:
    """Create storage for residual stream capture.

    Components captured:
        - 'residual': layer output
        - 'attn_out': attention contribution (o_proj output)
        - 'mlp_out': MLP contribution (down_proj output) - optional
    """
    components = ['residual', 'attn_out']
    if capture_mlp:
        components.append('mlp_out')
    return {i: {k: [] for k in components} for i in range(n_layers)}


def setup_residual_hooks(
    hook_manager: HookManager,
    storage: Dict,
    n_layers: int,
    mode: str,
    capture_mlp: bool = False,
    layer_prefix: str = "model.layers"
):
    """
    Register hooks for residual stream capture. Uses get_hook_path for consistency.

    Args:
        hook_manager: HookManager instance
        storage: Dict from create_residual_storage()
        n_layers: Number of layers
        mode: 'prompt' (capture all positions) or 'response' (capture last position)
        capture_mlp: Whether to also capture mlp_out
        layer_prefix: Path prefix (e.g. "model.layers" or "base_model.model.model.layers")

    Captures:
        - 'residual': layer output
        - 'attn_out': attention contribution (o_proj output)
        - 'mlp_out': MLP contribution (down_proj output) - if capture_mlp=True
    """
    def make_hook(layer_idx: int, component: str):
        def hook(module, inp, out):
            out_t = out[0] if isinstance(out, tuple) else out
            if mode == 'response':
                storage[layer_idx][component].append(out_t[:, -1, :].detach().cpu())
            else:
                storage[layer_idx][component].append(out_t.detach().cpu())
        return hook

    for i in range(n_layers):
        # Layer output (residual stream)
        hook_manager.add_forward_hook(
            get_hook_path(i, 'residual', layer_prefix),
            make_hook(i, 'residual')
        )
        # Attention contribution (o_proj output)
        hook_manager.add_forward_hook(
            get_hook_path(i, 'attn_out', layer_prefix),
            make_hook(i, 'attn_out')
        )
        # MLP contribution (down_proj output) - optional
        if capture_mlp:
            hook_manager.add_forward_hook(
                get_hook_path(i, 'mlp_out', layer_prefix),
                make_hook(i, 'mlp_out')
            )


def generate_with_capture(
    model,
    tokenizer,
    prompts: List[str],
    n_layers: int = None,
    batch_size: int = None,
    max_new_tokens: int = 50,
    temperature: float = 0.7,
    capture_mlp: bool = False,
    show_progress: bool = True,
    yield_per_batch: bool = False,
    add_special_tokens: bool = True,
):
    """
    Batched generation with per-token activation capture.

    Args:
        model: Loaded transformer model
        tokenizer: Model tokenizer
        prompts: List of formatted prompts
        n_layers: Number of layers to capture (default: all)
        batch_size: Batch size (default: auto-calculate from VRAM)
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        capture_mlp: Whether to also capture mlp_out (down_proj output)
        show_progress: Whether to show progress bars
        yield_per_batch: If True, yield List[CaptureResult] after each batch (generator mode).
                         If False, return all results at end (default, backwards compatible).
        add_special_tokens: Whether to add special tokens (BOS). Set False for chat-templated prompts.

    Returns:
        If yield_per_batch=False: List of CaptureResult, one per prompt
        If yield_per_batch=True: Generator yielding (batch_results, batch_prompts) tuples
    """
    if n_layers is None:
        n_layers = model.config.num_hidden_layers

    if batch_size is None:
        # Estimate max_seq_len from prompts + max_new_tokens
        max_input_len = max(len(tokenizer.encode(p)) for p in prompts) if prompts else 100
        max_seq_len = max_input_len + max_new_tokens
        batch_size = calculate_max_batch_size(model, max_seq_len)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    hidden_size = model.config.hidden_size

    # Process in batches
    batches = [prompts[i:i+batch_size] for i in range(0, len(prompts), batch_size)]

    from tqdm import tqdm

    if yield_per_batch:
        # Generator mode: yield after each batch for incremental saving
        def _generator():
            batch_iter = tqdm(batches, desc="Batches") if show_progress else batches
            for batch_prompts in batch_iter:
                batch_results = _capture_batch(
                    model, tokenizer, batch_prompts, n_layers, hidden_size,
                    max_new_tokens, temperature, capture_mlp, show_progress=False,
                    add_special_tokens=add_special_tokens
                )
                yield batch_results, batch_prompts
        return _generator()
    else:
        # Blocking mode: collect all results (backwards compatible)
        results = []
        for batch_prompts in (tqdm(batches, desc="Batches") if show_progress else batches):
            batch_results = _capture_batch(
                model, tokenizer, batch_prompts, n_layers, hidden_size,
                max_new_tokens, temperature, capture_mlp, show_progress,
                add_special_tokens=add_special_tokens
            )
            results.extend(batch_results)
        return results


def _capture_batch(
    model, tokenizer, prompts: List[str], n_layers: int, hidden_size: int,
    max_new_tokens: int, temperature: float, capture_mlp: bool, show_progress: bool,
    add_special_tokens: bool = True
) -> List[CaptureResult]:
    """Capture activations for a single batch with TRUE batching (1 forward pass)."""

    # Get layer path prefix (handles PeftModel wrapper)
    layer_prefix = get_layer_path_prefix(model)

    # Tokenize with padding (left-pad for generation)
    tokenizer.padding_side = 'left'
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=add_special_tokens).to(model.device)
    batch_size = inputs.input_ids.shape[0]

    # Track actual prompt lengths (excluding padding)
    prompt_lens = inputs.attention_mask.sum(dim=1).tolist()

    # Storage: [layer][component] -> list of tensors
    prompt_storage = create_residual_storage(n_layers, capture_mlp)
    response_storage = create_residual_storage(n_layers, capture_mlp)

    # Prompt phase: single forward pass
    with HookManager(model) as hooks:
        setup_residual_hooks(hooks, prompt_storage, n_layers, 'prompt', capture_mlp=capture_mlp, layer_prefix=layer_prefix)
        with torch.no_grad():
            model(**inputs)

    # Response phase: token-by-token generation
    context = inputs.input_ids.clone()
    attention_mask = inputs.attention_mask.clone()
    active = torch.ones(batch_size, dtype=torch.bool, device=model.device)
    generated_ids = [[] for _ in range(batch_size)]

    gen_iter = range(max_new_tokens)
    if show_progress and batch_size == 1:
        gen_iter = tqdm(gen_iter, desc="Generating", leave=False)

    for _ in gen_iter:
        # Single forward pass with hooks capturing all batch items
        with HookManager(model) as hooks:
            setup_residual_hooks(hooks, response_storage, n_layers, 'response', capture_mlp=capture_mlp, layer_prefix=layer_prefix)
            with torch.no_grad():
                outputs = model(input_ids=context, attention_mask=attention_mask)

        # Sample next tokens
        logits = outputs.logits[:, -1, :]
        if temperature == 0:
            next_ids = logits.argmax(dim=-1)
        else:
            probs = torch.softmax(logits / temperature, dim=-1)
            next_ids = torch.multinomial(probs, 1).squeeze(-1)

        # Update sequences
        for b in range(batch_size):
            if active[b]:
                generated_ids[b].append(next_ids[b].item())
                if next_ids[b].item() == tokenizer.eos_token_id:
                    active[b] = False

        # Extend context
        context = torch.cat([context, next_ids.unsqueeze(1)], dim=1)
        attention_mask = torch.cat([attention_mask, torch.ones(batch_size, 1, device=model.device, dtype=attention_mask.dtype)], dim=1)

        if not active.any():
            break

    # Package results
    results = []
    for b in range(batch_size):
        # Get actual prompt tokens (excluding padding)
        prompt_start = inputs.input_ids.shape[1] - prompt_lens[b]
        prompt_ids = inputs.input_ids[b, prompt_start:].tolist()
        prompt_tokens = [tokenizer.decode([tid]) for tid in prompt_ids]

        response_tokens = [tokenizer.decode([tid]) for tid in generated_ids[b]]
        response_text = tokenizer.decode(generated_ids[b], skip_special_tokens=True)

        # Extract this batch item's activations from shared storage
        prompt_acts = {}
        response_acts = {}

        for layer_idx in range(n_layers):
            prompt_acts[layer_idx] = {}
            response_acts[layer_idx] = {}

            for key in prompt_storage[layer_idx]:
                # Prompt: shape [batch, seq_len, hidden] -> extract [seq_len, hidden] for this batch
                if prompt_storage[layer_idx][key]:
                    full_prompt = prompt_storage[layer_idx][key][0]  # [batch, seq_len, hidden]
                    # Trim to actual prompt length (remove left padding)
                    prompt_acts[layer_idx][key] = full_prompt[b, -prompt_lens[b]:, :].clone()
                else:
                    prompt_acts[layer_idx][key] = torch.empty(0, hidden_size)

                # Response: list of [batch, hidden] tensors -> stack and extract
                # Skip first capture (duplicates prompt[-1], captured before any token generated)
                if response_storage[layer_idx][key] and len(response_storage[layer_idx][key]) > 1:
                    # Stack all response tokens: [n_tokens, batch, hidden]
                    stacked = torch.stack(response_storage[layer_idx][key][1:], dim=0)
                    # Extract this batch item and trim to actual generated length
                    n_gen = len(generated_ids[b])
                    response_acts[layer_idx][key] = stacked[:n_gen, b, :].clone()
                else:
                    response_acts[layer_idx][key] = torch.empty(0, hidden_size)

        results.append(CaptureResult(
            prompt_text=prompts[b],
            response_text=response_text,
            prompt_tokens=prompt_tokens,
            response_tokens=response_tokens,
            prompt_token_ids=prompt_ids,
            response_token_ids=generated_ids[b],
            prompt_activations=prompt_acts,
            response_activations=response_acts,
        ))

    return results


