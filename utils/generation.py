"""
Generation utilities with activation capture.

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

import os

import torch
from typing import List, Dict, Optional
from dataclasses import dataclass

from core import HookManager, get_hook_path
from utils.model import get_layer_path_prefix, tokenize_batch


from utils.vram import calculate_max_batch_size, get_gpu_stats


def get_think_end_token_id(tokenizer) -> Optional[int]:
    """Detect </think> token for reasoning models (DeepSeek-R1, Kimi-K2, etc).

    Returns the token ID if </think> is a single dedicated token in the
    vocabulary (thinking model), None otherwise (regular model).
    """
    ids = tokenizer.encode('</think>', add_special_tokens=False)
    if len(ids) == 1:
        return ids[0]
    return None


class GPUMonitor:
    """Track peak GPU memory during a stage."""

    def __init__(self, stage_name: str):
        self.stage_name = stage_name
        self.start_mem = 0.0
        self.peak_mem = 0.0
        self.start_time = 0.0

    def __enter__(self):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        self.start_mem = get_gpu_memory_gb()
        self.start_time = time.time()
        return self

    def __exit__(self, *args):
        pass

    def get_peak_gb(self) -> float:
        """Get peak memory usage since start (CUDA only)."""
        if torch.cuda.is_available():
            # PyTorch tracks peak across all devices
            peak_bytes = max(
                torch.cuda.max_memory_allocated(i)
                for i in range(torch.cuda.device_count())
            )
            return peak_bytes / (1024 ** 3)
        return self.start_mem

    def report(self, n_items: int = None) -> str:
        """Generate end-of-stage report string."""
        elapsed = time.time() - self.start_time
        current = get_gpu_memory_gb()
        peak = self.get_peak_gb()

        parts = [f"{elapsed:.1f}s"]
        if n_items:
            rate = n_items / elapsed if elapsed > 0 else 0
            parts.append(f"{rate:.1f}/s")
        if torch.cuda.is_available():
            parts.append(f"peak {peak:.1f}GB")
            parts.append(f"now {current:.1f}GB")

        return " | ".join(parts)


def print_gpu_memory_report():
    """Print detailed GPU memory breakdown for debugging."""
    if not torch.cuda.is_available():
        print("No CUDA available")
        return

    print("\n" + "="*60)
    print("GPU MEMORY REPORT")
    print("="*60)

    for i in range(torch.cuda.device_count()):
        free, total = torch.cuda.mem_get_info(i)
        allocated = torch.cuda.memory_allocated(i)
        reserved = torch.cuda.memory_reserved(i)
        cached = reserved - allocated

        print(f"\nGPU {i}:")
        print(f"  Total:     {total / 1e9:.1f} GB")
        print(f"  Free:      {free / 1e9:.1f} GB  (available for new allocations)")
        print(f"  Used:      {(total - free) / 1e9:.1f} GB")
        print(f"    ├─ Allocated: {allocated / 1e9:.2f} GB  (active tensors)")
        print(f"    └─ Cached:    {cached / 1e9:.2f} GB  (PyTorch cache, reclaimable)")

    print("\n" + "="*60)


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
    # components: 'residual' (layer output), 'attn_out' (raw attn output), optionally 'mlp_out'
    # Note: attn_out/mlp_out are raw outputs (before post-norm). For true contributions, use attn_contribution/mlp_contribution.
    prompt_activations: Dict[int, Dict[str, torch.Tensor]]
    response_activations: Dict[int, Dict[str, torch.Tensor]]


_GENERATION_MEMORY_LOG = bool(os.environ.get('GENERATION_MEMORY_LOG', ''))


def _gpu_mem_snapshot(label: str, device=None):
    """Print per-GPU memory snapshot if GENERATION_MEMORY_LOG is set."""
    if not _GENERATION_MEMORY_LOG or not torch.cuda.is_available():
        return
    n = torch.cuda.device_count()
    parts = []
    for i in range(n):
        torch.cuda.synchronize(i)
        free, total = torch.cuda.mem_get_info(i)
        alloc = torch.cuda.memory_allocated(i)
        resv = torch.cuda.memory_reserved(i)
        parts.append(f"[{i}]a={alloc/1e9:.1f}/r={resv/1e9:.1f}/f={free/1e9:.1f}")
    print(f"    MEM {label}: {' '.join(parts)}")


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

    _gpu_mem_snapshot(f"pre_tokenize(batch={len(prompts)})")

    batch = tokenize_batch(prompts, tokenizer)  # Auto-detects add_special_tokens
    inputs = {k: v.to(device) for k, v in batch.items() if k != 'lengths'}

    _gpu_mem_snapshot(f"pre_generate(seq_len={inputs['input_ids'].shape[1]})")

    # Reset peak stats for accurate peak tracking
    if _GENERATION_MEMORY_LOG and torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            torch.cuda.reset_peak_memory_stats(i)

    # For thinking models, stop generation at </think> to capture only the
    # reasoning block. This keeps scored content consistent (always thinking,
    # never a mix of thinking + answer).
    eos_ids = tokenizer.eos_token_id
    think_end_id = get_think_end_token_id(tokenizer)
    if think_end_id is not None:
        eos_ids = [tokenizer.eos_token_id, think_end_id]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature if temperature > 0 else None,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=eos_ids,
        )

    _gpu_mem_snapshot(f"post_generate(out_len={outputs.shape[1]})")

    # Show peak allocated during generate (the real high-water mark)
    if _GENERATION_MEMORY_LOG and torch.cuda.is_available():
        parts = []
        for i in range(torch.cuda.device_count()):
            peak = torch.cuda.max_memory_allocated(i)
            alloc = torch.cuda.memory_allocated(i)
            parts.append(f"[{i}]peak={peak/1e9:.1f}G/now={alloc/1e9:.1f}G")
        print(f"    MEM peak_during_generate: {' '.join(parts)}")

    # Decode each response, skipping the input tokens
    # With left-padding, all inputs are padded to same length, so use shape[1]
    input_len = inputs['input_ids'].shape[1]
    responses = []
    for output in outputs:
        response = tokenizer.decode(
            output[input_len:],
            skip_special_tokens=True,
        )
        # Strip thinking tags that survive skip_special_tokens (they're special=False)
        if think_end_id is not None:
            response = response.replace('</think>', '').replace('<think>', '')
        responses.append(response.strip())

    # Explicitly free generate outputs
    del outputs, inputs
    _gpu_mem_snapshot("post_cleanup")

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

    Tokenization auto-detects BOS from text content (via tokenize_batch).

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

    from utils.distributed import is_tp_mode

    # Calculate batch size from free VRAM
    max_input_len = max(len(tokenizer.encode(p)) for p in prompts)
    max_seq_len = max_input_len + max_new_tokens
    batch_size = calculate_max_batch_size(model, max_seq_len, mode='generation')

    # Use cached working batch size if sequence length is similar (within 2x)
    cached_bs = getattr(model, '_working_batch_size', None)
    cached_seq = getattr(model, '_working_batch_seq_len', 0)
    if cached_bs and cached_seq > 0 and max_seq_len <= cached_seq * 2:
        batch_size = min(batch_size, cached_bs)

    # Under TP, sync batch size across ranks (use min to prevent OOM divergence)
    tp = is_tp_mode()
    if tp:
        import torch.distributed as dist
        bs_tensor = torch.tensor([batch_size], device='cuda')
        dist.all_reduce(bs_tensor, op=dist.ReduceOp.MIN)
        batch_size = int(bs_tensor.item())

    all_responses = []
    i = 0

    while i < len(prompts):
        batch = prompts[i:i + batch_size]
        oom_flag = torch.zeros(1, device='cuda') if tp else None
        oom = False
        try:
            responses = _generate_batch_raw(model, tokenizer, batch, max_new_tokens, temperature)
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            # MPS raises RuntimeError for OOM, CUDA has specific error
            if "out of memory" not in str(e).lower() and not isinstance(e, torch.cuda.OutOfMemoryError):
                raise
            # Clear traceback chains that pin CUDA tensors via stack frame locals.
            # Must happen inside except block to access the exception, but gc/empty_cache
            # must happen OUTSIDE so Python thread state no longer roots the exception.
            import traceback as tb_mod
            if e.__traceback__:
                tb_mod.clear_frames(e.__traceback__)
            e.__traceback__ = None
            # Clear chained exceptions' tracebacks too
            for chained in (e.__context__, e.__cause__):
                if chained and hasattr(chained, '__traceback__') and chained.__traceback__:
                    tb_mod.clear_frames(chained.__traceback__)
                    chained.__traceback__ = None
            del e
            oom = True

        # OOM cleanup OUTSIDE except block — thread state no longer roots the exception
        if oom:
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if tp:
                oom_flag.fill_(1)
            else:
                if batch_size == 1:
                    raise RuntimeError("OOM even with batch_size=1")
                batch_size = max(1, batch_size // 2)
                if torch.cuda.is_available() and torch.cuda.device_count() > 1:
                    frees = []
                    for d in range(torch.cuda.device_count()):
                        os_free = torch.cuda.mem_get_info(d)[0]
                        alloc_free = torch.cuda.memory_reserved(d) - torch.cuda.memory_allocated(d)
                        frees.append((os_free + alloc_free) / 1e9)
                    print(f"  OOM, reducing batch_size to {batch_size} (effective free: min={min(frees):.1f}GB)")
                else:
                    print(f"  OOM, reducing batch_size to {batch_size}")
                continue  # retry same batch
            responses = None

        # Under TP, sync OOM status — if ANY rank OOM'd, all halve and retry
        if tp:
            dist.all_reduce(oom_flag, op=dist.ReduceOp.MAX)
            if oom_flag.item() > 0:
                if batch_size == 1:
                    raise RuntimeError("OOM even with batch_size=1 (across TP ranks)")
                batch_size = max(1, batch_size // 2)
                print(f"  OOM on some rank, reducing batch_size to {batch_size}")
                continue  # all ranks retry same batch

        all_responses.extend(responses)
        i += batch_size
        # Cache successful batch size with sequence length context
        model._working_batch_size = batch_size
        model._working_batch_seq_len = max_seq_len

    return all_responses




def create_residual_storage(n_layers: int, capture_mlp: bool = False, layers: List[int] = None) -> Dict:
    """Create storage for residual stream capture.

    Components captured:
        - 'residual': layer output
        - 'attn_out': raw attention output (o_proj, before post-norm)
        - 'mlp_out': raw MLP output (down_proj, before post-norm) - optional
    """
    components = ['residual', 'attn_out']
    if capture_mlp:
        components.append('mlp_out')
    layer_indices = layers if layers is not None else list(range(n_layers))
    return {i: {k: [] for k in components} for i in layer_indices}


def setup_residual_hooks(
    hook_manager: HookManager,
    storage: Dict,
    n_layers: int,
    mode: str,
    capture_mlp: bool = False,
    layer_prefix: str = "model.layers",
    layers: List[int] = None,
):
    """
    Register hooks for residual stream capture. Uses get_hook_path for consistency.

    Args:
        hook_manager: HookManager instance
        storage: Dict from create_residual_storage()
        n_layers: Number of layers (used only if layers is None)
        mode: 'prompt' (capture all positions) or 'response' (capture last position)
        capture_mlp: Whether to also capture mlp_out
        layer_prefix: Path prefix (e.g. "model.layers" or "base_model.model.model.layers")
        layers: Specific layer indices to hook (default: all n_layers)

    Captures:
        - 'residual': layer output
        - 'attn_out': raw attention output (o_proj, before post-norm)
        - 'mlp_out': raw MLP output (down_proj, before post-norm) - if capture_mlp=True
    """
    def make_hook(layer_idx: int, component: str):
        def hook(module, inp, out):
            out_t = out[0] if isinstance(out, tuple) else out
            if mode == 'response':
                storage[layer_idx][component].append(out_t[:, -1, :].detach().cpu())
            else:
                storage[layer_idx][component].append(out_t.detach().cpu())
        return hook

    layer_indices = layers if layers is not None else list(range(n_layers))
    for i in layer_indices:
        # Layer output (residual stream)
        hook_manager.add_forward_hook(
            get_hook_path(i, 'residual', layer_prefix),
            make_hook(i, 'residual')
        )
        # Raw attention output (o_proj, before post-norm)
        hook_manager.add_forward_hook(
            get_hook_path(i, 'attn_out', layer_prefix),
            make_hook(i, 'attn_out')
        )
        # Raw MLP output (down_proj, before post-norm) - optional
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
    layers: List[int] = None,
):
    """
    Batched generation with per-token activation capture.

    Tokenization auto-detects BOS from text content (via tokenize_batch).

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
        layers: Specific layer indices to capture (default: all). Reduces hook overhead.

    Returns:
        If yield_per_batch=False: List of CaptureResult, one per prompt
        If yield_per_batch=True: Generator yielding (batch_results, batch_prompts) tuples
    """
    # Handle nested text_config for multimodal models (e.g., Gemma 3)
    config = model.config
    if hasattr(config, 'text_config'):
        config = config.text_config

    if n_layers is None:
        n_layers = config.num_hidden_layers

    if batch_size is None:
        # Estimate max_seq_len from prompts + max_new_tokens
        max_input_len = max(len(tokenizer.encode(p)) for p in prompts) if prompts else 100
        max_seq_len = max_input_len + max_new_tokens
        batch_size = calculate_max_batch_size(model, max_seq_len, mode='generation')

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    hidden_size = config.hidden_size

    # Process in batches
    batches = [prompts[i:i+batch_size] for i in range(0, len(prompts), batch_size)]

    from tqdm import tqdm

    if yield_per_batch:
        # Generator mode: yield after each batch for incremental saving
        def _generator():
            pbar = tqdm(batches, desc="Batches") if show_progress else batches
            for batch_prompts in pbar:
                batch_results = _capture_batch(
                    model, tokenizer, batch_prompts, n_layers, hidden_size,
                    max_new_tokens, temperature, capture_mlp, show_progress=False,
                    layers=layers,
                )
                if show_progress and hasattr(pbar, 'set_postfix_str'):
                    pbar.set_postfix_str(get_gpu_stats())
                yield batch_results, batch_prompts
        return _generator()
    else:
        # Blocking mode: collect all results (backwards compatible)
        results = []
        pbar = tqdm(batches, desc="Batches") if show_progress else batches
        for batch_prompts in pbar:
            batch_results = _capture_batch(
                model, tokenizer, batch_prompts, n_layers, hidden_size,
                max_new_tokens, temperature, capture_mlp, show_progress,
                layers=layers,
            )
            if show_progress and hasattr(pbar, 'set_postfix_str'):
                pbar.set_postfix_str(get_gpu_stats())
            results.extend(batch_results)
        return results


def _capture_batch(
    model, tokenizer, prompts: List[str], n_layers: int, hidden_size: int,
    max_new_tokens: int, temperature: float, capture_mlp: bool, show_progress: bool,
    layers: List[int] = None,
) -> List[CaptureResult]:
    """Capture activations for a single batch with TRUE batching (1 forward pass)."""

    # Get layer path prefix (handles PeftModel wrapper)
    layer_prefix = get_layer_path_prefix(model)

    # Tokenize with left padding for generation (auto-detects add_special_tokens)
    batch = tokenize_batch(prompts, tokenizer)
    inputs = {k: v.to(model.device) for k, v in batch.items() if k != 'lengths'}
    batch_size = inputs['input_ids'].shape[0]
    prompt_lens = batch['lengths']

    # Storage: [layer][component] -> list of tensors
    prompt_storage = create_residual_storage(n_layers, capture_mlp, layers=layers)
    response_storage = create_residual_storage(n_layers, capture_mlp, layers=layers)

    # Prompt phase: single forward pass
    with HookManager(model) as hooks:
        setup_residual_hooks(hooks, prompt_storage, n_layers, 'prompt', capture_mlp=capture_mlp, layer_prefix=layer_prefix, layers=layers)
        with torch.no_grad():
            model(**inputs)

    # Response phase: token-by-token generation with KV cache
    context = inputs['input_ids'].clone()
    attention_mask = inputs['attention_mask'].clone()
    active = torch.ones(batch_size, dtype=torch.bool, device=model.device)
    generated_ids = [[] for _ in range(batch_size)]
    past_key_values = None  # KV cache for O(n) generation instead of O(n²)

    # Stop at EOS or </think> for reasoning models (DeepSeek-R1, Kimi-K2, etc.)
    stop_ids = {tokenizer.eos_token_id}
    think_end_id = get_think_end_token_id(tokenizer)
    if think_end_id is not None:
        stop_ids.add(think_end_id)

    gen_iter = range(max_new_tokens)
    if show_progress and batch_size == 1:
        gen_iter = tqdm(gen_iter, desc="Generating", leave=False)

    for _ in gen_iter:
        # Single forward pass with hooks capturing all batch items
        with HookManager(model) as hooks:
            setup_residual_hooks(hooks, response_storage, n_layers, 'response', capture_mlp=capture_mlp, layer_prefix=layer_prefix, layers=layers)
            with torch.no_grad():
                outputs = model(
                    input_ids=context,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                )

        # Update KV cache for next iteration
        past_key_values = outputs.past_key_values

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
                if next_ids[b].item() in stop_ids:
                    active[b] = False

        # Update context to just new tokens (KV cache handles history)
        context = next_ids.unsqueeze(1)
        # Extend attention mask (tells model which cached positions to attend to)
        attention_mask = torch.cat([attention_mask, torch.ones(batch_size, 1, device=model.device, dtype=attention_mask.dtype)], dim=1)

        if not active.any():
            break

    # Package results
    results = []
    for b in range(batch_size):
        # Get actual prompt tokens (excluding padding)
        prompt_start = inputs['input_ids'].shape[1] - prompt_lens[b]
        prompt_ids = inputs['input_ids'][b, prompt_start:].tolist()
        prompt_tokens = [tokenizer.decode([tid]) for tid in prompt_ids]

        # Filter out </think> stop token from generated IDs for clean decoding
        gen_ids = generated_ids[b]
        if think_end_id is not None:
            gen_ids = [tid for tid in gen_ids if tid != think_end_id]
        response_tokens = [tokenizer.decode([tid]) for tid in gen_ids]
        response_text = tokenizer.decode(gen_ids, skip_special_tokens=True)

        # Extract this batch item's activations from shared storage
        prompt_acts = {}
        response_acts = {}

        layer_indices = layers if layers is not None else list(range(n_layers))
        for layer_idx in layer_indices:
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


