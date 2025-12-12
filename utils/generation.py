"""
Generation utilities with VRAM management and activation capture.

Input:
    - model: Loaded transformer model
    - tokenizer: Model tokenizer
    - prompts: Text prompts to generate from

Output:
    - Generated response strings (simple generation)
    - CaptureResult with activations (capture mode)

Usage:
    from utils.generation import generate_response, generate_batch, generate_with_capture

    # Simple generation
    response = generate_response(model, tokenizer, prompt)
    responses = generate_batch(model, tokenizer, prompts)

    # Generation with activation capture
    results = generate_with_capture(model, tokenizer, prompts, n_layers=26)
    for result in results:
        print(result.response_text)
        print(result.activations[16]['residual_out'].shape)  # [n_tokens, hidden_dim]
"""

import torch
from typing import List, Dict, Optional
from dataclasses import dataclass

from traitlens import HookManager


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class CaptureResult:
    """Result from generate_with_capture() containing text, tokens, and activations."""
    prompt_text: str
    response_text: str
    prompt_tokens: List[str]
    response_tokens: List[str]
    prompt_token_ids: List[int]
    response_token_ids: List[int]
    # layer -> sublayer -> tensor[n_tokens, hidden_dim]
    # sublayers: 'after_attn', 'residual_out', optionally 'attn_out'
    prompt_activations: Dict[int, Dict[str, torch.Tensor]]
    response_activations: Dict[int, Dict[str, torch.Tensor]]


# ============================================================================
# Simple Generation (no hooks)
# ============================================================================

def generate_response(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 128,
    temperature: float = 0.0,
) -> str:
    """Generate a response from the model."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature if temperature > 0 else None,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1]:],
        skip_special_tokens=True,
    )
    return response.strip()


def generate_batch(
    model,
    tokenizer,
    prompts: List[str],
    max_new_tokens: int = 128,
    temperature: float = 0.0,
) -> List[str]:
    """Generate responses for a batch of prompts in parallel."""
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
    ).to(model.device)

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


# ============================================================================
# VRAM Utilities
# ============================================================================

def get_available_vram_gb() -> float:
    """Get available VRAM in GB. Falls back to conservative estimate."""
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        return props.total_memory / (1024 ** 3)
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        # MPS (Apple Silicon) - conservative estimate
        return 8.0
    return 8.0  # Fallback


def estimate_vram_gb(
    num_layers: int,
    hidden_size: int,
    num_kv_heads: int,
    head_dim: int,
    batch_size: int,
    seq_len: int,
    model_size_gb: float = 5.0,
    dtype_bytes: int = 2,
) -> float:
    """
    Estimate VRAM usage for batched generation.

    Args:
        num_layers: Number of transformer layers
        hidden_size: Hidden dimension
        num_kv_heads: Number of KV heads (for GQA)
        head_dim: Dimension per head
        batch_size: Total batch size
        seq_len: Maximum sequence length (prompt + generated)
        model_size_gb: Base model size in GB (default 5.0 for Gemma 2B bf16)
        dtype_bytes: Bytes per element (2 for bf16/fp16)

    Returns:
        Estimated VRAM in GB
    """
    # KV cache: 2 (K,V) x num_kv_heads x head_dim x seq_len x batch x layers x dtype
    kv_cache_bytes = 2 * num_kv_heads * head_dim * seq_len * batch_size * num_layers * dtype_bytes

    # Activation buffer (rough estimate): hidden_size x batch x seq_len x dtype x multiplier
    activation_bytes = hidden_size * batch_size * seq_len * dtype_bytes * 4  # 4x for intermediate

    total_bytes = kv_cache_bytes + activation_bytes
    total_gb = total_bytes / (1024 ** 3)

    return model_size_gb + total_gb


def calculate_max_batch_size(
    model,
    available_vram_gb: float,
    seq_len: int = 160,
    model_size_gb: float = 5.0,
) -> int:
    """
    Calculate maximum batch size that fits in available VRAM.

    Args:
        model: The transformer model (to get config)
        available_vram_gb: Available VRAM in GB
        seq_len: Expected max sequence length
        model_size_gb: Base model size

    Returns:
        Maximum safe batch size
    """
    config = model.config
    num_layers = config.num_hidden_layers
    hidden_size = config.hidden_size
    num_kv_heads = getattr(config, "num_key_value_heads", 4)
    head_dim = getattr(config, "head_dim", hidden_size // config.num_attention_heads)

    # Binary search for max batch size
    low, high = 1, 256
    while low < high:
        mid = (low + high + 1) // 2
        vram = estimate_vram_gb(
            num_layers, hidden_size, num_kv_heads, head_dim,
            mid, seq_len, model_size_gb
        )
        if vram <= available_vram_gb * 0.85:  # 85% safety margin
            low = mid
        else:
            high = mid - 1

    return low


# ============================================================================
# Generation with Activation Capture
# ============================================================================

def _create_storage(n_layers: int, capture_attn: bool = False) -> Dict:
    """Create storage for residual stream capture."""
    base = {'after_attn': [], 'residual_out': []}
    if capture_attn:
        base['attn_out'] = []
    return {i: {k: [] for k in base} for i in range(n_layers)}


def _setup_hooks(
    hook_manager: HookManager,
    storage: Dict,
    n_layers: int,
    mode: str,
    batch_idx: Optional[int] = None,
    capture_attn: bool = False
):
    """
    Register hooks for residual stream capture.

    Args:
        hook_manager: HookManager instance
        storage: Dict to store captured activations
        n_layers: Number of layers
        mode: 'prompt' (capture all positions) or 'response' (capture last position)
        batch_idx: If not None, only capture this batch index
        capture_attn: Whether to also capture attn_out
    """
    for i in range(n_layers):
        def make_layer_hook(layer_idx):
            def hook(module, inp, out):
                out_t = out[0] if isinstance(out, tuple) else out

                if batch_idx is not None:
                    out_t = out_t[batch_idx:batch_idx+1]

                if mode == 'response':
                    storage[layer_idx]['residual_out'].append(out_t[:, -1, :].detach().cpu())
                else:
                    storage[layer_idx]['residual_out'].append(out_t.detach().cpu())
            return hook
        hook_manager.add_forward_hook(f"model.layers.{i}", make_layer_hook(i))

        def make_mlp_hook(layer_idx):
            def hook(module, inp, out):
                inp_t = inp[0] if isinstance(inp, tuple) else inp

                if batch_idx is not None:
                    inp_t = inp_t[batch_idx:batch_idx+1]

                if mode == 'response':
                    storage[layer_idx]['after_attn'].append(inp_t[:, -1, :].detach().cpu())
                else:
                    storage[layer_idx]['after_attn'].append(inp_t.detach().cpu())
            return hook
        hook_manager.add_forward_hook(f"model.layers.{i}.mlp", make_mlp_hook(i))

        if capture_attn:
            def make_attn_hook(layer_idx):
                def hook(module, inp, out):
                    out_t = out[0] if isinstance(out, tuple) else out

                    if batch_idx is not None:
                        out_t = out_t[batch_idx:batch_idx+1]

                    if mode == 'response':
                        storage[layer_idx]['attn_out'].append(out_t[:, -1, :].detach().cpu())
                    else:
                        storage[layer_idx]['attn_out'].append(out_t.detach().cpu())
                return hook
            hook_manager.add_forward_hook(f"model.layers.{i}.self_attn", make_attn_hook(i))


def generate_with_capture(
    model,
    tokenizer,
    prompts: List[str],
    n_layers: int = None,
    batch_size: int = None,
    max_new_tokens: int = 50,
    temperature: float = 0.7,
    capture_attn: bool = False,
    show_progress: bool = True,
    yield_per_batch: bool = False,
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
        capture_attn: Whether to also capture attn_out
        show_progress: Whether to show progress bars
        yield_per_batch: If True, yield List[CaptureResult] after each batch (generator mode).
                         If False, return all results at end (default, backwards compatible).

    Returns:
        If yield_per_batch=False: List of CaptureResult, one per prompt
        If yield_per_batch=True: Generator yielding (batch_results, batch_prompts) tuples
    """
    from tqdm import tqdm

    if n_layers is None:
        n_layers = model.config.num_hidden_layers

    if batch_size is None:
        batch_size = min(8, calculate_max_batch_size(model, get_available_vram_gb()))

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    hidden_size = model.config.hidden_size

    # Process in batches
    batches = [prompts[i:i+batch_size] for i in range(0, len(prompts), batch_size)]

    if yield_per_batch:
        # Generator mode: yield after each batch for incremental saving
        def _generator():
            batch_iter = tqdm(batches, desc="Batches") if show_progress else batches
            for batch_prompts in batch_iter:
                batch_results = _capture_batch(
                    model, tokenizer, batch_prompts, n_layers, hidden_size,
                    max_new_tokens, temperature, capture_attn, show_progress=False
                )
                yield batch_results, batch_prompts
        return _generator()
    else:
        # Blocking mode: collect all results (backwards compatible)
        results = []
        for batch_prompts in (tqdm(batches, desc="Batches") if show_progress else batches):
            batch_results = _capture_batch(
                model, tokenizer, batch_prompts, n_layers, hidden_size,
                max_new_tokens, temperature, capture_attn, show_progress
            )
            results.extend(batch_results)
        return results


def _capture_batch(
    model, tokenizer, prompts: List[str], n_layers: int, hidden_size: int,
    max_new_tokens: int, temperature: float, capture_attn: bool, show_progress: bool
) -> List[CaptureResult]:
    """Capture activations for a single batch with TRUE batching (1 forward pass)."""

    # Tokenize with padding (left-pad for generation)
    tokenizer.padding_side = 'left'
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
    batch_size = inputs.input_ids.shape[0]

    # Track actual prompt lengths (excluding padding)
    prompt_lens = inputs.attention_mask.sum(dim=1).tolist()

    # Storage: [layer][sublayer] -> list of [batch, hidden] tensors
    prompt_storage = _create_storage(n_layers, capture_attn)
    response_storage = _create_storage(n_layers, capture_attn)

    # ================================================================
    # PROMPT PHASE: Single forward pass for entire batch
    # ================================================================
    with HookManager(model) as hooks:
        _setup_hooks(hooks, prompt_storage, n_layers, 'prompt', batch_idx=None, capture_attn=capture_attn)
        with torch.no_grad():
            model(**inputs)

    # ================================================================
    # RESPONSE PHASE: Token-by-token generation (1 forward pass per token)
    # ================================================================
    context = inputs.input_ids.clone()
    attention_mask = inputs.attention_mask.clone()
    active = torch.ones(batch_size, dtype=torch.bool, device=model.device)
    generated_ids = [[] for _ in range(batch_size)]

    from tqdm import tqdm
    gen_iter = range(max_new_tokens)
    if show_progress and batch_size == 1:
        gen_iter = tqdm(gen_iter, desc="Generating", leave=False)

    for _ in gen_iter:
        # Single forward pass with hooks capturing all batch items
        with HookManager(model) as hooks:
            _setup_hooks(hooks, response_storage, n_layers, 'response', batch_idx=None, capture_attn=capture_attn)
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

    # ================================================================
    # PACKAGE RESULTS - split batch storage into per-item results
    # ================================================================
    results = []
    n_response_tokens = len(response_storage[0]['residual_out']) if response_storage[0]['residual_out'] else 0

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
                if response_storage[layer_idx][key]:
                    # Stack all response tokens: [n_tokens, batch, hidden]
                    stacked = torch.stack(response_storage[layer_idx][key], dim=0)
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


# ============================================================================
# Legacy API (for backwards compatibility)
# ============================================================================

def capture_single(
    model, tokenizer, prompt_text: str, n_layers: int,
    max_new_tokens: int = 50, temperature: float = 0.7,
    capture_attn: bool = False
) -> CaptureResult:
    """
    Capture activations for a single prompt.

    Wrapper around generate_with_capture for single-prompt use case.
    """
    results = generate_with_capture(
        model, tokenizer, [prompt_text], n_layers,
        batch_size=1, max_new_tokens=max_new_tokens,
        temperature=temperature, capture_attn=capture_attn,
        show_progress=False
    )
    return results[0]
