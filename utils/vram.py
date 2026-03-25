"""GPU memory monitoring, VRAM estimation, and profiling utilities.

Input: Loaded model on GPU(s)
Output: Memory stats, batch size calculations, profiling results

Usage:
    from utils.vram import calculate_max_batch_size, get_free_vram_gb, format_duration
    from utils.vram import gpu_profile, memory_stats, find_cuda_tensors

    batch_size = calculate_max_batch_size(model, max_seq_len, mode='generation')

    with gpu_profile("forward pass"):
        model(**inputs)
    # Prints: [forward pass] 0.45s | peak 12.3GB | delta +2.1GB

    stats = memory_stats()  # {'allocated': 5.2, 'reserved': 8.0, 'free': 40.0, 'total': 80.0}
    leaked = find_cuda_tensors()  # [(shape, dtype, device, size_mb), ...]
"""

import gc
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass

import torch
import subprocess


def format_duration(seconds: float) -> str:
    """Format seconds as human-readable duration."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds / 60:.1f}m"
    return f"{seconds / 3600:.1f}h"


def get_gpu_stats() -> str:
    """Get GPU memory and utilization stats for progress bar display."""
    if not torch.cuda.is_available():
        return ""

    try:
        # Get stats for all GPUs
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.used,memory.total,utilization.gpu',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=1
        )
        if result.returncode != 0:
            return ""

        lines = result.stdout.strip().split('\n')
        stats = []
        for i, line in enumerate(lines):
            parts = [p.strip() for p in line.split(',')]
            if len(parts) >= 3:
                used_mb, total_mb, util = int(parts[0]), int(parts[1]), int(parts[2])
                used_gb = used_mb / 1024
                total_gb = total_mb / 1024
                stats.append(f"GPU{i}:{used_gb:.1f}/{total_gb:.0f}GB {util}%")

        return " | ".join(stats)
    except Exception:
        return ""


def get_gpu_memory_gb() -> float:
    """Get current GPU memory usage in GB (max across devices)."""
    if not torch.cuda.is_available():
        return 0.0
    max_used = 0.0
    for i in range(torch.cuda.device_count()):
        free, total = torch.cuda.mem_get_info(i)
        used = (total - free) / (1024 ** 3)
        max_used = max(max_used, used)
    return max_used


@dataclass
class ProfileResult:
    """Result from gpu_profile context manager."""
    elapsed: float = 0.0
    peak_memory_gb: float = 0.0
    start_memory_gb: float = 0.0
    end_memory_gb: float = 0.0
    delta_memory_gb: float = 0.0

    def __str__(self):
        return (
            f"{self.elapsed:.2f}s | "
            f"peak {self.peak_memory_gb:.1f}GB | "
            f"delta {self.delta_memory_gb:+.1f}GB"
        )


@contextmanager
def gpu_profile(name: str = "operation", print_result: bool = True):
    """Profile GPU time and memory for a code block.

    Uses torch.cuda.synchronize() before and after for accurate timing.

    Usage:
        with gpu_profile("forward pass"):
            model(**inputs)
        # Prints: [forward pass] 0.45s | peak 12.3GB | delta +2.1GB

        with gpu_profile("capture", print_result=False) as result:
            do_work()
        print(result.elapsed)
    """
    result = ProfileResult()

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        result.start_memory_gb = torch.cuda.memory_allocated() / 1e9

    start = time.perf_counter()

    try:
        yield result
    finally:
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        result.elapsed = time.perf_counter() - start

        if torch.cuda.is_available():
            result.peak_memory_gb = torch.cuda.max_memory_allocated() / 1e9
            result.end_memory_gb = torch.cuda.memory_allocated() / 1e9
            result.delta_memory_gb = result.end_memory_gb - result.start_memory_gb

        if print_result:
            print(f"[{name}] {result}")


def memory_stats() -> dict:
    """Get current GPU memory statistics.

    Returns dict with 'allocated', 'reserved', 'free', 'total' in GB.
    Returns zeros if CUDA not available.
    """
    if not torch.cuda.is_available():
        return {'allocated': 0.0, 'reserved': 0.0, 'free': 0.0, 'total': 0.0}

    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    free, total = torch.cuda.mem_get_info()

    return {
        'allocated': round(allocated, 2),
        'reserved': round(reserved, 2),
        'free': round(free / 1e9, 2),
        'total': round(total / 1e9, 2),
    }


def find_cuda_tensors() -> list:
    """Enumerate all live CUDA tensors for leak diagnosis.

    Walks gc.get_objects() and filters for CUDA tensors. Useful for answering
    "what's still on the GPU after I thought I cleaned up?"

    Returns list of (shape, dtype, device, size_mb) sorted by size descending.
    """
    tensors = []
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) and obj.is_cuda:
                size_mb = obj.element_size() * obj.nelement() / 1e6
                tensors.append((tuple(obj.shape), obj.dtype, str(obj.device), round(size_mb, 2)))
        except Exception:
            pass
    tensors.sort(key=lambda x: -x[3])
    return tensors


def tensor_size_gb(shape: tuple, dtype=torch.bfloat16) -> float:
    """Calculate tensor size in GB from shape and dtype."""
    numel = 1
    for dim in shape:
        numel *= dim
    bytes_per_elem = {
        torch.float32: 4, torch.float16: 2, torch.bfloat16: 2,
        torch.int64: 8, torch.int32: 4, torch.int16: 2, torch.int8: 1, torch.bool: 1,
    }.get(dtype, 2)
    return numel * bytes_per_elem / 1e9


def bandwidth_report(data_gb: float, elapsed: float) -> str:
    """Format bandwidth report: '4.6GB in 0.19s = 24.2 GB/s'."""
    bw = data_gb / elapsed if elapsed > 0 else 0
    return f"{data_gb:.1f}GB in {elapsed:.2f}s = {bw:.1f} GB/s"


def get_free_vram_gb(per_device: bool = True) -> float:
    """Get free VRAM in GB (after model loaded).

    Args:
        per_device: If True (default), return min across devices. This is correct
                    for batch size calculation since batches flow through each GPU.
                    If False, return sum (for total capacity checks).

    For MPS (Metal), queries available system memory (unified memory architecture).
    Override with MPS_MEMORY_GB environment variable if needed.

    Handles CUDA allocator fragmentation: after OOM recovery, the allocator may hold
    large "reserved" pools that mem_get_info reports as unavailable but are actually
    reusable. We use max(os_free, reserved - allocated) to account for this.
    """
    if torch.cuda.is_available():
        frees = []
        for i in range(torch.cuda.device_count()):
            os_free, total = torch.cuda.mem_get_info(i)
            # After OOM, reserved >> allocated. The gap is reusable by the allocator.
            allocated = torch.cuda.memory_allocated(i)
            reserved = torch.cuda.memory_reserved(i)
            allocator_free = reserved - allocated  # Free within the allocator's pool
            effective_free = os_free + allocator_free
            frees.append(effective_free)

        if per_device and len(frees) > 1:
            # Batch size limited by GPU with least free memory
            result = min(frees) / (1024 ** 3)
        else:
            result = sum(frees) / (1024 ** 3)
        return result
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
    # Handle nested text_config for multimodal models (e.g., Gemma 3)
    if hasattr(config, 'text_config'):
        config = config.text_config
    num_layers = config.num_hidden_layers
    num_kv_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
    head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads

    # MLA (Multi-head Latent Attention, e.g. DeepSeek V3 / Kimi K2):
    # K and V have different, larger head dimensions than the standard head_dim.
    # K caches qk_nope_head_dim + qk_rope_head_dim = 192, V caches v_head_dim = 128.
    k_head_dim = getattr(config, 'qk_nope_head_dim', 0) + getattr(config, 'qk_rope_head_dim', 0)
    v_head_dim = getattr(config, 'v_head_dim', 0)
    if k_head_dim > 0 and v_head_dim > 0:
        kv_per_tok_per_layer = num_kv_heads * (k_head_dim + v_head_dim) * dtype_bytes
    else:
        # Standard GQA/MHA: K and V share the same head_dim
        kv_per_tok_per_layer = 2 * num_kv_heads * head_dim * dtype_bytes

    # TP with attention sharding: KV heads are divided across GPUs
    from utils.distributed import is_tp_mode
    if is_tp_mode():
        import os
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        tp_plan = getattr(config, 'base_model_tp_plan', None) or {}
        if any('self_attn.kv_b_proj' in k or 'self_attn.k_proj' in k for k in tp_plan):
            kv_per_tok_per_layer //= world_size

    kv_bytes = kv_per_tok_per_layer * seq_len * batch_size * num_layers
    return kv_bytes / (1024 ** 3)


def estimate_forward_pass_gb(
    model,
    seq_len: int,
    batch_size: int = 1,
    dtype_bytes: int = None,
) -> float:
    """
    Estimate temporary GPU memory during forward pass (per GPU under TP).

    Accounts for attention scores (fp32 softmax), MLA projections, MLP/MoE
    intermediates, and MoE dispatch buffers. Under TP, uses sharded dimensions.
    """
    config = model.config
    if hasattr(config, 'text_config'):
        config = config.text_config

    if dtype_bytes is None:
        param = next(model.parameters())
        dtype_bytes = max(param.element_size(), 2)

    B, S = batch_size, seq_len
    hidden = config.hidden_size
    n_heads = config.num_attention_heads

    # TP sharding: detect world size and whether attention is sharded
    from utils.distributed import is_tp_mode
    tp = 1
    if is_tp_mode():
        tp = int(os.environ.get("WORLD_SIZE", "1"))
    tp_plan = getattr(config, 'base_model_tp_plan', None) or {}
    attn_sharded = any('self_attn' in k for k in tp_plan)
    heads_per_gpu = n_heads // tp if attn_sharded else n_heads

    # Attention scores: fp32 for eager (softmax upcast), bf16 for flash
    attn_impl = getattr(config, '_attn_implementation',
                        getattr(config, 'attn_implementation', 'eager'))
    uses_flash = 'flash' in str(attn_impl).lower()
    if uses_flash:
        attn_bytes = B * S * hidden * dtype_bytes
    else:
        attn_bytes = B * heads_per_gpu * S * S * 4  # fp32 softmax

    # MLA projection intermediates (if MLA architecture)
    q_lora = getattr(config, 'q_lora_rank', 0)
    kv_lora = getattr(config, 'kv_lora_rank', 0)
    if q_lora > 0 and kv_lora > 0:
        qk_nope = getattr(config, 'qk_nope_head_dim', 128)
        qk_rope = getattr(config, 'qk_rope_head_dim', 64)
        v_head = getattr(config, 'v_head_dim', 128)
        proj_bytes = B * S * dtype_bytes * (
            q_lora +                              # q_a (replicated)
            heads_per_gpu * (qk_nope + qk_rope) + # q_b (sharded)
            kv_lora + qk_rope +                   # kv_a (replicated)
            heads_per_gpu * (qk_nope + v_head) +  # kv_b (sharded)
            heads_per_gpu * v_head                 # attn output
        )
    else:
        proj_bytes = 0

    # Hidden states (input + output buffers)
    hidden_bytes = B * S * hidden * dtype_bytes * 2

    # MLP: MoE vs dense
    n_experts = getattr(config, 'n_routed_experts', 0)
    if n_experts > 0:
        moe_int = getattr(config, 'moe_intermediate_size', hidden) // max(tp, 1)
        num_tok = getattr(config, 'num_experts_per_tok', 8)
        # Expert FFN: sequential loop, 1 expert at a time, SwiGLU (gate+up)
        expert_bytes = B * S * moe_int * dtype_bytes * 2
        # Shared expert (also TP-sharded)
        shared_bytes = B * S * moe_int * dtype_bytes * 2
        # Dispatch: F.one_hot(topk_indices, n_experts) creates (B*S, num_tok, n_experts) int64
        dispatch_bytes = B * S * num_tok * n_experts * 8
        # Router logits (fp32)
        gate_bytes = B * S * n_experts * 4
        # Final hidden states buffer
        final_hidden = B * S * hidden * dtype_bytes
        mlp_bytes = expert_bytes + shared_bytes + dispatch_bytes + gate_bytes + final_hidden
    else:
        dense_int = getattr(config, 'intermediate_size', hidden * 4) // max(tp, 1)
        mlp_bytes = B * S * dense_int * dtype_bytes * 2  # SwiGLU gate+up

    per_layer_peak = attn_bytes + hidden_bytes + proj_bytes + mlp_bytes
    # Under TP, all-reduce sync between layers limits concurrency to ~2
    concurrent_layers = 2

    return (per_layer_peak * concurrent_layers) / (1024 ** 3)


def calculate_max_batch_size(
    model,
    max_seq_len: int,
    mode: str = 'inference',
    safety_margin: float = 0.9,
    num_capture_layers: int = 0,
) -> int:
    """
    Calculate maximum batch size from free VRAM after model is loaded.

    Accounts for KV cache, forward pass activations, and mode-specific overhead.

    Args:
        model: Loaded model (already using VRAM)
        max_seq_len: max_input_tokens + max_output_tokens
        mode: Memory profile mode:
            - 'inference': Basic forward pass (default)
            - 'generation': Includes KV cache growth, logits buffer, generate() overhead
            - 'extraction': Similar to inference (hooks offload to CPU)
        safety_margin: Fraction of free VRAM to use (default 90%)
        num_capture_layers: Number of layers with capture hooks (extraction mode).
            Each hook holds (batch, seq, hidden_size) on GPU during forward pass.

    Returns:
        Maximum safe batch size (at least 1)
    """
    free_gb = get_free_vram_gb(per_device=True)

    # Base memory: KV cache + forward pass activations
    kv_gb = estimate_kv_cache_gb(model, max_seq_len, batch_size=1)
    fwd_gb = estimate_forward_pass_gb(model, max_seq_len, batch_size=1)
    gb_per_seq = kv_gb + fwd_gb

    # Mode-specific adjustments
    hooks_gb = 0
    if mode == 'generation':
        # Logits buffer: seq × vocab_size × dtype
        vocab_size = getattr(model.config, 'vocab_size', 128000)
        logits_gb = (max_seq_len * vocab_size * 2) / (1024 ** 3)
        # Overhead for generate() internals (attention intermediates, framework buffers)
        overhead_factor = 1.15
        gb_per_seq = (gb_per_seq + logits_gb) * overhead_factor

    elif mode == 'extraction':
        # Extraction does a single forward pass with use_cache=False, so no KV cache.
        # Override gb_per_seq to exclude kv_gb.
        gb_per_seq = fwd_gb
        # Safety factor: MoE models have ~3x overhead beyond the analytical estimate
        # (triton kernel scratch space, allocator fragmentation, expert dispatch buffers,
        # FP8 matmul scratch). Measured: 22MB/seq estimate vs 53MB/seq actual on Kimi K2.
        gb_per_seq *= 3.0
        # Capture hooks hold (seq_len, hidden_size) per layer on GPU. These are
        # precisely known and should NOT be inflated by the safety factor.
        config = model.config
        if hasattr(config, 'text_config'):
            config = config.text_config
        hidden_size = config.hidden_size
        if num_capture_layers > 0:
            hooks_gb = (num_capture_layers * max_seq_len * hidden_size * 2) / (1024 ** 3)
            gb_per_seq += hooks_gb

    if gb_per_seq <= 0:
        return 1

    usable_gb = free_gb * safety_margin
    max_batch = int(usable_gb / gb_per_seq)

    # Allow env var override for batch size cap
    env_max = os.environ.get('MAX_BATCH_SIZE')
    if env_max:
        max_batch = min(max_batch, int(env_max))

    result = max(1, min(max_batch, 512))

    # Diagnostic output
    print(f"    Auto batch size: {result} (mode={mode}, free={free_gb:.1f}GB, "
          f"per_seq={gb_per_seq*1024:.0f}MB [kv={kv_gb*1024:.0f}+fwd={fwd_gb*1024:.0f}"
          f"{f'+hooks={hooks_gb*1024:.0f}' if hooks_gb else ''}])")

    # Per-GPU memory breakdown for debugging
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        print(f"    Per-GPU: ", end="")
        for i in range(torch.cuda.device_count()):
            free_i, total_i = torch.cuda.mem_get_info(i)
            alloc_i = torch.cuda.memory_allocated(i)
            resv_i = torch.cuda.memory_reserved(i)
            print(f"[{i}]f={free_i/1e9:.0f}G/a={alloc_i/1e9:.0f}G/r={resv_i/1e9:.0f}G ", end="")
        print()

    return result
