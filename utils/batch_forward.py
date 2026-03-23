"""
Shared helpers for batched forward passes: OOM recovery, TP sync, batch calibration.

Each pipeline keeps its own forward loop but uses these helpers instead of
copy-pasting the OOM traceback clearing, TP count agreement, etc.

Usage:
    from utils.batch_forward import check_oom_exception, recover_oom_batch_size, tp_agree_count, calibrate_batch_size
"""

import traceback as tb_mod
from typing import List, Optional

import torch

from utils.distributed import is_rank_zero, is_tp_mode


# =============================================================================
# TP helpers
# =============================================================================

def tp_agree_count(local_count: int, label: str = "", min_required: int = 0) -> int:
    """Synchronize an item count across TP ranks by taking the minimum.

    All ranks truncate to the smallest count so every rank processes
    the same number of items. No-op if not in TP mode.

    Args:
        min_required: If >0, raises RuntimeError when the agreed count is below
            this threshold but local_count is above it (detects rank disagreement
            that would silently corrupt downstream results).
    """
    if not is_tp_mode():
        return local_count

    import torch.distributed as dist
    min_t = torch.tensor([local_count], device='cuda')
    max_t = torch.tensor([local_count], device='cuda')
    dist.all_reduce(min_t, op=dist.ReduceOp.MIN)
    dist.all_reduce(max_t, op=dist.ReduceOp.MAX)

    agreed = int(min_t.item())
    if min_t.item() != max_t.item() and is_rank_zero():
        print(f"    WARNING: TP count mismatch{' in ' + label if label else ''}: "
              f"min={agreed}, max={int(max_t.item())}, this={local_count}. Truncating.")

    if min_required > 0 and agreed < min_required and local_count >= min_required:
        raise RuntimeError(
            f"TP rank disagreement{' in ' + label if label else ''}: "
            f"this rank has {local_count} items but another has {agreed}"
        )

    return agreed


def tp_agree_batch_size(batch_size: int) -> int:
    """Sync batch size across TP ranks (take minimum). No-op if not in TP mode."""
    if not is_tp_mode():
        return batch_size
    import torch.distributed as dist
    bs = torch.tensor([batch_size], device='cuda')
    dist.all_reduce(bs, op=dist.ReduceOp.MIN)
    return int(bs.item())


# =============================================================================
# OOM recovery
# =============================================================================

def clear_oom_traceback(exc: Exception) -> None:
    """Clear traceback frames from an OOM exception to release CUDA tensor references.

    PyTorch OOM tracebacks hold references to tensors via stack frame locals.
    Clearing them inside the except block (before gc.collect) allows the
    allocator to actually reclaim the memory. Call this, then del the exception,
    then gc.collect() + empty_cache() OUTSIDE the except block.
    """
    if exc.__traceback__:
        tb_mod.clear_frames(exc.__traceback__)
    exc.__traceback__ = None
    for chained in (exc.__context__, exc.__cause__):
        if chained and hasattr(chained, '__traceback__') and chained.__traceback__:
            tb_mod.clear_frames(chained.__traceback__)
            chained.__traceback__ = None


def check_oom_exception(exc: Exception, batch_size: int, tp_raises: bool = True) -> None:
    """Validate an exception is OOM and clear traceback for memory recovery.

    Call from except block. Re-raises if not OOM. Clears traceback frames
    so GPU tensor references are released before gc.collect.

    Args:
        exc: The caught exception.
        batch_size: Current batch size (for error message).
        tp_raises: If True (default), raise RuntimeError in TP mode
            (NCCL state is corrupted after OOM). Set False if caller
            handles TP recovery separately (e.g. via oom_flag sync).
    """
    if "out of memory" not in str(exc).lower() and not isinstance(exc, torch.cuda.OutOfMemoryError):
        raise exc
    if tp_raises and is_tp_mode():
        raise RuntimeError(
            f"OOM during TP forward pass (batch_size={batch_size}). "
            f"NCCL state is corrupted and cannot recover. "
            f"Re-run with fewer layers or a larger GPU."
        )
    clear_oom_traceback(exc)


def recover_oom_batch_size(batch_size: int) -> int:
    """Free GPU memory and halve batch size after OOM.

    Call OUTSIDE the except block (after setting oom flag and del-ing exception).
    Runs gc.collect + empty_cache, then returns halved batch size.
    Raises RuntimeError if batch_size is already 1.
    """
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if batch_size == 1:
        raise RuntimeError("OOM even with batch_size=1")
    new_size = max(1, batch_size // 2)
    if is_rank_zero():
        print(f"  OOM, reducing batch_size to {new_size}")
    return new_size


# =============================================================================
# Batch size calibration
# =============================================================================

def calibrate_batch_size(
    model: torch.nn.Module,
    seq_len: int,
    component: str = 'residual',
    layers: Optional[List[int]] = None,
    overhead_factor: float = 0.9,
) -> int:
    """Measure per-item GPU cost via a live forward pass and estimate safe batch size.

    Runs one forward pass on a dummy (1, seq_len) input with MultiLayerCapture,
    measures peak memory delta, then divides free VRAM by that delta.

    More accurate than the analytical estimate in vram.calculate_max_batch_size,
    especially for MoE models. Use tp_agree_batch_size() on the result under TP.
    """
    from core import MultiLayerCapture

    config = model.config
    if hasattr(config, 'text_config'):
        config = config.text_config
    pad_id = getattr(config, 'pad_token_id', None) or getattr(config, 'eos_token_id', 0) or 0

    dummy = torch.full((1, seq_len), pad_id, dtype=torch.long, device=next(model.parameters()).device)
    mask = torch.ones_like(dummy)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    baseline_mem = torch.cuda.memory_allocated()

    with MultiLayerCapture(model, component=component, layers=layers, keep_on_gpu=True):
        with torch.no_grad():
            model(input_ids=dummy, attention_mask=mask, use_cache=False)

    per_item = torch.cuda.max_memory_allocated() - baseline_mem
    del dummy, mask
    torch.cuda.empty_cache()

    free = torch.cuda.mem_get_info()[0]
    batch_size = max(1, int(free / per_item * overhead_factor)) if per_item > 0 else 1

    if is_rank_zero():
        print(f"    Calibrated: {per_item / 1024**2:.0f}MB/seq, "
              f"free={free / 1024**3:.1f}GB -> batch={batch_size}")

    return batch_size
