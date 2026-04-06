"""Tensor parallelism utilities for multi-GPU runs via torchrun.

Usage:
    from utils.distributed import is_tp_mode, is_rank_zero, tp_barrier, tp_lifecycle, flush_cuda

    if is_tp_mode():
        # Running under torchrun with multiple GPUs
        if is_rank_zero():
            print("Only rank 0 prints this")
        tp_barrier()  # Synchronize all ranks

    # Pipeline lifecycle: init TP, suppress non-rank-zero prints, cleanup
    with tp_lifecycle():
        run_pipeline(config)
"""

import gc
import os
from contextlib import contextmanager


def is_tp_mode():
    """Check if running under torchrun for tensor parallelism."""
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def get_rank():
    """Get distributed rank (0 if not distributed)."""
    if is_tp_mode():
        import torch.distributed as dist
        if dist.is_initialized():
            return dist.get_rank()
    return 0


def is_rank_zero():
    """Check if this is the primary process."""
    return get_rank() == 0


def tp_barrier():
    """Synchronize all TP ranks. No-op if not in TP mode."""
    if is_tp_mode():
        import torch.distributed as dist
        if dist.is_initialized():
            dist.barrier()


@contextmanager
def tp_lifecycle():
    """Init TP, suppress non-rank-zero prints, cleanup on exit."""
    import builtins
    _original_print = builtins.print
    if is_tp_mode():
        import torch.distributed as dist
        if not dist.is_initialized():
            dist.init_process_group("nccl")
        if not is_rank_zero():
            builtins.print = lambda *a, **k: None
    try:
        yield
    finally:
        builtins.print = _original_print
        if is_tp_mode():
            import torch.distributed as dist
            if dist.is_initialized():
                dist.destroy_process_group()


def flush_cuda():
    """Free GPU memory. Call after deleting models/tensors.

    Synchronizes first to ensure async CUDA kernels have finished,
    then collects Python garbage and releases the CUDA allocator cache.
    """
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
