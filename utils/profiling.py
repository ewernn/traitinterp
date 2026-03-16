"""
GPU profiling and monitoring utilities.

Input:
    - Code blocks to profile
    - Optional labels for identification

Output:
    - Timing, memory usage, effective bandwidth metrics

Usage:
    from utils.profiling import gpu_profile, gpu_timer, memory_stats

    # Context manager for detailed profiling
    with gpu_profile("batch capture"):
        results = capture_batch(...)
    # Prints: [batch capture] 0.45s | peak 12.3GB | delta +2.1GB

    # Simple timer
    with gpu_timer("forward pass") as t:
        model(**inputs)
    print(t.elapsed)  # 0.123

    # Memory snapshot
    stats = memory_stats()
    print(stats)  # {'allocated': 5.2, 'reserved': 8.0, 'free': 40.0}
"""

import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional
import torch


@dataclass
class ProfileResult:
    """Result from gpu_profile context manager."""
    elapsed: float  # seconds
    peak_memory_gb: float  # peak GPU memory during block
    start_memory_gb: float  # memory at start
    end_memory_gb: float  # memory at end
    delta_memory_gb: float  # end - start

    def __str__(self):
        return (
            f"{self.elapsed:.2f}s | "
            f"peak {self.peak_memory_gb:.1f}GB | "
            f"delta {self.delta_memory_gb:+.1f}GB"
        )


@dataclass
class TimerResult:
    """Result from gpu_timer context manager."""
    elapsed: float

    def __str__(self):
        return f"{self.elapsed:.3f}s"


def memory_stats() -> dict:
    """
    Get current GPU memory statistics.

    Returns:
        Dict with 'allocated', 'reserved', 'free' in GB.
        Returns zeros if CUDA not available.
    """
    if not torch.cuda.is_available():
        return {'allocated': 0.0, 'reserved': 0.0, 'free': 0.0}

    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    free, total = torch.cuda.mem_get_info()

    return {
        'allocated': round(allocated, 2),
        'reserved': round(reserved, 2),
        'free': round(free / 1e9, 2),
        'total': round(total / 1e9, 2),
    }


@contextmanager
def gpu_profile(name: str = "operation", print_result: bool = True):
    """
    Profile GPU time and memory for a code block.

    Args:
        name: Label for the operation
        print_result: Whether to print results on exit

    Yields:
        ProfileResult that gets populated on exit

    Usage:
        with gpu_profile("forward pass"):
            model(**inputs)

        # Or capture the result:
        with gpu_profile("capture", print_result=False) as result:
            do_work()
        print(result.elapsed)
    """
    result = ProfileResult(
        elapsed=0, peak_memory_gb=0, start_memory_gb=0,
        end_memory_gb=0, delta_memory_gb=0
    )

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


@contextmanager
def gpu_timer(name: str = None, print_result: bool = False):
    """
    Simple GPU-synchronized timer.

    Args:
        name: Optional label
        print_result: Whether to print on exit

    Yields:
        TimerResult with elapsed time

    Usage:
        with gpu_timer() as t:
            model(**inputs)
        print(f"Took {t.elapsed:.3f}s")
    """
    result = TimerResult(elapsed=0)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    start = time.perf_counter()

    try:
        yield result
    finally:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        result.elapsed = time.perf_counter() - start

        if print_result and name:
            print(f"[{name}] {result}")


def bandwidth_report(data_gb: float, elapsed: float) -> str:
    """
    Generate bandwidth report string.

    Args:
        data_gb: Data transferred in GB
        elapsed: Time in seconds

    Returns:
        Formatted string like "4.6GB in 0.19s = 24.2 GB/s"
    """
    bandwidth = data_gb / elapsed if elapsed > 0 else 0
    return f"{data_gb:.1f}GB in {elapsed:.2f}s = {bandwidth:.1f} GB/s"


def tensor_size_gb(shape: tuple, dtype=torch.bfloat16) -> float:
    """
    Calculate tensor size in GB.

    Args:
        shape: Tensor shape tuple
        dtype: Tensor dtype (default bfloat16 = 2 bytes)

    Returns:
        Size in GB
    """
    numel = 1
    for dim in shape:
        numel *= dim

    bytes_per_elem = {
        torch.float32: 4,
        torch.float16: 2,
        torch.bfloat16: 2,
        torch.int64: 8,
        torch.int32: 4,
        torch.int16: 2,
        torch.int8: 1,
        torch.bool: 1,
    }.get(dtype, 2)

    return numel * bytes_per_elem / 1e9
