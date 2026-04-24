"""Peak GPU memory tracking and KV-cache byte accounting.

Usage:
    with PeakMemory() as mem:
        ...work...
    print(mem.peak_mib)
"""

from __future__ import annotations

from typing import Any

import torch


class PeakMemory:
    """Context manager that records peak allocated CUDA memory."""

    def __init__(self, device: str | torch.device | None = None) -> None:
        self.device = device
        self.peak_bytes: int = 0

    def __enter__(self) -> PeakMemory:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if torch.cuda.is_available():
            self.peak_bytes = torch.cuda.max_memory_allocated(self.device)

    @property
    def peak_mib(self) -> float:
        return self.peak_bytes / (1024 * 1024)

    @property
    def peak_gib(self) -> float:
        return self.peak_bytes / (1024 * 1024 * 1024)


def cache_storage_bytes(cache: Any) -> int:
    """Sum the byte size of every K/V tensor stored by a cache object.

    Works for:
        - ``TieredKVCache`` (uses its ``storage_bytes`` method).
        - ``transformers.DynamicCache`` in both the old
          ``key_cache``/``value_cache`` list layout and the newer ``layers``
          layout (each layer exposes ``keys`` / ``values``).
    """
    # Prefer the compressed (essential) form when the cache exposes it so the
    # metric reflects at-rest compression, not the transient dequant buffer.
    fn = getattr(cache, "compressed_storage_bytes", None)
    if callable(fn):
        return int(fn())
    fn = getattr(cache, "storage_bytes", None)
    if callable(fn):
        return int(fn())
    total = 0
    if hasattr(cache, "layers") and cache.layers is not None:
        for layer in cache.layers:
            for name in ("keys", "values", "key_cache", "value_cache"):
                t = getattr(layer, name, None)
                if isinstance(t, torch.Tensor):
                    total += int(t.element_size() * t.numel())
        if total:
            return total
    for name in ("key_cache", "value_cache"):
        tensors = getattr(cache, name, None)
        if tensors:
            for t in tensors:
                if isinstance(t, torch.Tensor):
                    total += int(t.element_size() * t.numel())
    return total
