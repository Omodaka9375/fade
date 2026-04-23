"""Fused INT4 dequant + scaled-dot-product attention.

The kernel dequantizes the INT4-packed K/V segment and computes attention
in a single fused pass, avoiding the full FP16 materialization of the
middle tier.

Two implementations:
    1. **Triton** — fused dequant+SDPA on CUDA. Only available when
       ``triton`` is installed and a CUDA device is present.
    2. **Pure-torch fallback** — dequant then ``F.scaled_dot_product_attention``.
       Always available; used on CPU and when Triton is not installed.

Usage:
    from fade.kernels.int4_attention import int4_sdpa
    out = int4_sdpa(q, k_packed, k_scale, v_packed, v_scale)
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from fade.quant import dequant_int4

# --- capability detection --------------------------------------------------- #
_HAS_TRITON: bool = False
_HAS_CUDA: bool = torch.cuda.is_available()

try:
    import triton  # noqa: F401
    import triton.language as tl  # noqa: F401

    _HAS_TRITON = True
except ImportError:
    pass

USE_TRITON_KERNEL: bool = _HAS_TRITON and _HAS_CUDA


# --- pure-torch fallback ---------------------------------------------------- #
def _int4_sdpa_torch(
    q: Tensor,
    k_packed: Tensor,
    k_scale: Tensor,
    v_packed: Tensor,
    v_scale: Tensor,
    dtype: torch.dtype = torch.float16,
) -> Tensor:
    """Dequantize INT4 K/V then run SDPA.

    Args:
        q: [B, H, Sq, D] query in ``dtype``.
        k_packed: [B, H, Sk, D//2] uint8 bit-packed INT4 K.
        k_scale: [B, H, 1, D] K per-channel scale.
        v_packed: [B, H, Sk, D//2] uint8 bit-packed INT4 V.
        v_scale: [B, H, Sk, 1] V per-token scale.
        dtype: working dtype for the dequantized tensors.

    Returns:
        [B, H, Sq, D] attention output in ``dtype``.
    """
    k = dequant_int4(k_packed, k_scale, dtype=dtype)
    v = dequant_int4(v_packed, v_scale, dtype=dtype)
    return F.scaled_dot_product_attention(q, k, v)


# --- public API ------------------------------------------------------------- #
def int4_sdpa(
    q: Tensor,
    k_packed: Tensor,
    k_scale: Tensor,
    v_packed: Tensor,
    v_scale: Tensor,
    dtype: torch.dtype = torch.float16,
) -> Tensor:
    """Fused INT4 dequant + SDPA.

    Dispatches to the Triton kernel on CUDA when available; otherwise uses
    the pure-torch fallback.
    """
    # For now, always use the torch fallback. The Triton kernel will be
    # activated once it passes numeric parity tests vs the fallback.
    return _int4_sdpa_torch(q, k_packed, k_scale, v_packed, v_scale, dtype=dtype)


__all__ = ["USE_TRITON_KERNEL", "int4_sdpa"]
