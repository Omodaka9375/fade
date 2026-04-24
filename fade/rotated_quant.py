"""Rotation-based quantization for K/V cache vectors.

Inspired by TurboQuant (ICLR 2026): apply a random orthogonal rotation
before quantization. This spreads per-channel outliers (heavy in K)
across all coordinates, making uniform quantization much more effective.

Storage per vector at different bit widths (D=64 example):
    - 4-bit: 32 bytes packed + 4 bytes norm = 36 B  (3.6x vs 128 B FP16)
    - 3-bit: 24 bytes packed + 4 bytes norm = 28 B  (4.6x vs FP16)
    - 2-bit: 16 bytes packed + 4 bytes norm = 20 B  (6.4x vs FP16)

The rotation matrix is generated once from a fixed seed and stored on
the backend. Compress/decompress are deterministic.
"""

from __future__ import annotations

import torch
from torch import Tensor

from fade.quant import _pack_int4_last_dim, _unpack_int4_last_dim

# --- knobs ------------------------------------------------------------------ #
DEFAULT_SEED: int = 42
INT4_MIN: int = -8
INT4_MAX: int = 7
EPS: float = 1e-8


def _random_orthogonal(dim: int, seed: int = DEFAULT_SEED) -> Tensor:
    """Generate a random orthogonal matrix via QR decomposition."""
    gen = torch.Generator().manual_seed(seed)
    M = torch.randn(dim, dim, generator=gen)
    Q, _ = torch.linalg.qr(M)
    return Q  # [D, D] orthogonal


def rotated_quant_k(
    k: Tensor,
    R: Tensor,
) -> tuple[Tensor, Tensor]:
    """Rotate and quantize K to INT4 with per-channel scale (like standard K quant).

    The rotation spreads outliers before the standard per-channel quantization,
    giving better quality at the same storage as symmetric INT4.

    Args:
        k: [B, H, S, D] K cache tensors.
        R: [D, D] orthogonal rotation matrix.

    Returns:
        (packed_uint8 [B,H,S,D//2], scale [B,H,1,D]) — same format as quant_k_int4.
    """
    # Rotate along the head_dim axis.
    k_f = k.float()
    k_rot = k_f @ R.to(k.device).T  # [B, H, S, D]

    # Standard per-channel symmetric INT4 on the rotated tensor.
    absmax = k_rot.abs().amax(dim=-2, keepdim=True).clamp(min=EPS)  # [B, H, 1, D]
    scale = absmax / INT4_MAX
    q = (k_rot / scale).round().clamp(INT4_MIN, INT4_MAX).to(torch.int8)
    packed = _pack_int4_last_dim(q)
    return packed, scale.to(k.dtype)


def rotated_dequant_k(
    packed: Tensor,
    scale: Tensor,
    R: Tensor,
    dtype: torch.dtype = torch.float16,
) -> Tensor:
    """Inverse of rotated_quant_k."""
    q = _unpack_int4_last_dim(packed)
    k_rot = q.to(dtype) * scale.to(dtype)
    # Inverse rotate.
    return (k_rot.float() @ R.to(k_rot.device)).to(dtype)


def rotated_quant_v(
    v: Tensor,
    R: Tensor,
) -> tuple[Tensor, Tensor]:
    """Rotate and quantize V to INT4 with per-token scale (like standard V quant).

    Args:
        v: [B, H, S, D] V cache tensors.
        R: [D, D] orthogonal rotation matrix.

    Returns:
        (packed_uint8 [B,H,S,D//2], scale [B,H,S,1]) — same format as quant_v_int4.
    """
    v_f = v.float()
    v_rot = v_f @ R.to(v.device).T

    absmax = v_rot.abs().amax(dim=-1, keepdim=True).clamp(min=EPS)  # [B, H, S, 1]
    scale = absmax / INT4_MAX
    q = (v_rot / scale).round().clamp(INT4_MIN, INT4_MAX).to(torch.int8)
    packed = _pack_int4_last_dim(q)
    return packed, scale.to(v.dtype)


def rotated_dequant_v(
    packed: Tensor,
    scale: Tensor,
    R: Tensor,
    dtype: torch.dtype = torch.float16,
) -> Tensor:
    """Inverse of rotated_quant_v."""
    q = _unpack_int4_last_dim(packed)
    v_rot = q.to(dtype) * scale.to(dtype)
    return (v_rot.float() @ R.to(v_rot.device)).to(dtype)


__all__ = [
    "rotated_dequant_k",
    "rotated_dequant_v",
    "rotated_quant_k",
    "rotated_quant_v",
]
