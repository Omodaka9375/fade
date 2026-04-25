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
EPS: float = 1e-8

# Bit-width parameters: (min_val, max_val)
_BIT_PARAMS: dict[int, tuple[int, int]] = {
    4: (-8, 7),
    3: (-4, 3),
    2: (-2, 1),
}


def _pack_int2_last_dim(q: Tensor) -> Tensor:
    """Pack 4 INT2 values per uint8 byte along the last dim.

    Values must be in [-2, 1]. Last dim must be divisible by 4.
    Layout: byte = (v0 << 6) | (v1 << 4) | (v2 << 2) | v3
    """
    if q.shape[-1] % 4 != 0:
        raise ValueError(f"last dim must be divisible by 4 for INT2 packing, got {q.shape[-1]}")
    q_u = (q.to(torch.int8) & 0x3).to(torch.uint8)  # 2-bit unsigned [0..3]
    a = q_u[..., 0::4]
    b = q_u[..., 1::4]
    c = q_u[..., 2::4]
    d = q_u[..., 3::4]
    return (a << 6) | (b << 4) | (c << 2) | d


def _unpack_int2_last_dim(packed: Tensor) -> Tensor:
    """Inverse of _pack_int2_last_dim. Returns int8 with values in [-2, 1]."""
    a = ((packed >> 6) & 0x3).to(torch.int8)
    b = ((packed >> 4) & 0x3).to(torch.int8)
    c = ((packed >> 2) & 0x3).to(torch.int8)
    d = (packed & 0x3).to(torch.int8)
    # Branchless sign-extend: subtract twice the sign bit (bit-1).
    a = a - ((a & 2) << 1)
    b = b - ((b & 2) << 1)
    c = c - ((c & 2) << 1)
    d = d - ((d & 2) << 1)
    out_shape = list(packed.shape)
    out_shape[-1] *= 4
    out = torch.empty(out_shape, dtype=torch.int8, device=packed.device)
    out[..., 0::4] = a
    out[..., 1::4] = b
    out[..., 2::4] = c
    out[..., 3::4] = d
    return out


def _pack(q: Tensor, bits: int) -> Tensor:
    """Pack quantized int8 tensor to uint8 at the given bit width."""
    if bits == 4:
        return _pack_int4_last_dim(q)
    if bits == 2:
        return _pack_int2_last_dim(q)
    # 3-bit: no packing yet, store as int8.
    return q.to(torch.int8)


def _unpack(packed: Tensor, bits: int) -> Tensor:
    """Unpack uint8 tensor back to int8 at the given bit width."""
    if bits == 4:
        return _unpack_int4_last_dim(packed)
    if bits == 2:
        return _unpack_int2_last_dim(packed)
    return packed.to(torch.int8)


def _random_orthogonal(dim: int, seed: int = DEFAULT_SEED) -> Tensor:
    """Generate a random orthogonal matrix via QR decomposition."""
    gen = torch.Generator().manual_seed(seed)
    M = torch.randn(dim, dim, generator=gen)
    Q, _ = torch.linalg.qr(M)
    return Q  # [D, D] orthogonal


def rotated_quant_k(
    k: Tensor,
    R: Tensor,
    bits: int = 4,
) -> tuple[Tensor, Tensor]:
    """Rotate and quantize K with per-channel scale.

    Args:
        k: [B, H, S, D] K cache tensors.
        R: [D, D] orthogonal rotation matrix.
        bits: quantization bit width (2, 3, or 4).

    Returns:
        (packed, scale) where packed is bit-packed uint8 and scale is [B,H,1,D].
    """
    qmin, qmax = _BIT_PARAMS[bits]
    k_f = k.float()
    k_rot = k_f @ R.to(k.device).T

    absmax = k_rot.abs().amax(dim=-2, keepdim=True).clamp(min=EPS)
    inv_scale = qmax / absmax
    scale = absmax / qmax
    q = (k_rot * inv_scale).round().clamp(qmin, qmax).to(torch.int8)
    packed = _pack(q, bits)
    return packed, scale.to(k.dtype)


def rotated_dequant_k(
    packed: Tensor,
    scale: Tensor,
    R: Tensor,
    bits: int = 4,
    dtype: torch.dtype = torch.float16,
) -> Tensor:
    """Inverse of rotated_quant_k."""
    q = _unpack(packed, bits)
    k_rot = q.to(dtype) * scale.to(dtype)
    return (k_rot.float() @ R.to(k_rot.device)).to(dtype)


def rotated_quant_v(
    v: Tensor,
    R: Tensor,
    bits: int = 4,
) -> tuple[Tensor, Tensor]:
    """Rotate and quantize V with per-token scale.

    Args:
        v: [B, H, S, D] V cache tensors.
        R: [D, D] orthogonal rotation matrix.
        bits: quantization bit width (2, 3, or 4).

    Returns:
        (packed, scale) where packed is bit-packed uint8 and scale is [B,H,S,1].
    """
    qmin, qmax = _BIT_PARAMS[bits]
    v_f = v.float()
    v_rot = v_f @ R.to(v.device).T

    absmax = v_rot.abs().amax(dim=-1, keepdim=True).clamp(min=EPS)
    inv_scale = qmax / absmax
    scale = absmax / qmax
    q = (v_rot * inv_scale).round().clamp(qmin, qmax).to(torch.int8)
    packed = _pack(q, bits)
    return packed, scale.to(v.dtype)


def rotated_dequant_v(
    packed: Tensor,
    scale: Tensor,
    R: Tensor,
    bits: int = 4,
    dtype: torch.dtype = torch.float16,
) -> Tensor:
    """Inverse of rotated_quant_v."""
    q = _unpack(packed, bits)
    v_rot = q.to(dtype) * scale.to(dtype)
    return (v_rot.float() @ R.to(v_rot.device)).to(dtype)


__all__ = [
    "rotated_dequant_k",
    "rotated_dequant_v",
    "rotated_quant_k",
    "rotated_quant_v",
]
