"""Quantization primitives for K and V cache.

Design choices (from KIVI and follow-ups):
    - K has heavy channel-wise outliers -> quantize per-channel along head_dim.
    - V is well-behaved -> quantize per-token (last-dim-keeping).
    - INT4 is bit-packed along the last dim (two 4-bit values per uint8 byte),
      yielding a true 2x reduction vs the int8-holding-int4 intermediate form.
    - INT2 still uses group-wise scales along the sequence axis and is not
      yet bit-packed (Phase 2 work).
"""
from __future__ import annotations

import torch
from torch import Tensor

# --- knobs (top of file so they're easy to retune) --------------------------- #
INT4_MIN: int = -8
INT4_MAX: int = 7
INT2_MIN: int = -2
INT2_MAX: int = 1
EPS: float = 1e-8
DEFAULT_INT2_GROUP_SIZE: int = 64


def _pack_int4_last_dim(q_int8: Tensor) -> Tensor:
    """Pack pairs of INT4 values into uint8 bytes along the last dim.

    Args:
        q_int8: int8 tensor with values in [-8, 7]; last dim must be even.

    Returns:
        uint8 tensor with the last dim halved. Byte ``out[..., d]`` stores the
        pair (high nibble = ``q[..., 2d]``, low nibble = ``q[..., 2d+1]``).
    """
    if q_int8.shape[-1] % 2 != 0:
        raise ValueError(f"last dim must be even for INT4 packing, got {q_int8.shape[-1]}")
    # Two's-complement nibbles. Go via int16 to avoid any sign-ambiguity in
    # int8 bitwise ops, then down-cast to uint8.
    q_unsigned = (q_int8.to(torch.int16) & 0xF).to(torch.uint8)
    high = q_unsigned[..., 0::2]
    low = q_unsigned[..., 1::2]
    return (high << 4) | low


def _unpack_int4_last_dim(packed: Tensor) -> Tensor:
    """Inverse of ``_pack_int4_last_dim``. Returns int8 with values in [-8, 7]."""
    high = ((packed >> 4) & 0xF).to(torch.int16)
    low = (packed & 0xF).to(torch.int16)
    # Sign-extend 4-bit two's complement: values >= 8 are negative.
    high = torch.where(high >= 8, high - 16, high).to(torch.int8)
    low = torch.where(low >= 8, low - 16, low).to(torch.int8)
    out_shape = list(packed.shape)
    out_shape[-1] *= 2
    out = torch.empty(out_shape, dtype=torch.int8, device=packed.device)
    out[..., 0::2] = high
    out[..., 1::2] = low
    return out


def quant_k_int4(k: Tensor) -> tuple[Tensor, Tensor]:
    """Per-channel symmetric INT4 quantization for K cache, bit-packed.

    Args:
        k: Shape [B, H, S, D] with D even.

    Returns:
        (packed_uint8, scale) where ``packed_uint8`` has shape
        [B, H, S, D // 2] (uint8) and ``scale`` is [B, H, 1, D] (``k.dtype``).
    """
    absmax = k.abs().amax(dim=-2, keepdim=True).clamp(min=EPS)
    scale = absmax / INT4_MAX
    q = (k / scale).round().clamp(INT4_MIN, INT4_MAX).to(torch.int8)
    return _pack_int4_last_dim(q), scale


def quant_v_int4(v: Tensor) -> tuple[Tensor, Tensor]:
    """Per-token symmetric INT4 quantization for V cache, bit-packed.

    Args:
        v: Shape [B, H, S, D] with D even.

    Returns:
        (packed_uint8, scale) where ``packed_uint8`` has shape
        [B, H, S, D // 2] (uint8) and ``scale`` is [B, H, S, 1] (``v.dtype``).
    """
    absmax = v.abs().amax(dim=-1, keepdim=True).clamp(min=EPS)
    scale = absmax / INT4_MAX
    q = (v / scale).round().clamp(INT4_MIN, INT4_MAX).to(torch.int8)
    return _pack_int4_last_dim(q), scale


def quant_k_int2(k: Tensor, group_size: int = DEFAULT_INT2_GROUP_SIZE) -> tuple[Tensor, Tensor]:
    """Per-channel INT2 quantization for K cache, with groups along the sequence axis.

    The caller is responsible for padding S to a multiple of ``group_size``.

    Returns:
        (q_int8, scale) with q [B, H, S, D] and scale [B, H, S // group_size, D].
    """
    B, H, S, D = k.shape
    if S % group_size != 0:
        raise ValueError(f"S={S} must be divisible by group_size={group_size}")
    G = S // group_size
    k_g = k.view(B, H, G, group_size, D)
    absmax = k_g.abs().amax(dim=-2, keepdim=True).clamp(min=EPS)  # [B, H, G, 1, D]
    scale = absmax / INT2_MAX
    q = (k_g / scale).round().clamp(INT2_MIN, INT2_MAX).to(torch.int8)
    return q.view(B, H, S, D), scale.squeeze(-2)  # scale: [B, H, G, D]


def dequant(q: Tensor, scale: Tensor, dtype: torch.dtype = torch.float16) -> Tensor:
    """Dequantize broadcastable (q, scale) back to ``dtype``.

    For the *unpacked* int8 path (e.g. the INT2 grouped quantizer). For the
    bit-packed INT4 K/V path, use ``dequant_int4`` which unpacks first.
    """
    return q.to(dtype) * scale.to(dtype)


def dequant_int4(
    packed: Tensor, scale: Tensor, dtype: torch.dtype = torch.float16
) -> Tensor:
    """Unpack a bit-packed INT4 tensor and dequantize to ``dtype``."""
    q = _unpack_int4_last_dim(packed)
    return q.to(dtype) * scale.to(dtype)


def quant_v_int2(v: Tensor, group_size: int = DEFAULT_INT2_GROUP_SIZE) -> tuple[Tensor, Tensor]:
    """Per-token grouped INT2 quantization for V cache.

    Groups along the sequence axis with scales per-group per-head_dim element,
    symmetric scheme, same structure as ``quant_k_int2``.

    The caller must pad S to a multiple of ``group_size``.

    Returns:
        (q_int8, scale) with q [B, H, S, D] and scale [B, H, S // group_size, D].
    """
    B, H, S, D = v.shape
    if S % group_size != 0:
        raise ValueError(f"S={S} must be divisible by group_size={group_size}")
    G = S // group_size
    v_g = v.view(B, H, G, group_size, D)
    absmax = v_g.abs().amax(dim=-2, keepdim=True).clamp(min=EPS)  # [B, H, G, 1, D]
    scale = absmax / INT2_MAX
    q = (v_g / scale).round().clamp(INT2_MIN, INT2_MAX).to(torch.int8)
    return q.view(B, H, S, D), scale.squeeze(-2)  # scale: [B, H, G, D]


def dequant_k_int2(
    q: Tensor,
    scale: Tensor,
    group_size: int = DEFAULT_INT2_GROUP_SIZE,
    dtype: torch.dtype = torch.float16,
) -> Tensor:
    """Dequantize the grouped INT2 K tensor back to ``dtype``."""
    B, H, S, D = q.shape
    G = S // group_size
    scale_full = (
        scale.unsqueeze(-2).expand(B, H, G, group_size, D).reshape(B, H, S, D)
    )
    return q.to(dtype) * scale_full.to(dtype)


def dequant_v_int2(
    q: Tensor,
    scale: Tensor,
    group_size: int = DEFAULT_INT2_GROUP_SIZE,
    dtype: torch.dtype = torch.float16,
) -> Tensor:
    """Dequantize the grouped INT2 V tensor back to ``dtype``."""
    B, H, S, D = q.shape
    G = S // group_size
    scale_full = (
        scale.unsqueeze(-2).expand(B, H, G, group_size, D).reshape(B, H, S, D)
    )
    return q.to(dtype) * scale_full.to(dtype)


def pad_to_group(x: Tensor, group_size: int) -> tuple[Tensor, int]:
    """Pad sequence dim (-2) to a multiple of ``group_size``. Returns (padded, actual)."""
    S = x.shape[-2]
    if S % group_size == 0:
        return x, S
    pad_n = group_size - (S % group_size)
    padding = torch.zeros(
        *x.shape[:-2], pad_n, x.shape[-1], dtype=x.dtype, device=x.device
    )
    return torch.cat([x, padding], dim=-2), S
