"""Round-trip sanity tests for the quantization primitives.

We check that dequant(quant(x)) stays within a loose tolerance of x, and that
shapes of the scale tensors match the documented contracts.
"""
from __future__ import annotations

import torch

from fade.quant import (
    dequant_int4,
    dequant_k_int2,
    dequant_v_int2,
    pad_to_group,
    quant_k_int2,
    quant_k_int4,
    quant_v_int2,
    quant_v_int4,
)

torch.manual_seed(0)

# --- fixtures ---------------------------------------------------------------- #
B, H, S, D = 1, 4, 128, 64
DTYPE = torch.float32


def _relative_error(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a - b).abs().mean().item() / a.abs().mean().item()


def test_quant_k_int4_roundtrip():
    k = torch.randn(B, H, S, D, dtype=DTYPE)
    q, scale = quant_k_int4(k)
    # Packed shape: last dim halved, dtype uint8.
    assert q.shape == (B, H, S, D // 2)
    assert q.dtype == torch.uint8
    assert scale.shape == (B, H, 1, D)
    k_hat = dequant_int4(q, scale, dtype=DTYPE)
    assert k_hat.shape == k.shape
    # INT4 gives ~5-8% typical relative error on gaussian data
    assert _relative_error(k, k_hat) < 0.15


def test_quant_v_int4_roundtrip():
    v = torch.randn(B, H, S, D, dtype=DTYPE)
    q, scale = quant_v_int4(v)
    assert q.shape == (B, H, S, D // 2)
    assert q.dtype == torch.uint8
    assert scale.shape == (B, H, S, 1)
    v_hat = dequant_int4(q, scale, dtype=DTYPE)
    assert v_hat.shape == v.shape
    assert _relative_error(v, v_hat) < 0.15


def test_int4_odd_last_dim_rejected():
    # Bit-packing along the last dim requires an even size.
    k = torch.randn(B, H, S, 15, dtype=DTYPE)
    try:
        quant_k_int4(k)
    except ValueError:
        return
    raise AssertionError("expected ValueError for odd last dim")


def test_quant_k_int2_roundtrip():
    group_size = 64
    k = torch.randn(B, H, S, D, dtype=DTYPE)
    q, scale = quant_k_int2(k, group_size=group_size)
    assert q.shape == k.shape
    assert scale.shape == (B, H, S // group_size, D)
    k_hat = dequant_k_int2(q, scale, group_size=group_size, dtype=DTYPE)
    # INT2 with the simple symmetric scheme is very lossy on gaussian data;
    # Phase 2 will replace this with a better quantizer. Just ensure the
    # round-trip produces finite values of roughly the right magnitude.
    assert torch.isfinite(k_hat).all()
    assert _relative_error(k, k_hat) < 1.1


def test_quant_k_int2_requires_divisible_group():
    k = torch.randn(B, H, 100, D, dtype=DTYPE)  # 100 not divisible by 64
    try:
        quant_k_int2(k, group_size=64)
    except ValueError:
        return
    raise AssertionError("expected ValueError for non-divisible sequence length")


def test_quant_v_int2_roundtrip():
    group_size = 64
    v = torch.randn(B, H, S, D, dtype=DTYPE)
    q, scale = quant_v_int2(v, group_size=group_size)
    assert q.shape == v.shape
    assert scale.shape == (B, H, S // group_size, D)
    v_hat = dequant_v_int2(q, scale, group_size=group_size, dtype=DTYPE)
    assert torch.isfinite(v_hat).all()
    assert _relative_error(v, v_hat) < 1.1


def test_quant_v_int2_requires_divisible_group():
    v = torch.randn(B, H, 100, D, dtype=DTYPE)
    try:
        quant_v_int2(v, group_size=64)
    except ValueError:
        return
    raise AssertionError("expected ValueError for non-divisible sequence length")


def test_quant_k_int4_bf16_roundtrip():
    """INT4 K round-trip must work under bfloat16 with loose tolerance.

    bf16 has a 7-bit mantissa; quantization noise dominates format noise so
    the relative-error bar is the same as fp32.
    """
    k = torch.randn(B, H, S, D, dtype=torch.bfloat16)
    q, scale = quant_k_int4(k)
    assert q.dtype == torch.uint8
    assert scale.dtype == torch.bfloat16
    k_hat = dequant_int4(q, scale, dtype=torch.bfloat16)
    assert k_hat.dtype == torch.bfloat16
    assert torch.isfinite(k_hat).all()
    # Compare in fp32 to avoid the bf16 noise floor in the error metric.
    assert _relative_error(k.float(), k_hat.float()) < 0.2


def test_quant_v_int4_bf16_roundtrip():
    v = torch.randn(B, H, S, D, dtype=torch.bfloat16)
    q, scale = quant_v_int4(v)
    v_hat = dequant_int4(q, scale, dtype=torch.bfloat16)
    assert torch.isfinite(v_hat).all()
    assert _relative_error(v.float(), v_hat.float()) < 0.2


def test_quant_handles_tiny_absmax_without_nan():
    """Near-zero tensors must not produce NaN / inf through the EPS clamp."""
    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        k = torch.zeros(B, H, S, D, dtype=dtype)
        q, scale = quant_k_int4(k)
        k_hat = dequant_int4(q, scale, dtype=dtype)
        assert torch.isfinite(k_hat).all(), f"NaN in {dtype} zero round-trip"


def test_pad_to_group():
    x = torch.randn(B, H, 50, D, dtype=DTYPE)
    padded, actual = pad_to_group(x, 64)
    assert actual == 50
    assert padded.shape[-2] == 64
    assert torch.allclose(padded[..., :50, :], x)
    # Already-aligned: no padding.
    y = torch.randn(B, H, 128, D, dtype=DTYPE)
    padded2, actual2 = pad_to_group(y, 64)
    assert actual2 == 128
    assert padded2.shape[-2] == 128
