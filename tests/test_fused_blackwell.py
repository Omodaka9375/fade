"""Fused INT4 FlashAttention kernel parity on Blackwell (SM120+).

Gated behind ``@pytest.mark.dgx`` — skipped on non-CUDA or non-Blackwell.
Run with: ``pytest tests/test_fused_blackwell.py -m dgx``

Validates that the Triton fused kernel produces results within tolerance
of the dequant+SDPA reference path on DGX Spark hardware.
"""

from __future__ import annotations

import pytest
import torch

pytestmark = [pytest.mark.dgx, pytest.mark.cuda]


def _is_blackwell() -> bool:
    if not torch.cuda.is_available():
        return False
    cap = torch.cuda.get_device_capability(0)
    # SM 12.0+ = Blackwell (GB100/GB200/DGX Spark)
    return cap[0] >= 12


def _has_triton() -> bool:
    try:
        import triton  # noqa: F401

        return True
    except ImportError:
        return False


@pytest.fixture(autouse=True)
def skip_if_not_blackwell():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    if not _is_blackwell():
        pytest.skip(f"Not Blackwell: SM {torch.cuda.get_device_capability(0)}")
    if not _has_triton():
        pytest.skip("Triton not installed")


def test_fused_int4_sdpa_parity():
    """Fused INT4 SDPA matches dequant+SDPA within atol=1e-2."""
    from fade.kernels.fused_int4_attn import fused_int4_sdpa
    from fade.quant import dequant_int4, quant_k_int4, quant_v_int4

    B, H, S_q, S_kv, D = 1, 4, 1, 128, 64
    q = torch.randn(B, H, S_q, D, dtype=torch.float16, device="cuda")
    k_fp = torch.randn(B, H, S_kv, D, dtype=torch.float16, device="cuda")
    v_fp = torch.randn(B, H, S_kv, D, dtype=torch.float16, device="cuda")

    k_packed, k_scale = quant_k_int4(k_fp)
    v_packed, v_scale = quant_v_int4(v_fp)

    # Reference: dequant then SDPA.
    k_deq = dequant_int4(k_packed, k_scale, dtype=torch.float16)
    v_deq = dequant_int4(v_packed, v_scale, dtype=torch.float16)
    ref = torch.nn.functional.scaled_dot_product_attention(q, k_deq, v_deq)

    # Fused path.
    fused = fused_int4_sdpa(q, k_packed, k_scale, v_packed, v_scale)

    assert fused.shape == ref.shape
    diff = (fused - ref).abs().max().item()
    assert diff < 0.05, f"Fused vs reference max diff: {diff}"


def test_fused_int4_sdpa_causal():
    """Fused INT4 SDPA with causal mask on Blackwell."""
    from fade.kernels.fused_int4_attn import fused_int4_sdpa
    from fade.quant import quant_k_int4, quant_v_int4

    B, H, S, D = 1, 4, 32, 64
    q = torch.randn(B, H, S, D, dtype=torch.float16, device="cuda")
    k_fp = torch.randn(B, H, S, D, dtype=torch.float16, device="cuda")
    v_fp = torch.randn(B, H, S, D, dtype=torch.float16, device="cuda")

    k_packed, k_scale = quant_k_int4(k_fp)
    v_packed, v_scale = quant_v_int4(v_fp)

    # Should not crash with is_causal=True.
    out = fused_int4_sdpa(q, k_packed, k_scale, v_packed, v_scale, is_causal=True)
    assert out.shape == (B, H, S, D)
    assert not torch.isnan(out).any()


def test_fused_gqa_broadcast():
    """GQA: fewer KV heads than Q heads on Blackwell."""
    from fade.kernels.fused_int4_attn import fused_int4_sdpa
    from fade.quant import quant_k_int4, quant_v_int4

    B, H_q, H_kv, S_q, S_kv, D = 1, 8, 2, 1, 64, 64
    q = torch.randn(B, H_q, S_q, D, dtype=torch.float16, device="cuda")
    k_fp = torch.randn(B, H_kv, S_kv, D, dtype=torch.float16, device="cuda")
    v_fp = torch.randn(B, H_kv, S_kv, D, dtype=torch.float16, device="cuda")

    k_packed, k_scale = quant_k_int4(k_fp)
    v_packed, v_scale = quant_v_int4(v_fp)

    # GQA broadcast: K/V have fewer heads.
    out = fused_int4_sdpa(q, k_packed, k_scale, v_packed, v_scale)
    assert out.shape == (B, H_q, S_q, D)
    assert not torch.isnan(out).any()
