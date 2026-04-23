"""Tests for W5 performance features.

Covers:
    * Pre-allocated FP16 append buffer correctness.
    * Dequant-cache age-based eviction.
    * INT4 kernel fallback path.
"""
from __future__ import annotations

import torch

from fade.cache import TIER_FP16, TIER_INT4, TieredKVCache
from fade.kernels.int4_attention import int4_sdpa
from fade.quant import quant_k_int4, quant_v_int4

B, H, D = 1, 2, 32
DTYPE = torch.float32


def _make_cache(**kw) -> TieredKVCache:
    defaults = dict(
        n_sink=2, recent_window=3, int4_budget=None, int2_budget=0,
        dtype=DTYPE, rope_theta=10000.0, head_dim=D,
    )
    defaults.update(kw)
    return TieredKVCache(**defaults)


# --- pre-alloc buffer ------------------------------------------------------- #
def test_prealloc_buffer_grows_without_data_loss():
    """Append more tokens than the initial buffer capacity; data must survive."""
    cache = _make_cache()
    all_k = []
    all_v = []
    for _i in range(20):
        k = torch.randn(B, H, 1, D)
        v = torch.randn(B, H, 1, D)
        cache.update(k, v, layer_idx=0)
        all_k.append(k)
        all_v.append(v)

    assert cache.get_seq_length(0) == 20
    k_out, v_out = cache._materialize(0)
    expected_k = torch.cat(all_k, dim=-2)
    expected_v = torch.cat(all_v, dim=-2)
    assert torch.allclose(k_out, expected_k)
    assert torch.allclose(v_out, expected_v)


def test_prealloc_buffer_works_after_reassignment():
    """After tier reassignment resets the buffer, new appends are correct."""
    cache = _make_cache()
    k = torch.randn(B, H, 10, D)
    v = torch.randn(B, H, 10, D)
    cache.update(k, v, layer_idx=0)

    tiers = torch.tensor(
        [TIER_FP16, TIER_FP16,
         TIER_INT4, TIER_INT4, TIER_INT4, TIER_INT4, TIER_INT4,
         TIER_FP16, TIER_FP16, TIER_FP16]
    )
    cache.apply_tier_assignment(0, tiers)

    # Buffer is reset; append 3 more tokens.
    for _ in range(3):
        k2 = torch.randn(B, H, 1, D)
        v2 = torch.randn(B, H, 1, D)
        cache.update(k2, v2, layer_idx=0)

    assert cache.get_seq_length(0) == 10 + 3
    k_out, _v_out = cache._materialize(0)
    assert k_out.shape == (B, H, 13, D)


def test_prealloc_prefill_batch_append():
    """Prefill (multi-token) + single-token decode steps work correctly."""
    cache = _make_cache()
    # Prefill with 16 tokens at once.
    k_pf = torch.randn(B, H, 16, D)
    v_pf = torch.randn(B, H, 16, D)
    cache.update(k_pf, v_pf, layer_idx=0)
    assert cache.get_seq_length(0) == 16

    # 4 decode steps.
    for _ in range(4):
        cache.update(torch.randn(B, H, 1, D), torch.randn(B, H, 1, D), layer_idx=0)
    assert cache.get_seq_length(0) == 20

    k_out, _ = cache._materialize(0)
    # First 16 tokens must match the prefill.
    assert torch.allclose(k_out[:, :, :16, :], k_pf)


# --- dequant age eviction --------------------------------------------------- #
def test_dequant_age_eviction():
    """Dequant cache is refreshed after max_dequant_age updates.

    When max_dequant_age is set, the INT4 dequant cache is dropped and
    rebuilt periodically via _get_int4_dequant. We verify the age counter
    resets after exceeding the threshold.
    """
    cache = _make_cache()
    cache.max_dequant_age = 3

    k = torch.randn(B, H, 8, D)
    v = torch.randn(B, H, 8, D)
    cache.update(k, v, layer_idx=0)

    tiers = torch.tensor(
        [TIER_FP16, TIER_FP16,
         TIER_INT4, TIER_INT4, TIER_INT4, TIER_INT4,
         TIER_FP16, TIER_FP16]
    )
    cache.apply_tier_assignment(0, tiers)

    # Force dequant cache population.
    cache._materialize(0)
    state = cache._layers[0]
    assert state.int4_k_deq is not None
    assert state._dequant_age == 0  # freshly populated

    # 4 updates: age goes 1, 2, 3, 4. At age 4 > 3, the dequant is evicted
    # and repopulated in the same update() call.
    for _ in range(4):
        cache.update(torch.randn(B, H, 1, D), torch.randn(B, H, 1, D), layer_idx=0)

    # After the 4th update, age exceeded threshold during _get_int4_dequant
    # in _materialize, so dequant was evicted then rebuilt. Age should be 0.
    assert state._dequant_age == 0
    # The dequant is repopulated (not None) — the eviction + rebuild happened.
    assert state.int4_k_deq is not None


# --- INT4 kernel fallback --------------------------------------------------- #
def test_int4_sdpa_fallback_produces_valid_output():
    """The pure-torch INT4 SDPA fallback must produce finite attention output."""
    S_q, S_k = 4, 16
    q = torch.randn(B, H, S_q, D, dtype=DTYPE)
    k_fp = torch.randn(B, H, S_k, D, dtype=DTYPE)
    v_fp = torch.randn(B, H, S_k, D, dtype=DTYPE)

    k_packed, k_scale = quant_k_int4(k_fp)
    v_packed, v_scale = quant_v_int4(v_fp)

    out = int4_sdpa(q, k_packed, k_scale, v_packed, v_scale, dtype=DTYPE)
    assert out.shape == (B, H, S_q, D)
    assert torch.isfinite(out).all()


def test_int4_sdpa_matches_fp16_attention_approximately():
    """INT4 SDPA output should be close to full-precision SDPA."""
    import torch.nn.functional as F

    S_q, S_k = 1, 32
    q = torch.randn(B, H, S_q, D, dtype=DTYPE)
    k_fp = torch.randn(B, H, S_k, D, dtype=DTYPE)
    v_fp = torch.randn(B, H, S_k, D, dtype=DTYPE)

    ref = F.scaled_dot_product_attention(q, k_fp, v_fp)

    k_packed, k_scale = quant_k_int4(k_fp)
    v_packed, v_scale = quant_v_int4(v_fp)
    out = int4_sdpa(q, k_packed, k_scale, v_packed, v_scale, dtype=DTYPE)

    # INT4 quant introduces noise; relative error should be moderate.
    rel_err = (out - ref).abs().mean() / ref.abs().mean()
    assert rel_err < 0.3, f"INT4 SDPA diverges too much: rel_err={rel_err:.3f}"
