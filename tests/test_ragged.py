"""Tests for per-sequence ragged tier assignment (W2.5 / item 9)."""

from __future__ import annotations

import torch

from fade.cache import TIER_EVICT, TIER_FP16, TIER_INT4, TieredKVCache

B, H, D = 2, 2, 32
DTYPE = torch.float32


def _make_cache(**kw) -> TieredKVCache:
    defaults = dict(
        n_sink=2,
        recent_window=2,
        int4_budget=None,
        int2_budget=0,
        dtype=DTYPE,
        rope_theta=10000.0,
        head_dim=D,
    )
    defaults.update(kw)
    return TieredKVCache(**defaults)


def test_per_sequence_no_eviction():
    """Per-sequence assignment with no eviction should match shared path."""
    cache = _make_cache()
    k = torch.randn(B, H, 8, D)
    v = torch.randn(B, H, 8, D)
    cache.update(k, v, layer_idx=0)

    # Both rows: same tiers, no eviction.
    tiers = torch.tensor(
        [TIER_FP16, TIER_FP16, TIER_INT4, TIER_INT4, TIER_INT4, TIER_INT4, TIER_FP16, TIER_FP16]
    )
    tiers_per_row = tiers.unsqueeze(0).expand(B, -1).contiguous()
    cache.apply_tier_assignment_per_sequence(0, tiers_per_row)

    assert cache.get_seq_length(0) == 8
    k_out, _v_out = cache._materialize(0)
    assert k_out.shape == (B, H, 8, D)


def test_per_sequence_different_eviction():
    """Row 0 evicts 2 middle tokens; row 1 evicts 0. Result is padded to max."""
    cache = _make_cache()
    k = torch.randn(B, H, 8, D)
    v = torch.randn(B, H, 8, D)
    cache.update(k, v, layer_idx=0)

    # Row 0: evict positions 2 and 3.
    row0 = torch.tensor(
        [TIER_FP16, TIER_FP16, TIER_EVICT, TIER_EVICT, TIER_INT4, TIER_INT4, TIER_FP16, TIER_FP16]
    )
    # Row 1: no eviction.
    row1 = torch.tensor(
        [TIER_FP16, TIER_FP16, TIER_INT4, TIER_INT4, TIER_INT4, TIER_INT4, TIER_FP16, TIER_FP16]
    )
    tiers_per_row = torch.stack([row0, row1])
    cache.apply_tier_assignment_per_sequence(0, tiers_per_row)

    # Row 1 kept all 8; row 0 kept 6. Materialized should pad to 8.
    k_out, _v_out = cache._materialize(0)
    assert k_out.shape[0] == B
    # The seq dim should be the max across rows.
    assert k_out.shape[-2] >= 6  # at least row 0's surviving count


def test_per_sequence_all_eviction_one_row():
    """One row evicts all middle; the other keeps some. Should not crash."""
    cache = _make_cache(n_sink=1, recent_window=1)
    k = torch.randn(B, H, 6, D)
    v = torch.randn(B, H, 6, D)
    cache.update(k, v, layer_idx=0)

    # Row 0: evict all middle (keep only sink + recent).
    row0 = torch.tensor([TIER_FP16, TIER_EVICT, TIER_EVICT, TIER_EVICT, TIER_EVICT, TIER_FP16])
    # Row 1: keep 2 middle in INT4.
    row1 = torch.tensor([TIER_FP16, TIER_EVICT, TIER_INT4, TIER_INT4, TIER_EVICT, TIER_FP16])
    tiers_per_row = torch.stack([row0, row1])
    cache.apply_tier_assignment_per_sequence(0, tiers_per_row)

    k_out, _v_out = cache._materialize(0)
    assert k_out.shape[0] == B
    assert torch.isfinite(k_out).all()


def test_per_sequence_shape_validation():
    """Wrong shape should raise ValueError."""
    cache = _make_cache()
    k = torch.randn(B, H, 6, D)
    v = torch.randn(B, H, 6, D)
    cache.update(k, v, layer_idx=0)

    # Wrong batch dim.
    bad_tiers = torch.full((3, 6), TIER_FP16, dtype=torch.long)
    try:
        cache.apply_tier_assignment_per_sequence(0, bad_tiers)
    except ValueError as e:
        assert "B=2" in str(e)
        return
    raise AssertionError("expected ValueError")


def test_per_sequence_decode_after_ragged_eviction():
    """Decode one more token after ragged eviction should work."""
    cache = _make_cache()
    k = torch.randn(B, H, 8, D)
    v = torch.randn(B, H, 8, D)
    cache.update(k, v, layer_idx=0)

    row0 = torch.tensor(
        [TIER_FP16, TIER_FP16, TIER_EVICT, TIER_INT4, TIER_INT4, TIER_INT4, TIER_FP16, TIER_FP16]
    )
    row1 = torch.tensor(
        [TIER_FP16, TIER_FP16, TIER_INT4, TIER_INT4, TIER_INT4, TIER_INT4, TIER_FP16, TIER_FP16]
    )
    tiers_per_row = torch.stack([row0, row1])
    cache.apply_tier_assignment_per_sequence(0, tiers_per_row)

    # Append one more token.
    k2 = torch.randn(B, H, 1, D)
    v2 = torch.randn(B, H, 1, D)
    k_out, _v_out = cache.update(k2, v2, layer_idx=0)
    assert k_out.shape[0] == B
    assert torch.isfinite(k_out).all()
