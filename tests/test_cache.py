"""Smoke tests for ``TieredKVCache``.

These run without any LLM — they exercise the bookkeeping and the FP16/INT4/INT2
round-trip with synthetic K/V tensors, including Phase 2 pre-RoPE caching.
"""

from __future__ import annotations

import torch

from fade.cache import (
    TIER_EVICT,
    TIER_FP16,
    TIER_INT2,
    TIER_INT4,
    TieredKVCache,
)

torch.manual_seed(0)

# --- fixtures ---------------------------------------------------------------- #
B, H, D = 1, 2, 32
DTYPE = torch.float32
ROPE_THETA = 10000.0


def _make_cache() -> TieredKVCache:
    return TieredKVCache(
        n_sink=2,
        recent_window=3,
        int4_budget=None,
        int2_budget=0,
        dtype=DTYPE,
        rope_theta=ROPE_THETA,
        head_dim=D,
    )


def _make_rope_cos_sin(positions: torch.Tensor, head_dim: int, theta: float = ROPE_THETA):
    """Produce (cos, sin) in the shape the model passes to cache_kwargs: [1, S, D]."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    freqs = positions.float().unsqueeze(-1) * inv_freq.unsqueeze(0)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos().unsqueeze(0)  # [1, S, D]
    sin = emb.sin().unsqueeze(0)
    return cos, sin


def _rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rope(k, cos, sin):
    """Apply RoPE matching Qwen2's convention."""
    if cos.dim() == 3:
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
    return k * cos + _rotate_half(k) * sin


def test_update_appends_to_fp16_and_tracks_length():
    cache = _make_cache()
    k = torch.randn(B, H, 10, D)
    v = torch.randn(B, H, 10, D)
    positions = torch.arange(10)
    cos, sin = _make_rope_cos_sin(positions, D)
    k_rope = _apply_rope(k, cos, sin)
    k_out, v_out = cache.update(k_rope, v, layer_idx=0, cache_kwargs={"cos": cos, "sin": sin})
    assert k_out.shape == (B, H, 10, D)
    assert v_out.shape == (B, H, 10, D)
    assert cache.get_seq_length(0) == 10
    # second append continues the sequence
    k2 = torch.randn(B, H, 3, D)
    v2 = torch.randn(B, H, 3, D)
    pos2 = torch.arange(10, 13)
    cos2, sin2 = _make_rope_cos_sin(pos2, D)
    k2_rope = _apply_rope(k2, cos2, sin2)
    k_out2, _ = cache.update(k2_rope, v2, layer_idx=0, cache_kwargs={"cos": cos2, "sin": sin2})
    assert k_out2.shape == (B, H, 13, D)
    assert cache.get_seq_length(0) == 13


def test_apply_tier_assignment_splits_storage_without_loss():
    cache = _make_cache()
    k = torch.randn(B, H, 10, D)
    v = torch.randn(B, H, 10, D)
    positions = torch.arange(10)
    cos, sin = _make_rope_cos_sin(positions, D)
    k_rope = _apply_rope(k, cos, sin)
    cache.update(k_rope, v, layer_idx=0, cache_kwargs={"cos": cos, "sin": sin})

    tiers = torch.tensor(
        [
            TIER_FP16,
            TIER_FP16,
            TIER_INT4,
            TIER_INT4,
            TIER_INT4,
            TIER_INT4,
            TIER_INT4,
            TIER_FP16,
            TIER_FP16,
            TIER_FP16,
        ]
    )
    cache.apply_tier_assignment(0, tiers)

    state = cache._layers[0]
    assert state.fp16_k.shape[-2] == 5
    assert state.int4_kq.shape[-2] == 5
    assert cache.get_seq_length(0) == 10


def test_materialize_preserves_fp16_positions_approximately():
    """FP16 positions survive the RoPE undo+redo round-trip with low error."""
    cache = _make_cache()
    k = torch.randn(B, H, 8, D)
    v = torch.randn(B, H, 8, D)
    positions = torch.arange(8)
    cos, sin = _make_rope_cos_sin(positions, D)
    k_rope = _apply_rope(k, cos, sin)
    cache.update(k_rope, v, layer_idx=0, cache_kwargs={"cos": cos, "sin": sin})

    tiers = torch.tensor(
        [TIER_FP16, TIER_FP16, TIER_INT4, TIER_INT4, TIER_INT4, TIER_INT4, TIER_FP16, TIER_FP16]
    )
    cache.apply_tier_assignment(0, tiers)

    k_out, v_out = cache._materialize(0)
    assert k_out.shape == (B, H, 8, D)
    # V is untouched by RoPE — FP16 survivors are bit-exact.
    assert torch.allclose(v_out[:, :, :2], v[:, :, :2])
    assert torch.allclose(v_out[:, :, -2:], v[:, :, -2:])
    # K goes through inverse-RoPE then re-RoPE with StreamingLLM positions.
    # For FP16 entries that keep the same contiguous positions, the round-trip
    # error should be very small (float32 precision).
    assert (k_out[:, :, :2] - k_rope[:, :, :2]).abs().max().item() < 1e-4


def test_eviction_shrinks_cache_length():
    cache = _make_cache()
    k = torch.randn(B, H, 6, D)
    v = torch.randn(B, H, 6, D)
    positions = torch.arange(6)
    cos, sin = _make_rope_cos_sin(positions, D)
    k_rope = _apply_rope(k, cos, sin)
    cache.update(k_rope, v, layer_idx=0, cache_kwargs={"cos": cos, "sin": sin})

    tiers = torch.tensor([TIER_FP16, TIER_FP16, TIER_EVICT, TIER_EVICT, TIER_FP16, TIER_FP16])
    cache.apply_tier_assignment(0, tiers)
    assert cache.get_seq_length(0) == 4


def test_subsequent_update_after_reassign_works():
    cache = _make_cache()
    k = torch.randn(B, H, 8, D)
    v = torch.randn(B, H, 8, D)
    positions = torch.arange(8)
    cos, sin = _make_rope_cos_sin(positions, D)
    k_rope = _apply_rope(k, cos, sin)
    cache.update(k_rope, v, layer_idx=0, cache_kwargs={"cos": cos, "sin": sin})

    tiers = torch.tensor(
        [TIER_FP16, TIER_FP16, TIER_INT4, TIER_INT4, TIER_INT4, TIER_INT4, TIER_FP16, TIER_FP16]
    )
    cache.apply_tier_assignment(0, tiers)

    # now append 2 more tokens; materialized cache should have 10
    k2 = torch.randn(B, H, 2, D)
    v2 = torch.randn(B, H, 2, D)
    pos2 = torch.arange(8, 10)
    cos2, sin2 = _make_rope_cos_sin(pos2, D)
    k2_rope = _apply_rope(k2, cos2, sin2)
    k_out, _v_out = cache.update(k2_rope, v2, layer_idx=0, cache_kwargs={"cos": cos2, "sin": sin2})
    assert k_out.shape == (B, H, 10, D)
    assert cache.get_seq_length(0) == 10


# ------------------------------------------------------------------ #
# Phase 2 tests
# ------------------------------------------------------------------ #
def test_rope_inverse_is_exact():
    """Verify _inverse_rope undoes _apply_rope to float32 precision."""
    k = torch.randn(B, H, 16, D)
    positions = torch.arange(16)
    cos, sin = _make_rope_cos_sin(positions, D)
    cos4d = cos.unsqueeze(1)  # [1, 1, S, D]
    sin4d = sin.unsqueeze(1)
    k_rope = TieredKVCache._apply_rope(k, cos4d, sin4d)
    k_back = TieredKVCache._inverse_rope(k_rope, cos4d, sin4d)
    assert torch.allclose(k, k_back, atol=1e-5)


def test_eviction_with_rerope_produces_contiguous_positions():
    """After eviction, _materialize applies RoPE with StreamingLLM positions."""
    cache = _make_cache()
    k = torch.randn(B, H, 8, D)
    v = torch.randn(B, H, 8, D)
    positions = torch.arange(8)
    cos, sin = _make_rope_cos_sin(positions, D)
    k_rope = _apply_rope(k, cos, sin)
    cache.update(k_rope, v, layer_idx=0, cache_kwargs={"cos": cos, "sin": sin})

    # Evict positions 2 and 3.
    tiers = torch.tensor(
        [TIER_FP16, TIER_FP16, TIER_EVICT, TIER_EVICT, TIER_INT4, TIER_INT4, TIER_FP16, TIER_FP16]
    )
    cache.apply_tier_assignment(0, tiers)
    assert cache.get_seq_length(0) == 6

    # Append a new token (the model would assign position = get_seq_length = 6).
    k_new = torch.randn(B, H, 1, D)
    v_new = torch.randn(B, H, 1, D)
    pos_new = torch.tensor([6])
    cos_new, sin_new = _make_rope_cos_sin(pos_new, D)
    k_new_rope = _apply_rope(k_new, cos_new, sin_new)
    k_out, _v_out = cache.update(
        k_new_rope, v_new, layer_idx=0, cache_kwargs={"cos": cos_new, "sin": sin_new}
    )
    assert k_out.shape == (B, H, 7, D)
    assert cache.get_seq_length(0) == 7


def test_int2_tier_storage_round_trip():
    """TIER_INT2 tokens are stored and recovered (with quantization loss)."""
    cache = TieredKVCache(
        n_sink=2,
        recent_window=2,
        int4_budget=None,
        int2_budget=0,
        dtype=DTYPE,
        rope_theta=ROPE_THETA,
        head_dim=D,
    )
    # 68 tokens — middle = 64 which divides evenly by group_size=64.
    S = 68
    k = torch.randn(B, H, S, D)
    v = torch.randn(B, H, S, D)
    positions = torch.arange(S)
    cos, sin = _make_rope_cos_sin(positions, D)
    k_rope = _apply_rope(k, cos, sin)
    cache.update(k_rope, v, layer_idx=0, cache_kwargs={"cos": cos, "sin": sin})

    tiers = torch.full((S,), TIER_INT2, dtype=torch.long)
    tiers[:2] = TIER_FP16
    tiers[-2:] = TIER_FP16
    cache.apply_tier_assignment(0, tiers)

    state = cache._layers[0]
    assert state.int2_kq is not None
    assert state.int2_actual_count == 64
    assert cache.get_seq_length(0) == S

    k_out, v_out = cache._materialize(0)
    assert k_out.shape == (B, H, S, D)
    assert v_out.shape == (B, H, S, D)


def test_mixed_int4_int2_middle_ordering():
    """INT4 and INT2 middle tokens are merged in position order."""
    cache = TieredKVCache(
        n_sink=1,
        recent_window=1,
        int4_budget=None,
        int2_budget=0,
        dtype=DTYPE,
        rope_theta=ROPE_THETA,
        head_dim=D,
    )
    # 130 tokens: 1 sink + 128 middle + 1 recent.
    # We'll assign odd-indexed middle tokens to INT4 and even-indexed to INT2.
    S = 130
    k = torch.randn(B, H, S, D)
    v = torch.randn(B, H, S, D)
    positions = torch.arange(S)
    cos, sin = _make_rope_cos_sin(positions, D)
    k_rope = _apply_rope(k, cos, sin)
    cache.update(k_rope, v, layer_idx=0, cache_kwargs={"cos": cos, "sin": sin})

    tiers = torch.full((S,), TIER_INT4, dtype=torch.long)
    tiers[0] = TIER_FP16
    tiers[-1] = TIER_FP16
    # Even-indexed middle positions → INT2 (positions 2,4,6,...)
    for i in range(1, S - 1):
        if i % 2 == 0:
            tiers[i] = TIER_INT2
    cache.apply_tier_assignment(0, tiers)

    state = cache._layers[0]
    assert state.int4_kq is not None
    assert state.int2_kq is not None
    assert cache.get_seq_length(0) == S

    k_out, _v_out = cache._materialize(0)
    assert k_out.shape == (B, H, S, D)

    # Check positions come back in order via _all_in_position_order.
    _, _, pos_out = cache._all_in_position_order(0)
    diffs = pos_out[1:] - pos_out[:-1]
    assert (diffs > 0).all(), "positions must be strictly ascending"
