"""Batched ``TieredKVCache`` tests (shared-tier mode).

FADE's Phase-2 cache supports batch sizes > 1 when every row in the batch
shares the same positions and the same tier assignment. This is the common
case for decode-only batching where a single prompt is replicated or for
prompts that are left-padded to the same length.

Per-sequence tier decisions (ragged batches with different eviction per row)
are a future workstream; those require a `[B, S]` tiers tensor and validity
masking and are tracked separately. The tests here lock in the shared-tier
contract so we don't regress while W2.5 is being designed.
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
from fade.policy import _assign_one_layer
from fade.tracker import AttentionTracker

torch.manual_seed(0)

# --- fixtures --------------------------------------------------------------- #
H, D = 2, 32
DTYPE = torch.float32
ROPE_THETA = 10000.0


def _make_cache(**kw) -> TieredKVCache:
    defaults = dict(
        n_sink=2,
        recent_window=3,
        int4_budget=None,
        int2_budget=0,
        dtype=DTYPE,
        rope_theta=ROPE_THETA,
        head_dim=D,
    )
    defaults.update(kw)
    return TieredKVCache(**defaults)


def _make_rope_cos_sin(positions, head_dim, theta=ROPE_THETA):
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    freqs = positions.float().unsqueeze(-1) * inv_freq.unsqueeze(0)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos().unsqueeze(0)
    sin = emb.sin().unsqueeze(0)
    return cos, sin


def _rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rope(k, cos, sin):
    if cos.dim() == 3:
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
    return k * cos + _rotate_half(k) * sin


# --- tests ------------------------------------------------------------------ #
def test_update_batched_tracks_length():
    for B in (2, 4):
        cache = _make_cache()
        k = torch.randn(B, H, 10, D)
        v = torch.randn(B, H, 10, D)
        positions = torch.arange(10)
        cos, sin = _make_rope_cos_sin(positions, D)
        k_rope = _apply_rope(k, cos, sin)
        k_out, v_out = cache.update(k_rope, v, layer_idx=0)
        assert k_out.shape == (B, H, 10, D)
        assert v_out.shape == (B, H, 10, D)
        assert cache.get_seq_length(0) == 10


def test_tier_assignment_batched_splits_storage():
    B = 3
    cache = _make_cache()
    k = torch.randn(B, H, 10, D)
    v = torch.randn(B, H, 10, D)
    positions = torch.arange(10)
    cos, sin = _make_rope_cos_sin(positions, D)
    k_rope = _apply_rope(k, cos, sin)
    cache.update(k_rope, v, layer_idx=0)

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
    assert state.fp16_k.shape == (B, H, 5, D)
    # INT4 is bit-packed along the last dim (D -> D/2).
    assert state.int4_kq.shape == (B, H, 5, D // 2)
    assert state.int4_ks.shape == (B, H, 1, D)  # per-channel K scales survive
    assert state.int4_vs.shape == (B, H, 5, 1)  # per-token V scales survive
    assert cache.get_seq_length(0) == 10


def test_fp16_rows_stay_bit_exact_per_row_batched():
    """Tokens that stay FP16 across tier-assignment must survive per-row."""
    B = 4
    cache = _make_cache()
    k = torch.randn(B, H, 8, D)
    v = torch.randn(B, H, 8, D)
    positions = torch.arange(8)
    cos, sin = _make_rope_cos_sin(positions, D)
    k_rope = _apply_rope(k, cos, sin)
    cache.update(k_rope, v, layer_idx=0)

    tiers = torch.tensor(
        [TIER_FP16, TIER_FP16, TIER_INT4, TIER_INT4, TIER_INT4, TIER_INT4, TIER_FP16, TIER_FP16]
    )
    cache.apply_tier_assignment(0, tiers)
    _, v_out = cache._materialize(0)
    # V is never touched by RoPE; FP16 survivors must be bit-exact per row.
    assert torch.allclose(v_out[:, :, :2], v[:, :, :2])
    assert torch.allclose(v_out[:, :, -2:], v[:, :, -2:])


def test_eviction_batched_shrinks_length():
    B = 2
    cache = _make_cache()
    k = torch.randn(B, H, 6, D)
    v = torch.randn(B, H, 6, D)
    positions = torch.arange(6)
    cos, sin = _make_rope_cos_sin(positions, D)
    k_rope = _apply_rope(k, cos, sin)
    cache.update(k_rope, v, layer_idx=0)

    tiers = torch.tensor([TIER_FP16, TIER_FP16, TIER_EVICT, TIER_EVICT, TIER_FP16, TIER_FP16])
    cache.apply_tier_assignment(0, tiers)
    assert cache.get_seq_length(0) == 4

    k_out, v_out = cache._materialize(0)
    assert k_out.shape == (B, H, 4, D)
    assert v_out.shape == (B, H, 4, D)


def test_int2_tier_batched_round_trip():
    B = 2
    cache = _make_cache(n_sink=2, recent_window=2)
    S = 68  # middle = 64, divisible by group_size=64
    k = torch.randn(B, H, S, D)
    v = torch.randn(B, H, S, D)
    positions = torch.arange(S)
    cos, sin = _make_rope_cos_sin(positions, D)
    k_rope = _apply_rope(k, cos, sin)
    cache.update(k_rope, v, layer_idx=0)

    tiers = torch.full((S,), TIER_INT2, dtype=torch.long)
    tiers[:2] = TIER_FP16
    tiers[-2:] = TIER_FP16
    cache.apply_tier_assignment(0, tiers)

    state = cache._layers[0]
    assert state.int2_kq is not None
    assert state.int2_kq.shape[0] == B
    assert state.int2_ks.shape[0] == B
    k_out, v_out = cache._materialize(0)
    assert k_out.shape == (B, H, S, D)
    assert v_out.shape == (B, H, S, D)


def test_subsequent_update_after_reassign_batched():
    B = 2
    cache = _make_cache()
    k = torch.randn(B, H, 8, D)
    v = torch.randn(B, H, 8, D)
    positions = torch.arange(8)
    cos, sin = _make_rope_cos_sin(positions, D)
    k_rope = _apply_rope(k, cos, sin)
    cache.update(k_rope, v, layer_idx=0)

    tiers = torch.tensor(
        [TIER_FP16, TIER_FP16, TIER_INT4, TIER_INT4, TIER_INT4, TIER_INT4, TIER_FP16, TIER_FP16]
    )
    cache.apply_tier_assignment(0, tiers)

    k2 = torch.randn(B, H, 2, D)
    v2 = torch.randn(B, H, 2, D)
    pos2 = torch.arange(8, 10)
    cos2, sin2 = _make_rope_cos_sin(pos2, D)
    k2_rope = _apply_rope(k2, cos2, sin2)
    k_out, _ = cache.update(k2_rope, v2, layer_idx=0)
    assert k_out.shape == (B, H, 10, D)


def test_tracker_observes_batched_attentions():
    """AttentionTracker must handle [B, H, Q, K] for B>1 (already sums over B)."""
    B = 4
    num_layers = 2
    tracker = AttentionTracker(num_layers=num_layers)
    attn = torch.rand(B, H, 10, 10)
    tracker.observe(attn, layer_idx=0)
    scores = tracker.scores(0)
    assert scores is not None
    assert scores.shape == (10,)
    # Scores should be non-negative and finite.
    assert torch.all(torch.isfinite(scores))
    assert torch.all(scores >= 0)


def test_batch_size_is_pinned_on_first_update():
    cache = _make_cache()
    assert cache.batch_size is None
    k = torch.randn(2, H, 4, D)
    v = torch.randn(2, H, 4, D)
    cache.update(k, v, layer_idx=0)
    assert cache.batch_size == 2


def test_batch_size_mismatch_raises():
    cache = _make_cache(batch_size=2)
    k = torch.randn(3, H, 4, D)
    v = torch.randn(3, H, 4, D)
    try:
        cache.update(k, v, layer_idx=0)
    except ValueError as e:
        assert "batch size mismatch" in str(e)
        return
    raise AssertionError("expected ValueError on batch-size mismatch")


def test_kv_shape_mismatch_raises():
    cache = _make_cache()
    k = torch.randn(2, H, 4, D)
    v = torch.randn(2, H, 5, D)  # different seq len
    try:
        cache.update(k, v, layer_idx=0)
    except ValueError as e:
        assert "K/V shape mismatch" in str(e)
        return
    raise AssertionError("expected ValueError on K/V shape mismatch")


def test_non_4d_input_raises():
    cache = _make_cache()
    k = torch.randn(H, 4, D)  # missing batch dim
    v = torch.randn(H, 4, D)
    try:
        cache.update(k, v, layer_idx=0)
    except ValueError as e:
        assert "[B, H, S, D]" in str(e)
        return
    raise AssertionError("expected ValueError on non-4D input")


def test_assign_one_layer_unbatched_contract_preserved():
    """Sanity: the policy still works with [S]-shaped scores even at B>1 caches."""
    scores = torch.tensor([0.1, 0.2, 5.0, 4.0, 3.0, 0.5, 0.4, 0.3])
    tiers = _assign_one_layer(
        S=8,
        scores=scores,
        n_sink=1,
        recent_window=1,
        int4_budget=2,
        int2_budget=0,
    )
    assert tiers.shape == (8,)
    assert tiers[0].item() == TIER_FP16  # sink
    assert tiers[-1].item() == TIER_FP16  # recent
    # The two highest-scoring middle positions survive as INT4.
    int4_positions = (tiers == TIER_INT4).nonzero(as_tuple=False).squeeze(-1).tolist()
    assert 2 in int4_positions  # score 5.0
    assert 3 in int4_positions  # score 4.0
