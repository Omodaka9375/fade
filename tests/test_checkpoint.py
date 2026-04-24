"""Cache checkpoint round-trip tests (W6)."""

from __future__ import annotations

import torch

from fade.cache import TIER_FP16, TIER_INT4, TieredKVCache

B, H, D = 1, 2, 32
DTYPE = torch.float32


def _make_cache(**kw) -> TieredKVCache:
    defaults = dict(
        n_sink=2,
        recent_window=3,
        int4_budget=None,
        int2_budget=0,
        dtype=DTYPE,
        rope_theta=10000.0,
        head_dim=D,
    )
    defaults.update(kw)
    return TieredKVCache(**defaults)


def test_state_dict_round_trip_fp16_only():
    """Save/restore a cache that only has FP16 tokens."""
    cache = _make_cache()
    k = torch.randn(B, H, 8, D)
    v = torch.randn(B, H, 8, D)
    cache.update(k, v, layer_idx=0)
    assert cache.get_seq_length(0) == 8

    sd = cache.cache_state_dict()
    assert sd["n_layers"] == 1
    assert "layer.0.fp16_k" in sd

    # Restore into a fresh cache.
    cache2 = _make_cache()
    cache2.load_cache_state_dict(sd)
    assert cache2.get_seq_length(0) == 8

    # Materialized output must be identical.
    k_out1, v_out1 = cache._materialize(0)
    k_out2, v_out2 = cache2._materialize(0)
    assert torch.equal(k_out1, k_out2)
    assert torch.equal(v_out1, v_out2)


def test_state_dict_round_trip_with_int4():
    """Save/restore after tier assignment splits tokens into FP16 + INT4."""
    cache = _make_cache()
    k = torch.randn(B, H, 10, D)
    v = torch.randn(B, H, 10, D)
    cache.update(k, v, layer_idx=0)

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

    sd = cache.cache_state_dict()
    assert "layer.0.int4_kq" in sd
    assert "layer.0.int4_ks" in sd

    cache2 = _make_cache()
    cache2.load_cache_state_dict(sd)
    assert cache2.get_seq_length(0) == 10

    # Materialized output must be identical (within dequant precision).
    k1, v1 = cache._materialize(0)
    k2, v2 = cache2._materialize(0)
    assert torch.allclose(k1, k2, atol=1e-6)
    assert torch.allclose(v1, v2, atol=1e-6)


def test_state_dict_excludes_dequant_caches():
    """Ephemeral dequant buffers must NOT appear in the state dict."""
    cache = _make_cache()
    k = torch.randn(B, H, 8, D)
    v = torch.randn(B, H, 8, D)
    cache.update(k, v, layer_idx=0)

    tiers = torch.tensor(
        [TIER_FP16, TIER_FP16, TIER_INT4, TIER_INT4, TIER_INT4, TIER_INT4, TIER_FP16, TIER_FP16]
    )
    cache.apply_tier_assignment(0, tiers)
    # Force dequant cache population.
    cache._materialize(0)

    sd = cache.cache_state_dict()
    for key in sd:
        assert "_deq" not in key, f"dequant buffer leaked: {key}"


def test_state_dict_preserves_scalars():
    cache = _make_cache()
    k = torch.randn(B, H, 6, D)
    v = torch.randn(B, H, 6, D)
    cache.update(k, v, layer_idx=0)

    sd = cache.cache_state_dict()
    assert sd["layer.0.sink_count"] == 0
    assert sd["layer.0.next_position"] == 6

    cache2 = _make_cache()
    cache2.load_cache_state_dict(sd)
    assert cache2._layers[0].next_position == 6
