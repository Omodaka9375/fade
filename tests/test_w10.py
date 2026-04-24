"""Tests for W10: asymmetric K/V compression + learned eviction policy."""

from __future__ import annotations

import pytest
import torch

from fade.cache import TIER_FP16, TIER_INT4, TieredKVCache
from fade.config import FadeConfig
from fade.learned_policy import EvictionMLP, _build_features, reassign_tiers_learned

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


# --- asymmetric K/V -------------------------------------------------------- #
def test_asymmetric_config_k4_v2():
    c = FadeConfig(phase="2", int4_budget=100, middle_k_bits=4, middle_v_bits=2)
    assert c.middle_k_bits == 4
    assert c.middle_v_bits == 2


def test_asymmetric_config_rejects_bad_bits():
    with pytest.raises(ValueError, match="middle_k_bits"):
        FadeConfig(middle_k_bits=3)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="middle_v_bits"):
        FadeConfig(middle_v_bits=8)  # type: ignore[arg-type]


def test_asymmetric_k4_v2_round_trip():
    """K at INT4, V at INT2: materialize produces correct shape."""
    cache = _make_cache(middle_k_bits=4, middle_v_bits=2)
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

    state = cache._layers[0]
    # K is still INT4-packed.
    assert state.int4_kq is not None
    assert state.int4_kq.dtype == torch.uint8
    # V is INT2 (stored in int4_vq slot as int8).
    assert state.int4_vq is not None
    assert state.int4_vq.dtype == torch.int8

    k_out, v_out = cache._materialize(0)
    assert k_out.shape == (B, H, 10, D)
    assert v_out.shape == (B, H, 10, D)
    assert torch.isfinite(k_out).all()
    assert torch.isfinite(v_out).all()


def test_asymmetric_k4_v4_is_default():
    """Default symmetric K4/V4 still works."""
    cache = _make_cache(middle_k_bits=4, middle_v_bits=4)
    k = torch.randn(B, H, 8, D)
    v = torch.randn(B, H, 8, D)
    cache.update(k, v, layer_idx=0)

    tiers = torch.tensor(
        [TIER_FP16, TIER_FP16, TIER_INT4, TIER_INT4, TIER_INT4, TIER_INT4, TIER_FP16, TIER_FP16]
    )
    cache.apply_tier_assignment(0, tiers)
    k_out, _v_out = cache._materialize(0)
    assert k_out.shape == (B, H, 8, D)


def test_to_cache_kwargs_includes_bits():
    c = FadeConfig(phase="2", int4_budget=100, middle_k_bits=4, middle_v_bits=2)
    kw = c.to_cache_kwargs()
    assert kw["middle_k_bits"] == 4
    assert kw["middle_v_bits"] == 2


# --- learned eviction policy ------------------------------------------------ #
def test_eviction_mlp_forward_shape():
    mlp = EvictionMLP()
    features = torch.randn(20, 4)
    out = mlp(features)
    assert out.shape == (20,)
    assert (out >= 0).all() and (out <= 1).all()


def test_build_features_shape():
    features = _build_features(
        S=16,
        scores=torch.rand(16),
        layer_idx=1,
        num_layers=4,
        step=100,
        device=torch.device("cpu"),
    )
    assert features.shape == (16, 4)
    assert torch.isfinite(features).all()
    # All features should be in [0, 1].
    assert features.min() >= 0
    assert features.max() <= 1.0 + 1e-6


def test_build_features_no_scores():
    """When scores is None, mass feature should be zero."""
    features = _build_features(
        S=8,
        scores=None,
        layer_idx=0,
        num_layers=2,
        step=50,
        device=torch.device("cpu"),
    )
    assert (features[:, 1] == 0).all()


def test_reassign_tiers_learned_runs():
    """Learned policy with a random MLP should not crash."""
    cache = _make_cache(int4_budget=4, int2_budget=0)
    k = torch.randn(B, H, 12, D)
    v = torch.randn(B, H, 12, D)
    cache.update(k, v, layer_idx=0)

    mlp = EvictionMLP()
    reassign_tiers_learned(cache, mlp, num_layers=1, step=10)

    # Should have evicted some tokens (budget=4 < 12-2-3=7 middle).
    assert cache.get_seq_length(0) < 12
    state = cache._layers[0]
    assert state.fp16_k is not None
    assert state.int4_kq is not None


def test_eviction_mlp_save_load(tmp_path):
    mlp = EvictionMLP()
    path = tmp_path / "test_mlp.pt"
    mlp.save(path)

    loaded = EvictionMLP.load(path)
    # Weights should match.
    for p1, p2 in zip(mlp.parameters(), loaded.parameters(), strict=False):
        assert torch.equal(p1, p2)


def test_config_accepts_learned_policy():
    c = FadeConfig(phase="2", int4_budget=100, eviction_policy="learned")
    assert c.eviction_policy == "learned"
