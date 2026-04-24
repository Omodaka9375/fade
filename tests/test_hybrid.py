"""Tests for hybrid attention support (Qwen 3.5/3.6 DeltaNet skip)."""

from __future__ import annotations

import types

import torch

from fade.cache import TieredKVCache
from fade.policy import reassign_tiers, reassign_tiers_by_position
from fade.tracker import AttentionTracker

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


def test_skip_layers_excludes_from_tier_assignment():
    """Layers in skip_layers should not be tiered."""
    cache = _make_cache()
    cache.set_skip_layers({0, 1, 2})  # simulate 3 DeltaNet + 1 full_attn

    # Populate all 4 layers.
    for layer in range(4):
        k = torch.randn(B, H, 10, D)
        v = torch.randn(B, H, 10, D)
        cache.update(k, v, layer_idx=layer)

    # Reassign by position — should only touch layer 3.
    reassign_tiers_by_position(cache, num_layers=4)

    # Skipped layers: still pure FP16, no INT4.
    for layer in (0, 1, 2):
        state = cache._layers[layer]
        assert state.int4_kq is None, f"layer {layer} should be skipped"
        assert state.fp16_k is not None

    # Managed layer: should have INT4 middle.
    state3 = cache._layers[3]
    assert state3.fp16_k is not None  # sinks + recent
    # With int4_budget=None, all middle goes to INT4.
    assert state3.int4_kq is not None


def test_is_managed_and_managed_layers():
    cache = _make_cache()
    cache.set_skip_layers({1, 3})
    for i in range(4):
        cache.update(torch.randn(B, H, 4, D), torch.randn(B, H, 4, D), layer_idx=i)

    assert cache.is_managed(0) is True
    assert cache.is_managed(1) is False
    assert cache.is_managed(2) is True
    assert cache.is_managed(3) is False
    assert cache.managed_layers == {0, 2}


def test_ema_policy_skips_deltanet_layers():
    """EMA tracker + reassign_tiers should skip non-managed layers."""
    cache = _make_cache()
    cache.set_skip_layers({0, 1, 2})
    tracker = AttentionTracker(num_layers=4)

    for layer in range(4):
        k = torch.randn(B, H, 10, D)
        v = torch.randn(B, H, 10, D)
        cache.update(k, v, layer_idx=layer)
        # Simulate attention observation only for the full-attn layer.
        if layer == 3:
            attn = torch.rand(B, H, 10, 10)
            tracker.observe(attn, layer)

    reassign_tiers(cache, tracker, num_layers=4)

    # Only layer 3 should be tiered.
    for layer in (0, 1, 2):
        assert cache._layers[layer].int4_kq is None
    assert cache._layers[3].int4_kq is not None


def test_layer_types_detection():
    """Simulate Qwen 3.5 layer_types config."""
    # 3:1 pattern: linear, linear, linear, full, repeated.
    layer_types = ["linear_attention"] * 3 + ["full_attention"]
    layer_types = layer_types * 2  # 8 layers total

    types.SimpleNamespace(
        layer_types=layer_types,
        rope_theta=10000.0,
        hidden_size=D,
        num_attention_heads=1,
        head_dim=D,
    )

    # Verify the skip set would contain linear_attention layers.
    skip = {
        i for i, lt in enumerate(layer_types) if lt not in ("full_attention", "sliding_attention")
    }
    assert skip == {0, 1, 2, 4, 5, 6}
    assert {i for i in range(8) if i not in skip} == {3, 7}


def test_skip_layers_dont_affect_seq_length():
    """Total seq length should include all layers (managed + skipped)."""
    cache = _make_cache()
    cache.set_skip_layers({0})

    cache.update(torch.randn(B, H, 8, D), torch.randn(B, H, 8, D), layer_idx=0)
    cache.update(torch.randn(B, H, 8, D), torch.randn(B, H, 8, D), layer_idx=1)

    assert cache.get_seq_length(0) == 8
    assert cache.get_seq_length(1) == 8
