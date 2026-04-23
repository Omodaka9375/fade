"""Tests for ``fade.config.FadeConfig`` validation and presets."""
from __future__ import annotations

import pytest

from fade.config import FadeConfig


def test_safe_preset_is_phase_1a():
    c = FadeConfig.safe()
    assert c.phase == "1a"
    assert c.int4_budget is None
    assert c.int2_budget == 0


def test_balanced_preset_uses_h2o():
    c = FadeConfig.balanced()
    assert c.phase == "2"
    assert c.eviction_policy == "h2o"
    assert isinstance(c.int4_budget, int)


def test_aggressive_preset_has_smaller_budget():
    balanced = FadeConfig.balanced()
    aggressive = FadeConfig.aggressive()
    assert aggressive.int4_budget < (balanced.int4_budget or 10_000)
    assert aggressive.recent_window <= balanced.recent_window


def test_phase_1a_rejects_int4_budget():
    with pytest.raises(ValueError, match="phase='1a'"):
        FadeConfig(phase="1a", int4_budget=100)


def test_phase_2_requires_budget():
    with pytest.raises(ValueError, match="phase='2'"):
        FadeConfig(phase="2", int4_budget=None, int2_budget=0)


def test_unknown_policy_rejected():
    with pytest.raises(ValueError, match="unknown eviction_policy"):
        FadeConfig(phase="1a", eviction_policy="random")  # type: ignore[arg-type]


def test_negative_fields_rejected():
    with pytest.raises(ValueError):
        FadeConfig(n_sink=-1)
    with pytest.raises(ValueError):
        FadeConfig(recent_window=-1)
    with pytest.raises(ValueError):
        FadeConfig(phase="2", int4_budget=-5)
    with pytest.raises(ValueError):
        FadeConfig(int2_budget=-1)
    with pytest.raises(ValueError):
        FadeConfig(reassign_every=0)
    with pytest.raises(ValueError):
        FadeConfig(int2_group_size=0)


def test_with_overrides_revalidates():
    c = FadeConfig.balanced()
    # Valid override.
    c2 = c.with_overrides(int4_budget=128)
    assert c2.int4_budget == 128
    # Invalid override — overrides re-run __post_init__.
    with pytest.raises(ValueError):
        c.with_overrides(phase="1a")  # int4_budget is still set


def test_to_cache_kwargs_roundtrip():
    c = FadeConfig.balanced()
    kw = c.to_cache_kwargs()
    assert kw == {
        "n_sink": c.n_sink,
        "recent_window": c.recent_window,
        "int4_budget": c.int4_budget,
        "int2_budget": c.int2_budget,
        "middle_k_bits": c.middle_k_bits,
        "middle_v_bits": c.middle_v_bits,
    }


def test_is_importable_from_top_level():
    from fade import FadeConfig as TopLevel

    assert TopLevel is FadeConfig
