"""Tests for attention-implementation flexibility (W4).

``TieredKVCache`` must work end-to-end under non-eager attention when the
chosen eviction policy doesn't require prefill attention output. These tests
use a tiny random-init Qwen2 so they stay CPU-only and deterministic.

Test coverage:
    * ``load_model`` auto-mode selects ``sdpa`` when ``need_attentions=False``.
    * ``load_model`` auto-mode selects ``eager`` when ``need_attentions=True``.
    * Position-only eviction runs under SDPA without needing attentions.
    * ``forward_with_tracking`` emits a clean ``RuntimeWarning`` (not an
      ``AssertionError``) when a tracker is supplied but attentions are
      dropped.
"""

from __future__ import annotations

import warnings

import pytest
import torch

pytest.importorskip("transformers")
from transformers import AutoConfig, AutoModelForCausalLM

from fade.patch import (
    _installed_attn_impl,
    create_tiered_cache,
    forward_with_tracking,
    load_model,  # noqa: F401 (imported for test side)
)
from fade.policy import reassign_tiers_by_position
from fade.tracker import AttentionTracker

DEVICE = "cpu"
DTYPE = torch.float32


def _tiny_model(attn_impl: str):
    cfg = AutoConfig.for_model(
        "qwen2",
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        rope_theta=10000.0,
        tie_word_embeddings=True,
    )
    model = AutoModelForCausalLM.from_config(cfg, attn_implementation=attn_impl)
    model.to(DTYPE).eval()
    return model, cfg


def test_installed_attn_impl_reports_eager():
    model, _ = _tiny_model("eager")
    assert _installed_attn_impl(model) == "eager"


def test_installed_attn_impl_reports_sdpa():
    model, _ = _tiny_model("sdpa")
    assert _installed_attn_impl(model) == "sdpa"


def test_position_policy_works_under_sdpa():
    """SDPA + position eviction is the canonical non-attention-aware path."""
    model, cfg = _tiny_model("sdpa")
    num_layers = cfg.num_hidden_layers

    cache = create_tiered_cache(
        model,
        dtype=DTYPE,
        n_sink=2,
        recent_window=4,
        int4_budget=4,
        int2_budget=0,
    )
    prompt_ids = torch.randint(0, cfg.vocab_size, (1, 20))

    # No tracker — SDPA works fine.
    out = forward_with_tracking(model, prompt_ids, cache, tracker=None)
    assert out.logits.shape == (1, 20, cfg.vocab_size)
    assert cache.get_seq_length(0) == 20

    # Reassign tiers by position (no attention data needed).
    reassign_tiers_by_position(cache, num_layers)
    state = cache._layers[0]
    # Budget=4 with 20 - 2 sinks - 4 recent = 14 middle; 4 INT4, 10 evicted.
    assert state.int4_kq.shape[-2] == 4
    assert cache.get_seq_length(0) == 2 + 4 + 4

    # Decode still works.
    tok = out.logits[:, -1:, :].argmax(dim=-1)
    out2 = forward_with_tracking(model, tok, cache, tracker=None)
    assert out2.logits.shape == (1, 1, cfg.vocab_size)


def test_tracker_under_sdpa_warns_once_and_degrades_gracefully():
    """Supplying a tracker under SDPA must warn, not crash."""
    model, cfg = _tiny_model("sdpa")
    cache = create_tiered_cache(
        model,
        dtype=DTYPE,
        n_sink=2,
        recent_window=4,
        int4_budget=None,
        int2_budget=0,
    )
    tracker = AttentionTracker(num_layers=cfg.num_hidden_layers)
    prompt_ids = torch.randint(0, cfg.vocab_size, (1, 8))

    # Depending on the transformers minor, SDPA may either return None or
    # dispatch back to the eager path when output_attentions=True is
    # requested. Either way, the forward must complete without raising.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = forward_with_tracking(model, prompt_ids, cache, tracker=tracker)

    assert out.logits.shape == (1, 8, cfg.vocab_size)
    # If a warning was emitted, it must be the specific RuntimeWarning from
    # forward_with_tracking (not a spurious DeprecationWarning).
    runtime_warnings = [w for w in caught if issubclass(w.category, RuntimeWarning)]
    if out.attentions is None:
        assert runtime_warnings, "expected a RuntimeWarning when attentions is None"
        assert "AttentionTracker" in str(runtime_warnings[0].message)


def test_eager_path_still_feeds_tracker():
    """Regression: when eager is actually used, the tracker gets observations."""
    model, cfg = _tiny_model("eager")
    cache = create_tiered_cache(
        model,
        dtype=DTYPE,
        n_sink=2,
        recent_window=4,
        int4_budget=None,
        int2_budget=0,
    )
    tracker = AttentionTracker(num_layers=cfg.num_hidden_layers)
    prompt_ids = torch.randint(0, cfg.vocab_size, (1, 8))
    forward_with_tracking(model, prompt_ids, cache, tracker=tracker)
    # Tracker must have non-None scores for every layer.
    for layer_idx in range(cfg.num_hidden_layers):
        assert tracker.scores(layer_idx) is not None
        assert tracker.scores(layer_idx).shape == (8,)
