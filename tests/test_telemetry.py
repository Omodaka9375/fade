"""Tests for W8 observability: telemetry events, exporters, and debug dump."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import torch

from fade.cache import TIER_EVICT, TIER_FP16, TIER_INT4, TieredKVCache
from fade.telemetry import (
    JsonlExporter,
    ListExporter,
    TierEvent,
    attach_telemetry,
)

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


# --- TierEvent ------------------------------------------------------------- #
def test_tier_event_to_dict():
    e = TierEvent(layer_idx=1, fp16_count=5, int4_count=10)
    d = e.to_dict()
    assert d["layer_idx"] == 1
    assert d["fp16_count"] == 5
    assert "timestamp" in d


# --- ListExporter + attach -------------------------------------------------- #
def test_attach_telemetry_emits_events():
    cache = _make_cache()
    exporter = ListExporter()
    attach_telemetry(cache, exporter)

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

    assert len(exporter.events) == 1
    ev = exporter.events[0]
    assert ev.layer_idx == 0
    assert ev.fp16_count == 5
    assert ev.int4_count == 5
    assert ev.total_count == 10
    assert ev.evicted_count == 0


def test_telemetry_captures_eviction_counts():
    cache = _make_cache()
    exporter = ListExporter()
    attach_telemetry(cache, exporter)

    k = torch.randn(B, H, 6, D)
    v = torch.randn(B, H, 6, D)
    cache.update(k, v, layer_idx=0)

    tiers = torch.tensor([TIER_FP16, TIER_FP16, TIER_EVICT, TIER_EVICT, TIER_FP16, TIER_FP16])
    cache.apply_tier_assignment(0, tiers)

    ev = exporter.events[0]
    assert ev.evicted_count == 2
    assert ev.fp16_count == 4


def test_telemetry_with_scores():
    cache = _make_cache()
    exporter = ListExporter()
    attach_telemetry(cache, exporter)

    k = torch.randn(B, H, 8, D)
    v = torch.randn(B, H, 8, D)
    cache.update(k, v, layer_idx=0)

    tiers = torch.full((8,), TIER_FP16, dtype=torch.long)
    scores = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    cache.apply_tier_assignment(0, tiers, scores=scores)

    ev = exporter.events[0]
    assert ev.score_mean > 0
    assert ev.score_p50 > 0
    assert ev.score_p99 > 0


# --- JsonlExporter ---------------------------------------------------------- #
def test_jsonl_exporter_writes_valid_json():
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode="w") as f:
        path = f.name

    try:
        cache = _make_cache()
        exporter = JsonlExporter(path)
        attach_telemetry(cache, exporter)

        k = torch.randn(B, H, 6, D)
        v = torch.randn(B, H, 6, D)
        cache.update(k, v, layer_idx=0)
        tiers = torch.full((6,), TIER_FP16, dtype=torch.long)
        cache.apply_tier_assignment(0, tiers)
        exporter.close()

        lines = Path(path).read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data["total_count"] == 6
    finally:
        Path(path).unlink(missing_ok=True)


# --- dump_debug ------------------------------------------------------------- #
def test_dump_debug_creates_valid_json():
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

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name
    try:
        cache.dump_debug(path)
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        assert data["n_layers"] == 1
        layer = data["layers"][0]
        assert layer["fp16_count"] == 5
        assert layer["int4_count"] == 5
        assert layer["total_seq_length"] == 10
        assert "fp16_positions" in layer
        assert "int4_positions" in layer
    finally:
        Path(path).unlink(missing_ok=True)
