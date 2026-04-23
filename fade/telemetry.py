"""Structured telemetry for tier-assignment events.

Emits a ``TierEvent`` on every ``apply_tier_assignment`` call, carrying
per-layer counts and score statistics. Exporters consume these events.

Usage:
    from fade.telemetry import JsonlExporter, attach_telemetry

    exporter = JsonlExporter("tier_events.jsonl")
    attach_telemetry(cache, exporter)
    # ... run inference ...
    exporter.close()
"""
from __future__ import annotations

import json
import sys
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path

from torch import Tensor


@dataclass
class TierEvent:
    """Emitted on every ``apply_tier_assignment`` call."""

    timestamp: float = field(default_factory=time.time)
    layer_idx: int = 0
    fp16_count: int = 0
    int4_count: int = 0
    int2_count: int = 0
    evicted_count: int = 0
    total_count: int = 0
    score_mean: float = 0.0
    score_p50: float = 0.0
    score_p99: float = 0.0
    dequant_cache_bytes: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


def _build_event(
    layer_idx: int,
    tiers: Tensor,
    scores: Tensor | None,
    dequant_bytes: int,
) -> TierEvent:
    """Build a TierEvent from the tier assignment tensor and optional scores."""
    from fade.cache import TIER_EVICT, TIER_FP16, TIER_INT2, TIER_INT4

    S = int(tiers.shape[0])
    fp16 = int((tiers == TIER_FP16).sum().item())
    int4 = int((tiers == TIER_INT4).sum().item())
    int2 = int((tiers == TIER_INT2).sum().item())
    evicted = int((tiers == TIER_EVICT).sum().item())

    score_mean = score_p50 = score_p99 = 0.0
    if scores is not None and scores.numel() > 0:
        s = scores.float()
        score_mean = s.mean().item()
        score_p50 = s.median().item()
        score_p99 = s.quantile(0.99).item() if s.numel() >= 2 else s.max().item()

    return TierEvent(
        layer_idx=layer_idx,
        fp16_count=fp16,
        int4_count=int4,
        int2_count=int2,
        evicted_count=evicted,
        total_count=S,
        score_mean=score_mean,
        score_p50=score_p50,
        score_p99=score_p99,
        dequant_cache_bytes=dequant_bytes,
    )


# --- exporters -------------------------------------------------------------- #
class MetricsExporter(ABC):
    """Base class for telemetry exporters."""

    @abstractmethod
    def export(self, event: TierEvent) -> None: ...

    def close(self) -> None:  # noqa: B027
        """Called when the exporter is no longer needed."""


class StdoutExporter(MetricsExporter):
    """Print events to stdout."""

    def export(self, event: TierEvent) -> None:
        parts = [
            f"layer={event.layer_idx}",
            f"fp16={event.fp16_count}",
            f"int4={event.int4_count}",
            f"int2={event.int2_count}",
            f"evict={event.evicted_count}",
            f"score_mean={event.score_mean:.4f}",
        ]
        print(f"[FADE] {' '.join(parts)}", file=sys.stderr)


class JsonlExporter(MetricsExporter):
    """Append events as JSON lines to a file."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._fh = open(self._path, "a", encoding="utf-8")  # noqa: SIM115

    def export(self, event: TierEvent) -> None:
        self._fh.write(json.dumps(event.to_dict()) + "\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()


class ListExporter(MetricsExporter):
    """Collect events in memory (useful for testing)."""

    def __init__(self) -> None:
        self.events: list[TierEvent] = []

    def export(self, event: TierEvent) -> None:
        self.events.append(event)


# --- attach to cache -------------------------------------------------------- #
def attach_telemetry(cache, exporter: MetricsExporter) -> None:
    """Monkey-patch ``cache.apply_tier_assignment`` to emit telemetry.

    The original method is preserved and called first; the exporter
    receives a ``TierEvent`` after each call.
    """
    original = cache.apply_tier_assignment

    def _instrumented(layer_idx: int, tiers: Tensor, scores: Tensor | None = None) -> None:
        # Compute dequant bytes before the assignment (they'll be invalidated).
        dequant_bytes = 0
        if layer_idx < len(cache._layers):
            state = cache._layers[layer_idx]
            for t in (state.int4_k_deq, state.int4_v_deq, state.int2_k_deq, state.int2_v_deq):
                if t is not None:
                    dequant_bytes += int(t.element_size() * t.numel())
        event = _build_event(layer_idx, tiers, scores, dequant_bytes)
        original(layer_idx, tiers, scores=scores)
        exporter.export(event)

    cache.apply_tier_assignment = _instrumented


__all__ = [
    "JsonlExporter",
    "ListExporter",
    "MetricsExporter",
    "StdoutExporter",
    "TierEvent",
    "attach_telemetry",
]
