"""Tuning configuration for the tiered KV cache.

All of FADE's knobs collected in one validated dataclass, with three named
presets so callers don't have to reason about the interaction between budgets
and window sizes.

Example:
    from fade.config import FadeConfig
    from fade.patch import create_tiered_cache

    cache = create_tiered_cache(model, config=FadeConfig.balanced())
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Literal

# --- knobs (top of module for easy override) ------------------------------- #
DEFAULT_N_SINK: int = 4
DEFAULT_RECENT_WINDOW: int = 64
DEFAULT_INT2_GROUP_SIZE: int = 64
DEFAULT_REASSIGN_EVERY: int = 64
DEFAULT_PREFILL_TRACK_LIMIT: int = 1024

EvictionPolicy = Literal["h2o", "ema", "position", "learned"]
Phase = Literal["1a", "2"]
QuantBits = Literal[4, 2]
QuantBackendName = Literal["int4", "turbo"]


@dataclass(frozen=True)
class FadeConfig:
    """Validated FADE configuration.

    Attributes:
        phase: ``"1a"`` for no-eviction INT4 compression, ``"2"`` for
            budget-bounded eviction with re-RoPE.
        n_sink: number of leading attention-sink tokens kept in FP16.
        recent_window: number of trailing tokens kept in FP16.
        int4_budget: maximum middle tokens retained in INT4. ``None`` means
            unlimited (Phase 1-A). Set to an int for Phase 2 eviction.
        int2_budget: maximum middle tokens retained in INT2 after INT4 is
            full. Currently recommended to be ``0`` — INT2 is too lossy on
            small models.
        eviction_policy: ``"h2o"`` (best quality, needs prefill attention),
            ``"ema"`` (decode-only EMA tracker), or ``"position"`` (FIFO).
        reassign_every: run tier assignment every N decode steps.
        prefill_track_limit: max prompt length for which
            ``output_attentions=True`` is safe during prefill.
        int2_group_size: group size for INT2 quantization along the
            sequence axis.
        middle_k_bits: quantization bits for K in the middle tier. K has
            heavy per-channel outliers so 4 is recommended.
        middle_v_bits: quantization bits for V in the middle tier. V
            tolerates INT2 better than K; set to 2 for asymmetric
            compression.
    """

    phase: Phase = "1a"
    n_sink: int = DEFAULT_N_SINK
    recent_window: int = DEFAULT_RECENT_WINDOW
    int4_budget: int | None = None
    int2_budget: int = 0
    eviction_policy: EvictionPolicy = "h2o"
    reassign_every: int = DEFAULT_REASSIGN_EVERY
    prefill_track_limit: int = DEFAULT_PREFILL_TRACK_LIMIT
    int2_group_size: int = DEFAULT_INT2_GROUP_SIZE
    middle_k_bits: QuantBits = 4
    middle_v_bits: QuantBits = 4
    quant_backend: QuantBackendName = "int4"
    _extras: dict = field(default_factory=dict, repr=False)

    # --- validation -------------------------------------------------------- #
    def __post_init__(self) -> None:
        if self.n_sink < 0:
            raise ValueError(f"n_sink must be >= 0, got {self.n_sink}")
        if self.recent_window < 0:
            raise ValueError(f"recent_window must be >= 0, got {self.recent_window}")
        if self.int4_budget is not None and self.int4_budget < 0:
            raise ValueError(f"int4_budget must be >= 0 or None, got {self.int4_budget}")
        if self.int2_budget < 0:
            raise ValueError(f"int2_budget must be >= 0, got {self.int2_budget}")
        if self.reassign_every < 1:
            raise ValueError(f"reassign_every must be >= 1, got {self.reassign_every}")
        if self.int2_group_size < 1:
            raise ValueError(f"int2_group_size must be >= 1, got {self.int2_group_size}")
        if self.phase == "1a" and self.int4_budget is not None:
            raise ValueError(
                "phase='1a' requires int4_budget=None (no eviction); "
                f"got int4_budget={self.int4_budget}"
            )
        if self.phase == "2" and self.int4_budget is None and self.int2_budget == 0:
            raise ValueError("phase='2' requires at least one of int4_budget / int2_budget > 0")
        if self.eviction_policy not in ("h2o", "ema", "position", "learned"):
            raise ValueError(f"unknown eviction_policy: {self.eviction_policy!r}")
        if self.middle_k_bits not in (4, 2):
            raise ValueError(f"middle_k_bits must be 4 or 2, got {self.middle_k_bits}")
        if self.middle_v_bits not in (4, 2):
            raise ValueError(f"middle_v_bits must be 4 or 2, got {self.middle_v_bits}")

    # --- presets ----------------------------------------------------------- #
    @classmethod
    def safe(cls) -> FadeConfig:
        """Phase 1-A: compress the middle to INT4, evict nothing.

        Safest setting. ~3-4x KV compression, 100% greedy match vs baseline
        for prompts up to a few thousand tokens.
        """
        return cls(phase="1a", n_sink=4, recent_window=64)

    @classmethod
    def balanced(cls) -> FadeConfig:
        """Phase 2 with H2O eviction: ~5x compression, near-baseline quality.

        Recommended default for production inference.
        """
        return cls(
            phase="2",
            n_sink=4,
            recent_window=64,
            int4_budget=400,
            int2_budget=0,
            eviction_policy="h2o",
        )

    @classmethod
    def aggressive(cls) -> FadeConfig:
        """Phase 2 with smaller INT4 budget: ~7-8x compression.

        Accepts some quality degradation in exchange for memory. Only use
        after validating on your specific workload.
        """
        return cls(
            phase="2",
            n_sink=4,
            recent_window=32,
            int4_budget=200,
            int2_budget=0,
            eviction_policy="h2o",
        )

    def with_overrides(self, **kwargs) -> FadeConfig:
        """Return a copy with the specified fields replaced; re-validated."""
        return replace(self, **kwargs)

    def to_cache_kwargs(self) -> dict:
        """Arguments suitable for ``TieredKVCache(**config.to_cache_kwargs())``."""
        return {
            "n_sink": self.n_sink,
            "recent_window": self.recent_window,
            "int4_budget": self.int4_budget,
            "int2_budget": self.int2_budget,
            "middle_k_bits": self.middle_k_bits,
            "middle_v_bits": self.middle_v_bits,
            "quant_backend": self.quant_backend,
        }


__all__ = [
    "EvictionPolicy",
    "FadeConfig",
    "Phase",
]
