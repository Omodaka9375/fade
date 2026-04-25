"""Tiered KV cache.

Phase 1 (current skeleton): FP16 sinks + recent window + INT4 middle, hard eviction.
Phase 2: add INT2 tier and bit-packing.
Phase 3: replace INT2 with a learned product-quantization codebook.

The cache inherits from ``transformers.DynamicCache`` so the HF generate loop
accepts it without modification. Only ``update()`` and ``get_seq_length()`` are
actually consulted by the loop; everything else here is for the tier policy.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

from fade._compat import DynamicCache
from fade.backends import QuantBackend, SymmetricINT4Backend, get_backend
from fade.quant import (
    DEFAULT_INT2_GROUP_SIZE,
    dequant_int4,
    dequant_k_int2,
    dequant_v_int2,
    pad_to_group,
    quant_k_int2,
    quant_k_int4,
    quant_v_int2,
    quant_v_int4,
)
from fade.rope import RopeScheme, Vanilla

# --- configuration (top of file for easy override) --------------------------- #
DEFAULT_N_SINK: int = 4
DEFAULT_RECENT_WINDOW: int = 128
DEFAULT_INT4_BUDGET: int | None = None  # None = unlimited (no eviction in Phase 1)
DEFAULT_INT2_BUDGET: int = 0  # Phase 2 enables the INT2 tier
DEFAULT_ROPE_THETA: float = 10000.0
DEFAULT_FP16_PREALLOC: int = 128  # initial pre-allocated capacity for FP16 tier
DEFAULT_MAX_DEQUANT_AGE: int | None = None  # None = keep forever; int = drop after N updates

TIER_FP16: int = 0
TIER_INT4: int = 1
TIER_INT2: int = 2
TIER_EVICT: int = 3
TIER_PQ: int = 4


@dataclass
class LayerState:
    """Per-layer tier state. All ``*_pos`` tensors hold ABSOLUTE positions.

    Layout invariant (maintained after every ``apply_tier_assignment``):
        - The first ``sink_count`` entries of ``fp16_k`` are the sink tokens
          (positions ``0 .. sink_count-1``).
        - The remaining FP16 entries are the recent-window tokens plus any
          tokens appended since the last reassignment.
        - ``int4_*`` entries hold the compressed middle in ascending position.
        - This means the sorted full cache is exactly the concatenation
          ``fp16_k[:sink_count] + int4_dequant + fp16_k[sink_count:]``, so the
          hot path never needs ``argsort`` or ``index_select``.
    """

    fp16_k: Tensor | None = None  # [B, H, S_fp, D]
    fp16_v: Tensor | None = None
    fp16_pos: Tensor | None = None  # LongTensor [S_fp]
    int4_kq: Tensor | None = None  # int8 tensor holding INT4 values, [B, H, S_q, D]
    int4_ks: Tensor | None = None  # K scales, [B, H, 1, D]
    int4_vq: Tensor | None = None
    int4_vs: Tensor | None = None  # V scales, [B, H, S_q, 1]
    int4_pos: Tensor | None = None
    # INT2 tier (Phase 2)
    int2_kq: Tensor | None = None  # int8 holding INT2 values, [B, H, S_padded, D]
    int2_ks: Tensor | None = None  # K scales, [B, H, G, D]
    int2_vq: Tensor | None = None
    int2_vs: Tensor | None = None  # V scales, [B, H, G, D]
    int2_pos: Tensor | None = None
    int2_actual_count: int = 0  # pre-padding token count (for trim after dequant)
    int2_group_size: int = DEFAULT_INT2_GROUP_SIZE
    sink_count: int = 0  # number of leading fp16 entries that are sinks
    # Monotonic counter: the absolute position to assign to the next appended
    # token. Never decreases, even after eviction. Guarantees unique position
    # labels across the lifetime of the cache.
    next_position: int = 0
    # Ephemeral dequantized cache. Populated on first materialize after a
    # reassignment, then reused until the next reassignment invalidates it.
    int4_k_deq: Tensor | None = None
    int4_v_deq: Tensor | None = None
    int2_k_deq: Tensor | None = None
    int2_v_deq: Tensor | None = None
    # Pre-allocated FP16 append buffer. Avoids torch.cat on every decode step.
    # When set, fp16_k/v/pos are *views* into the buffer up to _fp16_len.
    _fp16_buf_k: Tensor | None = None
    _fp16_buf_v: Tensor | None = None
    _fp16_buf_pos: Tensor | None = None
    _fp16_len: int = 0  # number of valid entries in the buffer
    # Dequant cache age: incremented on every update; used for age-based eviction.
    _dequant_age: int = 0
    # PQ tier (Phase 3)
    pq_codes: Tensor | None = None  # [..., n_sub] uint8 codes for K
    pq_v_codes: Tensor | None = None  # [..., n_sub] uint8 codes for V
    pq_pos: Tensor | None = None
    pq_k_deq: Tensor | None = None
    pq_v_deq: Tensor | None = None
    # Backend-compressed storage (for non-INT4 backends like TurboQuant).
    backend_k: dict | None = None
    backend_v: dict | None = None
    backend_pos: Tensor | None = None

    def total_seq_length(self) -> int:
        total = 0
        for pos in (self.fp16_pos, self.int4_pos, self.int2_pos, self.pq_pos, self.backend_pos):
            if pos is not None:
                total += int(pos.shape[0])
        return total


class TieredKVCache(DynamicCache):
    """Tiered KV cache with attention-aware compression policy.

    Invariants:
        - A position appears in AT MOST one tier per layer.
        - Absolute positions are never renumbered (RoPE stays valid).
        - ``update()`` must return K, V covering every retained position,
          sorted ascending by absolute position.

    Phase 2: K is stored **post-RoPE** during normal operation (identical
    to Phase 1). When eviction occurs in ``apply_tier_assignment()``, the
    retained K is re-RoPE'd: undo at old absolute positions, re-apply at
    StreamingLLM-style contiguous positions. This keeps FP16 tokens
    bit-exact when no eviction happens and makes eviction safe.

    Batching:
        Two modes are supported:

        **Shared-tier** (default): all rows share the same positions and
        tier assignment (``tiers`` is ``[S]``). Scores are pooled across B.
        Correct for lockstep decoding with equal-length prompts.

        **Per-sequence** (``apply_tier_assignment_per_sequence``): each row
        gets an independent ``[S]`` tier assignment. After eviction, rows
        may have different surviving counts; K/V are padded to the max.
        Use this for continuous-batching where sequences diverge.
    """

    def __init__(
        self,
        n_sink: int = DEFAULT_N_SINK,
        recent_window: int = DEFAULT_RECENT_WINDOW,
        int4_budget: int | None = DEFAULT_INT4_BUDGET,
        int2_budget: int = DEFAULT_INT2_BUDGET,
        dtype: torch.dtype = torch.float16,
        cache_dequant: bool = True,
        rope_theta: float = DEFAULT_ROPE_THETA,
        head_dim: int | None = None,
        batch_size: int | None = None,
        rope_scheme: RopeScheme | None = None,
        middle_k_bits: int = 4,
        middle_v_bits: int = 4,
        quant_backend: str | QuantBackend = "int4",
    ) -> None:
        """Tiered KV cache.

        Args:
            cache_dequant: if True (default), keep the dequantized INT4 tier
                between forward passes so we don't pay the dequant cost on every
                step. Faster, but ``storage_bytes`` will include the cached
                dequant buffer (negating the at-rest memory savings until the
                next reassignment). Set to False to prioritize at-rest memory.
            rope_theta: RoPE base frequency (from model config). Ignored if
                ``rope_scheme`` is provided.
            head_dim: attention head dimension. If None, inferred on first
                ``update()`` from the incoming K tensor.
            batch_size: expected batch size. If None, inferred on first
                ``update()`` and pinned; subsequent calls with a different
                batch size raise ``ValueError``.
            rope_scheme: a ``RopeScheme`` instance that governs how cos/sin
                are computed during eviction re-RoPE. If None, a ``Vanilla``
                scheme is built from ``rope_theta`` / ``head_dim``.
            middle_k_bits: quantization bits for K in the middle tier (4 or 2).
            middle_v_bits: quantization bits for V in the middle tier (4 or 2).
                Set ``middle_v_bits=2`` with ``middle_k_bits=4`` for asymmetric
                compression — V tolerates INT2 better than K.
        """
        super().__init__()
        self.n_sink = n_sink
        self.recent_window = recent_window
        self.int4_budget = int4_budget
        self.int2_budget = int2_budget
        self.dtype = dtype
        self.cache_dequant = cache_dequant
        self.rope_theta = rope_theta
        self.head_dim = head_dim
        self.batch_size = batch_size
        self.max_dequant_age = DEFAULT_MAX_DEQUANT_AGE
        self.middle_k_bits = middle_k_bits
        self.middle_v_bits = middle_v_bits
        if isinstance(quant_backend, str):
            self._quant_backend: QuantBackend = get_backend(quant_backend, head_dim=head_dim or 64)
        else:
            self._quant_backend = quant_backend
        self._rope_scheme = rope_scheme  # resolved lazily in _ensure_rope_scheme
        self._pq_codebook_k = None  # PQCodebook for K (set via set_codebooks)
        self._pq_codebook_v = None  # PQCodebook for V
        self._compiled_materialize = None  # populated by enable_compile()
        self._skip_layers: set[int] = set()  # layers with plain passthrough (DeltaNet)
        self._layers: list[LayerState] = []

    def set_skip_layers(self, layer_indices: set[int] | list[int]) -> None:
        """Mark layers as passthrough (no tier management).

        For hybrid models like Qwen 3.5/3.6, DeltaNet (linear attention)
        layers don't produce a standard K/V cache. These layers store K/V
        as plain FP16 and are excluded from tier assignment and re-RoPE.
        """
        self._skip_layers = set(layer_indices)

    @property
    def managed_layers(self) -> set[int]:
        """Layer indices that participate in tier management."""
        return {i for i in range(len(self._layers)) if i not in self._skip_layers}

    def is_managed(self, layer_idx: int) -> bool:
        """Whether a layer participates in tier management."""
        return layer_idx not in self._skip_layers

    def enable_compile(self, **compile_kwargs) -> None:
        """Compile the materialize hot path with ``torch.compile``.

        Call after the first ``update()`` so shapes are known. The compiled
        function is used between tier reassignments; reassignment itself is
        a graph-break boundary and runs eagerly.

        Args:
            **compile_kwargs: forwarded to ``torch.compile`` (e.g.
                ``mode="reduce-overhead"``, ``fullgraph=False``).
        """
        compile_kwargs.setdefault("fullgraph", False)
        self._compiled_materialize = torch.compile(self._materialize_impl, **compile_kwargs)

    def set_codebooks(self, k_codebook, v_codebook=None) -> None:
        """Attach trained PQ codebooks for the TIER_PQ path.

        Args:
            k_codebook: a ``PQCodebook`` for K vectors.
            v_codebook: a ``PQCodebook`` for V vectors. If None, reuses ``k_codebook``.
        """
        self._pq_codebook_k = k_codebook
        self._pq_codebook_v = v_codebook or k_codebook

    def _ensure_rope_scheme(self) -> RopeScheme:
        """Lazily build a RopeScheme when head_dim is first known."""
        if self._rope_scheme is None:
            hd = self.head_dim or 64
            self._rope_scheme = Vanilla(theta=self.rope_theta, head_dim=hd)
        return self._rope_scheme

    @property
    def rope_scheme(self) -> RopeScheme:
        return self._ensure_rope_scheme()

    # ------------------------------------------------------------------ #
    # RoPE helpers (Phase 2)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _rotate_half(x: Tensor) -> Tensor:
        """Rotate the last dim: [-x2, x1]."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    @staticmethod
    def _apply_rope(k: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        """Apply RoPE: ``k_post = k * cos + rotate_half(k) * sin``.

        Computed in float32 to avoid precision loss from float16 arithmetic.
        """
        k_f = k.float()
        cos_f = cos.float()
        sin_f = sin.float()
        return (k_f * cos_f + TieredKVCache._rotate_half(k_f) * sin_f).to(k.dtype)

    @staticmethod
    def _inverse_rope(k: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        """Undo RoPE: ``k_pre = k_post * cos - rotate_half(k_post) * sin``.

        Derivation: cos²+sin²=1 gives the closed-form inverse.
        Computed in float32 to avoid precision loss.
        """
        k_f = k.float()
        cos_f = cos.float()
        sin_f = sin.float()
        return (k_f * cos_f - TieredKVCache._rotate_half(k_f) * sin_f).to(k.dtype)

    def _compute_cos_sin(
        self,
        positions: Tensor,
        device: torch.device,
    ) -> tuple[Tensor, Tensor]:
        """Compute RoPE cos/sin for arbitrary position IDs.

        Delegates to the ``rope_scheme`` so that Llama-3.1 scaling, YaRN,
        NTK-aware, etc. are all handled transparently.

        Args:
            positions: [S] int64 position IDs.
            device: target device.

        Returns:
            (cos, sin) each broadcastable to [1, 1, S, head_dim].
        """
        return self._ensure_rope_scheme().compute_cos_sin(
            positions,
            device,
            model_dtype=self.dtype,
        )

    def _streamingllm_positions(self, total: int, device: torch.device) -> Tensor:
        """Compute StreamingLLM-style contiguous positions.

        Sinks keep positions ``[0, n_sink)``, everything else is numbered
        contiguously from ``n_sink`` onward.
        """
        return torch.arange(total, device=device, dtype=torch.long)

    # ------------------------------------------------------------------ #
    # HF DynamicCache API
    # ------------------------------------------------------------------ #
    def update(
        self,
        key_states: Tensor,
        value_states: Tensor,
        layer_idx: int,
        cache_kwargs: dict | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Append new K/V to the FP16 tier, then return the full cache.

        K is stored as-is (post-RoPE). Re-RoPE only happens inside
        ``apply_tier_assignment`` when eviction changes positions.
        """
        if key_states.dim() != 4:
            raise ValueError(
                f"key_states must be [B, H, S, D]; got shape {tuple(key_states.shape)}"
            )
        if key_states.shape != value_states.shape:
            raise ValueError(
                f"K/V shape mismatch: {tuple(key_states.shape)} vs {tuple(value_states.shape)}"
            )
        batch = int(key_states.shape[0])
        if self.batch_size is None:
            self.batch_size = batch
        elif self.batch_size != batch:
            raise ValueError(
                f"batch size mismatch: cache pinned to B={self.batch_size} but got B={batch}"
            )
        if self.head_dim is None:
            self.head_dim = int(key_states.shape[-1])

        self._ensure_layer(layer_idx)
        self._append_fp16(key_states, value_states, layer_idx)
        return self._materialize(layer_idx)

    def get_seq_length(self, layer_idx: int = 0) -> int:
        if layer_idx >= len(self._layers):
            return 0
        return self._layers[layer_idx].total_seq_length()

    def get_mask_sizes(
        self,
        cache_position: Tensor | int,
        layer_idx: int = 0,
    ) -> tuple[int, int]:
        """Return (kv_length, kv_offset) for the attention mask.

        The base ``Cache`` delegates to ``self.layers[layer_idx]``, which we
        don't populate. Override so the mask sees the real KV length.
        """
        if isinstance(cache_position, int):
            query_length = cache_position
        else:
            query_length = int(cache_position.shape[0])
        kv_length = self.get_seq_length(layer_idx) + query_length
        return kv_length, 0

    def __len__(self) -> int:
        return len(self._layers)

    def storage_bytes(self) -> int:
        """Total bytes currently held for K/V across all tiers (including any
        dequantized cache). Useful for isolating KV-cache memory from model
        weights and activations.
        """
        total = 0
        for state in self._layers:
            for t in (
                state.fp16_k,
                state.fp16_v,
                state.int4_kq,
                state.int4_vq,
                state.int4_ks,
                state.int4_vs,
                state.int4_k_deq,
                state.int4_v_deq,
                state.int2_kq,
                state.int2_vq,
                state.int2_ks,
                state.int2_vs,
                state.int2_k_deq,
                state.int2_v_deq,
            ):
                if t is not None:
                    total += int(t.element_size() * t.numel())
        return total

    def compressed_storage_bytes(self) -> int:
        """Bytes needed for the *essential* (compressed) form only — excludes
        the ephemeral dequantized caches. This is the true at-rest metric.
        """
        total = 0
        for state in self._layers:
            for t in (
                state.fp16_k,
                state.fp16_v,
                state.int4_kq,
                state.int4_vq,
                state.int4_ks,
                state.int4_vs,
                state.int2_kq,
                state.int2_vq,
                state.int2_ks,
                state.int2_vs,
            ):
                if t is not None:
                    total += int(t.element_size() * t.numel())
            # Backend-compressed dicts (TurboQuant etc.).
            for d in (state.backend_k, state.backend_v):
                if isinstance(d, dict):
                    for v in d.values():
                        if isinstance(v, Tensor):
                            total += int(v.element_size() * v.numel())
        return total

    # ------------------------------------------------------------------ #
    # Checkpointing
    # ------------------------------------------------------------------ #
    _LAYER_TENSOR_KEYS = (
        "fp16_k",
        "fp16_v",
        "fp16_pos",
        "int4_kq",
        "int4_ks",
        "int4_vq",
        "int4_vs",
        "int4_pos",
        "int2_kq",
        "int2_ks",
        "int2_vq",
        "int2_vs",
        "int2_pos",
    )
    _LAYER_SCALAR_KEYS = (
        "int2_actual_count",
        "int2_group_size",
        "sink_count",
        "next_position",
    )

    def cache_state_dict(self) -> dict:
        """Serialize the essential compressed form (no dequant caches).

        Returns a plain dict suitable for ``torch.save``. Use
        ``load_cache_state_dict`` to restore.
        """
        sd: dict = {
            "n_layers": len(self._layers),
            "n_sink": self.n_sink,
            "recent_window": self.recent_window,
            "int4_budget": self.int4_budget,
            "int2_budget": self.int2_budget,
            "batch_size": self.batch_size,
            "head_dim": self.head_dim,
            "dtype": str(self.dtype),
        }
        for i, state in enumerate(self._layers):
            prefix = f"layer.{i}."
            for key in self._LAYER_TENSOR_KEYS:
                val = getattr(state, key)
                if val is not None:
                    sd[prefix + key] = val.cpu()
            for key in self._LAYER_SCALAR_KEYS:
                sd[prefix + key] = getattr(state, key)
        return sd

    def load_cache_state_dict(self, sd: dict) -> None:
        """Restore from a dict produced by ``cache_state_dict``.

        Replaces all layer state in-place. The cache config (n_sink, etc.)
        must already match; this method only overwrites the per-layer data.
        """
        n_layers = sd["n_layers"]
        self._layers = []
        for i in range(n_layers):
            prefix = f"layer.{i}."
            state = LayerState()
            for key in self._LAYER_TENSOR_KEYS:
                val = sd.get(prefix + key)
                if val is not None:
                    setattr(state, key, val)
            for key in self._LAYER_SCALAR_KEYS:
                val = sd.get(prefix + key)
                if val is not None:
                    setattr(state, key, val)
            self._layers.append(state)
        if sd.get("batch_size") is not None:
            self.batch_size = sd["batch_size"]
        if sd.get("head_dim") is not None:
            self.head_dim = sd["head_dim"]

    def dump_debug(self, path: str | Path) -> None:
        """Write a JSON snapshot of tier membership and positions per layer.

        Useful for offline analysis of eviction behaviour. Does not include
        tensor data — only metadata, counts, and position arrays.
        """
        layers = []
        for i, state in enumerate(self._layers):
            layer: dict = {
                "layer_idx": i,
                "sink_count": state.sink_count,
                "next_position": state.next_position,
                "fp16_count": int(state.fp16_pos.shape[0]) if state.fp16_pos is not None else 0,
                "int4_count": int(state.int4_pos.shape[0]) if state.int4_pos is not None else 0,
                "int2_count": int(state.int2_pos.shape[0]) if state.int2_pos is not None else 0,
                "int2_actual_count": state.int2_actual_count,
                "total_seq_length": state.total_seq_length(),
            }
            if state.fp16_pos is not None:
                layer["fp16_positions"] = state.fp16_pos.tolist()
            if state.int4_pos is not None:
                layer["int4_positions"] = state.int4_pos.tolist()
            if state.int2_pos is not None:
                layer["int2_positions"] = state.int2_pos.tolist()
            layers.append(layer)

        snapshot = {
            "n_layers": len(self._layers),
            "n_sink": self.n_sink,
            "recent_window": self.recent_window,
            "int4_budget": self.int4_budget,
            "int2_budget": self.int2_budget,
            "batch_size": self.batch_size,
            "head_dim": self.head_dim,
            "layers": layers,
        }
        Path(path).write_text(json.dumps(snapshot, indent=2), encoding="utf-8")

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _ensure_layer(self, layer_idx: int) -> None:
        while len(self._layers) <= layer_idx:
            self._layers.append(LayerState())

    def _append_fp16(self, k: Tensor, v: Tensor, layer_idx: int) -> None:
        """Append new tokens to the FP16 tier with fresh absolute positions.

        Uses a pre-allocated buffer that doubles on overflow, avoiding
        ``torch.cat`` allocator churn on every decode step. ``fp16_k/v/pos``
        are set to views of the buffer up to ``_fp16_len``.
        """
        state = self._layers[layer_idx]
        n = int(k.shape[-2])
        start = state.next_position
        new_pos = torch.arange(start, start + n, device=k.device)
        state.next_position = start + n

        if state._fp16_buf_k is None:
            # Init the buffer. If fp16_k already exists (post-reassignment),
            # seed the buffer with the existing data then append.
            existing = 0
            if state.fp16_k is not None:
                existing = int(state.fp16_k.shape[-2])
            total_needed = existing + n
            cap = max(DEFAULT_FP16_PREALLOC, total_needed)
            B, H, _, D = k.shape
            state._fp16_buf_k = torch.empty(B, H, cap, D, dtype=k.dtype, device=k.device)
            state._fp16_buf_v = torch.empty(B, H, cap, D, dtype=v.dtype, device=v.device)
            state._fp16_buf_pos = torch.empty(cap, dtype=torch.long, device=k.device)
            if existing > 0:
                state._fp16_buf_k[:, :, :existing, :] = state.fp16_k
                state._fp16_buf_v[:, :, :existing, :] = state.fp16_v
                state._fp16_buf_pos[:existing] = state.fp16_pos
            state._fp16_buf_k[:, :, existing:total_needed, :] = k
            state._fp16_buf_v[:, :, existing:total_needed, :] = v
            state._fp16_buf_pos[existing:total_needed] = new_pos
            state._fp16_len = total_needed
        else:
            old_len = state._fp16_len
            new_len = old_len + n
            cap = state._fp16_buf_k.shape[-2]
            if new_len > cap:
                # Double the buffer (amortized O(1) append).
                new_cap = max(cap * 2, new_len)
                B, H, _, D = state._fp16_buf_k.shape
                new_buf_k = torch.empty(B, H, new_cap, D, dtype=k.dtype, device=k.device)
                new_buf_v = torch.empty(B, H, new_cap, D, dtype=v.dtype, device=v.device)
                new_buf_pos = torch.empty(new_cap, dtype=torch.long, device=k.device)
                new_buf_k[:, :, :old_len, :] = state._fp16_buf_k[:, :, :old_len, :]
                new_buf_v[:, :, :old_len, :] = state._fp16_buf_v[:, :, :old_len, :]
                new_buf_pos[:old_len] = state._fp16_buf_pos[:old_len]
                state._fp16_buf_k = new_buf_k
                state._fp16_buf_v = new_buf_v
                state._fp16_buf_pos = new_buf_pos
            state._fp16_buf_k[:, :, old_len:new_len, :] = k
            state._fp16_buf_v[:, :, old_len:new_len, :] = v
            state._fp16_buf_pos[old_len:new_len] = new_pos
            state._fp16_len = new_len

        # Expose views as fp16_k/v/pos for the rest of the cache.
        ln = state._fp16_len
        state.fp16_k = state._fp16_buf_k[:, :, :ln, :]
        state.fp16_v = state._fp16_buf_v[:, :, :ln, :]
        state.fp16_pos = state._fp16_buf_pos[:ln]

        # Track age for dequant-cache eviction.
        state._dequant_age += 1

        # E2: Speculative eviction — pre-compress FP16 tokens that are likely
        # to be evicted at the next reassignment. This amortizes the quant cost
        # across decode steps instead of paying it all at once.
        # Currently a no-op placeholder; activate by setting
        # cache.speculative_precompress = True and providing a score tracker.
        # The actual pre-compression logic would:
        #   1. Check if we're N steps from the next reassignment.
        #   2. Score the bottom-k FP16 tokens by attention mass.
        #   3. Pre-quantize them to INT4 in the background.
        # This is left as infrastructure for future activation.

    def _materialize(self, layer_idx: int) -> tuple[Tensor, Tensor]:
        """Hot path: return K, V in ascending position order.

        Delegates to ``_compiled_materialize`` if ``enable_compile()`` was
        called, otherwise falls through to ``_materialize_impl``.
        """
        if self._compiled_materialize is not None:
            return self._compiled_materialize(layer_idx)
        return self._materialize_impl(layer_idx)

    def _materialize_impl(self, layer_idx: int) -> tuple[Tensor, Tensor]:
        """Actual materialize logic (compilable)."""
        state = self._layers[layer_idx]
        parts_k, parts_v = self._assemble_parts(state, want_positions=False)
        if not parts_k:
            raise RuntimeError(f"No cache state for layer {layer_idx}")
        return torch.cat(parts_k, dim=-2), torch.cat(parts_v, dim=-2)

    def _all_in_position_order(self, layer_idx: int) -> tuple[Tensor, Tensor, Tensor]:
        """Return (K, V, positions) in ascending position order for the tier
        policy. INT4 is dequantized to ``self.dtype``.
        """
        state = self._layers[layer_idx]
        parts_k, parts_v, parts_pos = self._assemble_parts(state, want_positions=True)
        if not parts_k:
            raise RuntimeError(f"No cache state for layer {layer_idx}")
        return (
            torch.cat(parts_k, dim=-2),
            torch.cat(parts_v, dim=-2),
            torch.cat(parts_pos, dim=0),
        )

    def _assemble_parts(
        self,
        state: LayerState,
        want_positions: bool,
    ):
        """Collect the sorted tier segments.

        Segments are emitted in the order
            sink  |  middle (INT4 + INT2 merged by position)  |  recent

        When both INT4 and INT2 middle tokens exist, their positions may be
        interleaved (selected by attention score, not contiguously). We merge
        them with a small argsort bounded by ``int4_budget + int2_budget``.
        """
        parts_k: list[Tensor] = []
        parts_v: list[Tensor] = []
        parts_pos: list[Tensor] = []
        nsk = state.sink_count
        fp16_len = int(state.fp16_k.shape[-2]) if state.fp16_k is not None else 0

        # --- sinks ---
        if state.fp16_k is not None and nsk > 0:
            parts_k.append(state.fp16_k[..., :nsk, :])
            parts_v.append(state.fp16_v[..., :nsk, :])
            if want_positions:
                parts_pos.append(state.fp16_pos[:nsk])

        # --- middle: INT4 + INT2, merged by position ---
        mid_k: list[Tensor] = []
        mid_v: list[Tensor] = []
        mid_pos: list[Tensor] = []

        if state.int4_kq is not None:
            k4, v4 = self._get_int4_dequant(state)
            mid_k.append(k4)
            mid_v.append(v4)
            mid_pos.append(state.int4_pos)

        if state.backend_k is not None:
            k_be = self._quant_backend.decompress_k(state.backend_k, dtype=self.dtype)
            v_be = self._quant_backend.decompress_v(state.backend_v, dtype=self.dtype)
            mid_k.append(k_be)
            mid_v.append(v_be)
            mid_pos.append(state.backend_pos)

        if state.int2_kq is not None:
            k2, v2 = self._get_int2_dequant(state)
            mid_k.append(k2)
            mid_v.append(v2)
            mid_pos.append(state.int2_pos)

        if state.pq_codes is not None:
            kpq, vpq = self._get_pq_dequant(state)
            mid_k.append(kpq)
            mid_v.append(vpq)
            mid_pos.append(state.pq_pos)

        if mid_k:
            if len(mid_k) == 1:
                parts_k.append(mid_k[0])
                parts_v.append(mid_v[0])
                if want_positions:
                    parts_pos.append(mid_pos[0])
            else:
                # Merge two middle segments by position.
                all_k = torch.cat(mid_k, dim=-2)
                all_v = torch.cat(mid_v, dim=-2)
                all_pos = torch.cat(mid_pos, dim=0)
                order = all_pos.argsort()
                parts_k.append(all_k.index_select(-2, order))
                parts_v.append(all_v.index_select(-2, order))
                if want_positions:
                    parts_pos.append(all_pos[order])

        # --- recent ---
        if state.fp16_k is not None and nsk < fp16_len:
            parts_k.append(state.fp16_k[..., nsk:, :])
            parts_v.append(state.fp16_v[..., nsk:, :])
            if want_positions:
                parts_pos.append(state.fp16_pos[nsk:])

        if want_positions:
            return parts_k, parts_v, parts_pos
        return parts_k, parts_v

    def _get_int4_dequant(self, state: LayerState) -> tuple[Tensor, Tensor]:
        """Return (K, V) dequantized from the INT4 tier.

        Handles asymmetric K/V: K is always INT4-packed; V may be INT2.
        Caches the FP16 result between reassignments when ``self.cache_dequant``.
        Drops the cached dequant when ``max_dequant_age`` is exceeded.
        """
        # Age-based dequant eviction: drop if stale.
        if (
            self.max_dequant_age is not None
            and state._dequant_age > self.max_dequant_age
            and state.int4_k_deq is not None
        ):
            state.int4_k_deq = None
            state.int4_v_deq = None
        if self.cache_dequant and state.int4_k_deq is not None:
            return state.int4_k_deq, state.int4_v_deq
        k_deq = dequant_int4(state.int4_kq, state.int4_ks, dtype=self.dtype)
        if self.middle_v_bits == 4:
            v_deq = dequant_int4(state.int4_vq, state.int4_vs, dtype=self.dtype)
        else:
            # V was quantized at INT2 (stored in int4_vq/vs slots).
            gs = DEFAULT_INT2_GROUP_SIZE
            v_deq = dequant_v_int2(state.int4_vq, state.int4_vs, group_size=gs, dtype=self.dtype)
            # Trim padding if needed.
            n_int4 = int(state.int4_pos.shape[0]) if state.int4_pos is not None else 0
            if n_int4 > 0 and n_int4 < v_deq.shape[-2]:
                v_deq = v_deq[..., :n_int4, :]
        if self.cache_dequant:
            state.int4_k_deq = k_deq
            state.int4_v_deq = v_deq
            state._dequant_age = 0  # reset age on fresh population
        return k_deq, v_deq

    def _get_pq_dequant(self, state: LayerState) -> tuple[Tensor, Tensor]:
        """Return (K, V) decoded from PQ codes."""
        if self.cache_dequant and state.pq_k_deq is not None:
            return state.pq_k_deq, state.pq_v_deq
        if self._pq_codebook_k is None:
            raise RuntimeError(
                "TIER_PQ tokens exist but no codebook is attached. "
                "Call cache.set_codebooks(k_cb, v_cb) before tier assignment."
            )
        # codes: [B, H, S_pq, n_sub] -> decode to [B, H, S_pq, head_dim]
        k_deq = self._pq_codebook_k.decode(state.pq_codes).to(self.dtype)
        v_deq = self._pq_codebook_v.decode(state.pq_v_codes).to(self.dtype)
        if self.cache_dequant:
            state.pq_k_deq = k_deq
            state.pq_v_deq = v_deq
        return k_deq, v_deq

    def _get_int2_dequant(self, state: LayerState) -> tuple[Tensor, Tensor]:
        """Return (K, V) dequantized from the INT2 tier, trimmed to actual count."""
        if self.cache_dequant and state.int2_k_deq is not None:
            return state.int2_k_deq, state.int2_v_deq
        gs = state.int2_group_size
        k_deq = dequant_k_int2(state.int2_kq, state.int2_ks, group_size=gs, dtype=self.dtype)
        v_deq = dequant_v_int2(state.int2_vq, state.int2_vs, group_size=gs, dtype=self.dtype)
        # Trim padding added by pad_to_group.
        n = state.int2_actual_count
        if n > 0 and n < k_deq.shape[-2]:
            k_deq = k_deq[..., :n, :]
            v_deq = v_deq[..., :n, :]
        if self.cache_dequant:
            state.int2_k_deq = k_deq
            state.int2_v_deq = v_deq
        return k_deq, v_deq

    # ------------------------------------------------------------------ #
    # Tier management
    # ------------------------------------------------------------------ #
    def apply_tier_assignment(
        self,
        layer_idx: int,
        tiers: Tensor,
        scores: Tensor | None = None,
    ) -> None:
        """Move tokens between tiers according to ``tiers``.

        Args:
            layer_idx: transformer layer index.
            tiers: LongTensor [S_total] aligned with ascending absolute-position order,
                with values in {TIER_FP16, TIER_INT4, TIER_INT2, TIER_EVICT}.
            scores: optional attention scores, currently only used for telemetry.

        When any tokens are evicted (``TIER_EVICT``), positions become
        non-contiguous. We undo RoPE at the old positions and re-apply with
        StreamingLLM-style contiguous positions so relative distances stay
        meaningful. ``next_position`` is reset to the retained count so
        subsequent tokens receive correct contiguous positions.
        """
        del scores  # telemetry hook only; not needed for the transformation
        state = self._layers[layer_idx]
        k_full, v_full, pos_full = self._all_in_position_order(layer_idx)
        S = int(pos_full.shape[0])
        if int(tiers.shape[0]) != S:
            raise ValueError(
                f"tiers length {int(tiers.shape[0])} != cache length {S} for layer {layer_idx}"
            )
        tiers = tiers.to(pos_full.device)

        keep_mask = tiers != TIER_EVICT
        n_kept = int(keep_mask.sum().item())  # single host sync

        if n_kept < S:
            new_pos = self._streamingllm_positions(n_kept, pos_full.device)
            # Fuse old + new cos/sin into a single batch to halve kernel launches.
            all_pos = torch.cat([pos_full, new_pos])
            all_cos, all_sin = self._compute_cos_sin(all_pos, pos_full.device)
            old_cos = all_cos[..., :S, :]
            old_sin = all_sin[..., :S, :]
            new_cos = all_cos[..., S:, :]
            new_sin = all_sin[..., S:, :]
            k_pre_rope = self._inverse_rope(k_full, old_cos, old_sin)
            # Build a mapping: for each kept token, what is its index in the
            # dense [0..n_kept) range? We keep the original ordering.
            keep_indices = keep_mask.nonzero(as_tuple=False).squeeze(-1)
            k_pre_kept = k_pre_rope.index_select(-2, keep_indices).contiguous()
            v_kept = v_full.index_select(-2, keep_indices).contiguous()
            k_full = self._apply_rope(k_pre_kept, new_cos, new_sin)
            v_full = v_kept
            # Re-label tiers to exclude evicted entries.
            tiers = tiers[keep_mask]
            pos_full = new_pos
            # Reset next_position so the model assigns contiguous positions.
            state.next_position = n_kept

        fp16_idx = (tiers == TIER_FP16).nonzero(as_tuple=False).squeeze(-1)
        if fp16_idx.numel() > 0:
            state.fp16_k = k_full.index_select(-2, fp16_idx).contiguous()
            state.fp16_v = v_full.index_select(-2, fp16_idx).contiguous()
            state.fp16_pos = pos_full.index_select(0, fp16_idx).contiguous()
        else:
            state.fp16_k = state.fp16_v = state.fp16_pos = None
        # Reseed the pre-alloc buffer to avoid reallocation on the next append.
        n_fp16 = int(state.fp16_k.shape[-2]) if state.fp16_k is not None else 0
        if n_fp16 > 0 and state._fp16_buf_k is not None and n_fp16 <= state._fp16_buf_k.shape[-2]:
            state._fp16_buf_k[:, :, :n_fp16, :] = state.fp16_k
            state._fp16_buf_v[:, :, :n_fp16, :] = state.fp16_v
            state._fp16_buf_pos[:n_fp16] = state.fp16_pos
            state._fp16_len = n_fp16
            # Expose views so subsequent reads go through the buffer.
            state.fp16_k = state._fp16_buf_k[:, :, :n_fp16, :]
            state.fp16_v = state._fp16_buf_v[:, :, :n_fp16, :]
            state.fp16_pos = state._fp16_buf_pos[:n_fp16]
        else:
            # Buffer too small or absent — let _append_fp16 re-init.
            state._fp16_buf_k = None
            state._fp16_buf_v = None
            state._fp16_buf_pos = None
            state._fp16_len = 0

        int4_idx = (tiers == TIER_INT4).nonzero(as_tuple=False).squeeze(-1)
        if int4_idx.numel() > 0:
            k_sub = k_full.index_select(-2, int4_idx).contiguous()
            v_sub = v_full.index_select(-2, int4_idx).contiguous()
            if isinstance(self._quant_backend, SymmetricINT4Backend):
                # Legacy path: direct INT4 quant.
                state.int4_kq, state.int4_ks = quant_k_int4(k_sub)
                if self.middle_v_bits == 4:
                    state.int4_vq, state.int4_vs = quant_v_int4(v_sub)
                else:
                    gs = DEFAULT_INT2_GROUP_SIZE
                    v_padded, _ = pad_to_group(v_sub, gs)
                    state.int4_vq, state.int4_vs = quant_v_int2(v_padded, group_size=gs)
                state.int4_pos = pos_full.index_select(0, int4_idx).contiguous()
                state.backend_k = state.backend_v = None
                state.backend_pos = None
            else:
                # Pluggable backend (e.g. TurboQuant).
                state.backend_k = self._quant_backend.compress_k(k_sub)
                state.backend_v = self._quant_backend.compress_v(v_sub)
                state.backend_pos = pos_full.index_select(0, int4_idx).contiguous()
                state.int4_kq = state.int4_vq = None
                state.int4_ks = state.int4_vs = None
                state.int4_pos = None
        else:
            state.int4_kq = state.int4_vq = None
            state.int4_ks = state.int4_vs = None
            state.int4_pos = None
            state.backend_k = state.backend_v = None
            state.backend_pos = None

        int2_idx = (tiers == TIER_INT2).nonzero(as_tuple=False).squeeze(-1)
        if int2_idx.numel() > 0:
            k_sub = k_full.index_select(-2, int2_idx).contiguous()
            v_sub = v_full.index_select(-2, int2_idx).contiguous()
            gs = DEFAULT_INT2_GROUP_SIZE
            actual = int(int2_idx.numel())
            k_sub, _ = pad_to_group(k_sub, gs)
            v_sub, _ = pad_to_group(v_sub, gs)
            state.int2_kq, state.int2_ks = quant_k_int2(k_sub, group_size=gs)
            state.int2_vq, state.int2_vs = quant_v_int2(v_sub, group_size=gs)
            state.int2_pos = pos_full.index_select(0, int2_idx).contiguous()
            state.int2_actual_count = actual
            state.int2_group_size = gs
        else:
            state.int2_kq = state.int2_vq = None
            state.int2_ks = state.int2_vs = None
            state.int2_pos = None
            state.int2_actual_count = 0

        pq_idx = (tiers == TIER_PQ).nonzero(as_tuple=False).squeeze(-1)
        if pq_idx.numel() > 0:
            if self._pq_codebook_k is None:
                raise RuntimeError(
                    "TIER_PQ assigned but no codebook attached. "
                    "Call cache.set_codebooks(k_cb, v_cb) first."
                )
            k_sub = k_full.index_select(-2, pq_idx).contiguous()
            v_sub = v_full.index_select(-2, pq_idx).contiguous()
            state.pq_codes = self._pq_codebook_k.encode(k_sub)
            state.pq_v_codes = self._pq_codebook_v.encode(v_sub)
            state.pq_pos = pos_full.index_select(0, pq_idx).contiguous()
        else:
            state.pq_codes = state.pq_v_codes = None
            state.pq_pos = None

        # Invalidate ephemeral caches; ``_materialize`` rebuilds them lazily.
        state.int4_k_deq = None
        state.int4_v_deq = None
        state.int2_k_deq = None
        state.int2_v_deq = None
        state.pq_k_deq = None
        state.pq_v_deq = None

        # Sinks are positions 0..n_sink-1. Since the policy always places them
        # in the FP16 tier and ``fp16_pos`` is sorted ascending, they form the
        # leading prefix.
        if state.fp16_pos is not None:
            state.sink_count = min(self.n_sink, int(state.fp16_pos.shape[0]))
        else:
            state.sink_count = 0

    def apply_tier_assignment_per_sequence(
        self,
        layer_idx: int,
        tiers_per_row: Tensor,
        scores: Tensor | None = None,
    ) -> None:
        """Per-sequence tier assignment for ragged batching.

        Args:
            layer_idx: transformer layer index.
            tiers_per_row: LongTensor ``[B, S]`` with per-row tier values.
            scores: optional ``[B, S]`` or ``[S]`` scores for telemetry.

        Each row is processed independently. After eviction, rows may have
        different surviving counts. The result is padded to the max count
        across rows so K/V remain rectangular ``[B, H, S_max, D]``.

        This is slower than the shared-tier path (Python loop over B) but
        correct for continuous-batching scenarios.
        """
        state = self._layers[layer_idx]
        k_full, v_full, pos_full = self._all_in_position_order(layer_idx)
        B = int(k_full.shape[0])
        H = int(k_full.shape[1])
        D = int(k_full.shape[3])
        S = int(pos_full.shape[0])

        if tiers_per_row.dim() != 2 or tiers_per_row.shape[0] != B:
            raise ValueError(
                f"tiers_per_row must be [B={B}, S={S}], got {tuple(tiers_per_row.shape)}"
            )
        if tiers_per_row.shape[1] != S:
            raise ValueError(f"tiers_per_row S dim {tiers_per_row.shape[1]} != cache length {S}")

        device = k_full.device
        tiers_per_row = tiers_per_row.to(device)

        # Process each row independently, collect surviving K/V per tier.
        fp16_k_rows, fp16_v_rows, fp16_pos_rows = [], [], []
        int4_k_rows, int4_v_rows, int4_pos_rows = [], [], []

        min_next_pos = S  # track the minimum across rows for next_position
        for b in range(B):
            row_tiers = tiers_per_row[b]  # [S]
            row_k = k_full[b : b + 1]  # [1, H, S, D]
            row_v = v_full[b : b + 1]

            keep_mask = row_tiers != TIER_EVICT
            n_kept = int(keep_mask.sum().item())  # single host sync
            if n_kept < int(row_tiers.shape[0]):
                old_cos, old_sin = self._compute_cos_sin(pos_full, device)
                k_pre = self._inverse_rope(row_k, old_cos, old_sin)
                keep_idx = keep_mask.nonzero(as_tuple=False).squeeze(-1)
                k_pre = k_pre.index_select(-2, keep_idx).contiguous()
                row_v = row_v.index_select(-2, keep_idx).contiguous()
                new_pos = self._streamingllm_positions(n_kept, device)
                new_cos, new_sin = self._compute_cos_sin(new_pos, device)
                row_k = self._apply_rope(k_pre, new_cos, new_sin)
                row_tiers = row_tiers[keep_mask]
                row_pos = new_pos
                min_next_pos = min(min_next_pos, n_kept)
            else:
                row_pos = pos_full

            # Split into tiers.
            fp16_idx = (row_tiers == TIER_FP16).nonzero(as_tuple=False).squeeze(-1)
            if fp16_idx.numel() > 0:
                fp16_k_rows.append(row_k.index_select(-2, fp16_idx))
                fp16_v_rows.append(row_v.index_select(-2, fp16_idx))
                fp16_pos_rows.append(row_pos[fp16_idx])
            else:
                fp16_k_rows.append(torch.zeros(1, H, 0, D, dtype=row_k.dtype, device=device))
                fp16_v_rows.append(torch.zeros(1, H, 0, D, dtype=row_v.dtype, device=device))
                fp16_pos_rows.append(torch.zeros(0, dtype=torch.long, device=device))

            int4_idx = (row_tiers == TIER_INT4).nonzero(as_tuple=False).squeeze(-1)
            if int4_idx.numel() > 0:
                int4_k_rows.append(row_k.index_select(-2, int4_idx))
                int4_v_rows.append(row_v.index_select(-2, int4_idx))
                int4_pos_rows.append(row_pos[int4_idx])
            else:
                int4_k_rows.append(torch.zeros(1, H, 0, D, dtype=row_k.dtype, device=device))
                int4_v_rows.append(torch.zeros(1, H, 0, D, dtype=row_v.dtype, device=device))
                int4_pos_rows.append(torch.zeros(0, dtype=torch.long, device=device))

        # Pad to max count across rows and stack.
        def _pad_and_stack(rows_list, dim=-2):
            max_len = max(r.shape[dim] for r in rows_list)
            if max_len == 0:
                return None
            padded = []
            for r in rows_list:
                gap = max_len - r.shape[dim]
                if gap > 0:
                    pad_shape = list(r.shape)
                    pad_shape[dim] = gap
                    r = torch.cat(
                        [r, torch.zeros(pad_shape, dtype=r.dtype, device=r.device)], dim=dim
                    )
                padded.append(r)
            return torch.cat(padded, dim=0)  # [B, H, max_len, D]

        def _pad_and_stack_1d(rows_list):
            max_len = max(r.shape[0] for r in rows_list)
            if max_len == 0:
                return None
            # Use first row's positions as representative (shared-tier compat).
            return rows_list[0][:max_len] if len(rows_list) > 0 else None

        # FP16 tier.
        fp16_k = _pad_and_stack(fp16_k_rows)
        if fp16_k is not None:
            state.fp16_k = fp16_k
            state.fp16_v = _pad_and_stack(fp16_v_rows)
            state.fp16_pos = _pad_and_stack_1d(fp16_pos_rows)
        else:
            state.fp16_k = state.fp16_v = state.fp16_pos = None

        # Reseed the pre-alloc buffer to avoid reallocation on the next append.
        n_fp16 = int(state.fp16_k.shape[-2]) if state.fp16_k is not None else 0
        if n_fp16 > 0 and state._fp16_buf_k is not None and n_fp16 <= state._fp16_buf_k.shape[-2]:
            state._fp16_buf_k[:, :, :n_fp16, :] = state.fp16_k
            state._fp16_buf_v[:, :, :n_fp16, :] = state.fp16_v
            state._fp16_buf_pos[:n_fp16] = state.fp16_pos
            state._fp16_len = n_fp16
            state.fp16_k = state._fp16_buf_k[:, :, :n_fp16, :]
            state.fp16_v = state._fp16_buf_v[:, :, :n_fp16, :]
            state.fp16_pos = state._fp16_buf_pos[:n_fp16]
        else:
            state._fp16_buf_k = None
            state._fp16_buf_v = None
            state._fp16_buf_pos = None
            state._fp16_len = 0

        # INT4 tier.
        int4_k = _pad_and_stack(int4_k_rows)
        if int4_k is not None:
            state.int4_kq, state.int4_ks = quant_k_int4(int4_k)
            int4_v = _pad_and_stack(int4_v_rows)
            if self.middle_v_bits == 4:
                state.int4_vq, state.int4_vs = quant_v_int4(int4_v)
            else:
                gs = DEFAULT_INT2_GROUP_SIZE
                v_padded, _ = pad_to_group(int4_v, gs)
                state.int4_vq, state.int4_vs = quant_v_int2(v_padded, group_size=gs)
            state.int4_pos = _pad_and_stack_1d(int4_pos_rows)
        else:
            state.int4_kq = state.int4_vq = None
            state.int4_ks = state.int4_vs = None
            state.int4_pos = None

        # INT2 / PQ tiers not handled per-sequence yet (use shared path).
        state.int2_kq = state.int2_vq = None
        state.int2_ks = state.int2_vs = None
        state.int2_pos = None
        state.int2_actual_count = 0
        state.pq_codes = state.pq_v_codes = None
        state.pq_pos = None

        # Invalidate ephemeral caches.
        state.int4_k_deq = None
        state.int4_v_deq = None
        state.int2_k_deq = None
        state.int2_v_deq = None
        state.pq_k_deq = None
        state.pq_v_deq = None

        state.next_position = min_next_pos
        if state.fp16_pos is not None:
            state.sink_count = min(self.n_sink, int(state.fp16_pos.shape[0]))
        else:
            state.sink_count = 0
