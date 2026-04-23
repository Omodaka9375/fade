"""RoPE scheme abstraction for the re-RoPE eviction path.

Each scheme knows how to compute ``inv_freq`` given a ``head_dim`` and
``rope_theta``. The cache delegates to the scheme when it needs cos/sin
for arbitrary position IDs (during eviction re-RoPE). This keeps the
scheme-specific math out of ``cache.py``.

Supported schemes:
    * ``Vanilla``       — standard RoPE (Qwen, Mistral, Phi-3, most models).
    * ``LinearScaled``  — ``factor``-divided positions (CodeLlama, etc.).
    * ``Llama3``        — frequency-dependent linear interpolation with
                          ``low_freq_factor`` / ``high_freq_factor`` (Llama-3.1).
    * ``NtkAware``      — base-frequency scaling (CodeLlama NTK-aware).
    * ``Yarn``          — inv-freq from YaRN (Yet another RoPE extensioN).
    * ``NoRope``        — sentinel for non-RoPE models (ALiBi, absolute, none).
                          ``compute_cos_sin`` returns ones/zeros so that
                          ``_apply_rope`` / ``_inverse_rope`` are identity.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

# --- knobs ------------------------------------------------------------------ #
DEFAULT_ROPE_THETA: float = 10000.0
DEFAULT_FACTOR: float = 1.0
DEFAULT_LOW_FREQ_FACTOR: float = 1.0
DEFAULT_HIGH_FREQ_FACTOR: float = 4.0
DEFAULT_ORIGINAL_MAX_POS: int = 8192
DEFAULT_YARN_BETA_FAST: float = 32.0
DEFAULT_YARN_BETA_SLOW: float = 1.0


def _vanilla_inv_freq(
    head_dim: int,
    theta: float,
    device: torch.device,
) -> Tensor:
    """Standard RoPE inverse frequencies."""
    return 1.0 / (
        theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim)
    )


@dataclass
class RopeScheme:
    """Base RoPE scheme. Subclass to override ``inv_freq``."""

    theta: float = DEFAULT_ROPE_THETA
    head_dim: int = 64

    def inv_freq(self, device: torch.device) -> Tensor:
        """Return ``[head_dim // 2]`` inverse-frequency tensor."""
        return _vanilla_inv_freq(self.head_dim, self.theta, device)

    def compute_cos_sin(
        self,
        positions: Tensor,
        device: torch.device,
        model_dtype: torch.dtype = torch.float16,
    ) -> tuple[Tensor, Tensor]:
        """Compute ``(cos, sin)`` broadcastable to ``[1, 1, S, head_dim]``.

        Mimics the model's RotaryEmbedding by casting through ``model_dtype``
        so the inverse exactly cancels the model's RoPE.
        """
        inv = self.inv_freq(device)
        freqs = positions.float().unsqueeze(-1) * inv.unsqueeze(0)  # [S, hd//2]
        emb = torch.cat((freqs, freqs), dim=-1)  # [S, hd]
        cos = emb.cos().to(model_dtype).float().unsqueeze(0).unsqueeze(0)
        sin = emb.sin().to(model_dtype).float().unsqueeze(0).unsqueeze(0)
        return cos, sin

    @property
    def is_rope(self) -> bool:
        """False only for the ``NoRope`` sentinel."""
        return True


@dataclass
class Vanilla(RopeScheme):
    """Standard RoPE. Used by Qwen2, Mistral, Phi-3, Gemma-2, etc."""


@dataclass
class LinearScaled(RopeScheme):
    """Linearly-scaled positions: ``pos / factor``."""

    factor: float = DEFAULT_FACTOR

    def compute_cos_sin(
        self,
        positions: Tensor,
        device: torch.device,
        model_dtype: torch.dtype = torch.float16,
    ) -> tuple[Tensor, Tensor]:
        scaled = positions.float() / self.factor
        inv = self.inv_freq(device)
        freqs = scaled.unsqueeze(-1) * inv.unsqueeze(0)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().to(model_dtype).float().unsqueeze(0).unsqueeze(0)
        sin = emb.sin().to(model_dtype).float().unsqueeze(0).unsqueeze(0)
        return cos, sin


@dataclass
class Llama3(RopeScheme):
    """Llama-3.1 frequency-dependent linear interpolation.

    Frequencies are partitioned into three bands by comparing wavelength to
    ``original_max_position_embeddings``. Low-frequency dims are interpolated
    by ``factor``, high-frequency dims are kept unchanged, and mid-range dims
    are linearly blended.
    """

    factor: float = DEFAULT_FACTOR
    low_freq_factor: float = DEFAULT_LOW_FREQ_FACTOR
    high_freq_factor: float = DEFAULT_HIGH_FREQ_FACTOR
    original_max_position_embeddings: int = DEFAULT_ORIGINAL_MAX_POS

    def inv_freq(self, device: torch.device) -> Tensor:
        base_inv = _vanilla_inv_freq(self.head_dim, self.theta, device)
        old_ctx = float(self.original_max_position_embeddings)
        low_len = old_ctx / self.low_freq_factor
        high_len = old_ctx / self.high_freq_factor

        new_inv = torch.empty_like(base_inv)
        for i in range(base_inv.shape[0]):
            freq = base_inv[i].item()
            wavelength = 2.0 * math.pi / freq
            if wavelength < high_len:
                new_inv[i] = freq  # high-frequency: keep
            elif wavelength > low_len:
                new_inv[i] = freq / self.factor  # low-frequency: interpolate
            else:
                # Mid-range: smooth linear blend.
                t = (old_ctx / wavelength - self.low_freq_factor) / (
                    self.high_freq_factor - self.low_freq_factor
                )
                new_inv[i] = freq * (1 - t) / self.factor + freq * t
        return new_inv


@dataclass
class NtkAware(RopeScheme):
    """NTK-aware scaled RoPE: theta is scaled by ``factor`` before inv_freq."""

    factor: float = DEFAULT_FACTOR

    def inv_freq(self, device: torch.device) -> Tensor:
        scaled_theta = self.theta * self.factor
        return _vanilla_inv_freq(self.head_dim, scaled_theta, device)


@dataclass
class Yarn(RopeScheme):
    """YaRN (Yet another RoPE extensioN).

    Per-dimension frequency scaling based on wavelength, blending between
    NTK-aware and linear interpolation using a ramp function.
    """

    factor: float = DEFAULT_FACTOR
    beta_fast: float = DEFAULT_YARN_BETA_FAST
    beta_slow: float = DEFAULT_YARN_BETA_SLOW
    original_max_position_embeddings: int = DEFAULT_ORIGINAL_MAX_POS

    def _yarn_ramp(self, dim_idx: float) -> float:
        """Linear ramp between beta_fast and beta_slow."""
        low = self.beta_fast * self.head_dim / (2 * math.pi * self.original_max_position_embeddings)
        high = self.beta_slow * self.head_dim / (2 * math.pi * self.original_max_position_embeddings)
        if dim_idx < low:
            return 0.0
        if dim_idx > high:
            return 1.0
        return (dim_idx - low) / (high - low)

    def inv_freq(self, device: torch.device) -> Tensor:
        base_inv = _vanilla_inv_freq(self.head_dim, self.theta, device)
        new_inv = torch.empty_like(base_inv)
        for i in range(base_inv.shape[0]):
            ramp_val = self._yarn_ramp(float(i))
            # Blend between original and NTK-scaled.
            ntk_inv = _vanilla_inv_freq(self.head_dim, self.theta * self.factor, device)[i]
            new_inv[i] = (1.0 - ramp_val) * base_inv[i] / self.factor + ramp_val * ntk_inv
        return new_inv


@dataclass
class Proportional(RopeScheme):
    """Gemma 4 proportional RoPE with partial rotation.

    Computes inv_freq using ``head_dim`` as the denominator (not
    ``rotary_dim``), then zero-pads the non-rotated dimensions so that
    ``_apply_rope`` / ``_inverse_rope`` act as identity on those dims.

    When ``partial_rotary_factor=1.0`` this is equivalent to vanilla RoPE
    with head_dim-scaled frequencies.
    """

    partial_rotary_factor: float = 1.0

    @property
    def rotary_dim(self) -> int:
        return int(self.head_dim * self.partial_rotary_factor)

    def inv_freq(self, device: torch.device) -> Tensor:
        # Rotated dims: use head_dim as denominator (not rotary_dim).
        rope_angles = self.rotary_dim // 2
        freq_exponents = (
            torch.arange(0, 2 * rope_angles, 2, dtype=torch.float32, device=device)
            / self.head_dim
        )
        inv = 1.0 / (self.theta ** freq_exponents)
        # Zero-pad for non-rotated dims (cos=1, sin=0 → identity rotation).
        nope_angles = (self.head_dim // 2) - rope_angles
        if nope_angles > 0:
            inv = torch.cat([inv, torch.zeros(nope_angles, dtype=torch.float32, device=device)])
        return inv


@dataclass
class NoRope(RopeScheme):
    """Sentinel for non-RoPE models (ALiBi, absolute positional embeddings).

    Returns ones/zeros so ``_apply_rope`` / ``_inverse_rope`` are identity ops.
    The cache still tracks positions for eviction ordering, but no RoPE math
    is applied during re-RoPE.
    """

    def compute_cos_sin(
        self,
        positions: Tensor,
        device: torch.device,
        model_dtype: torch.dtype = torch.float16,
    ) -> tuple[Tensor, Tensor]:
        S = positions.shape[0]
        hd = self.head_dim
        cos = torch.ones(1, 1, S, hd, dtype=torch.float32, device=device)
        sin = torch.zeros(1, 1, S, hd, dtype=torch.float32, device=device)
        return cos, sin

    @property
    def is_rope(self) -> bool:
        return False


def _scheme_from_rope_params(
    rp: dict,
    head_dim: int,
    fallback_theta: float = DEFAULT_ROPE_THETA,
) -> RopeScheme:
    """Build a RopeScheme from a single rope_parameters dict."""
    theta = float(rp.get("rope_theta", fallback_theta))
    rope_type = rp.get("rope_type", rp.get("type", "default")).lower()
    prf = float(rp.get("partial_rotary_factor", 1.0))
    if rope_type == "proportional":
        return Proportional(theta=theta, head_dim=head_dim, partial_rotary_factor=prf)
    if rope_type in ("", "default"):
        if prf < 1.0:
            # Default type with partial rotation → proportional.
            return Proportional(theta=theta, head_dim=head_dim, partial_rotary_factor=prf)
        return Vanilla(theta=theta, head_dim=head_dim)
    # Delegate to the main extract logic for other types.
    import types
    mock = types.SimpleNamespace(
        rope_theta=theta,
        rope_scaling={"type": rope_type, **{k: v for k, v in rp.items() if k not in ("rope_theta", "rope_type", "type")}},
        hidden_size=head_dim,
        num_attention_heads=1,
    )
    return extract_rope_scheme(mock, head_dim=head_dim)


def extract_rope_schemes_per_layer(
    cfg,
    head_dim: int | None = None,
) -> dict[str, RopeScheme] | None:
    """For models with per-layer-type RoPE (e.g. Gemma 4), return a dict.

    Returns ``{"sliding_attention": scheme, "full_attention": scheme}`` or
    ``None`` if the config uses a single global scheme.
    """
    rp = getattr(cfg, "rope_parameters", None)
    if not isinstance(rp, dict):
        return None
    # Check if it's a nested dict (per-layer-type) vs a flat dict (global).
    first_val = next(iter(rp.values()), None)
    if not isinstance(first_val, dict):
        return None  # flat dict — global scheme

    if head_dim is None:
        head_dim = getattr(cfg, "head_dim", None)
        if head_dim is None:
            head_dim = getattr(cfg, "hidden_size", 64) // getattr(cfg, "num_attention_heads", 1)

    fallback_theta = float(getattr(cfg, "rope_theta", DEFAULT_ROPE_THETA))
    return {
        layer_type: _scheme_from_rope_params(params, head_dim, fallback_theta)
        for layer_type, params in rp.items()
        if isinstance(params, dict)
    }


def extract_rope_scheme(cfg, head_dim: int | None = None) -> RopeScheme:
    """Auto-detect the RoPE scheme from a HuggingFace model config.

    Inspects ``config.rope_scaling`` and ``config.rope_theta`` (or the
    transformers-5.x ``config.rope_parameters`` layout).

    For models with per-layer-type RoPE (Gemma 4), returns the
    ``full_attention`` scheme as the default (it's the one used for the
    global KV cache that FADE compresses).
    """
    # --- check for per-layer-type schemes (Gemma 4) ---
    per_layer = extract_rope_schemes_per_layer(cfg, head_dim)
    if per_layer is not None:
        # Prefer full_attention scheme; fall back to first available.
        return per_layer.get("full_attention", next(iter(per_layer.values())))

    # --- resolve theta ---
    theta = getattr(cfg, "rope_theta", None)
    rp = getattr(cfg, "rope_parameters", None)
    if theta is None and isinstance(rp, dict):
        theta = rp.get("rope_theta")
    if theta is None:
        theta = DEFAULT_ROPE_THETA
    theta = float(theta)

    # --- resolve head_dim ---
    if head_dim is None:
        head_dim = getattr(cfg, "head_dim", None)
        if head_dim is None:
            head_dim = getattr(cfg, "hidden_size", 64) // getattr(cfg, "num_attention_heads", 1)

    # --- detect non-RoPE models ---
    # Falcon (ALiBi) sets alibi=True or uses the FalconConfig with no rope_theta.
    if getattr(cfg, "alibi", False):
        return NoRope(theta=theta, head_dim=head_dim)
    model_type = getattr(cfg, "model_type", "")
    if model_type in ("bloom", "mpt"):
        return NoRope(theta=theta, head_dim=head_dim)

    # --- inspect rope_scaling dict ---
    rope_scaling = getattr(cfg, "rope_scaling", None)
    if rope_scaling is None:
        return Vanilla(theta=theta, head_dim=head_dim)

    if not isinstance(rope_scaling, dict):
        return Vanilla(theta=theta, head_dim=head_dim)

    scale_type = rope_scaling.get("type", rope_scaling.get("rope_type", "")).lower()
    # Transformers 5.x may set type="default" to mean "vanilla, no scaling".
    if scale_type in ("", "default"):
        return Vanilla(theta=theta, head_dim=head_dim)
    # Gemma 4 proportional RoPE.
    if scale_type == "proportional":
        prf = float(rope_scaling.get("partial_rotary_factor", 1.0))
        return Proportional(theta=theta, head_dim=head_dim, partial_rotary_factor=prf)
    factor = float(rope_scaling.get("factor", 1.0))
    orig_max = int(rope_scaling.get(
        "original_max_position_embeddings",
        getattr(cfg, "original_max_position_embeddings", DEFAULT_ORIGINAL_MAX_POS),
    ))

    if scale_type == "linear":
        return LinearScaled(theta=theta, head_dim=head_dim, factor=factor)
    if scale_type in ("llama3", "longrope"):
        return Llama3(
            theta=theta,
            head_dim=head_dim,
            factor=factor,
            low_freq_factor=float(rope_scaling.get("low_freq_factor", DEFAULT_LOW_FREQ_FACTOR)),
            high_freq_factor=float(rope_scaling.get("high_freq_factor", DEFAULT_HIGH_FREQ_FACTOR)),
            original_max_position_embeddings=orig_max,
        )
    if scale_type in ("ntk", "dynamic"):
        return NtkAware(theta=theta, head_dim=head_dim, factor=factor)
    if scale_type == "yarn":
        return Yarn(
            theta=theta,
            head_dim=head_dim,
            factor=factor,
            beta_fast=float(rope_scaling.get("beta_fast", DEFAULT_YARN_BETA_FAST)),
            beta_slow=float(rope_scaling.get("beta_slow", DEFAULT_YARN_BETA_SLOW)),
            original_max_position_embeddings=orig_max,
        )

    # Unknown scaling type — fall back to vanilla + a warning.
    import warnings
    warnings.warn(
        f"Unknown rope_scaling type {scale_type!r}; falling back to vanilla RoPE.",
        RuntimeWarning,
        stacklevel=2,
    )
    return Vanilla(theta=theta, head_dim=head_dim)


__all__ = [
    "LinearScaled",
    "Llama3",
    "NoRope",
    "NtkAware",
    "Proportional",
    "RopeScheme",
    "Vanilla",
    "Yarn",
    "extract_rope_scheme",
    "extract_rope_schemes_per_layer",
]
