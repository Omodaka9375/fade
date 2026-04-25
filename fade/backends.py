"""Pluggable quantization backends for the tiered KV cache.

The ``QuantBackend`` protocol defines how K/V tensors are compressed and
decompressed. FADE ships two backends:

    * ``SymmetricINT4Backend`` — the original per-channel K / per-token V
      symmetric INT4 quantization with bit-packing. Default.
    * ``TurboQuantBackend`` — wraps ``turboquant-kv`` (ICLR 2026) for
      rotation-based quantization at 3-4 bits. Requires ``pip install
      fade-kv[turbo]``.

Usage:
    from fade.backends import get_backend
    backend = get_backend("turbo", head_dim=128, bits=4)
    compressed = backend.compress_k(k_tensor)
    k_restored = backend.decompress_k(compressed)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import torch
from torch import Tensor

from fade.quant import (
    dequant_int4,
    quant_k_int4,
    quant_v_int4,
)


# --- protocol --------------------------------------------------------------- #
@runtime_checkable
class QuantBackend(Protocol):
    """Protocol for KV cache quantization backends."""

    def compress_k(self, k: Tensor) -> dict[str, Tensor]:
        """Compress K tensor [B, H, S, D] -> dict of compressed tensors."""
        ...

    def decompress_k(self, compressed: dict[str, Tensor], dtype: torch.dtype) -> Tensor:
        """Decompress K back to [B, H, S, D] in ``dtype``."""
        ...

    def compress_v(self, v: Tensor) -> dict[str, Tensor]:
        """Compress V tensor [B, H, S, D] -> dict of compressed tensors."""
        ...

    def decompress_v(self, compressed: dict[str, Tensor], dtype: torch.dtype) -> Tensor:
        """Decompress V back to [B, H, S, D] in ``dtype``."""
        ...

    @property
    def name(self) -> str:
        """Human-readable name for logging."""
        ...


# --- SymmetricINT4 (default) ------------------------------------------------ #
@dataclass
class SymmetricINT4Backend:
    """Original FADE quantization: per-channel K, per-token V, bit-packed INT4."""

    @property
    def name(self) -> str:
        return "symmetric_int4"

    def compress_k(self, k: Tensor) -> dict[str, Tensor]:
        packed, scale = quant_k_int4(k)
        return {"packed": packed, "scale": scale}

    def decompress_k(
        self, compressed: dict[str, Tensor], dtype: torch.dtype = torch.float16
    ) -> Tensor:
        return dequant_int4(compressed["packed"], compressed["scale"], dtype=dtype)

    def compress_v(self, v: Tensor) -> dict[str, Tensor]:
        packed, scale = quant_v_int4(v)
        return {"packed": packed, "scale": scale}

    def decompress_v(
        self, compressed: dict[str, Tensor], dtype: torch.dtype = torch.float16
    ) -> Tensor:
        return dequant_int4(compressed["packed"], compressed["scale"], dtype=dtype)


# --- RotatedINT4 (TurboQuant-inspired, native) ----------------------------- #
@dataclass
class RotatedINT4Backend:
    """Rotation-based INT4: random orthogonal rotation before quantization.

    Inspired by TurboQuant (ICLR 2026). Spreads per-channel outliers across
    all coordinates, making uniform INT4 quantization much more effective.
    Storage: uint8 packed INT4 + float32 combined_scale per vector.
    No external dependencies.
    """

    head_dim: int = 64
    bits: int = 4
    seed: int = 42
    _R_per_device: dict = None  # type: ignore[assignment]  # cached per device

    def __post_init__(self) -> None:
        self._R_per_device = {}

    @property
    def name(self) -> str:
        return f"rotated_int{self.bits}"

    def _get_R(self, device: torch.device | None = None) -> Tensor:
        device = device or torch.device("cpu")
        cached = self._R_per_device.get(device)
        if cached is not None:
            return cached
        from fade.rotated_quant import _random_orthogonal

        R = _random_orthogonal(self.head_dim, self.seed).to(device)
        self._R_per_device[device] = R
        return R

    def compress_k(self, k: Tensor) -> dict[str, Tensor]:
        from fade.rotated_quant import rotated_quant_k

        packed, scale = rotated_quant_k(k, self._get_R(k.device), bits=self.bits)
        return {"packed": packed, "scale": scale}

    def decompress_k(
        self, compressed: dict[str, Tensor], dtype: torch.dtype = torch.float16
    ) -> Tensor:
        from fade.rotated_quant import rotated_dequant_k

        return rotated_dequant_k(
            compressed["packed"],
            compressed["scale"],
            self._get_R(compressed["packed"].device),
            bits=self.bits,
            dtype=dtype,
        )

    def compress_v(self, v: Tensor) -> dict[str, Tensor]:
        from fade.rotated_quant import rotated_quant_v

        packed, scale = rotated_quant_v(v, self._get_R(v.device), bits=self.bits)
        return {"packed": packed, "scale": scale}

    def decompress_v(
        self, compressed: dict[str, Tensor], dtype: torch.dtype = torch.float16
    ) -> Tensor:
        from fade.rotated_quant import rotated_dequant_v

        return rotated_dequant_v(
            compressed["packed"],
            compressed["scale"],
            self._get_R(compressed["packed"].device),
            bits=self.bits,
            dtype=dtype,
        )


# --- TurboQuant ------------------------------------------------------------- #
@dataclass
class TurboQuantBackend:
    """Wraps ``turboquant-kv`` (TurboQuantProd) for rotation-based quantization.

    Requires ``pip install turboquant-kv``.
    """

    head_dim: int = 128
    bits: float = 4
    device: str | None = None
    _tq: Any = None  # lazy-init TurboQuantProd

    @property
    def name(self) -> str:
        return f"turboquant_{self.bits}bit"

    def _ensure_tq(self, device: torch.device | None = None) -> Any:
        if self._tq is None:
            try:
                from turboquant import TurboQuantProd
            except ImportError as e:
                raise ImportError(
                    "TurboQuantBackend requires turboquant-kv. "
                    "Install with: pip install fade-kv[turbo]"
                ) from e
            dev = self.device or (str(device) if device else "cpu")
            self._tq = TurboQuantProd(head_dim=self.head_dim, bits=self.bits, device=dev)
        return self._tq

    def compress_k(self, k: Tensor) -> dict[str, Tensor]:
        tq = self._ensure_tq(k.device)
        B, H, S, D = k.shape
        flat = k.reshape(-1, D)
        # compress(k, v) returns keys like k_idx, k_norm, k_sign, k_gamma.
        # We pass k as both args and extract the k-side.
        result = tq.compress(flat, flat)
        compressed = {key: val for key, val in result.items() if key.startswith("k_")}
        compressed["shape"] = torch.tensor([B, H, S, D])
        return compressed

    def decompress_k(
        self, compressed: dict[str, Tensor], dtype: torch.dtype = torch.float16
    ) -> Tensor:
        tq = self._ensure_tq()
        shape = compressed["shape"].tolist()
        B, H, S, D = int(shape[0]), int(shape[1]), int(shape[2]), int(shape[3])
        # Build the dict decompress expects (k_* and v_* keys).
        kv_dict = {key: val for key, val in compressed.items() if key != "shape"}
        # Add dummy v entries so decompress doesn't fail.
        for k_key in list(kv_dict.keys()):
            v_key = "v_" + k_key[2:]
            if v_key not in kv_dict:
                kv_dict[v_key] = kv_dict[k_key]
        k_flat, _ = tq.decompress(kv_dict)
        return k_flat.reshape(B, H, S, D).to(dtype)

    def compress_v(self, v: Tensor) -> dict[str, Tensor]:
        tq = self._ensure_tq(v.device)
        B, H, S, D = v.shape
        flat = v.reshape(-1, D)
        result = tq.compress(flat, flat)
        compressed = {key: val for key, val in result.items() if key.startswith("v_")}
        compressed["shape"] = torch.tensor([B, H, S, D])
        return compressed

    def decompress_v(
        self, compressed: dict[str, Tensor], dtype: torch.dtype = torch.float16
    ) -> Tensor:
        tq = self._ensure_tq()
        shape = compressed["shape"].tolist()
        B, H, S, D = int(shape[0]), int(shape[1]), int(shape[2]), int(shape[3])
        kv_dict = {key: val for key, val in compressed.items() if key != "shape"}
        for v_key in list(kv_dict.keys()):
            k_key = "k_" + v_key[2:]
            if k_key not in kv_dict:
                kv_dict[k_key] = kv_dict[v_key]
        _, v_flat = tq.decompress(kv_dict)
        return v_flat.reshape(B, H, S, D).to(dtype)


# --- factory
_BACKENDS: dict[str, type] = {
    "symmetric_int4": SymmetricINT4Backend,
    "int4": SymmetricINT4Backend,
    "rotated_int4": RotatedINT4Backend,
    "rotated": RotatedINT4Backend,
    "turbo": TurboQuantBackend,
    "turboquant": TurboQuantBackend,
}


def get_backend(name: str = "int4", **kwargs) -> QuantBackend:
    """Create a quantization backend by name.

    Args:
        name: ``"int4"`` (default), ``"turbo"`` / ``"turboquant"``.
        **kwargs: forwarded to the backend constructor (e.g. ``head_dim``,
            ``bits`` for TurboQuant).
    """
    cls = _BACKENDS.get(name.lower())
    if cls is None:
        raise ValueError(f"Unknown quant backend {name!r}. Available: {list(_BACKENDS.keys())}")
    # SymmetricINT4Backend takes no kwargs; filter them out.
    if cls is SymmetricINT4Backend:
        return cls()
    return cls(**{k: v for k, v in kwargs.items() if k in cls.__dataclass_fields__})


__all__ = [
    "QuantBackend",
    "SymmetricINT4Backend",
    "TurboQuantBackend",
    "get_backend",
]
