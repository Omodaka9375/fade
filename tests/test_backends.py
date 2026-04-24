"""Tests for pluggable quantization backends (Phase A1)."""

from __future__ import annotations

import pytest
import torch

from fade.backends import SymmetricINT4Backend, get_backend

B, H, S, D = 1, 2, 16, 64
DTYPE = torch.float32


def _random_kv():
    return torch.randn(B, H, S, D, dtype=DTYPE), torch.randn(B, H, S, D, dtype=DTYPE)


# --- SymmetricINT4Backend --------------------------------------------------- #
def test_int4_backend_round_trip():
    backend = SymmetricINT4Backend()
    k, v = _random_kv()

    ck = backend.compress_k(k)
    cv = backend.compress_v(v)
    assert "packed" in ck and "scale" in ck
    assert "packed" in cv and "scale" in cv

    k_hat = backend.decompress_k(ck, dtype=DTYPE)
    v_hat = backend.decompress_v(cv, dtype=DTYPE)
    assert k_hat.shape == k.shape
    assert v_hat.shape == v.shape

    # INT4 round-trip should be reasonably close.
    k_err = (k - k_hat).abs().mean() / k.abs().mean()
    v_err = (v - v_hat).abs().mean() / v.abs().mean()
    assert k_err < 0.15, f"K round-trip error too high: {k_err:.3f}"
    assert v_err < 0.15, f"V round-trip error too high: {v_err:.3f}"


def test_int4_backend_name():
    assert SymmetricINT4Backend().name == "symmetric_int4"


def test_get_backend_int4():
    b = get_backend("int4")
    assert isinstance(b, SymmetricINT4Backend)


def test_get_backend_unknown_raises():
    with pytest.raises(ValueError, match="Unknown"):
        get_backend("nonexistent")


# --- TurboQuantBackend ------------------------------------------------------ #
def test_get_backend_turbo_import():
    """Verify TurboQuant backend can be instantiated (if turboquant installed)."""
    try:
        b = get_backend("turbo", head_dim=D, bits=4)
        assert b.name == "turboquant_4bit"
    except ImportError:
        pytest.skip("turboquant-kv not installed")


def test_turbo_backend_round_trip():
    """Full compress/decompress round-trip with TurboQuant."""
    try:
        b = get_backend("turbo", head_dim=D, bits=4)
    except ImportError:
        pytest.skip("turboquant-kv not installed")

    k, v = _random_kv()
    ck = b.compress_k(k)
    k_hat = b.decompress_k(ck, dtype=DTYPE)
    assert k_hat.shape == k.shape
    assert torch.isfinite(k_hat).all()

    cv = b.compress_v(v)
    v_hat = b.decompress_v(cv, dtype=DTYPE)
    assert v_hat.shape == v.shape
    assert torch.isfinite(v_hat).all()


# --- FadeConfig integration ------------------------------------------------- #
def test_config_quant_backend_field():
    from fade.config import FadeConfig

    c = FadeConfig.safe()
    assert c.quant_backend == "int4"
    assert "quant_backend" in c.to_cache_kwargs()


def test_cache_accepts_quant_backend():
    from fade.cache import TieredKVCache

    cache = TieredKVCache(head_dim=D, quant_backend="int4")
    assert cache._quant_backend.name == "symmetric_int4"
