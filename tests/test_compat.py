"""Smoke tests for the ``fade._compat`` shim.

These run offline and lock in the contract that the rest of the package depends on.
"""
from __future__ import annotations

from fade import _compat


def test_dynamic_cache_is_importable():
    assert _compat.DynamicCache is not None


def test_cache_base_is_importable():
    assert _compat.Cache is not None


def test_transformers_version_is_parsed():
    major, minor = _compat.TX_MAJOR, _compat.TX_MINOR
    assert isinstance(major, int)
    assert isinstance(minor, int)
    assert major >= 4, f"unsupported transformers major {major}"


def test_is_tx_5_plus_matches_major():
    assert (_compat.TX_MAJOR >= 5) == _compat.IS_TX_5_PLUS


def test_version_string_is_non_empty():
    v = _compat.get_transformers_version()
    assert isinstance(v, str)
    assert v  # non-empty
