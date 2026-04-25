"""Tests for fade.server: SessionStore, H2O downgrade, module imports.

These tests validate the server infrastructure without requiring a running
FastAPI instance or GPU. The actual endpoint tests need ``pip install
fade-kv[server]`` and are skipped if fastapi is not installed.
"""

from __future__ import annotations

import time

import pytest

from fade.server import Session, SessionStore, _maybe_downgrade_h2o


# --- SessionStore ----------------------------------------------------------- #
class TestSessionStore:
    def test_put_and_get(self):
        store = SessionStore(max_entries=10, ttl=60)
        s = Session()
        store.put("a", s)
        assert store.get("a") is s
        assert len(store) == 1

    def test_get_missing_returns_none(self):
        store = SessionStore()
        assert store.get("nonexistent") is None

    def test_lru_eviction(self):
        store = SessionStore(max_entries=3, ttl=600)
        store.put("a", Session())
        store.put("b", Session())
        store.put("c", Session())
        assert len(store) == 3
        # Adding a 4th evicts the oldest (a).
        store.put("d", Session())
        assert len(store) == 3
        assert store.get("a") is None
        assert store.get("b") is not None

    def test_ttl_eviction(self):
        store = SessionStore(max_entries=10, ttl=0.01)
        store.put("a", Session())
        time.sleep(0.02)
        # Stale entry should be evicted on next get.
        assert store.get("a") is None
        assert len(store) == 0

    def test_access_refreshes_lru_order(self):
        store = SessionStore(max_entries=2, ttl=600)
        store.put("a", Session())
        store.put("b", Session())
        # Access "a" to make it most-recently-used.
        store.get("a")
        # Adding "c" should evict "b" (least recently used), not "a".
        store.put("c", Session())
        assert store.get("a") is not None
        assert store.get("b") is None

    def test_put_updates_existing(self):
        store = SessionStore(max_entries=10, ttl=600)
        s1 = Session()
        s2 = Session()
        store.put("a", s1)
        store.put("a", s2)
        assert store.get("a") is s2
        assert len(store) == 1


# --- H2O downgrade --------------------------------------------------------- #
class TestH2ODowngrade:
    def test_short_prompt_keeps_h2o(self):
        from fade.config import FadeConfig

        config = FadeConfig.balanced()  # h2o policy, prefill_track_limit=1024
        result = _maybe_downgrade_h2o(config, prompt_len=100)
        assert result.eviction_policy == "h2o"

    def test_long_prompt_downgrades_to_position(self):
        from fade.config import FadeConfig

        config = FadeConfig.balanced()
        result = _maybe_downgrade_h2o(config, prompt_len=2000)
        assert result.eviction_policy == "position"

    def test_non_h2o_policy_unchanged(self):
        from fade.config import FadeConfig

        config = FadeConfig.balanced().with_overrides(eviction_policy="position")
        result = _maybe_downgrade_h2o(config, prompt_len=2000)
        assert result.eviction_policy == "position"


# --- FastAPI app build ------------------------------------------------------ #
def test_build_app_imports():
    """Verify _build_app doesn't crash when fastapi is installed."""
    try:
        from fade.server import _build_app

        app = _build_app()
        assert app is not None
    except ImportError:
        pytest.skip("fastapi not installed")


def test_server_module_importable():
    """The server module must be importable even without fastapi."""
    import fade.server

    assert hasattr(fade.server, "main")
    assert hasattr(fade.server, "SessionStore")
    assert hasattr(fade.server, "_maybe_downgrade_h2o")
