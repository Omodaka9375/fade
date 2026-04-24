"""Transformers compatibility shim.

The Cache API moved several times between `transformers` 4.45 and 5.3. This
module is the single place where FADE reaches into `transformers` internals,
so everything else in `fade/` imports from here.

Supported minors:
    - 4.45.x
    - 5.3.x

If a third minor needs support, add a branch here rather than scattering
``try/except ImportError`` across the package.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version
from typing import TYPE_CHECKING

# --- transformers version detection ---------------------------------------- #
try:
    _TX_VERSION: str = _pkg_version("transformers")
except PackageNotFoundError:  # pragma: no cover - transformers is a hard dep
    _TX_VERSION = "0.0.0"


def _parse_minor(v: str) -> tuple[int, int]:
    """Return ``(major, minor)`` as ints. Ignores patch and pre-release tags."""
    parts = v.split(".")
    try:
        return int(parts[0]), int(parts[1])
    except (IndexError, ValueError):
        return 0, 0


TX_MAJOR, TX_MINOR = _parse_minor(_TX_VERSION)
IS_TX_5_PLUS: bool = TX_MAJOR >= 5


# --- DynamicCache import (stable across 4.45 and 5.3) ---------------------- #
# Both minors keep ``DynamicCache`` in ``transformers.cache_utils``.
# If that ever moves, gate it here on ``IS_TX_5_PLUS``.
from transformers.cache_utils import DynamicCache  # noqa: E402

# --- Cache base class ------------------------------------------------------ #
# 4.45 exposes ``transformers.cache_utils.Cache``; 5.x keeps it but also
# re-exports from ``transformers``. We anchor on the stable path.
try:
    from transformers.cache_utils import Cache  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover - extremely defensive
    Cache = DynamicCache  # type: ignore[misc,assignment]


def get_transformers_version() -> str:
    """Return the installed ``transformers`` version string."""
    return _TX_VERSION


if TYPE_CHECKING:
    # Re-export types for downstream typing without triggering import cycles.
    from transformers.cache_utils import DynamicCache as DynamicCache


__all__ = [
    "IS_TX_5_PLUS",
    "TX_MAJOR",
    "TX_MINOR",
    "Cache",
    "DynamicCache",
    "get_transformers_version",
]
