"""FADE: attention-aware tiered KV cache compression."""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

from fade.cache import TieredKVCache
from fade.config import FadeConfig
from fade.patch import create_tiered_cache
from fade.policy import reassign_tiers
from fade.tracker import AttentionTracker

try:
    __version__ = _pkg_version("fade")
except PackageNotFoundError:  # editable install without metadata yet
    __version__ = "0.0.0+unknown"

__all__ = [
    "AttentionTracker",
    "FadeConfig",
    "TieredKVCache",
    "create_tiered_cache",
    "reassign_tiers",
]
