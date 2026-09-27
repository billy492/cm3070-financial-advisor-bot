"""Data layer: ticker universe and offline-safe price loading.

Provides the curated US large-cap :data:`UNIVERSE` (grouped by :data:`SECTORS`)
and the cached yfinance price loader. Imports are network-free; fetching only
occurs on cache miss inside :func:`load_prices`.
"""

from __future__ import annotations

__all__: list[str] = []

try:  # Re-export common entry points when submodules are importable.
    from .universe import SECTORS, UNIVERSE  # noqa: F401

    __all__ += ["SECTORS", "UNIVERSE"]
except Exception:  # pragma: no cover - keep package import resilient.
    pass

try:
    from .loader import clear_cache, load_prices  # noqa: F401

    __all__ += ["load_prices", "clear_cache"]
except Exception:  # pragma: no cover
    pass
