"""Feature layer: classical technical-analysis indicators.

Computes lookahead-free features (returns, SMAs, momentum, volatility, RSI)
from price data for consumption by the LLM recommender (ADR-0002).
"""

from __future__ import annotations

__all__: list[str] = []

try:  # Re-export the public feature functions when available.
    from .indicators import add_features_by_ticker, compute_features  # noqa: F401

    __all__ += ["compute_features", "add_features_by_ticker"]
except Exception:  # pragma: no cover - keep package import resilient.
    pass
