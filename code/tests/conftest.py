"""Shared ``pytest`` fixtures for the advisor test suite.

These fixtures construct small, fully deterministic, **offline** datasets so the
working components of the ``advisor`` package can be exercised without any
network access or external service. Synthetic prices follow a smooth,
reproducible geometric path (seeded RNG) so technical indicators and evaluation
metrics produce stable, hand-checkable values.

All fixtures live here so that no individual test module duplicates data-setup
logic. See the SHARED INTERFACE CONTRACT for the long-form OHLCV DataFrame shape
(columns: date, ticker, open, high, low, close, adj_close, volume).
"""

from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest

if TYPE_CHECKING:  # pragma: no cover - typing only
    from pathlib import Path


# Column order matching the loader contract (long-form OHLCV).
OHLCV_COLUMNS: list[str] = [
    "date",
    "ticker",
    "open",
    "high",
    "low",
    "close",
    "adj_close",
    "volume",
]


def _make_single_ticker_frame(
    ticker: str,
    *,
    n: int,
    start: date,
    seed: int,
    start_price: float = 100.0,
) -> pd.DataFrame:
    """Build a deterministic single-ticker long-form OHLCV frame.

    The close series follows a seeded geometric random walk with a small daily
    drift so that derived returns/indicators are well-defined and stable. OHLC
    values are derived from the close so that the usual ordering invariants
    (low <= open/close <= high) hold.

    Parameters
    ----------
    ticker:
        Ticker symbol to stamp on every row.
    n:
        Number of trading days (rows) to generate.
    start:
        Calendar date of the first row. Subsequent rows use business-day spacing.
    seed:
        Seed for the per-ticker RNG to keep generation reproducible.
    start_price:
        Initial close price.

    Returns:
    -------
    pandas.DataFrame
        Long-form OHLCV frame with :data:`OHLCV_COLUMNS`, sorted by date.
    """
    rng = np.random.default_rng(seed)
    # Small daily log-returns: tiny positive drift + modest noise.
    daily_ret = rng.normal(loc=0.0005, scale=0.01, size=n)
    close = start_price * np.exp(np.cumsum(daily_ret))

    dates = pd.bdate_range(start=pd.Timestamp(start), periods=n)

    # Derive OHLC from close in a way that preserves low <= {open, close} <= high.
    open_ = np.empty(n, dtype=float)
    open_[0] = start_price
    open_[1:] = close[:-1]  # open = previous close (gap-free synthetic series)

    span = np.abs(rng.normal(loc=0.0, scale=0.5, size=n)) + 0.25
    high = np.maximum(open_, close) + span
    low = np.minimum(open_, close) - span
    volume = rng.integers(1_000_000, 5_000_000, size=n).astype("int64")

    frame = pd.DataFrame(
        {
            "date": dates.date,
            "ticker": ticker,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "adj_close": close,  # treat unadjusted == adjusted for synthetic data
            "volume": volume,
        }
    )
    return frame[OHLCV_COLUMNS]


@pytest.fixture(scope="session")
def n_days() -> int:
    """Number of synthetic trading days used across the suite."""
    return 80


@pytest.fixture
def single_ticker_prices(n_days: int) -> pd.DataFrame:
    """A single-ticker long-form OHLCV frame (``AAA``), sorted ascending by date.

    Long enough (default 80 rows) that the longest indicator window (sma_50)
    yields non-NaN values for the tail rows.
    """
    return _make_single_ticker_frame(
        "AAA", n=n_days, start=date(2021, 1, 4), seed=1
    )


@pytest.fixture
def multi_ticker_prices(n_days: int) -> pd.DataFrame:
    """A multi-ticker long-form OHLCV frame for ``AAA``, ``BBB``, ``CCC``.

    Returned concatenated and globally sorted by (ticker, date) to mirror the
    ``loader.load_prices`` output for several tickers.
    """
    frames = [
        _make_single_ticker_frame("AAA", n=n_days, start=date(2021, 1, 4), seed=1),
        _make_single_ticker_frame("BBB", n=n_days, start=date(2021, 1, 4), seed=2),
        _make_single_ticker_frame("CCC", n=n_days, start=date(2021, 1, 4), seed=3),
    ]
    combined = pd.concat(frames, ignore_index=True)
    return combined.sort_values(["ticker", "date"]).reset_index(drop=True)


@pytest.fixture
def returns_series() -> pd.Series:
    """A small, deterministic daily-returns series with known statistics.

    Hand-chosen values so tests can assert against analytically computed
    Sharpe/Sortino/drawdown targets.
    """
    return pd.Series([0.01, -0.02, 0.03, 0.00, -0.01, 0.02, 0.015, -0.005])


@pytest.fixture
def features_dict() -> dict:
    """A representative feature dict as passed to the LLM advisor / schema.

    Mirrors the columns produced by ``compute_features`` for a single as-of row.
    """
    return {
        "ret_1d": 0.012,
        "sma_10": 101.5,
        "sma_50": 98.2,
        "mom_10": 0.034,
        "vol_20": 0.018,
        "rsi_14": 57.3,
    }


@pytest.fixture
def synthetic_cache_dir(tmp_path: Path) -> Path:
    """An empty temporary cache directory for loader cache tests."""
    cache = tmp_path / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    return cache
