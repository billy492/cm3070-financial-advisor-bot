"""Technical-analysis feature engineering for the financial advisor bot.

This module turns raw OHLCV price history into the classical technical-analysis
(TA) feature set that is fed to the LLM advisor (ADR-0002). All indicators are
implemented from scratch with ``pandas``/``numpy`` only -- the project
deliberately avoids a hard dependency on ``pandas-ta`` so that the test suite
runs in a minimal offline environment.

Design constraints (ADR-0002):
    * **No lookahead bias.** Every rolling / momentum / volatility window only
      ever looks *backward* in time. A feature row dated ``t`` is computable
      using information available up to and including the close of day ``t``.
    * **Stable schema.** ``compute_features`` always appends the same six
      columns so downstream code (LLM prompt builder, backtester, calibration)
      can rely on a fixed feature contract.

Feature columns added:
    ret_1d   -- 1-day simple return of ``close`` (``close.pct_change(1)``).
    sma_10   -- 10-day simple moving average of ``close``.
    sma_50   -- 50-day simple moving average of ``close``.
    mom_10   -- 10-day momentum, ``close.pct_change(10)``.
    vol_20   -- 20-day rolling standard deviation of ``ret_1d`` (realised vol).
    rsi_14   -- 14-period Wilder Relative Strength Index of ``close``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "compute_features",
    "add_features_by_ticker",
    "FEATURE_COLUMNS",
]

#: Canonical, ordered list of feature columns this module appends.
FEATURE_COLUMNS: list[str] = [
    "ret_1d",
    "sma_10",
    "sma_50",
    "mom_10",
    "vol_20",
    "rsi_14",
]

# Window lengths (kept as module constants for clarity / single source of truth).
_SMA_FAST: int = 10
_SMA_SLOW: int = 50
_MOM_WINDOW: int = 10
_VOL_WINDOW: int = 20
_RSI_PERIOD: int = 14


def _wilder_rsi(close: pd.Series, period: int = _RSI_PERIOD) -> pd.Series:
    """Compute the Wilder Relative Strength Index (RSI) of a price series.

    Implements the original J. Welles Wilder (1978) smoothing: average gains and
    average losses are smoothed with an exponential moving average whose decay
    corresponds to ``alpha = 1 / period`` (i.e. a Wilder / RMA smoother, *not*
    the simple-rolling-mean variant). This is the convention used by most charting
    platforms and by ``pandas-ta``'s ``rsi``.

    The smoother is causal (``adjust=False`` EMA over backward-looking deltas), so
    the result introduces no lookahead bias: ``rsi[t]`` depends only on closes up
    to and including day ``t``.

    Args:
        close: Price series (typically adjusted/unadjusted close), indexed in
            chronological order.
        period: Lookback period for the Wilder smoothing. Defaults to 14.

    Returns:
        A ``pd.Series`` aligned to ``close`` holding RSI values in ``[0, 100]``.
        The first ``period`` rows are ``NaN`` (insufficient history); flat
        windows where every loss is zero yield ``100.0``.
    """
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)

    # Wilder smoothing == EMA with alpha = 1/period, computed causally.
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))

    # When avg_loss == 0 (uninterrupted gains) RS is +inf -> RSI -> 100.
    # When both averages are 0 (perfectly flat) RS is NaN; treat as neutral 50.
    rsi = rsi.where(avg_loss != 0.0, other=100.0)
    flat = (avg_gain == 0.0) & (avg_loss == 0.0)
    rsi = rsi.mask(flat, other=50.0)

    # Re-mask the warm-up period (ewm min_periods already NaNs it, but the
    # .where above can resurrect values where avg_loss==0 during warm-up).
    warmup = close.diff().isna().cumsum() <= period
    rsi = rsi.mask(warmup & avg_gain.isna(), other=np.nan)

    return rsi.rename("rsi_14")


def compute_features(prices: pd.DataFrame) -> pd.DataFrame:
    """Append classical TA features to a single-ticker price frame.

    The input is expected to be one ticker's history, sorted ascending by date,
    with at least a ``close`` column (OHLCV columns are preserved if present but
    only ``close`` is required). The function never reorders or mutates the input
    in place: it operates on a defensive copy and returns a new frame with the
    six :data:`FEATURE_COLUMNS` appended.

    All features are strictly backward-looking (no lookahead): each row uses only
    data available up to and including that row's date. Leading rows where a
    window has insufficient history are ``NaN``.

    Args:
        prices: Single-ticker ``pd.DataFrame`` sorted by date containing a
            ``close`` column. If a ``date`` column is present it is used to
            enforce chronological ordering; otherwise the existing index order is
            assumed to be chronological.

    Returns:
        A new ``pd.DataFrame``: the original columns followed by
        ``ret_1d, sma_10, sma_50, mom_10, vol_20, rsi_14``.

    Raises:
        ValueError: If ``prices`` is empty or has no ``close`` column.
    """
    if prices is None or len(prices) == 0:
        raise ValueError("compute_features received an empty price frame.")
    if "close" not in prices.columns:
        raise ValueError(
            "compute_features requires a 'close' column; "
            f"got columns={list(prices.columns)!r}"
        )

    out = prices.copy()

    # Enforce chronological order without lookahead: sort by 'date' if available.
    if "date" in out.columns:
        out = out.sort_values("date").reset_index(drop=True)

    close = out["close"].astype("float64")

    # 1-day simple return.
    out["ret_1d"] = close.pct_change(periods=1)

    # Simple moving averages (backward rolling means).
    out["sma_10"] = close.rolling(window=_SMA_FAST, min_periods=_SMA_FAST).mean()
    out["sma_50"] = close.rolling(window=_SMA_SLOW, min_periods=_SMA_SLOW).mean()

    # 10-day momentum (percentage change over the window).
    out["mom_10"] = close.pct_change(periods=_MOM_WINDOW)

    # 20-day realised volatility = rolling std of daily returns.
    out["vol_20"] = (
        out["ret_1d"].rolling(window=_VOL_WINDOW, min_periods=_VOL_WINDOW).std()
    )

    # 14-period Wilder RSI.
    out["rsi_14"] = _wilder_rsi(close, period=_RSI_PERIOD)

    return out


def add_features_by_ticker(prices_long: pd.DataFrame) -> pd.DataFrame:
    """Apply :func:`compute_features` independently to each ticker in a long frame.

    Groups the long-form price frame by ``ticker`` and applies
    :func:`compute_features` per group, so rolling windows never bleed across
    ticker boundaries (which would otherwise introduce both contamination and
    lookahead at the seams). Groups are processed in sorted ticker order and the
    results are concatenated back into a single long frame.

    Args:
        prices_long: Long-form ``pd.DataFrame`` with at least ``ticker``, ``date``
            and ``close`` columns (the standard schema emitted by
            ``advisor.data.loader.load_prices``).

    Returns:
        A long-form ``pd.DataFrame`` with the original columns plus the six
        :data:`FEATURE_COLUMNS`, sorted by ``(ticker, date)`` and re-indexed.

    Raises:
        ValueError: If ``prices_long`` is empty or is missing the ``ticker`` or
            ``close`` columns.
    """
    if prices_long is None or len(prices_long) == 0:
        raise ValueError("add_features_by_ticker received an empty price frame.")
    missing = {"ticker", "close"} - set(prices_long.columns)
    if missing:
        raise ValueError(
            "add_features_by_ticker requires columns "
            f"{sorted({'ticker', 'close'})}; missing={sorted(missing)!r}"
        )

    frames: list[pd.DataFrame] = []
    # sort=True keeps deterministic, sorted-ticker output ordering.
    for _ticker, group in prices_long.groupby("ticker", sort=True):
        frames.append(compute_features(group))

    result = pd.concat(frames, ignore_index=True)

    # Stable final ordering by (ticker, date) when a date column is present.
    sort_cols = [c for c in ("ticker", "date") if c in result.columns]
    if sort_cols:
        result = result.sort_values(sort_cols).reset_index(drop=True)

    return result
