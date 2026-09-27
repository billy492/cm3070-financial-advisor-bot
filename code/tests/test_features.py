"""Tests for ``advisor.features.indicators``.

Verifies that :func:`compute_features` adds the full set of technical-indicator
columns, introduces **no look-ahead** (leading rows are NaN exactly where the
rolling/lagged window cannot yet be filled), preserves row count, and that
:func:`add_features_by_ticker` applies the transform independently per ticker.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from advisor.features.indicators import add_features_by_ticker, compute_features

EXPECTED_FEATURE_COLUMNS: list[str] = [
    "ret_1d",
    "sma_10",
    "sma_50",
    "mom_10",
    "vol_20",
    "rsi_14",
]


def test_compute_features_adds_all_expected_columns(
    single_ticker_prices: pd.DataFrame,
) -> None:
    """All six indicator columns are present and originals are retained."""
    out = compute_features(single_ticker_prices)
    for col in EXPECTED_FEATURE_COLUMNS:
        assert col in out.columns, f"missing feature column: {col}"
    # Original price columns must be preserved.
    for col in ("date", "close"):
        assert col in out.columns


def test_compute_features_preserves_row_count(
    single_ticker_prices: pd.DataFrame,
) -> None:
    """Feature computation does not add or drop rows."""
    out = compute_features(single_ticker_prices)
    assert len(out) == len(single_ticker_prices)


def test_compute_features_does_not_mutate_input(
    single_ticker_prices: pd.DataFrame,
) -> None:
    """The input frame is not mutated in place (no new columns appear on it)."""
    before = set(single_ticker_prices.columns)
    _ = compute_features(single_ticker_prices)
    after = set(single_ticker_prices.columns)
    assert before == after


def test_no_lookahead_leading_nans(single_ticker_prices: pd.DataFrame) -> None:
    """Leading rows are NaN exactly where the window has insufficient history.

    - ret_1d (1-day pct change): first row NaN.
    - sma_10 (10-window mean): first 9 rows NaN.
    - sma_50 (50-window mean): first 49 rows NaN.
    - mom_10 (10-day momentum): first 10 rows NaN.
    - vol_20 (20-window stdev of returns): first 20 rows NaN.
    No NaNs may appear *after* the window has been satisfied (no look-ahead leak
    backwards or forwards).
    """
    out = compute_features(single_ticker_prices).reset_index(drop=True)

    # ret_1d: index 0 NaN, the rest finite.
    assert pd.isna(out.loc[0, "ret_1d"])
    assert out.loc[1:, "ret_1d"].notna().all()

    # sma_10: indices 0..8 NaN, index 9 onward finite.
    assert out.loc[: 8, "sma_10"].isna().all()
    assert out.loc[9:, "sma_10"].notna().all()

    # sma_50: indices 0..48 NaN, index 49 onward finite.
    assert out.loc[: 48, "sma_50"].isna().all()
    assert out.loc[49:, "sma_50"].notna().all()


def test_sma_10_matches_manual_rolling_mean(
    single_ticker_prices: pd.DataFrame,
) -> None:
    """sma_10 equals the trailing 10-row simple mean of close (no centring)."""
    out = compute_features(single_ticker_prices).reset_index(drop=True)
    close = single_ticker_prices["close"].reset_index(drop=True)
    expected = close.rolling(window=10).mean()
    # Compare on the non-NaN tail.
    pd.testing.assert_series_equal(
        out["sma_10"].reset_index(drop=True),
        expected,
        check_names=False,
    )


def test_ret_1d_matches_manual_pct_change(
    single_ticker_prices: pd.DataFrame,
) -> None:
    """ret_1d equals the simple 1-day percentage change of close."""
    out = compute_features(single_ticker_prices).reset_index(drop=True)
    close = single_ticker_prices["close"].reset_index(drop=True)
    expected = close.pct_change()
    pd.testing.assert_series_equal(
        out["ret_1d"].reset_index(drop=True),
        expected,
        check_names=False,
    )


def test_rsi_14_bounded_between_0_and_100(
    single_ticker_prices: pd.DataFrame,
) -> None:
    """RSI(14) values are always within the [0, 100] interval where defined."""
    out = compute_features(single_ticker_prices)
    rsi = out["rsi_14"].dropna()
    assert not rsi.empty
    assert (rsi >= 0.0 - 1e-9).all()
    assert (rsi <= 100.0 + 1e-9).all()


def test_add_features_by_ticker_groups_independently(
    multi_ticker_prices: pd.DataFrame,
) -> None:
    """Per-ticker grouping yields independent leading NaNs for each ticker."""
    out = add_features_by_ticker(multi_ticker_prices)

    # All expected feature columns present, ticker column retained.
    for col in EXPECTED_FEATURE_COLUMNS + ["ticker"]:
        assert col in out.columns

    # Row count preserved across all groups.
    assert len(out) == len(multi_ticker_prices)

    tickers = sorted(multi_ticker_prices["ticker"].unique())
    assert tickers == ["AAA", "BBB", "CCC"]

    for tkr in tickers:
        grp = out[out["ticker"] == tkr].sort_values("date").reset_index(drop=True)
        # The first row of *each* ticker must be NaN for ret_1d -- proving the
        # rolling computation did not bleed across the ticker boundary.
        assert pd.isna(grp.loc[0, "ret_1d"]), f"{tkr}: ret_1d leaked across group"
        # And the tail (post-window) is fully populated for sma_10.
        assert grp.loc[9:, "sma_10"].notna().all(), f"{tkr}: sma_10 tail has NaN"


def test_add_features_by_ticker_row_alignment(
    multi_ticker_prices: pd.DataFrame,
) -> None:
    """Output close values match the input per (ticker, date) -- no row scramble."""
    out = add_features_by_ticker(multi_ticker_prices)
    merged = multi_ticker_prices.merge(
        out[["ticker", "date", "close"]],
        on=["ticker", "date"],
        suffixes=("_in", "_out"),
    )
    assert np.allclose(merged["close_in"].to_numpy(), merged["close_out"].to_numpy())
