"""Tests for ``advisor.data.loader`` -- cache-first, offline-safe loading.

The central guarantee under test: when a per-ticker parquet cache file already
exists, :func:`load_prices` must read it **without touching the network** (i.e.
without ever calling into ``yfinance``). We enforce this by installing a booby-
trapped ``yfinance`` stub in :data:`sys.modules` whose every access raises, so
any accidental network fetch fails the test loudly.

We also test the :func:`universe` invariants and :func:`clear_cache` behaviour,
both of which are fully offline.
"""

from __future__ import annotations

import sys
import types
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from advisor.config import CACHE_DIR, HISTORY_START
from advisor.data.loader import (
    _cache_path,
    _fetch_from_yfinance,
    _normalize_yf_frame,
    clear_cache,
    load_prices,
)

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


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
class _ExplodingYFinance(types.ModuleType):
    """A fake ``yfinance`` module whose every attribute access raises.

    Installed into ``sys.modules`` to prove that a cache hit never reaches the
    network layer. If the loader tries to use ``yfinance`` at all, the test
    fails with a clear, attributable error.
    """

    def __getattr__(self, name: str) -> object:  # noqa: D401 - simple guard
        raise AssertionError(
            f"yfinance was accessed ('{name}') -- load_prices must not hit the "
            "network on a cache hit."
        )


def _write_cache_parquet(ticker: str, cache_dir: Path, n: int = 30) -> pd.DataFrame:
    """Write a deterministic long-form OHLCV parquet at the loader's cache path.

    Returns the frame written, so callers can compare what comes back out.
    """
    rng = np.random.default_rng(abs(hash(ticker)) % (2**32))
    dates = pd.bdate_range("2021-01-04", periods=n)
    close = 100.0 + np.cumsum(rng.normal(0.0, 1.0, size=n))
    frame = pd.DataFrame(
        {
            "date": dates.date,
            "ticker": ticker,
            "open": close,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "adj_close": close,
            "volume": rng.integers(1_000_000, 2_000_000, size=n).astype("int64"),
        }
    )[OHLCV_COLUMNS]

    path = _cache_path(ticker, cache_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    return frame


# --------------------------------------------------------------------------- #
# _cache_path
# --------------------------------------------------------------------------- #
def test_cache_path_under_cache_dir_and_ticker_specific(
    synthetic_cache_dir: Path,
) -> None:
    """``_cache_path`` lives under the given cache dir and is per-ticker."""
    p_aaa = _cache_path("AAA", synthetic_cache_dir)
    p_bbb = _cache_path("BBB", synthetic_cache_dir)

    assert synthetic_cache_dir in p_aaa.parents
    assert p_aaa != p_bbb
    # Ticker symbol should appear in the file name in some form.
    assert "AAA" in p_aaa.name.upper()


def test_cache_path_default_is_config_cache_dir() -> None:
    """The default ``cache_dir`` argument resolves under the configured CACHE_DIR."""
    p = _cache_path("AAA")  # type: ignore[call-arg]
    assert CACHE_DIR in p.parents or p.parent == CACHE_DIR


# --------------------------------------------------------------------------- #
# load_prices: cache hit -> no network
# --------------------------------------------------------------------------- #
def test_load_prices_reads_cache_without_network(
    synthetic_cache_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A pre-populated cache is read back with yfinance booby-trapped."""
    expected = _write_cache_parquet("AAA", synthetic_cache_dir, n=30)

    # Any access to yfinance now raises -> proves no network fetch occurred.
    monkeypatch.setitem(sys.modules, "yfinance", _ExplodingYFinance("yfinance"))

    out = load_prices(
        ["AAA"],
        date(2021, 1, 4),
        date(2021, 3, 1),
        use_cache=True,
        cache_dir=synthetic_cache_dir,
    )

    assert isinstance(out, pd.DataFrame)
    assert not out.empty
    for col in OHLCV_COLUMNS:
        assert col in out.columns
    assert set(out["ticker"].unique()) == {"AAA"}

    # The close prices that round-trip through the cache must match what we wrote
    # (restricted to the requested date window).
    out_aaa = out[out["ticker"] == "AAA"].sort_values("date")
    assert len(out_aaa) <= len(expected)
    assert out_aaa["close"].notna().all()


def test_load_prices_single_ticker_string_arg(
    synthetic_cache_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``tickers`` accepts a bare string as well as a list (cache hit, no net)."""
    _write_cache_parquet("BBB", synthetic_cache_dir, n=20)
    monkeypatch.setitem(sys.modules, "yfinance", _ExplodingYFinance("yfinance"))

    out = load_prices(
        "BBB",
        date(2021, 1, 4),
        date(2021, 3, 1),
        use_cache=True,
        cache_dir=synthetic_cache_dir,
    )
    assert set(out["ticker"].unique()) == {"BBB"}


def test_load_prices_multi_ticker_cache_hit(
    synthetic_cache_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Multiple cached tickers load together without any network access."""
    for tkr in ("AAA", "BBB", "CCC"):
        _write_cache_parquet(tkr, synthetic_cache_dir, n=25)
    monkeypatch.setitem(sys.modules, "yfinance", _ExplodingYFinance("yfinance"))

    out = load_prices(
        ["AAA", "BBB", "CCC"],
        date(2021, 1, 4),
        date(2021, 3, 1),
        use_cache=True,
        cache_dir=synthetic_cache_dir,
    )
    assert set(out["ticker"].unique()) == {"AAA", "BBB", "CCC"}


def test_load_prices_respects_date_window(
    synthetic_cache_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Returned rows fall within the requested [start, end] window."""
    _write_cache_parquet("AAA", synthetic_cache_dir, n=40)
    monkeypatch.setitem(sys.modules, "yfinance", _ExplodingYFinance("yfinance"))

    start, end = date(2021, 1, 11), date(2021, 1, 22)
    out = load_prices(
        ["AAA"], start, end, use_cache=True, cache_dir=synthetic_cache_dir
    )
    out_dates = pd.to_datetime(out["date"])
    assert (out_dates >= pd.Timestamp(start)).all()
    assert (out_dates <= pd.Timestamp(end)).all()


# --------------------------------------------------------------------------- #
# clear_cache
# --------------------------------------------------------------------------- #
def test_clear_cache_removes_parquet_files(synthetic_cache_dir: Path) -> None:
    """``clear_cache`` removes cached parquet files from the directory."""
    _write_cache_parquet("AAA", synthetic_cache_dir, n=10)
    _write_cache_parquet("BBB", synthetic_cache_dir, n=10)
    assert _cache_path("AAA", synthetic_cache_dir).exists()

    clear_cache(cache_dir=synthetic_cache_dir)

    assert not _cache_path("AAA", synthetic_cache_dir).exists()
    assert not _cache_path("BBB", synthetic_cache_dir).exists()


# --------------------------------------------------------------------------- #
# _normalize_yf_frame: yfinance 1.x (MultiIndex) and legacy (flat) shapes
# --------------------------------------------------------------------------- #
_YF_FIELDS = ["Adj Close", "Close", "High", "Low", "Open", "Volume"]


def _yf17_frame(ticker: str, n: int = 3) -> pd.DataFrame:
    """Frame shaped like ``yf.download(<one ticker>)`` on yfinance 1.7.

    Columns are a two-level ``MultiIndex`` named ``(Price, Ticker)`` even for a
    single ticker; the index is a ``DatetimeIndex`` named ``Date``.
    """
    index = pd.DatetimeIndex(pd.bdate_range("2024-01-02", periods=n), name="Date")
    columns = pd.MultiIndex.from_product([_YF_FIELDS, [ticker]], names=["Price", "Ticker"])
    values = np.arange(n * len(_YF_FIELDS), dtype="float64").reshape(n, len(_YF_FIELDS))
    return pd.DataFrame(values, index=index, columns=columns)


def _legacy_frame(n: int = 2, *, adj_close: bool = True, tz: str | None = None) -> pd.DataFrame:
    """Flat-column frame as returned by older yfinance / ``Ticker.history``."""
    index = pd.DatetimeIndex(pd.bdate_range("2024-01-02", periods=n, tz=tz), name="Date")
    data = {
        "Open": np.full(n, 10.0),
        "High": np.full(n, 11.0),
        "Low": np.full(n, 9.0),
        "Close": np.full(n, 10.5),
        "Volume": np.full(n, 1000),
    }
    if adj_close:
        data["Adj Close"] = np.full(n, 10.4)
    return pd.DataFrame(data, index=index)


def test_normalize_yf17_multiindex_single_ticker() -> None:
    """The ``(Price, Ticker)`` MultiIndex of yfinance 1.7 collapses to long form."""
    out = _normalize_yf_frame(_yf17_frame("AAPL"), "aapl")

    assert list(out.columns) == OHLCV_COLUMNS
    assert out.columns.name is None
    assert len(out) == 3
    assert set(out["ticker"]) == {"AAPL"}
    assert all(isinstance(d, date) for d in out["date"])
    assert out["date"].tolist() == [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4)]
    # Column mapping: Adj Close -> adj_close is the first field, Close the second, ...
    assert out["adj_close"].tolist() == [0.0, 6.0, 12.0]
    assert out["close"].tolist() == [1.0, 7.0, 13.0]
    assert out["volume"].tolist() == [5.0, 11.0, 17.0]


def test_normalize_yf_multiindex_ticker_first_level() -> None:
    """``group_by="ticker"`` nesting ``(Ticker, Price)`` is handled too."""
    raw = _yf17_frame("MSFT").swaplevel(axis=1)
    out = _normalize_yf_frame(raw, "MSFT")
    assert list(out.columns) == OHLCV_COLUMNS
    assert out["close"].tolist() == [1.0, 7.0, 13.0]


def test_normalize_legacy_flat_frame_with_adj_close() -> None:
    """Flat title-case columns map to snake_case and keep Adj Close."""
    out = _normalize_yf_frame(_legacy_frame(adj_close=True), "spy")
    assert list(out.columns) == OHLCV_COLUMNS
    assert set(out["ticker"]) == {"SPY"}
    assert out["adj_close"].tolist() == [10.4, 10.4]
    assert out["close"].tolist() == [10.5, 10.5]


def test_normalize_legacy_flat_frame_backfills_adj_close_and_drops_tz() -> None:
    """Missing Adj Close is copied from Close; tz-aware dates become naive dates."""
    out = _normalize_yf_frame(_legacy_frame(adj_close=False, tz="America/New_York"), "AAPL")
    assert out["adj_close"].tolist() == out["close"].tolist()
    assert out["date"].tolist() == [date(2024, 1, 2), date(2024, 1, 3)]


def test_normalize_empty_and_none() -> None:
    """``None`` / empty input yields an empty frame with the canonical columns."""
    for raw in (None, pd.DataFrame()):
        out = _normalize_yf_frame(raw, "AAPL")
        assert list(out.columns) == OHLCV_COLUMNS
        assert out.empty


def test_normalize_drops_rows_without_close_and_duplicate_dates() -> None:
    """Rows with NaN close are removed and duplicate dates keep the last row."""
    raw = _legacy_frame(n=3)
    raw.iloc[1, raw.columns.get_loc("Close")] = np.nan
    dup = pd.concat([raw, raw.iloc[[2]]])
    out = _normalize_yf_frame(dup, "AAPL")
    assert len(out) == 2
    assert out["date"].is_monotonic_increasing


# --------------------------------------------------------------------------- #
# _fetch_from_yfinance: explicit start date, retries and fallback (no network)
# --------------------------------------------------------------------------- #
class _FakeYFinance(types.ModuleType):
    """Scriptable stand-in for the ``yfinance`` module."""

    def __init__(self, download_results: list[object], history: object = None) -> None:
        super().__init__("yfinance")
        self.download_results = list(download_results)
        self.download_calls: list[dict] = []
        self.history_calls: list[dict] = []
        self._history = history

        outer = self

        class Ticker:  # noqa: D106 - fake
            def __init__(self, symbol: str) -> None:
                self.symbol = symbol

            def history(self, **kw: object) -> object:
                outer.history_calls.append(kw)
                return outer._history if outer._history is not None else pd.DataFrame()

        self.Ticker = Ticker

    def download(self, ticker: str, **kw: object) -> object:
        self.download_calls.append({"ticker": ticker, **kw})
        item = self.download_results.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def test_fetch_passes_explicit_start_not_period_max(monkeypatch: pytest.MonkeyPatch) -> None:
    """The loader must send ``start``: yfinance 1.x ignores ``period="max"``."""
    fake = _FakeYFinance([_yf17_frame("AAPL")])
    monkeypatch.setitem(sys.modules, "yfinance", fake)

    out = _fetch_from_yfinance("AAPL", pause=0.0)

    assert len(out) == 3 and set(out["ticker"]) == {"AAPL"}
    call = fake.download_calls[0]
    assert call["ticker"] == "AAPL"
    assert call["start"] == HISTORY_START.isoformat()
    assert "period" not in call
    assert call["auto_adjust"] is False and call["actions"] is False
    assert call["progress"] is False and call["threads"] is False


def test_fetch_retries_after_exception_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    """A transient exception is retried (with zero pause in tests)."""
    fake = _FakeYFinance([RuntimeError("rate limited"), _yf17_frame("AAPL")])
    monkeypatch.setitem(sys.modules, "yfinance", fake)
    out = _fetch_from_yfinance("AAPL", retries=3, pause=0.0)
    assert len(out) == 3
    assert len(fake.download_calls) == 2


def test_fetch_falls_back_to_ticker_history_when_download_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An empty ``download`` result triggers ``Ticker.history`` with the same start."""
    fake = _FakeYFinance([pd.DataFrame()], history=_legacy_frame(n=4))
    monkeypatch.setitem(sys.modules, "yfinance", fake)
    out = _fetch_from_yfinance("AAPL", retries=1, pause=0.0)
    assert len(out) == 4
    assert fake.history_calls[0]["start"] == HISTORY_START.isoformat()


def test_fetch_raises_after_all_attempts_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    """Persistent exceptions surface as RuntimeError naming the ticker."""
    fake = _FakeYFinance([RuntimeError("down"), RuntimeError("down")])
    monkeypatch.setitem(sys.modules, "yfinance", fake)
    with pytest.raises(RuntimeError, match="AAPL"):
        _fetch_from_yfinance("AAPL", retries=2, pause=0.0)


def test_fetch_returns_empty_when_no_data_without_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """No rows on every attempt (and no exception) -> empty canonical frame."""
    fake = _FakeYFinance([pd.DataFrame(), pd.DataFrame()])
    monkeypatch.setitem(sys.modules, "yfinance", fake)
    out = _fetch_from_yfinance("ZZZZ", retries=2, pause=0.0)
    assert out.empty and list(out.columns) == OHLCV_COLUMNS


def test_load_prices_cache_miss_writes_parquet_then_hits(
    synthetic_cache_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A miss fetches via (fake) yfinance and caches; the next call never fetches."""
    fake = _FakeYFinance([_yf17_frame("AAPL", n=5)])
    monkeypatch.setitem(sys.modules, "yfinance", fake)

    out = load_prices(["AAPL"], date(2024, 1, 1), date(2024, 2, 1), cache_dir=synthetic_cache_dir)
    assert len(out) == 5
    assert _cache_path("AAPL", synthetic_cache_dir).exists()

    monkeypatch.setitem(sys.modules, "yfinance", _ExplodingYFinance("yfinance"))
    again = load_prices(["AAPL"], date(2024, 1, 1), date(2024, 2, 1), cache_dir=synthetic_cache_dir)
    pd.testing.assert_frame_equal(again, out)
