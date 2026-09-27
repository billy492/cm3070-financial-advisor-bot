"""Price-data loader with a per-ticker parquet cache.

Loads daily OHLCV price history for one or more US-equity tickers and returns
it in the canonical *long form* used throughout the advisor pipeline::

    columns = [date, ticker, open, high, low, close, adj_close, volume]

Design notes:
    * **Offline-safe import.** ``yfinance`` is imported lazily *inside* the
      fetch helper, never at module import time, so importing this module
      never touches the network. Tests and downstream modules can import
      :func:`load_prices` with no network and no third-party data dependency
      present.
    * **Per-ticker parquet cache.** Each ticker is cached as a single parquet
      file under ``cache_dir`` (default :data:`advisor.config.CACHE_DIR`). On a
      cache hit the file is read and sliced to the requested ``[start, end)``
      window; yfinance is only contacted on a cache miss. The cache stores the
      full fetched history per ticker (from :data:`advisor.config.HISTORY_START`
      onwards), so overlapping date requests reuse one file.
    * **Robust normalisation.** yfinance returns different column shapes
      depending on the library version and the number of tickers: yfinance
      1.x returns ``MultiIndex`` columns ``(Price, Ticker)`` even for a single
      ticker, older versions return flat columns, and ``Adj Close`` may be
      absent when ``auto_adjust=True``. :func:`_normalize_yf_frame` collapses
      every variant into the canonical long-form schema.
    * **yfinance 1.x quirk.** ``yf.download(period="max")`` returns an empty
      frame ("possibly delisted") on yfinance 1.7, so the fetch helper always
      passes an explicit ``start`` date and retries transient failures.

This module owns the data-access boundary referenced by ADR-0001
(market = US stocks via yfinance).
"""

from __future__ import annotations

import logging
import time
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING

from advisor.config import CACHE_DIR, HISTORY_START

if TYPE_CHECKING:  # pragma: no cover - typing only, no runtime import
    import pandas as pd

__all__ = ["load_prices", "clear_cache", "LONG_COLUMNS"]

_log = logging.getLogger(__name__)

#: Canonical long-form column order returned by :func:`load_prices`.
LONG_COLUMNS: list[str] = [
    "date",
    "ticker",
    "open",
    "high",
    "low",
    "close",
    "adj_close",
    "volume",
]
_LONG_COLUMNS = LONG_COLUMNS  # backwards-compatible private alias

# Mapping from yfinance's (title-case, spaced) OHLCV field names to our
# snake_case canonical names.
_FIELD_MAP: dict[str, str] = {
    "open": "open",
    "high": "high",
    "low": "low",
    "close": "close",
    "adj close": "adj_close",
    "adjclose": "adj_close",
    "adj_close": "adj_close",
    "volume": "volume",
}
_PRICE_COLUMNS: tuple[str, ...] = ("open", "high", "low", "close", "adj_close", "volume")


def _cache_path(ticker: str, cache_dir: Path = CACHE_DIR) -> Path:
    """Return the parquet cache path for a single ticker.

    Args:
        ticker: Ticker symbol (case-insensitive). Normalised to upper case so
            that ``"aapl"`` and ``"AAPL"`` map to the same cache file.
        cache_dir: Directory under which per-ticker parquet files are stored.

    Returns:
        ``cache_dir / "<TICKER>.parquet"``. The file is not guaranteed to exist.
    """
    return Path(cache_dir) / f"{ticker.upper()}.parquet"


def clear_cache(cache_dir: Path = CACHE_DIR) -> None:
    """Delete all cached per-ticker parquet files.

    Removes every ``*.parquet`` file directly under ``cache_dir``. The
    directory itself is left in place. Missing directories are a no-op, so this
    is safe to call before the cache has ever been populated.

    Args:
        cache_dir: Directory holding the per-ticker parquet cache.
    """
    cache_dir = Path(cache_dir)
    if not cache_dir.exists():
        return
    for parquet_file in cache_dir.glob("*.parquet"):
        parquet_file.unlink()


def _flatten_columns(frame: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Collapse yfinance ``MultiIndex`` columns down to the OHLCV field level.

    yfinance may nest columns as ``(field, ticker)`` (1.x default, also for a
    single ticker) or ``(ticker, field)`` (``group_by="ticker"``). The level
    holding the ticker is sliced away; if the ticker is not found in any level
    the level containing recognisable OHLCV field names is kept.

    Args:
        frame: Raw frame with ``MultiIndex`` columns.
        ticker: Ticker symbol to slice out.

    Returns:
        A frame with flat, single-level columns.
    """
    import pandas as pd

    if not isinstance(frame.columns, pd.MultiIndex):
        return frame

    upper_ticker = ticker.upper()
    for level in range(frame.columns.nlevels):
        labels = [str(v).upper() for v in frame.columns.get_level_values(level)]
        if upper_ticker in labels:
            key = frame.columns.get_level_values(level)[labels.index(upper_ticker)]
            frame = frame.xs(key, axis=1, level=level, drop_level=True)
            break
    else:
        level_to_keep = 0
        for level in range(frame.columns.nlevels):
            fields = {str(v).lower() for v in frame.columns.get_level_values(level)}
            if fields & set(_FIELD_MAP):
                level_to_keep = level
                break
        frame.columns = frame.columns.get_level_values(level_to_keep)

    if isinstance(frame.columns, pd.MultiIndex):  # more than two levels: flatten
        frame.columns = frame.columns.get_level_values(0)
    return frame


def _normalize_yf_frame(raw: pd.DataFrame | None, ticker: str) -> pd.DataFrame:
    """Normalise a raw yfinance frame into canonical long form.

    Handles the column-shape variations yfinance produces:

    * flat columns (legacy single-ticker download),
    * ``MultiIndex`` columns keyed by ``(field, ticker)`` or ``(ticker, field)``
      (yfinance 1.x, even for one ticker),
    * presence or absence of an ``Adj Close`` column (when absent it is
      back-filled from ``Close``),
    * a ``Date``/``Datetime`` index that may be timezone-aware.

    Args:
        raw: DataFrame as returned by ``yfinance.download`` or
            ``Ticker.history`` for a single ticker, indexed by date.
        ticker: The ticker symbol this frame belongs to (used to fill the
            ``ticker`` column and to pick the right slice out of ``MultiIndex``
            columns).

    Returns:
        Long-form frame with columns :data:`LONG_COLUMNS`, sorted by date,
        duplicate dates dropped, rows without a close price removed. Empty
        (with those columns) if ``raw`` is ``None`` or empty.
    """
    import pandas as pd

    if raw is None or len(raw) == 0:
        return pd.DataFrame(columns=LONG_COLUMNS)

    frame = _flatten_columns(raw.copy(), ticker)

    # Reset the date index into a column.
    frame = frame.reset_index()

    # Rename columns to canonical snake_case where recognised.
    rename: dict[str, str] = {}
    for col in frame.columns:
        key = str(col).strip().lower()
        if key in ("date", "datetime", "index"):
            rename[col] = "date"
        elif key in _FIELD_MAP:
            rename[col] = _FIELD_MAP[key]
    frame = frame.rename(columns=rename)
    # Drop duplicate column labels (e.g. both "Adj Close" and "adj_close").
    frame = frame.loc[:, ~frame.columns.duplicated()]

    if "date" not in frame.columns:
        raise ValueError(
            f"yfinance frame for {ticker!r} has no date index/column: {list(frame.columns)!r}"
        )

    # Guarantee every expected OHLCV column exists.
    if "adj_close" not in frame.columns and "close" in frame.columns:
        frame["adj_close"] = frame["close"]
    for col in _PRICE_COLUMNS:
        if col not in frame.columns:
            frame[col] = float("nan")
        frame[col] = pd.to_numeric(frame[col], errors="coerce")

    frame["ticker"] = ticker.upper()
    dates = pd.to_datetime(frame["date"], errors="coerce")
    if getattr(dates.dt, "tz", None) is not None:
        dates = dates.dt.tz_localize(None)
    frame["date"] = dates.dt.date

    frame = frame[LONG_COLUMNS]
    frame.columns.name = None  # drop the "Price" axis name yfinance leaves behind
    frame = frame.dropna(subset=["date", "close"])
    frame = frame.drop_duplicates(subset=["date"], keep="last")
    return frame.sort_values("date").reset_index(drop=True)


def _fetch_from_yfinance(
    ticker: str,
    *,
    start: date = HISTORY_START,
    retries: int = 3,
    pause: float = 1.5,
) -> pd.DataFrame:
    """Download daily history for a ticker via yfinance, with retries.

    ``yfinance`` is imported here (not at module level) so that importing this
    module never requires the package or network access. The download always
    passes an explicit ``start`` (``period="max"`` is unreliable on yfinance
    1.x). If ``yf.download`` returns nothing, ``Ticker.history`` is tried as a
    fallback before the attempt counts as failed.

    Args:
        ticker: Ticker symbol to download.
        start: First calendar date to request (default
            :data:`advisor.config.HISTORY_START`).
        retries: Total attempts before giving up.
        pause: Base seconds slept between attempts (linear back-off).

    Returns:
        Normalised long-form history for ``ticker`` (possibly empty if the
        symbol returned no data on every attempt).

    Raises:
        RuntimeError: If every attempt raised an exception (network down,
            rate-limited, ...). An empty-but-successful download does not
            raise; it returns an empty frame.
    """
    import yfinance as yf  # lazy import: keeps module import offline-safe

    last_exc: Exception | None = None
    for attempt in range(1, max(1, retries) + 1):
        try:
            raw = yf.download(
                ticker,
                start=start.isoformat(),
                interval="1d",
                auto_adjust=False,
                actions=False,
                progress=False,
                threads=False,
            )
            frame = _normalize_yf_frame(raw, ticker)
            if not frame.empty:
                return frame
            history = yf.Ticker(ticker).history(
                start=start.isoformat(),
                interval="1d",
                auto_adjust=False,
                actions=False,
            )
            frame = _normalize_yf_frame(history, ticker)
            if not frame.empty:
                return frame
            last_exc = None
            _log.warning(
                "yfinance returned no rows for %s (attempt %d/%d)", ticker, attempt, retries
            )
        except AssertionError:  # test booby-traps must surface immediately
            raise
        except Exception as exc:  # noqa: BLE001 - network/library errors are retried
            last_exc = exc
            _log.warning("yfinance error for %s (attempt %d/%d): %s", ticker, attempt, retries, exc)
        if attempt < retries and pause > 0:
            time.sleep(pause * attempt)

    if last_exc is not None:
        raise RuntimeError(
            f"yfinance download failed for {ticker!r} after {retries} attempt(s): {last_exc}"
        ) from last_exc
    import pandas as pd

    return pd.DataFrame(columns=LONG_COLUMNS)


def _load_one_ticker(
    ticker: str,
    start: date,
    end: date,
    *,
    use_cache: bool,
    cache_dir: Path,
) -> pd.DataFrame:
    """Load a single ticker's long-form history, using/refreshing the cache.

    On a cache hit (and ``use_cache``) the cached parquet is read; otherwise
    yfinance is queried and the full history is written to the cache. The
    returned frame is always sliced to the ``[start, end)`` window.

    Args:
        ticker: Ticker symbol.
        start: Inclusive start date of the window to return.
        end: Exclusive end date of the window to return.
        use_cache: When ``True``, read from and write to the parquet cache.
            When ``False``, always fetch fresh and skip writing.
        cache_dir: Directory holding the per-ticker parquet cache.

    Returns:
        Long-form rows for ``ticker`` within ``[start, end)``.
    """
    import pandas as pd

    path = _cache_path(ticker, cache_dir)
    frame: pd.DataFrame | None = None

    if use_cache and path.exists():
        frame = pd.read_parquet(path)
        # Stored dates may round-trip as Timestamp; normalise back to date.
        frame["date"] = pd.to_datetime(frame["date"]).dt.date

    if frame is None or frame.empty:
        frame = _fetch_from_yfinance(ticker)
        if use_cache and not frame.empty:
            path.parent.mkdir(parents=True, exist_ok=True)
            frame.to_parquet(path, index=False)

    mask = (frame["date"] >= start) & (frame["date"] < end)
    return frame.loc[mask].reset_index(drop=True)


def load_prices(
    tickers: list[str] | str,
    start: date,
    end: date,
    *,
    use_cache: bool = True,
    cache_dir: Path = CACHE_DIR,
) -> pd.DataFrame:
    """Load long-form daily OHLCV price history for one or more tickers.

    Args:
        tickers: A single ticker string (e.g. ``"AAPL"``) or a list of tickers.
            Order is not significant; the output is sorted by ``(ticker, date)``.
        start: Inclusive start date of the window to return.
        end: Exclusive end date of the window to return.
        use_cache: When ``True`` (default), read each ticker from its parquet
            cache when present and fetch + cache on a miss. When ``False``,
            always fetch fresh from yfinance and do not touch the cache.
        cache_dir: Directory under which per-ticker parquet files are stored.
            Defaults to :data:`advisor.config.CACHE_DIR`.

    Returns:
        Long-form frame with columns
        ``[date, ticker, open, high, low, close, adj_close, volume]``, sorted
        by ``ticker`` then ``date``, with ``date`` holding
        :class:`datetime.date` objects. Empty (with those columns) if no data
        is available for any requested ticker in the window.

    Note:
        ``yfinance`` is imported lazily on cache misses only, so importing this
        module and reading from a populated cache never require network access.
    """
    import pandas as pd

    if isinstance(tickers, str):
        tickers = [tickers]
    # De-duplicate while preserving a deterministic order.
    requested = sorted({t.upper() for t in tickers})

    frames: list[pd.DataFrame] = []
    for ticker in requested:
        one = _load_one_ticker(
            ticker,
            start,
            end,
            use_cache=use_cache,
            cache_dir=Path(cache_dir),
        )
        if not one.empty:
            frames.append(one)

    if not frames:
        return pd.DataFrame(columns=LONG_COLUMNS)

    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values(["ticker", "date"]).reset_index(drop=True)
    return out[LONG_COLUMNS]
