"""CLI to warm the local price cache for the advisor universe.

This script calls :func:`advisor.data.loader.load_prices` one ticker at a time
for every symbol in ``advisor.data.universe.UNIVERSE`` plus the ``SPY``
benchmark (or a user-supplied subset) over a configured date window,
populating the per-ticker parquet cache under ``CACHE_DIR``. Once warmed, the
rest of the system (features, backtest, Streamlit UI) runs fully offline.

Tickers are fetched sequentially with a short pause between network fetches
and a small number of retries per ticker, so that Yahoo Finance rate limits
(which yfinance surfaces as empty "possibly delisted" responses) do not abort
the whole run. Failures are collected and reported at the end rather than
raised part-way through.

Examples:
    Warm the full universe (+SPY) for the default train+test span::

        python scripts/fetch_data.py

    Warm just two tickers for a custom window::

        python scripts/fetch_data.py --tickers AAPL MSFT --start 2024-01-01 --end 2025-01-01

    Force a re-fetch, ignoring any existing cache::

        python scripts/fetch_data.py --no-cache

This module performs network I/O (via yfinance inside ``load_prices``) only when
explicitly run; importing it has no side effects.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import date
from pathlib import Path

# Make ``advisor`` importable when run as ``python scripts/fetch_data.py``
# (Python puts scripts/ on sys.path, not the code root).
_CODE_ROOT = Path(__file__).resolve().parent.parent
if str(_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODE_ROOT))

from advisor.config import CACHE_DIR, TEST_END, TRAIN_START  # noqa: E402
from advisor.data.loader import _cache_path, load_prices  # noqa: E402
from advisor.data.universe import UNIVERSE  # noqa: E402

#: Market benchmark cached alongside the universe (buy-and-hold S&P 500 baseline).
BENCHMARK_TICKER: str = "SPY"


def _parse_date(value: str) -> date:
    """Parse an ISO ``YYYY-MM-DD`` string into a :class:`datetime.date`.

    Args:
        value: The date string from the command line.

    Returns:
        The parsed date.

    Raises:
        argparse.ArgumentTypeError: If ``value`` is not a valid ISO date.
    """
    try:
        return date.fromisoformat(value)
    except ValueError as exc:  # pragma: no cover - argparse surfaces this
        raise argparse.ArgumentTypeError(
            f"Invalid date {value!r}; expected ISO format YYYY-MM-DD."
        ) from exc


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser for the data-fetch CLI.

    Returns:
        Parser exposing ``--tickers``, ``--start``, ``--end``, ``--no-cache``,
        ``--cache-dir``, ``--pause`` and ``--retries``.
    """
    parser = argparse.ArgumentParser(
        prog="fetch_data",
        description=(
            "Warm the local price cache for the financial advisor bot by "
            "fetching OHLCV data over the configured window."
        ),
    )
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=None,
        metavar="TICKER",
        help=f"Subset of tickers to fetch. Defaults to the full UNIVERSE plus {BENCHMARK_TICKER}.",
    )
    parser.add_argument(
        "--start",
        type=_parse_date,
        default=TRAIN_START,
        help=f"Start date (YYYY-MM-DD). Default: {TRAIN_START.isoformat()}.",
    )
    parser.add_argument(
        "--end",
        type=_parse_date,
        default=TEST_END,
        help=f"End date (YYYY-MM-DD, exclusive). Default: {TEST_END.isoformat()}.",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Bypass the cache and force a fresh fetch for every ticker.",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=str(CACHE_DIR),
        help=f"Directory for per-ticker parquet caches. Default: {CACHE_DIR}.",
    )
    parser.add_argument(
        "--pause",
        type=float,
        default=1.0,
        help="Seconds to sleep between network fetches (rate-limit courtesy). Default: 1.0.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Attempts per ticker before it is reported as failed. Default: 3.",
    )
    return parser


def fetch_one(
    ticker: str,
    start: date,
    end: date,
    *,
    use_cache: bool,
    cache_dir: Path,
    retries: int,
    pause: float,
) -> tuple[int, str | None]:
    """Fetch (or read from cache) one ticker, retrying on failure.

    Args:
        ticker: Ticker symbol.
        start: Inclusive start of the window to report on.
        end: Exclusive end of the window to report on.
        use_cache: Forwarded to :func:`load_prices`.
        cache_dir: Forwarded to :func:`load_prices`.
        retries: Total attempts before giving up.
        pause: Base seconds slept between attempts (linear back-off).

    Returns:
        ``(rows, error)`` where ``rows`` is the number of rows returned inside
        the window and ``error`` is ``None`` on success or a short message.
    """
    error: str | None = None
    for attempt in range(1, max(1, retries) + 1):
        try:
            frame = load_prices([ticker], start, end, use_cache=use_cache, cache_dir=cache_dir)
        except Exception as exc:  # noqa: BLE001 - reported, not raised
            error = f"{type(exc).__name__}: {exc}"
        else:
            if not frame.empty:
                return len(frame), None
            error = "no rows returned"
        if attempt < retries and pause > 0:
            time.sleep(pause * attempt)
    return 0, error


def main(argv: list[str] | None = None) -> int:
    """Entry point: fetch prices for the requested tickers and report status.

    Args:
        argv: Optional argument vector (defaults to ``sys.argv[1:]``). Provided
            for testability.

    Returns:
        Process exit code (``0`` when every ticker was cached, ``1`` if any
        ticker failed).
    """
    args = build_parser().parse_args(argv)
    tickers = [t.upper() for t in args.tickers] if args.tickers else [*UNIVERSE, BENCHMARK_TICKER]
    tickers = list(dict.fromkeys(tickers))  # de-duplicate, keep order
    cache_dir = Path(args.cache_dir)
    use_cache = not args.no_cache

    print(
        f"Fetching {len(tickers)} ticker(s) from {args.start.isoformat()} "
        f"to {args.end.isoformat()} (use_cache={use_cache}) into {cache_dir} ..."
    )

    started = time.time()
    total_rows = 0
    failures: dict[str, str] = {}
    for index, ticker in enumerate(tickers, start=1):
        was_cached = use_cache and _cache_path(ticker, cache_dir).exists()
        rows, error = fetch_one(
            ticker,
            args.start,
            args.end,
            use_cache=use_cache,
            cache_dir=cache_dir,
            retries=args.retries,
            pause=args.pause,
        )
        if error is None:
            total_rows += rows
            source = "cache" if was_cached else "yfinance"
            print(f"[{index:2d}/{len(tickers)}] {ticker:<6} {rows:5d} rows ({source})")
        else:
            failures[ticker] = error
            print(f"[{index:2d}/{len(tickers)}] {ticker:<6} FAILED: {error}", file=sys.stderr)
        if not was_cached and index < len(tickers) and args.pause > 0:
            time.sleep(args.pause)

    elapsed = time.time() - started
    ok = len(tickers) - len(failures)
    print(
        f"Done in {elapsed:.0f}s. Cached {total_rows} rows across {ok}/{len(tickers)} "
        f"ticker(s) in {cache_dir}."
    )
    if failures:
        print("Failed tickers: " + ", ".join(f"{t} ({e})" for t, e in failures.items()))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
