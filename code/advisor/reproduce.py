"""Offline-safe, end-to-end reproducibility demo for the Financial Advisor Bot.

This module provides a tiny, fully deterministic demonstration of the advisor
pipeline that runs with **no network access** and **no LLM call**. Its purpose is
to give graders and CI a single command that exercises the core data ->
features -> recommendation path and prints a JSON recommendation::

    python -m advisor.reproduce

Design notes:
    * If a cached parquet file for the default ticker already exists under
      ``CACHE_DIR`` (see ``scripts/fetch_data.py``), it is loaded and used.
      Otherwise a small synthetic OHLCV frame is generated from a fixed random
      seed so the output is reproducible byte-for-byte.
    * The recommendation comes from the transparent rule-based
      :class:`~advisor.recommender.heuristic.HeuristicAdvisor` (the "no AI"
      baseline of the evaluation), so the demo never touches Ollama. The
      LLM-backed path lives in :mod:`advisor.recommender.llm` and is exercised
      by the Streamlit UI and the evaluation runner.
    * If the evaluation has been run and ``results/summary_table.csv`` exists,
      the backtest summary table is printed after the recommendation so the
      headline numbers of the report can be reproduced from one command.
"""

from __future__ import annotations

import json
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from advisor.config import CACHE_DIR, RANDOM_SEED, RESULTS_DIR
from advisor.features.indicators import compute_features
from advisor.recommender.heuristic import HeuristicAdvisor
from advisor.recommender.prompts import select_features
from advisor.recommender.schema import Recommendation

__all__ = ["main", "demo_recommendation"]

# A reasonable default ticker to look for in the cache before falling back to
# synthetic data. Kept local so reproduce.py has no hard dependency on the
# universe module ordering.
_DEFAULT_TICKER: str = "AAPL"

# Number of synthetic trading days. Must comfortably exceed the longest
# look-back window used by compute_features (sma_50) so features are non-NaN.
_N_SYNTHETIC_DAYS: int = 120

# Feature columns that must be present (non-NaN) on the decision day.
_REQUIRED: tuple[str, ...] = ("sma_10", "sma_50", "mom_10", "vol_20", "rsi_14", "ret_1d")


def _synthetic_prices(
    ticker: str = _DEFAULT_TICKER,
    *,
    n_days: int = _N_SYNTHETIC_DAYS,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Build a small, deterministic single-ticker OHLCV DataFrame.

    A geometric-random-walk close series is generated from a fixed seed so the
    demo is byte-for-byte reproducible across runs and machines. Open/high/low
    and volume are derived from the close so the frame matches the OHLCV schema
    expected by :func:`advisor.features.indicators.compute_features`.

    Args:
        ticker: Ticker symbol to stamp on every row.
        n_days: Number of synthetic trading days to generate.
        seed: Seed for the local NumPy generator (no global state mutated).

    Returns:
        Long-form frame with columns
        ``[date, ticker, open, high, low, close, adj_close, volume]``, sorted
        ascending by ``date``.
    """
    rng = np.random.default_rng(seed)

    # Business-day index ending "today-ish"; exact dates are irrelevant for the
    # smoke test but we keep them plausible and monotonic.
    end = date(2026, 6, 1)
    start = end - timedelta(days=int(n_days * 1.6))  # pad for weekend skips
    dates = pd.bdate_range(start=start, end=end)[-n_days:]

    # Geometric random walk for the close.
    daily_ret = rng.normal(loc=0.0005, scale=0.012, size=len(dates))
    close = 100.0 * np.exp(np.cumsum(daily_ret))

    # Derive a plausible OHLC envelope around the close.
    intraday = np.abs(rng.normal(loc=0.0, scale=0.006, size=len(dates)))
    open_ = close * (1.0 - rng.normal(loc=0.0, scale=0.004, size=len(dates)))
    high = np.maximum(open_, close) * (1.0 + intraday)
    low = np.minimum(open_, close) * (1.0 - intraday)
    volume = rng.integers(low=1_000_000, high=5_000_000, size=len(dates))

    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(dates).date,
            "ticker": ticker,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "adj_close": close,
            "volume": volume,
        }
    )
    return frame.sort_values("date").reset_index(drop=True)


def _load_or_synthesize(cache_dir: Path = CACHE_DIR) -> tuple[pd.DataFrame, str]:
    """Return a single-ticker OHLCV frame, preferring cached data if present.

    Looks for a cached parquet for ``_DEFAULT_TICKER``. If found and readable it
    is used (proving the cache path works end-to-end); otherwise a synthetic
    frame is generated. Either way the function performs **no network I/O**.

    Args:
        cache_dir: Directory holding per-ticker parquet caches (see
            :func:`advisor.data.loader._cache_path`).

    Returns:
        ``(frame, source)`` where ``source`` is ``"cache"`` or ``"synthetic"``.
    """
    cache_path = cache_dir / f"{_DEFAULT_TICKER}.parquet"
    if cache_path.exists():
        try:
            cached = pd.read_parquet(cache_path)
            if not cached.empty and "close" in cached.columns:
                cached["date"] = pd.to_datetime(cached["date"]).dt.date
                cached = cached.sort_values("date").reset_index(drop=True)
                if "ticker" not in cached.columns:
                    cached["ticker"] = _DEFAULT_TICKER
                return cached, "cache"
        except Exception:  # pragma: no cover - corrupt cache -> fall back
            pass
    return _synthetic_prices(), "synthetic"


def demo_recommendation(cache_dir: Path = CACHE_DIR) -> tuple[Recommendation, str]:
    """Run the offline pipeline once and return the rule-based recommendation.

    Args:
        cache_dir: Directory searched for a cached ``AAPL.parquet``.

    Returns:
        ``(recommendation, source)`` where ``source`` says whether cached or
        synthetic prices were used.
    """
    prices, source = _load_or_synthesize(cache_dir)
    ticker = str(prices["ticker"].iloc[0]) if "ticker" in prices.columns else _DEFAULT_TICKER

    feats = compute_features(prices)
    latest = feats.dropna(subset=list(_REQUIRED)).iloc[-1]
    as_of = latest["date"]
    if isinstance(as_of, pd.Timestamp):
        as_of = as_of.date()
    elif not isinstance(as_of, date):
        as_of = date.fromisoformat(str(as_of)[:10])

    rec = HeuristicAdvisor().recommend(ticker, select_features(latest), as_of=as_of)
    return rec, source


def _print_summary_table(results_dir: Path = RESULTS_DIR) -> bool:
    """Print ``results/summary_table.csv`` if the evaluation has produced it.

    Args:
        results_dir: Directory the evaluation runner writes to.

    Returns:
        ``True`` if a table was printed, ``False`` if none exists yet.
    """
    path = results_dir / "summary_table.csv"
    if not path.exists():
        return False
    table = pd.read_csv(path)
    print(f"\n# Backtest summary ({path.relative_to(results_dir.parent)}):")
    with pd.option_context("display.width", 200, "display.max_columns", 50):
        print(table.to_string(index=False))
    return True


def main() -> None:
    """Run the offline demo and print a JSON recommendation to stdout.

    Steps:
      1. Load the cached AAPL frame if available, else synthesize one.
      2. Compute technical features (no look-ahead).
      3. Ask the rule-based advisor about the latest feature row.
      4. Print the recommendation as pretty JSON, then the backtest summary
         table if ``results/summary_table.csv`` exists.

    The JSON is written to stdout first so CI can parse it; provenance notes go
    to stderr.
    """
    rec, source = demo_recommendation()
    print(
        f"# reproduce: {rec.ticker} as of {rec.as_of.isoformat()} from {source} prices "
        "(rule-based advisor, no network, no LLM)",
        file=sys.stderr,
    )
    print(json.dumps(rec.to_dict(), indent=2, sort_keys=True))
    if not _print_summary_table():
        print(
            "\n# No backtest results yet: run "
            "`python -m advisor.evaluation.run_evaluation --advisors llama qwen heuristic` "
            "to produce results/summary_table.csv.",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
