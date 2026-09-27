"""Is one strategy's Sharpe ratio really higher than another's?

**Status: post-hoc.** Not one of the pre-registered hypotheses; the report
labels these numbers as a robustness check on H1 and H3.

Two strategies traded over the same days have correlated returns, so their
Sharpe ratios cannot be compared with two independent confidence intervals.
Following the idea behind Ledoit and Wolf (2008), this module resamples the
*pairs* of daily returns together, in blocks so that short-term dependence
(volatility clustering, overlapping positions) is kept, and recomputes the
Sharpe difference on every resample. The spread of those differences gives a
confidence interval and a two-sided bootstrap p-value for "no difference".

Outputs: ``results/sharpe_differences.csv`` (one row per comparison) and
``results/sharpe_by_year.csv`` (each strategy's Sharpe ratio per calendar year).

Example:
    ``python -m advisor.evaluation.sharpe_difference``

Reference:
    Ledoit, O. & Wolf, M. (2008). Robust performance hypothesis testing with
    the Sharpe ratio. *Journal of Empirical Finance*, 15(5), 850-859.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from advisor.config import RANDOM_SEED
from advisor.evaluation.metrics import sharpe_ratio

__all__ = [
    "block_bootstrap_indices",
    "sharpe_difference_test",
    "sharpe_by_year",
    "load_daily_returns",
    "main",
]

log = logging.getLogger(__name__)

#: The comparisons reported in Chapter 5: each advisor against the rule
#: (H3) and against buy-and-hold (H1), plus the two models against each other.
COMPARISONS: tuple[tuple[str, str], ...] = (
    ("qwen", "heuristic"),
    ("llama", "heuristic"),
    ("qwen", "llama"),
    ("qwen", "buy_and_hold"),
    ("llama", "buy_and_hold"),
    ("markowitz_mean_variance", "buy_and_hold"),
)


def block_bootstrap_indices(
    n: int, block_length: int, rng: np.random.Generator
) -> np.ndarray:
    """Row indices for one circular block-bootstrap resample of length ``n``.

    Blocks of ``block_length`` consecutive days start at random positions and
    wrap around the end of the sample, so every day is equally likely to be
    drawn (the circular block bootstrap of Politis and Romano).

    Raises:
        ValueError: If ``n`` or ``block_length`` is below 1.
    """
    if n < 1 or block_length < 1:
        raise ValueError("n and block_length must both be at least 1.")
    n_blocks = -(-n // block_length)  # ceiling division
    starts = rng.integers(0, n, size=n_blocks)
    offsets = np.arange(block_length)
    return ((starts[:, None] + offsets[None, :]) % n).ravel()[:n]


def sharpe_difference_test(
    returns_a: Sequence[float] | np.ndarray,
    returns_b: Sequence[float] | np.ndarray,
    *,
    n_resamples: int = 10_000,
    block_length: int = 10,
    seed: int = RANDOM_SEED,
) -> dict[str, float]:
    """Bootstrap the difference in annualised Sharpe ratio, A minus B.

    Args:
        returns_a: Daily returns of strategy A.
        returns_b: Daily returns of strategy B on the same days.
        n_resamples: Number of bootstrap resamples.
        block_length: Days per block (about two trading weeks by default).
        seed: Random seed, so the interval is reproducible.

    Returns:
        ``sharpe_a``, ``sharpe_b``, ``difference`` and the 95% interval
        ``ci_low``/``ci_high``, plus ``p_value``: the two-sided share of
        resampled differences on the other side of zero (doubled, capped at 1).

    Raises:
        ValueError: If the two series have different lengths.
    """
    a = np.asarray(returns_a, dtype=float)
    b = np.asarray(returns_b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("Both return series must cover the same days.")

    observed = sharpe_ratio(a) - sharpe_ratio(b)
    rng = np.random.default_rng(seed)
    differences = np.empty(n_resamples)
    for k in range(n_resamples):
        idx = block_bootstrap_indices(len(a), block_length, rng)
        differences[k] = sharpe_ratio(a[idx]) - sharpe_ratio(b[idx])

    below = np.mean(differences <= 0)
    above = np.mean(differences >= 0)
    return {
        "sharpe_a": sharpe_ratio(a),
        "sharpe_b": sharpe_ratio(b),
        "difference": observed,
        "ci_low": float(np.percentile(differences, 2.5)),
        "ci_high": float(np.percentile(differences, 97.5)),
        "p_value": float(min(1.0, 2 * min(below, above))),
    }


def load_daily_returns(results_dir: str | Path) -> pd.DataFrame:
    """Daily returns of every advisor and baseline, aligned on the same dates."""
    root = Path(results_dir)
    columns = {}
    for tag in ("llama", "qwen", "heuristic"):
        path = root / f"{tag}_equity.csv"
        if path.exists():
            equity = pd.read_csv(path, index_col="date", parse_dates=True)["equity"]
            columns[tag] = equity.pct_change()
    baselines = pd.read_csv(root / "baselines_equity.csv", index_col="date", parse_dates=True)
    for name in ("buy_and_hold", "momentum_12_1", "markowitz_mean_variance"):
        columns[name] = baselines[name].pct_change()
    return pd.DataFrame(columns).dropna()


def sharpe_by_year(returns: pd.DataFrame) -> pd.DataFrame:
    """Annualised Sharpe ratio of every strategy within each calendar year.

    A quick check of whether the ranking of strategies is stable over time,
    which a single held-out window cannot show on its own.
    """
    table = {}
    for year, part in returns.groupby(returns.index.year):
        table[int(year)] = {name: sharpe_ratio(part[name]) for name in returns.columns}
    return pd.DataFrame(table)


def main(argv: Sequence[str] | None = None) -> int:
    """Run every comparison in :data:`COMPARISONS` and write the CSV."""
    parser = argparse.ArgumentParser(
        prog="python -m advisor.evaluation.sharpe_difference",
        description="POST-HOC: block-bootstrap confidence intervals for Sharpe differences.",
    )
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--resamples", type=int, default=10_000)
    parser.add_argument("--block-length", type=int, default=10)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    returns = load_daily_returns(args.results_dir)
    rows = []
    for a, b in COMPARISONS:
        result = sharpe_difference_test(returns[a], returns[b], n_resamples=args.resamples,
                                        block_length=args.block_length)
        rows.append({"strategy_a": a, "strategy_b": b, **result})
        log.info("%s vs %s: diff %.2f, 95%% CI [%.2f, %.2f], p = %.2f", a, b,
                 result["difference"], result["ci_low"], result["ci_high"], result["p_value"])

    table = pd.DataFrame(rows)
    out = Path(args.results_dir) / "sharpe_differences.csv"
    table.to_csv(out, index=False)
    print(table.round(3).to_string(index=False))

    by_year = sharpe_by_year(returns)
    by_year.to_csv(Path(args.results_dir) / "sharpe_by_year.csv")
    print("\nSharpe ratio by calendar year:")
    print(by_year.round(2).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
