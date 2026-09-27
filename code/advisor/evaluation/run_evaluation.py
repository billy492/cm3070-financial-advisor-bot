"""Experiment runner: walk-forward evaluation of advisors against the baselines.

Runs the complete evaluation promised in the preliminary report (§3.4,
Table 1) and ADR-0002: each advisor is backtested walk-forward over the
held-out test window with :class:`~advisor.evaluation.backtest.WalkForwardBacktest`,
the four baselines (buy-and-hold S&P 500, random allocation, 12-1 momentum,
Markowitz mean-variance) and the random null ensemble are simulated under
the same cost model, and everything is written to a results directory as
CSV/JSON tables plus PNG+SVG figures.

Command line::

    python -m advisor.evaluation.run_evaluation \
        --advisors llama qwen heuristic --tickers ALL --freq W-FRI \
        --results-dir results/ --cost-bps 10 --seeds 100

Outputs (under ``--results-dir``):

* ``<tag>_recommendations.csv`` -- every recommendation with its label;
* ``<tag>_equity.csv`` -- equity, daily return and drawdown per day;
* ``<tag>_monthly.csv`` / ``<tag>_trades.csv`` -- per-month metrics, trades;
* ``calibration_<tag>.json`` -- ECE raw vs calibrated, fitted temperatures;
* ``baselines_equity.csv`` -- the four baselines and the random band;
* ``random_null.csv`` -- terminal statistics of every random-ensemble member;
* ``summary_table.csv`` -- one row per strategy with Sharpe, Deflated Sharpe
  (``n_trials`` = number of strategies compared, Bailey & Lopez de Prado
  2014), Sortino, max drawdown, Calmar and a permutation p-value against the
  random null;
* ``ablation.csv`` -- the LLM-ablation comparison of the advisors;
* ``figures/`` -- equity curves, drawdowns, monthly-return heatmap and a
  reliability diagram per advisor (colour-blind-safe Okabe-Ito palette);
* ``run_manifest.json`` -- timestamp, git hash, configuration, package
  versions and seed, for reproducibility.

Advisor tags: ``llama`` and ``qwen`` map to :class:`OllamaAdvisor` with the
models in :data:`advisor.config.ADVISOR_MODELS` (wrapped in
``CachedAdvisor`` so LLM answers are memoised in
``<results-dir>/llm_cache_<tag>.jsonl`` and a run can be resumed);
``heuristic`` is the no-LLM ablation arm. With ``--resume`` an advisor whose
result files already exist is loaded from disk instead of being re-run. All
imports of the recommender package are lazy, so this module imports without
an LLM installed.

References:
    Bailey, D. H. & Lopez de Prado, M. (2014). The Deflated Sharpe Ratio.
        *Journal of Portfolio Management*, 40(5), 94-107.
    Arnott, R., Harvey, C. R. & Markowitz, H. (2019). A Backtesting Protocol
        in the Era of Machine Learning. *Journal of Financial Data Science*.
"""

from __future__ import annotations

import argparse
import functools
import json
import logging
import platform
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from advisor import config as _config
from advisor.config import (
    CACHE_DIR,
    PROJECT_ROOT,
    RANDOM_SEED,
    RESULTS_DIR,
    TEST_END,
    TEST_START,
    TRAIN_START,
)
from advisor.data.universe import UNIVERSE
from advisor.evaluation.backtest import (
    WalkForwardBacktest,
    ablation_table,
    evaluate_calibration,
)
from advisor.evaluation.baselines import (
    BASELINE_NAMES,
    permutation_p_value,
    run_all_baselines,
)
from advisor.evaluation.metrics import sharpe_ratio
from advisor.evaluation.portfolio import drawdown_series, period_returns, summarise

__all__ = [
    "ADVISOR_MODELS",
    "BENCHMARK_TICKER",
    "OKABE_ITO",
    "STRATEGY_LABELS",
    "parse_tickers",
    "build_advisor",
    "wrap_cached",
    "make_progress_logger",
    "run_advisor",
    "advisor_result_paths",
    "save_advisor_results",
    "load_advisor_results",
    "build_summary_table",
    "build_baselines_equity",
    "build_random_null_table",
    "make_figures",
    "package_versions",
    "git_hash",
    "write_manifest",
    "build_parser",
    "main",
]

log = logging.getLogger(__name__)

#: Advisor tag -> Ollama model (from config when present).
ADVISOR_MODELS: dict[str, str] = dict(
    getattr(_config, "ADVISOR_MODELS", {"llama": "llama3.1:8b", "qwen": "qwen3:8b"})
)
#: The no-LLM ablation arm.
HEURISTIC_TAG: str = "heuristic"
#: Benchmark ticker for the buy-and-hold S&P 500 baseline.
BENCHMARK_TICKER: str = "SPY"
#: Okabe & Ito (2008) colour-blind-safe qualitative palette.
OKABE_ITO: tuple[str, ...] = (
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#009E73",  # bluish green
    "#E69F00",  # orange
    "#CC79A7",  # reddish purple
    "#56B4E9",  # sky blue
    "#F0E442",  # yellow
    "#000000",  # black
)
#: Human-readable strategy labels for figures.
STRATEGY_LABELS: dict[str, str] = {
    "buy_and_hold": "Buy & hold (S&P 500 / SPY)",
    "random_allocation": "Random allocation",
    "momentum_12_1": "12-1 momentum",
    "markowitz_mean_variance": "Markowitz mean-variance",
    "random_ensemble_mean": "Random ensemble (mean)",
    "heuristic": "Heuristic (no LLM)",
    "llama": "LLM advisor (Llama 3.1 8B)",
    "qwen": "LLM advisor (Qwen3 8B)",
}
_RANDOM_MEAN: str = "random_ensemble_mean"
_SUMMARY_COLUMNS: tuple[str, ...] = (
    "kind",
    "total_return",
    "cagr",
    "annualised_vol",
    "sharpe",
    "deflated_sharpe",
    "sortino",
    "max_drawdown",
    "calmar",
    "p_value_vs_random",
    "n_days",
    "n_trials",
    "initial_equity",
    "final_equity",
)

AdvisorFactory = Callable[[str], Any]


# --------------------------------------------------------------------------- #
# Small pure helpers
# --------------------------------------------------------------------------- #
def parse_tickers(
    spec: str | Sequence[str] | None, universe: Sequence[str] = UNIVERSE
) -> list[str]:
    """Resolve the ``--tickers`` argument into a list of symbols.

    Args:
        spec: ``None`` or ``"ALL"`` for the whole universe; an integer string
            ``"N"`` for the first ``N`` symbols of the (sorted) universe; or
            a comma/space-separated list of explicit symbols.
        universe: The reference universe (default :data:`UNIVERSE`).

    Returns:
        Upper-cased, de-duplicated symbols in a deterministic order.

    Raises:
        ValueError: For ``N < 1`` or an empty explicit list.
    """
    if spec is None:
        return list(universe)
    if not isinstance(spec, str):
        symbols = [str(s).strip().upper() for s in spec if str(s).strip()]
    else:
        text = spec.strip()
        if text.upper() == "ALL":
            return list(universe)
        if text.isdigit():
            n = int(text)
            if n < 1:
                raise ValueError("--tickers N must be >= 1.")
            return list(universe[:n])
        symbols = [s.strip().upper() for s in text.replace(",", " ").split() if s.strip()]
    if not symbols:
        raise ValueError("--tickers resolved to an empty list.")
    return sorted(set(symbols))


def _parse_date(value: str) -> date:
    """Parse ``YYYY-MM-DD`` for argparse."""
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid date {value!r}; expected YYYY-MM-DD.") from exc


def wrap_cached(advisor: Any, cache_path: Path) -> Any:
    """Wrap an advisor in ``CachedAdvisor`` when the recommender provides one.

    Args:
        advisor: The advisor to memoise.
        cache_path: JSONL file for the cache.

    Returns:
        The wrapped advisor, or ``advisor`` unchanged (with a log line) when
        no ``CachedAdvisor`` is importable.
    """
    cached_cls: Any = None
    for module_name in ("advisor.recommender.cache", "advisor.recommender"):
        try:
            module = __import__(module_name, fromlist=["CachedAdvisor"])
            cached_cls = getattr(module, "CachedAdvisor", None)
        except Exception:  # pragma: no cover - depends on concurrent modules
            cached_cls = None
        if cached_cls is not None:
            break
    if cached_cls is None:
        log.warning("CachedAdvisor not available; LLM calls will not be memoised.")
        return advisor
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    log.info("LLM cache: %s", cache_path)
    return cached_cls(advisor, cache_path)


def build_advisor(
    tag: str,
    *,
    results_dir: Path = RESULTS_DIR,
    base_url: str | None = None,
    use_cache: bool = True,
) -> Any:
    """Instantiate the advisor behind a CLI tag (lazy imports).

    Args:
        tag: ``"heuristic"``, a key of :data:`ADVISOR_MODELS` (``llama``,
            ``qwen``), or ``"ollama:<model>"`` for an ad-hoc Ollama model.
        results_dir: Where ``llm_cache_<tag>.jsonl`` is kept.
        base_url: Optional Ollama base URL override.
        use_cache: Wrap LLM advisors in ``CachedAdvisor`` when available.

    Returns:
        An object exposing ``recommend(ticker, features, *, as_of)``.

    Raises:
        ValueError: For an unknown tag.
    """
    if tag == HEURISTIC_TAG:
        from advisor.recommender.heuristic import HeuristicAdvisor  # lazy import

        return HeuristicAdvisor()

    if tag in ADVISOR_MODELS:
        model = ADVISOR_MODELS[tag]
    elif tag.startswith("ollama:"):
        model = tag.split(":", 1)[1]
    else:
        known = ", ".join([HEURISTIC_TAG, *ADVISOR_MODELS, "ollama:<model>"])
        raise ValueError(f"Unknown advisor tag {tag!r}; expected one of: {known}.")

    from advisor.recommender.llm import OllamaAdvisor  # lazy import

    kwargs: dict[str, Any] = {"model": model}
    if base_url:
        kwargs["base_url"] = base_url
    advisor = OllamaAdvisor(**kwargs)
    if use_cache:
        safe_tag = tag.replace(":", "_").replace("/", "_")
        advisor = wrap_cached(advisor, Path(results_dir) / f"llm_cache_{safe_tag}.jsonl")
    return advisor


def make_progress_logger(tag: str, every: int = 5) -> Callable[[dict[str, Any]], None]:
    """Build a progress callback that logs throughput and an ETA.

    Args:
        tag: Advisor tag used as the log prefix.
        every: Log every ``every`` decision dates (and always the last one).

    Returns:
        A callable accepting the progress dict emitted by
        :class:`WalkForwardBacktest`.
    """

    def _callback(info: dict[str, Any]) -> None:
        index = int(info.get("decision_index", 0)) + 1
        total = int(info.get("n_decisions", 0))
        if index % max(1, every) != 0 and index != total:
            return
        eta = info.get("eta_s", float("nan"))
        eta_txt = f"{eta / 60:.1f} min" if eta == eta else "n/a"
        log.info(
            "[%s] decision %d/%d (%s) calls=%d errors=%d elapsed=%.0fs ETA=%s",
            tag,
            index,
            total,
            info.get("date", "?"),
            int(info.get("calls_done", 0)),
            int(info.get("errors", 0)),
            float(info.get("elapsed_s", 0.0)),
            eta_txt,
        )

    return _callback


# --------------------------------------------------------------------------- #
# Advisor runs and persistence
# --------------------------------------------------------------------------- #
def run_advisor(
    tag: str,
    advisor: Any,
    prices_long: pd.DataFrame,
    *,
    tickers: Sequence[str],
    start: date,
    end: date,
    decision_freq: str | int = "W-FRI",
    horizon_days: int = 5,
    cost_bps: float = 10.0,
    calibrate: bool = True,
    calibration_min_samples: int = 50,
    hold_threshold: float = 0.01,
    initial_capital: float = 100_000.0,
    max_workers: int = 1,
) -> dict[str, Any]:
    """Backtest one advisor walk-forward and return its result dict.

    Args:
        tag: Advisor tag (for logging).
        advisor: The advisor object.
        prices_long: Long-form prices with pre-window history.
        tickers: Universe subset to evaluate.
        start: Inclusive window start.
        end: Exclusive window end.
        decision_freq: Decision cadence.
        horizon_days: Label horizon in trading days.
        cost_bps: Turnover cost in basis points.
        calibrate: Fit temperature scaling walk-forward.
        calibration_min_samples: Samples needed before the first fit.
        hold_threshold: HOLD-correctness threshold.
        initial_capital: Starting capital.
        max_workers: Threads for concurrent advisor calls.

    Returns:
        The dict returned by :meth:`WalkForwardBacktest.run`, plus ``tag``.
    """
    backtest = WalkForwardBacktest(
        advisor,
        decision_freq=decision_freq,
        horizon_days=horizon_days,
        cost_bps=cost_bps,
        calibrate=calibrate,
        calibration_min_samples=calibration_min_samples,
        start=start,
        end=end,
        tickers=list(tickers),
        progress_callback=make_progress_logger(tag),
        max_workers=max_workers,
        hold_threshold=hold_threshold,
        initial_capital=initial_capital,
    )
    log.info("[%s] running walk-forward backtest over %d tickers", tag, len(tickers))
    result = backtest.run(prices_long)
    result["tag"] = tag
    cache = getattr(advisor, "hits", None)
    if cache is not None:
        log.info("[%s] LLM cache hits=%s misses=%s", tag, cache, getattr(advisor, "misses", "?"))
    return result


def advisor_result_paths(tag: str, results_dir: Path) -> dict[str, Path]:
    """Return the canonical output paths for one advisor."""
    results_dir = Path(results_dir)
    return {
        "recommendations": results_dir / f"{tag}_recommendations.csv",
        "equity": results_dir / f"{tag}_equity.csv",
        "monthly": results_dir / f"{tag}_monthly.csv",
        "trades": results_dir / f"{tag}_trades.csv",
        "calibration": results_dir / f"calibration_{tag}.json",
    }


def _json_default(value: Any) -> Any:
    """Fallback JSON encoder for numpy, pandas and path objects."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.ndarray, pd.Series)):
        return value.tolist()
    return str(value)


def save_advisor_results(tag: str, result: dict[str, Any], results_dir: Path) -> dict[str, Path]:
    """Persist an advisor's backtest result to CSV/JSON files.

    Args:
        tag: Advisor tag.
        result: Dict from :meth:`WalkForwardBacktest.run`.
        results_dir: Output directory (created if missing).

    Returns:
        The paths written (see :func:`advisor_result_paths`).
    """
    paths = advisor_result_paths(tag, results_dir)
    Path(results_dir).mkdir(parents=True, exist_ok=True)

    recs = result["recommendations"].copy()
    for col in ("date", "label_date"):
        if col in recs.columns:
            recs[col] = pd.to_datetime(recs[col]).dt.strftime("%Y-%m-%d")
    recs.to_csv(paths["recommendations"], index=False)

    equity = pd.DataFrame(
        {
            "equity": result["equity_curve"],
            "daily_return": result["equity_curve"].pct_change(),
            "drawdown": drawdown_series(result["equity_curve"]),
        }
    )
    equity.index = pd.DatetimeIndex(equity.index).strftime("%Y-%m-%d")
    equity.index.name = "date"
    equity.to_csv(paths["equity"])

    monthly = result.get("per_period_metrics")
    if isinstance(monthly, pd.DataFrame) and not monthly.empty:
        monthly.to_csv(paths["monthly"])
    trades = result.get("trades")
    if isinstance(trades, pd.DataFrame):
        trades.to_csv(paths["trades"], index=False)

    payload = {
        "tag": tag,
        "metrics": result.get("calibration_metrics", {}),
        "folds": result.get("calibration", []),
        "config": result.get("config", {}),
        "n_llm_calls": result.get("n_llm_calls", 0),
        "n_errors": result.get("n_errors", 0),
        "elapsed_s": result.get("elapsed_s"),
    }
    paths["calibration"].write_text(json.dumps(payload, indent=2, default=_json_default))
    return paths


def load_advisor_results(tag: str, results_dir: Path) -> dict[str, Any] | None:
    """Rebuild an advisor result dict from files written by :func:`save_advisor_results`.

    Args:
        tag: Advisor tag.
        results_dir: Directory holding the files.

    Returns:
        A result dict compatible with the summary/ablation builders, or
        ``None`` when the recommendation or equity file is missing.
    """
    paths = advisor_result_paths(tag, results_dir)
    if not (paths["recommendations"].exists() and paths["equity"].exists()):
        return None

    recs = pd.read_csv(paths["recommendations"])
    for col in ("date", "label_date"):
        if col in recs.columns:
            recs[col] = pd.to_datetime(recs[col], errors="coerce")
    equity = pd.read_csv(paths["equity"], index_col="date", parse_dates=True)
    equity_curve = equity["equity"].astype("float64").rename("equity")
    daily_returns = equity_curve.pct_change().dropna().rename("return")

    calibration: list[dict[str, Any]] = []
    n_llm_calls = int(len(recs))
    n_errors = int(recs["error"].fillna(False).astype(bool).sum()) if "error" in recs else 0
    config: dict[str, Any] = {}
    if paths["calibration"].exists():
        try:
            payload = json.loads(paths["calibration"].read_text())
            calibration = list(payload.get("folds", []))
            n_llm_calls = int(payload.get("n_llm_calls", n_llm_calls))
            n_errors = int(payload.get("n_errors", n_errors))
            config = dict(payload.get("config", {}))
        except (OSError, ValueError) as exc:  # pragma: no cover - corrupt file
            log.warning("Could not read %s: %s", paths["calibration"], exc)

    return {
        "tag": tag,
        "equity_curve": equity_curve,
        "daily_returns": daily_returns,
        "recommendations": recs,
        "calibration": calibration,
        "calibration_metrics": evaluate_calibration(recs),
        "summary": summarise(equity_curve, daily_returns),
        "n_llm_calls": n_llm_calls,
        "n_errors": n_errors,
        "config": config,
        "resumed": True,
    }


# --------------------------------------------------------------------------- #
# Tables
# --------------------------------------------------------------------------- #
def _kind_of(name: str) -> str:
    """Classify a strategy name as advisor / baseline / null."""
    if name in BASELINE_NAMES:
        return "baseline"
    if name.startswith("random_ensemble"):
        return "null"
    return "advisor"


def build_summary_table(
    strategies: dict[str, dict[str, Any]],
    *,
    null_sharpes: pd.Series | np.ndarray | None = None,
    n_trials: int | None = None,
) -> pd.DataFrame:
    """Build the headline comparison table (one row per strategy).

    Args:
        strategies: ``{name: {"equity_curve": Series, "daily_returns": Series}}``
            (``daily_returns`` optional -- derived from the curve when absent).
        null_sharpes: Terminal Sharpe ratios of the random ensemble; when
            given, ``p_value_vs_random`` is the one-sided permutation p-value
            of each strategy's Sharpe against that null.
        n_trials: Trials used for the Deflated Sharpe Ratio. Defaults to the
            number of advisor + baseline strategies in the table (the null
            mean is not a candidate strategy), following Bailey & Lopez de
            Prado (2014).

    Returns:
        A ``pd.DataFrame`` indexed by strategy with :data:`_SUMMARY_COLUMNS`.
    """
    kinds = {name: _kind_of(name) for name in strategies}
    if n_trials is None:
        n_trials = max(1, sum(1 for k in kinds.values() if k != "null"))

    # Daily returns per strategy, then the annualised Sharpe of every candidate
    # (advisors + baselines, not the null mean): their cross-trial spread sets
    # the Deflated Sharpe benchmark (Bailey & Lopez de Prado 2014).
    returns_by_name: dict[str, pd.Series] = {}
    curves_by_name: dict[str, pd.Series] = {}
    for name, res in strategies.items():
        curve = pd.Series(res["equity_curve"], dtype="float64")
        rets = res.get("daily_returns")
        if rets is None:
            rets = curve.pct_change().dropna()
        curves_by_name[name] = curve
        returns_by_name[name] = pd.Series(rets, dtype="float64")
    trial_sharpes = [
        sharpe_ratio(returns_by_name[name]) for name, kind in kinds.items() if kind != "null"
    ]

    rows: list[dict[str, Any]] = []
    for name in strategies:
        curve = curves_by_name[name]
        rets = returns_by_name[name]
        stats = summarise(curve, rets, n_trials=n_trials, trial_sharpes=trial_sharpes)
        stats["strategy"] = name
        stats["kind"] = kinds[name]
        stats["p_value_vs_random"] = (
            permutation_p_value(stats["sharpe"], null_sharpes)
            if null_sharpes is not None
            else float("nan")
        )
        rows.append(stats)

    table = pd.DataFrame(rows).set_index("strategy")
    return table.reindex(columns=list(_SUMMARY_COLUMNS))


def build_baselines_equity(baselines: dict[str, dict[str, Any]]) -> pd.DataFrame:
    """Combine the baseline equity curves (and random band) into one frame.

    Args:
        baselines: Output of :func:`run_all_baselines`.

    Returns:
        A ``pd.DataFrame`` indexed by date with one column per baseline plus
        ``random_ensemble_mean``, ``random_ensemble_p05``, ``_p50``, ``_p95``.
    """
    columns: dict[str, pd.Series] = {}
    for name in BASELINE_NAMES:
        if name in baselines:
            columns[name] = baselines[name]["equity_curve"]
    ensemble = baselines.get("random_ensemble")
    if ensemble is not None:
        columns[_RANDOM_MEAN] = ensemble["mean_equity_curve"]
        for pct in ("p05", "p50", "p95"):
            columns[f"random_ensemble_{pct}"] = ensemble["percentiles"][pct]
    frame = pd.DataFrame(columns)
    frame.index.name = "date"
    return frame


def build_random_null_table(ensemble: dict[str, Any]) -> pd.DataFrame:
    """Per-seed terminal statistics of the random ensemble (the null sample)."""
    table = ensemble["terminal_stats"].copy()
    table.index.name = "seed"
    return table.reset_index()


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def _save_figure(fig: Any, out_dir: Path, stem: str) -> list[Path]:
    """Save a figure as PNG and SVG and return the paths."""
    paths = []
    for ext in ("png", "svg"):
        path = Path(out_dir) / f"{stem}.{ext}"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        paths.append(path)
    return paths


def make_figures(
    curves: pd.DataFrame,
    out_dir: Path,
    *,
    band: pd.DataFrame | None = None,
    recommendations_by_tag: dict[str, pd.DataFrame] | None = None,
    n_bins: int = 10,
) -> list[Path]:
    """Render the evaluation figures with the Agg backend (PNG + SVG).

    Args:
        curves: Equity curves indexed by date, one column per strategy
            (advisors first). Curves are normalised to 1.0 at their first
            observation before plotting.
        out_dir: Directory for the image files (created if missing).
        band: Optional frame with ``p05`` and ``p95`` columns (random
            ensemble percentile curves) shaded behind the equity lines.
        recommendations_by_tag: Optional ``{advisor_tag: recommendations}``
            used for one reliability diagram per advisor.
        n_bins: Confidence bins in the reliability diagrams.

    Returns:
        The list of files written.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    palette = list(OKABE_ITO)

    curves = curves.dropna(how="all").astype("float64")
    normalised = curves / curves.apply(lambda s: s.dropna().iloc[0] if s.notna().any() else np.nan)

    # 1. Equity curves --------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5.5))
    if band is not None and {"p05", "p95"}.issubset(band.columns):
        first = band["p50"].dropna().iloc[0] if "p50" in band else band["p05"].dropna().iloc[0]
        ax.fill_between(
            band.index,
            band["p05"] / first,
            band["p95"] / first,
            color="#BBBBBB",
            alpha=0.35,
            label="Random ensemble 5-95%",
            linewidth=0,
        )
    from matplotlib import patheffects

    for i, name in enumerate(normalised.columns):
        style = "--" if _kind_of(name) != "advisor" else "-"
        width = 2.0 if _kind_of(name) == "advisor" else 1.4
        colour = palette[i % len(palette)]
        # The palette's yellow is hard to see on white paper, so give it a thin dark edge.
        edge = None
        if colour == "#F0E442":
            edge = [patheffects.Stroke(linewidth=width + 1.2, foreground="#666666"),
                    patheffects.Normal()]
        ax.plot(
            normalised.index,
            normalised[name],
            style,
            color=colour,
            linewidth=width,
            label=STRATEGY_LABELS.get(name, name),
            path_effects=edge,
        )
    ax.axhline(1.0, color="#777777", linewidth=0.8)
    ax.set_title("Growth of 1 unit, held-out test window")
    ax.set_ylabel("Equity (normalised)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8, frameon=False)
    fig.autofmt_xdate()
    written += _save_figure(fig, out_dir, "equity_curves")
    plt.close(fig)

    # 2. Drawdowns ------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for i, name in enumerate(curves.columns):
        dd = drawdown_series(curves[name].dropna())
        ax.plot(
            dd.index,
            dd * 100.0,
            color=palette[i % len(palette)],
            linewidth=1.4 if _kind_of(name) == "advisor" else 1.0,
            label=STRATEGY_LABELS.get(name, name),
        )
    ax.set_title("Drawdown from running peak")
    ax.set_ylabel("Drawdown (%)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left", fontsize=8, frameon=False)
    fig.autofmt_xdate()
    written += _save_figure(fig, out_dir, "drawdowns")
    plt.close(fig)

    # 3. Monthly returns heatmap ----------------------------------------------
    monthly = {}
    for name in curves.columns:
        rets = curves[name].dropna().pct_change().dropna()
        monthly[name] = period_returns(rets, "M")
    heat = pd.DataFrame(monthly).T * 100.0
    if not heat.empty:
        heat.columns = [str(c) for c in heat.columns]
        fig, ax = plt.subplots(figsize=(max(6, 0.45 * heat.shape[1] + 3), 0.5 * heat.shape[0] + 2))
        limit = float(np.nanmax(np.abs(heat.to_numpy()))) if heat.notna().any().any() else 1.0
        image = ax.imshow(heat.to_numpy(), cmap="PuOr", vmin=-limit, vmax=limit, aspect="auto")
        ax.set_xticks(range(heat.shape[1]))
        ax.set_xticklabels(heat.columns, rotation=90, fontsize=7)
        ax.set_yticks(range(heat.shape[0]))
        ax.set_yticklabels([STRATEGY_LABELS.get(n, n) for n in heat.index], fontsize=8)
        if heat.shape[1] <= 24:
            for r in range(heat.shape[0]):
                for c in range(heat.shape[1]):
                    value = heat.iat[r, c]
                    if pd.notna(value):
                        ax.text(c, r, f"{value:.1f}", ha="center", va="center", fontsize=6)
        fig.colorbar(image, ax=ax, label="Monthly return (%)")
        ax.set_title("Monthly returns by strategy")
        written += _save_figure(fig, out_dir, "monthly_returns_heatmap")
        plt.close(fig)

    # 4. Reliability diagrams ---------------------------------------------------
    for tag, recs in (recommendations_by_tag or {}).items():
        if recs is None or recs.empty or "correct" not in recs.columns:
            continue
        labelled = recs[pd.to_numeric(recs["correct"], errors="coerce").notna()]
        if labelled.empty:
            continue
        correct = pd.to_numeric(labelled["correct"], errors="coerce").to_numpy(dtype="float64")
        fig, ax = plt.subplots(figsize=(5.5, 5))
        ax.plot([0, 1], [0, 1], "--", color="#777777", linewidth=1, label="Perfect calibration")
        for j, (col, label) in enumerate(
            (
                ("raw_confidence", "Raw verbal confidence"),
                ("calibrated_confidence", "Temperature-scaled"),
            )
        ):
            if col not in labelled.columns:
                continue
            conf = pd.to_numeric(labelled[col], errors="coerce").to_numpy(dtype="float64")
            ok = np.isfinite(conf)
            if not ok.any():
                continue
            edges = np.linspace(0.0, 1.0, n_bins + 1)
            bins = np.clip(np.digitize(conf[ok], edges[1:-1]), 0, n_bins - 1)
            xs, ys = [], []
            for b in range(n_bins):
                mask = bins == b
                if mask.any():
                    xs.append(float(conf[ok][mask].mean()))
                    ys.append(float(correct[ok][mask].mean()))
            ax.plot(xs, ys, "o-", color=palette[j], label=label)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Mean confidence in bin")
        ax.set_ylabel("Empirical hit rate")
        ax.set_title(f"Reliability diagram: {STRATEGY_LABELS.get(tag, tag)}")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left", fontsize=8, frameon=False)
        written += _save_figure(fig, out_dir, f"reliability_{tag}")
        plt.close(fig)

    return written


# --------------------------------------------------------------------------- #
# Manifest
# --------------------------------------------------------------------------- #
def package_versions() -> dict[str, str]:
    """Versions of the packages that determine the numerical results."""
    from importlib import metadata

    versions: dict[str, str] = {"python": platform.python_version()}
    for name in ("pandas", "numpy", "scipy", "scikit-learn", "matplotlib"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:  # pragma: no cover - optional
            versions[name] = "not installed"
    return versions


def git_hash(root: Path = PROJECT_ROOT) -> str | None:
    """Current git commit hash of ``root``, or ``None`` when unavailable."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None if out.returncode == 0 else None


def write_manifest(path: Path, *, config: dict[str, Any], extras: dict[str, Any]) -> dict[str, Any]:
    """Write ``run_manifest.json`` and return its content.

    Args:
        path: Destination file.
        config: The resolved run configuration (CLI arguments and constants).
        extras: Additional entries (advisors run/resumed, outputs, timing).

    Returns:
        The manifest dict that was written.
    """
    manifest = {
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_hash": git_hash(),
        "platform": platform.platform(),
        "package_versions": package_versions(),
        "seed": config.get("seed", RANDOM_SEED),
        "config": config,
        **extras,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(manifest, indent=2, default=_json_default)
    path.write_text(text)
    return json.loads(text)  # the JSON-normalised content that is on disk


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    """Construct the command-line parser."""
    parser = argparse.ArgumentParser(
        prog="python -m advisor.evaluation.run_evaluation",
        description=(
            "Walk-forward evaluation of the advisor(s) against the four baselines "
            "(buy-and-hold SPY, random allocation, 12-1 momentum, Markowitz) with "
            "Deflated Sharpe, Sortino, max drawdown, ECE and an LLM ablation."
        ),
    )
    parser.add_argument(
        "--advisors",
        nargs="+",
        default=[HEURISTIC_TAG],
        metavar="TAG",
        help=f"Advisor tags to evaluate: {HEURISTIC_TAG}, {', '.join(ADVISOR_MODELS)}, "
        "or ollama:<model>. Default: heuristic.",
    )
    parser.add_argument(
        "--tickers",
        default="ALL",
        help="ALL, an integer N (first N universe tickers) or explicit symbols. Default: ALL.",
    )
    parser.add_argument("--freq", default="W-FRI", help="Decision cadence. Default: W-FRI.")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=RESULTS_DIR,
        help=f"Output directory. Default: {RESULTS_DIR}.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=CACHE_DIR,
        help=f"Price parquet cache. Default: {CACHE_DIR}.",
    )
    parser.add_argument(
        "--cost-bps", type=float, default=10.0, help="Turnover cost (bps). Default: 10."
    )
    parser.add_argument(
        "--seeds", type=int, default=100, help="Random-ensemble size. Default: 100."
    )
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="Base random seed.")
    parser.add_argument(
        "--n-holdings", type=int, default=10, help="Names held by the random baseline."
    )
    parser.add_argument("--horizon-days", type=int, default=5, help="Label horizon (trading days).")
    parser.add_argument(
        "--hold-threshold", type=float, default=0.01, help="HOLD is correct when |return| < this."
    )
    parser.add_argument("--start", type=_parse_date, default=TEST_START, help="Test window start.")
    parser.add_argument(
        "--end", type=_parse_date, default=TEST_END, help="Test window end (exclusive)."
    )
    parser.add_argument(
        "--history-start",
        type=_parse_date,
        default=TRAIN_START,
        help="Earliest price date loaded (indicator and baseline warm-up). Default: TRAIN_START.",
    )
    parser.add_argument(
        "--benchmark", default=BENCHMARK_TICKER, help="Buy-and-hold benchmark ticker."
    )
    parser.add_argument(
        "--calibration-min-samples", type=int, default=50, help="Labels needed before calibrating."
    )
    parser.add_argument("--no-calibrate", action="store_true", help="Disable temperature scaling.")
    parser.add_argument(
        "--initial-capital", type=float, default=100_000.0, help="Starting capital."
    )
    parser.add_argument("--max-workers", type=int, default=1, help="Concurrent advisor calls.")
    parser.add_argument("--ollama-base-url", default=None, help="Override the Ollama base URL.")
    parser.add_argument("--no-llm-cache", action="store_true", help="Do not memoise LLM answers.")
    parser.add_argument(
        "--resume", action="store_true", help="Skip advisors with existing results."
    )
    parser.add_argument("--no-figures", action="store_true", help="Skip figure rendering.")
    parser.add_argument("--log-level", default="INFO", help="Logging level. Default: INFO.")
    return parser


def main(
    argv: Sequence[str] | None = None, *, advisor_factory: AdvisorFactory | None = None
) -> int:
    """Run the full evaluation from the command line.

    Args:
        argv: Argument vector (defaults to ``sys.argv[1:]``).
        advisor_factory: Optional ``tag -> advisor`` callable replacing
            :func:`build_advisor` (used by tests to inject a fake advisor).

    Returns:
        Process exit code (``0`` on success, ``1`` on a data error).
    """
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    t_start = time.monotonic()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = results_dir / "figures"
    tickers = parse_tickers(args.tickers)
    benchmark = str(args.benchmark).upper()
    log.info(
        "Evaluation: advisors=%s tickers=%d window=%s..%s history from %s freq=%s cost=%.1fbps",
        args.advisors,
        len(tickers),
        args.start,
        args.end,
        args.history_start,
        args.freq,
        args.cost_bps,
    )

    from advisor.data.loader import load_prices  # lazy: keeps module import light

    prices_all = load_prices(
        sorted(set(tickers) | {benchmark}),
        args.history_start,
        args.end,
        use_cache=True,
        cache_dir=Path(args.cache_dir),
    )
    if prices_all is None or len(prices_all) == 0:
        log.error("No price data loaded; run scripts/fetch_data.py first.")
        return 1
    prices_all["ticker"] = prices_all["ticker"].astype(str).str.upper()
    benchmark_prices = prices_all[prices_all["ticker"] == benchmark]
    universe_prices = prices_all[prices_all["ticker"].isin(tickers)]
    missing = sorted(set(tickers) - set(universe_prices["ticker"].unique()))
    if missing:
        log.warning("No data for %d ticker(s): %s", len(missing), ", ".join(missing))
        tickers = [t for t in tickers if t not in missing]
    if universe_prices.empty:
        log.error("No universe prices available.")
        return 1
    if benchmark_prices.empty:
        log.warning("Benchmark %s not loaded; buy-and-hold falls back to equal weight.", benchmark)
        benchmark_prices = None

    make_advisor: AdvisorFactory = (
        advisor_factory
        if advisor_factory is not None
        else functools.partial(
            build_advisor,
            results_dir=results_dir,
            base_url=args.ollama_base_url,
            use_cache=not args.no_llm_cache,
        )
    )

    advisor_results: dict[str, dict[str, Any]] = {}
    resumed: list[str] = []
    for tag in args.advisors:
        if args.resume:
            loaded = load_advisor_results(tag, results_dir)
            if loaded is not None:
                log.info("[%s] --resume: loaded existing results, skipping backtest", tag)
                advisor_results[tag] = loaded
                resumed.append(tag)
                continue
        advisor = make_advisor(tag)
        result = run_advisor(
            tag,
            advisor,
            universe_prices,
            tickers=tickers,
            start=args.start,
            end=args.end,
            decision_freq=args.freq,
            horizon_days=args.horizon_days,
            cost_bps=args.cost_bps,
            calibrate=not args.no_calibrate,
            calibration_min_samples=args.calibration_min_samples,
            hold_threshold=args.hold_threshold,
            initial_capital=args.initial_capital,
            max_workers=args.max_workers,
        )
        save_advisor_results(tag, result, results_dir)
        advisor_results[tag] = result
        # Drop bulky per-day frames we no longer need to keep memory modest.
        for key in ("holdings", "weights", "trades", "turnover"):
            result.pop(key, None)

    baselines = run_all_baselines(
        universe_prices,
        benchmark_prices=benchmark_prices,
        start=args.start,
        end=args.end,
        cost_bps=args.cost_bps,
        initial_capital=args.initial_capital,
        seed=args.seed,
        n_seeds=args.seeds,
        n_holdings=args.n_holdings,
        random_freq=args.freq,
    )
    ensemble = baselines["random_ensemble"]
    build_baselines_equity(baselines).to_csv(results_dir / "baselines_equity.csv")
    build_random_null_table(ensemble).to_csv(results_dir / "random_null.csv", index=False)

    strategies: dict[str, dict[str, Any]] = {}
    for tag, res in advisor_results.items():
        strategies[tag] = {
            "equity_curve": res["equity_curve"],
            "daily_returns": res["daily_returns"],
        }
    for name in BASELINE_NAMES:
        strategies[name] = {
            "equity_curve": baselines[name]["equity_curve"],
            "daily_returns": baselines[name]["daily_returns"],
        }
    strategies[_RANDOM_MEAN] = {"equity_curve": ensemble["mean_equity_curve"]}

    summary = build_summary_table(strategies, null_sharpes=ensemble["terminal_sharpe"])
    summary.to_csv(results_dir / "summary_table.csv")
    ablation = ablation_table(advisor_results, reference=HEURISTIC_TAG)
    ablation.to_csv(results_dir / "ablation.csv")

    figures: list[Path] = []
    if not args.no_figures:
        curves = pd.DataFrame({name: s["equity_curve"] for name, s in strategies.items()})
        figures = make_figures(
            curves,
            figures_dir,
            band=ensemble["percentiles"],
            recommendations_by_tag={t: r["recommendations"] for t, r in advisor_results.items()},
        )

    elapsed = time.monotonic() - t_start
    config = {
        **{k: (v if not isinstance(v, Path) else str(v)) for k, v in vars(args).items()},
        "train_start": TRAIN_START.isoformat(),
        "test_start": TEST_START.isoformat(),
        "test_end": TEST_END.isoformat(),
        "advisor_models": ADVISOR_MODELS,
        "n_trials_for_deflation": int(summary["n_trials"].iloc[0]) if len(summary) else 1,
    }
    write_manifest(
        results_dir / "run_manifest.json",
        config=config,
        extras={
            "advisors_run": [t for t in advisor_results if t not in resumed],
            "advisors_resumed": resumed,
            "tickers": tickers,
            "benchmark": benchmark,
            "n_strategies": int(len(summary)),
            "n_llm_calls": {t: int(r.get("n_llm_calls", 0)) for t, r in advisor_results.items()},
            "outputs": sorted(
                str(p.relative_to(results_dir)) for p in results_dir.rglob("*") if p.is_file()
            ),
            "figures": [str(p) for p in figures],
            "elapsed_s": round(elapsed, 1),
        },
    )
    with pd.option_context("display.width", 200, "display.float_format", "{:.4f}".format):
        log.info(
            "Summary table:\n%s",
            summary.drop(columns=["initial_equity", "final_equity"]).to_string(),
        )
        if not ablation.empty:
            log.info("Ablation:\n%s", ablation.T.to_string())
    log.info("Done in %.1fs; results in %s", elapsed, results_dir)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    sys.exit(main())
