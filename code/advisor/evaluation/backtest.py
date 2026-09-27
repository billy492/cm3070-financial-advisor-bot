"""Walk-forward backtest of the advisor over the held-out test window.

:class:`WalkForwardBacktest` is the out-of-sample evaluation engine of the
thesis. It drives any advisor exposing
``recommend(ticker, features, *, as_of) -> Recommendation`` across the
held-out test period and turns the resulting BUY/HOLD/SELL stream into a
portfolio, an equity curve and a labelled recommendation log from which both
the performance metrics (Sharpe, Sortino, max drawdown, Deflated Sharpe) and
the calibration metrics (ECE raw vs calibrated) are computed.

Protocol (report §3.3-3.4; Lopez de Prado 2018; Arnott, Harvey & Markowitz 2019)
--------------------------------------------------------------------------------
* **Expanding-window walk-forward.** The advisor is a pretrained reasoner and
  is never fitted; the only fitted component is the confidence calibrator.
  Its calibration set *expands* as labels realise: at each fold the
  temperature is refitted on every earlier recommendation whose outcome is
  already known (``label_date < current decision date``), then applied to the
  current fold's raw confidences. The stub-era constructor arguments are kept
  and re-interpreted accordingly: ``train_window`` optionally caps the number
  of most-recent labelled samples used for calibration (``None`` = fully
  expanding), ``test_window`` optionally sets the fold length in decision
  dates (``None`` = one fold per calendar month), and ``step`` must be
  ``None`` or equal to ``test_window`` (folds never overlap).
* **Weekly decisions.** Recommendations are requested only on decision dates
  (last trading day of each week by default), once per ticker, with the
  feature row of that date -- ``close, ret_1d, sma_10, sma_50, mom_10,
  vol_20, rsi_14`` from :func:`advisor.features.indicators.add_features_by_ticker`.
  Features on date ``t`` use data up to and including ``t`` only.
* **Labels.** The realised return of a recommendation is the forward
  ``horizon_days``-trading-day return of ``adj_close``; it is *correct* when
  ``BUY`` and the return is positive, ``SELL`` and it is negative, or ``HOLD``
  and ``|return| < hold_threshold`` (default 1%). The last decision dates of
  the window have no realised label yet (``NaN``).
* **Portfolio rule (the "dynamic investment strategy").** At each decision
  date the held set becomes ``BUY_t ∪ (held_{t-1} ∩ HOLD_t)``: BUY adds a
  name, HOLD keeps a name that is already held (a HOLD on an unheld name does
  nothing), SELL -- or the absence of a usable recommendation -- removes it.
  The held set is equal-weighted (fully invested) and the portfolio is in
  cash when it is empty. Re-equalising rather than letting HOLD names keep
  their drifted weight makes the target weights a pure function of the
  recommendation history, so the simulation is vectorised and reproducible.
* **Costs.** Ten basis points of one-way turnover per rebalance by default,
  through the shared :func:`advisor.evaluation.portfolio.simulate_weights`.
* **Deflation.** The per-strategy summary reports a Deflated Sharpe Ratio
  with ``n_trials=1``; the experiment runner recomputes it with ``n_trials``
  equal to the number of strategies compared (Bailey & Lopez de Prado 2014).

References:
    Bailey, D. H. & Lopez de Prado, M. (2014). The Deflated Sharpe Ratio.
        *Journal of Portfolio Management*, 40(5), 94-107.
    Lopez de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley.
    Arnott, R., Harvey, C. R. & Markowitz, H. (2019). A Backtesting Protocol
        in the Era of Machine Learning. *Journal of Financial Data Science*.
    Guo, C., Pleiss, G., Sun, Y. & Weinberger, K. Q. (2017). On Calibration of
        Modern Neural Networks. *ICML*.
    Naeini, M. P., Cooper, G. F. & Hauskrecht, M. (2015). Obtaining Well
        Calibrated Probabilities Using Bayesian Binning. *AAAI*.
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from advisor.config import TEST_END, TEST_START
from advisor.evaluation import metrics as _metrics
from advisor.evaluation.metrics import (
    expected_calibration_error,
    max_drawdown,
    sharpe_ratio,
)
from advisor.evaluation.portfolio import (
    decision_dates,
    pivot_prices,
    simulate_weights,
    slice_window,
    summarise,
    to_timestamp,
)
from advisor.features.indicators import FEATURE_COLUMNS, add_features_by_ticker

__all__ = [
    "FEATURE_KEYS",
    "RECOMMENDATION_COLUMNS",
    "WalkForwardBacktest",
    "label_correctness",
    "evaluate_calibration",
    "ablation_table",
]

log = logging.getLogger(__name__)

#: Keys of the feature dict handed to ``advisor.recommend`` (all floats).
FEATURE_KEYS: tuple[str, ...] = ("close", *FEATURE_COLUMNS)

#: Column order of the recommendation log returned by :meth:`WalkForwardBacktest.run`.
RECOMMENDATION_COLUMNS: tuple[str, ...] = (
    "date",
    "ticker",
    "action",
    "raw_confidence",
    "calibrated_confidence",
    "reason",
    "counterfactual",
    "realised_return",
    "label_date",
    "correct",
    "fold",
    "error",
)

_ACTIONS: tuple[str, ...] = ("BUY", "HOLD", "SELL")

ProgressCallback = Callable[[dict[str, Any]], None]


# --------------------------------------------------------------------------- #
# Pure helpers
# --------------------------------------------------------------------------- #
def label_correctness(
    actions: Sequence[str] | pd.Series,
    realised_returns: Sequence[float] | pd.Series,
    hold_threshold: float = 0.01,
) -> pd.Series:
    """Score recommendations against their realised forward returns.

    Args:
        actions: ``"BUY"`` / ``"HOLD"`` / ``"SELL"`` per recommendation.
        realised_returns: Forward simple return over the label horizon
            (``NaN`` when not yet realised).
        hold_threshold: A ``HOLD`` counts as correct when the absolute
            realised return is strictly below this fraction (default 1%).

    Returns:
        A float ``pd.Series`` with ``1.0`` (correct), ``0.0`` (incorrect) or
        ``NaN`` (no label yet / unknown action), aligned with the inputs.
    """
    acts = pd.Series(actions, dtype="object").astype(str).str.upper().to_numpy()
    rets = pd.Series(realised_returns, dtype="float64").to_numpy()
    out = np.full(len(acts), np.nan, dtype="float64")
    known = np.isfinite(rets)
    buy = (acts == "BUY") & known
    sell = (acts == "SELL") & known
    hold = (acts == "HOLD") & known
    out[buy] = (rets[buy] > 0.0).astype("float64")
    out[sell] = (rets[sell] < 0.0).astype("float64")
    out[hold] = (np.abs(rets[hold]) < float(hold_threshold)).astype("float64")
    index = actions.index if isinstance(actions, pd.Series) else None
    return pd.Series(out, index=index, name="correct")


def _brier_score(confidences: np.ndarray, correct: np.ndarray) -> float:
    """Mean squared error between confidence and 0/1 correctness."""
    if confidences.size == 0:
        return float("nan")
    return float(np.mean((confidences - correct) ** 2))


def evaluate_calibration(recommendations: pd.DataFrame, n_bins: int = 10) -> dict[str, Any]:
    """Calibration and hit-rate metrics of a labelled recommendation log.

    Compares the advisor's raw verbal confidence with the temperature-scaled
    confidence on the recommendations whose label has realised. Reports the
    Expected Calibration Error (Naeini et al. 2015) for both, the Brier score,
    and -- when the metrics module provides them -- an adaptive (equal-mass)
    ECE variant. Everything beyond ECE is imported guardedly so this function
    keeps working while the metrics module evolves.

    Args:
        recommendations: The ``recommendations`` frame produced by
            :meth:`WalkForwardBacktest.run` (needs ``action``,
            ``raw_confidence``, ``calibrated_confidence``, ``correct``).
        n_bins: Number of equal-width confidence bins for ECE.

    Returns:
        A dict with ``n`` (labelled rows), ``n_total``, ``hit_rate``,
        ``ece_raw``, ``ece_calibrated``, ``ece_improvement``, ``brier_raw``,
        ``brier_calibrated``, ``mean_raw_confidence``,
        ``mean_calibrated_confidence``, per-action counts and hit rates, and
        optionally ``adaptive_ece_raw`` / ``adaptive_ece_calibrated``.
    """
    nan = float("nan")
    result: dict[str, Any] = {
        "n": 0,
        "n_total": int(len(recommendations)) if recommendations is not None else 0,
        "n_bins": int(n_bins),
        "hit_rate": nan,
        "ece_raw": nan,
        "ece_calibrated": nan,
        "ece_improvement": nan,
        "brier_raw": nan,
        "brier_calibrated": nan,
        "mean_raw_confidence": nan,
        "mean_calibrated_confidence": nan,
    }
    for action in _ACTIONS:
        result[f"n_{action}"] = 0
        result[f"hit_rate_{action}"] = nan
    if recommendations is None or len(recommendations) == 0:
        return result

    df = recommendations
    raw = pd.to_numeric(df["raw_confidence"], errors="coerce").to_numpy(dtype="float64")
    cal_col = "calibrated_confidence" if "calibrated_confidence" in df.columns else "raw_confidence"
    cal = pd.to_numeric(df[cal_col], errors="coerce").to_numpy(dtype="float64")
    correct = pd.to_numeric(df["correct"], errors="coerce").to_numpy(dtype="float64")
    actions = df["action"].astype(str).str.upper().to_numpy()

    for action in _ACTIONS:
        mask = actions == action
        result[f"n_{action}"] = int(mask.sum())
        labelled = mask & np.isfinite(correct)
        if labelled.any():
            result[f"hit_rate_{action}"] = float(np.mean(correct[labelled]))

    keep = np.isfinite(raw) & np.isfinite(correct)
    cal = np.where(np.isfinite(cal), cal, raw)
    raw, cal, correct = raw[keep], cal[keep], correct[keep]
    n = int(raw.size)
    result["n"] = n
    if n == 0:
        return result

    ece_raw = float(expected_calibration_error(raw, correct, n_bins=n_bins))
    ece_cal = float(expected_calibration_error(cal, correct, n_bins=n_bins))
    result.update(
        {
            "hit_rate": float(np.mean(correct)),
            "ece_raw": ece_raw,
            "ece_calibrated": ece_cal,
            "ece_improvement": ece_raw - ece_cal,
            "mean_raw_confidence": float(np.mean(raw)),
            "mean_calibrated_confidence": float(np.mean(cal)),
        }
    )

    brier = getattr(_metrics, "brier_score", None)
    try:
        if callable(brier):
            result["brier_raw"] = float(brier(raw, correct))
            result["brier_calibrated"] = float(brier(cal, correct))
        else:
            result["brier_raw"] = _brier_score(raw, correct)
            result["brier_calibrated"] = _brier_score(cal, correct)
    except Exception as exc:  # pragma: no cover - defensive against API drift
        log.debug("Brier score unavailable: %s", exc)
        result["brier_raw"] = _brier_score(raw, correct)
        result["brier_calibrated"] = _brier_score(cal, correct)

    adaptive = getattr(_metrics, "adaptive_expected_calibration_error", None) or getattr(
        _metrics, "adaptive_ece", None
    )
    if callable(adaptive):
        try:
            result["adaptive_ece_raw"] = float(adaptive(raw, correct, n_bins))
            result["adaptive_ece_calibrated"] = float(adaptive(cal, correct, n_bins))
        except Exception as exc:  # pragma: no cover - optional metric
            log.debug("Adaptive ECE unavailable: %s", exc)
    return result


def ablation_table(
    results: dict[str, dict[str, Any]], reference: str = "heuristic"
) -> pd.DataFrame:
    """Tabulate advisor variants side by side for the LLM-ablation study.

    Args:
        results: ``{advisor_tag: run_result}`` where each value is the dict
            returned by :meth:`WalkForwardBacktest.run` (``summary``,
            ``calibration_metrics``, ``recommendations``, ``n_llm_calls``).
        reference: Tag of the no-LLM variant; when present, relative columns
            ``sharpe_pct_vs_reference`` and ``dsr_delta_vs_reference`` are
            added so hypothesis H3 ("removing the LLM degrades risk-adjusted
            return by >= 20%") can be read off directly.

    Returns:
        A ``pd.DataFrame`` indexed by advisor tag with performance,
        hit-rate, calibration and volume columns.
    """
    rows: list[dict[str, Any]] = []
    for tag, res in results.items():
        summary = res.get("summary", {})
        calib = res.get("calibration_metrics")
        if calib is None:
            calib = evaluate_calibration(res.get("recommendations", pd.DataFrame()))
        row: dict[str, Any] = {"advisor": tag}
        for key in (
            "total_return",
            "cagr",
            "annualised_vol",
            "sharpe",
            "sortino",
            "max_drawdown",
            "deflated_sharpe",
            "calmar",
        ):
            row[key] = summary.get(key, float("nan"))
        for key in ("n", "hit_rate", "ece_raw", "ece_calibrated", "brier_raw", "brier_calibrated"):
            row[f"{key}" if key != "n" else "n_labelled"] = calib.get(key, float("nan"))
        for action in _ACTIONS:
            row[f"n_{action}"] = calib.get(f"n_{action}", 0)
        row["n_llm_calls"] = int(res.get("n_llm_calls", 0))
        row["n_errors"] = int(res.get("n_errors", 0))
        rows.append(row)

    table = pd.DataFrame(rows).set_index("advisor") if rows else pd.DataFrame()
    if reference in table.index and len(table) > 0:
        ref_sharpe = float(table.loc[reference, "sharpe"])
        ref_dsr = float(table.loc[reference, "deflated_sharpe"])
        if ref_sharpe != 0 and math.isfinite(ref_sharpe):
            table["sharpe_pct_vs_reference"] = table["sharpe"] / ref_sharpe - 1.0
        else:
            table["sharpe_pct_vs_reference"] = float("nan")
        table["dsr_delta_vs_reference"] = table["deflated_sharpe"] - ref_dsr
        table["reference"] = reference
    return table


# --------------------------------------------------------------------------- #
# The engine
# --------------------------------------------------------------------------- #
class WalkForwardBacktest:
    """Expanding-window walk-forward backtest with weekly decisions.

    See the module docstring for the full protocol. The advisor is called
    *only* on decision dates, once per ticker with NaN-free features, through
    ``advisor.recommend(ticker, features, as_of=<date>)``. Exceptions raised
    by the advisor are caught per ticker, logged, counted in ``n_errors`` and
    treated as ``HOLD`` with an unknown confidence.

    Args:
        advisor: Object exposing ``recommend(ticker, features, *, as_of)``.
        train_window: Optional cap on the number of most-recent labelled
            recommendations used to refit the calibrator (``None`` = fully
            expanding window).
        test_window: Optional fold length in decision dates (``None`` = one
            fold per calendar month). The calibrator is refitted at the start
            of every fold.
        step: Kept for interface compatibility; must be ``None`` or equal to
            ``test_window`` because folds never overlap.
        decision_freq: Decision cadence understood by
            :func:`advisor.evaluation.portfolio.decision_dates`
            (default ``"W-FRI"``).
        horizon_days: Forward horizon, in trading days, used to label
            recommendations (default 5, i.e. one week).
        cost_bps: One-way turnover cost in basis points.
        calibrate: Whether to fit and apply temperature scaling.
        calibration_min_samples: Minimum labelled samples before the first
            calibrator is fitted; earlier folds use ``calibrated = raw``.
        start: Inclusive start of the evaluation window.
        end: Exclusive end of the evaluation window.
        tickers: Optional subset of tickers to evaluate (default: all in the
            price frame).
        progress_callback: Optional callable receiving a progress dict after
            every decision date (``decision_index``, ``n_decisions``, ``date``,
            ``calls_done``, ``calls_total_estimate``, ``elapsed_s``, ``eta_s``).
        max_workers: Threads used to query the advisor concurrently within a
            decision date (the LLM call is I/O bound). Results are always
            assembled in deterministic ticker order.
        hold_threshold: Absolute realised return below which a ``HOLD`` is
            counted correct.
        initial_capital: Starting capital of the simulated portfolio.
        price_column: Price field used for mark-to-market and labels.
        scaler_factory: Optional zero-argument callable returning a
            calibrator with ``fit(confidences, labels)`` and
            ``transform(confidences)``; defaults to a lazily imported
            :class:`advisor.calibration.temperature.TemperatureScaler`.
        n_bins: Bins for the ECE computation in ``calibration_metrics``.
    """

    def __init__(
        self,
        advisor: Any,
        *,
        train_window: int | None = None,
        test_window: int | None = None,
        step: int | None = None,
        decision_freq: str | int = "W-FRI",
        horizon_days: int = 5,
        cost_bps: float = 10.0,
        calibrate: bool = True,
        calibration_min_samples: int = 50,
        start: date | str | pd.Timestamp = TEST_START,
        end: date | str | pd.Timestamp = TEST_END,
        tickers: Sequence[str] | None = None,
        progress_callback: ProgressCallback | None = None,
        max_workers: int = 1,
        hold_threshold: float = 0.01,
        initial_capital: float = 100_000.0,
        price_column: str = "adj_close",
        scaler_factory: Callable[[], Any] | None = None,
        n_bins: int = 10,
    ) -> None:
        """Configure the backtest; every argument is described on the class."""
        if horizon_days < 1:
            raise ValueError("horizon_days must be >= 1.")
        if test_window is not None and test_window < 1:
            raise ValueError("test_window must be >= 1 when given.")
        if step is not None and step != test_window:
            raise ValueError("step must be None or equal to test_window (non-overlapping folds).")
        if train_window is not None and train_window < 1:
            raise ValueError("train_window must be >= 1 when given.")
        if max_workers < 1:
            raise ValueError("max_workers must be >= 1.")

        self.advisor = advisor
        self.train_window = train_window
        self.test_window = test_window
        self.step = step
        self.decision_freq = decision_freq
        self.horizon_days = int(horizon_days)
        self.cost_bps = float(cost_bps)
        self.calibrate = bool(calibrate)
        self.calibration_min_samples = int(calibration_min_samples)
        self.start = to_timestamp(start)
        self.end = to_timestamp(end)
        self.tickers = [str(t).upper() for t in tickers] if tickers is not None else None
        self.progress_callback = progress_callback
        self.max_workers = int(max_workers)
        self.hold_threshold = float(hold_threshold)
        self.initial_capital = float(initial_capital)
        self.price_column = price_column
        self.scaler_factory = scaler_factory
        self.n_bins = int(n_bins)

        self.n_llm_calls = 0
        self.n_errors = 0
        self._calibration_disabled = False

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def run(self, prices_long: pd.DataFrame) -> dict[str, Any]:
        """Execute the walk-forward backtest.

        Args:
            prices_long: Long-form price history with columns
                ``date, ticker, open, high, low, close, adj_close, volume``.
                Pass history that starts before ``start`` (e.g. from
                ``TRAIN_START``) so the 50-day indicator windows are warm on
                the first decision date.

        Returns:
            A dict with keys ``equity_curve``, ``daily_returns``, ``trades``,
            ``turnover``, ``weights``, ``holdings``, ``per_period_metrics``
            (per calendar month), ``recommendations`` (see
            :data:`RECOMMENDATION_COLUMNS`), ``calibration`` (one entry per
            fold with the fitted temperature and sample count),
            ``calibration_metrics`` (see :func:`evaluate_calibration`),
            ``summary`` (see :func:`advisor.evaluation.portfolio.summarise`),
            ``n_llm_calls``, ``n_errors``, ``decision_dates``, ``config`` and
            ``elapsed_s``.

        Raises:
            ValueError: If the window holds no trading days or no ticker has
                usable features on any decision date.
        """
        t_start = time.monotonic()
        self.n_llm_calls = 0
        self.n_errors = 0
        self._calibration_disabled = False

        frame = self._prepare_frame(prices_long)
        features = add_features_by_ticker(frame)
        features["date"] = pd.to_datetime(features["date"])
        features["ticker"] = features["ticker"].astype(str)

        prices_full = pivot_prices(frame, self.price_column, ffill=True)
        window = slice_window(prices_full, self.start, self.end)
        if window.empty:
            raise ValueError(f"No trading days between {self.start.date()} and {self.end.date()}.")

        dates = decision_dates(window.index, self.decision_freq)
        first = pd.Timestamp(window.index[0])
        if not dates or dates[0] != first:
            dates.insert(0, first)
        folds = self._assign_folds(dates)

        # Forward labels (positional shift on the full calendar; label dates
        # beyond the loaded history are NaT and the return NaN).
        forward = prices_full.shift(-self.horizon_days) / prices_full - 1.0
        label_index = pd.Series(
            list(prices_full.index[self.horizon_days :]) + [pd.NaT] * self.horizon_days,
            index=prices_full.index,
        )

        decision_index = pd.DatetimeIndex(dates)
        feature_subset = features[features["date"].isin(decision_index)]
        features_by_date: dict[pd.Timestamp, pd.DataFrame] = {
            pd.Timestamp(d): grp.set_index("ticker") for d, grp in feature_subset.groupby("date")
        }

        n_tickers = int(frame["ticker"].nunique())
        calls_total_estimate = len(dates) * n_tickers
        tickers_all = list(window.columns)

        records: list[dict[str, Any]] = []
        weight_rows: list[np.ndarray] = []
        calibration_log: list[dict[str, Any]] = []
        held: set[str] = set()
        scaler: Any = None
        current_fold: str | None = None
        col_index = {c: i for i, c in enumerate(tickers_all)}

        for i, d in enumerate(dates):
            fold = folds[i]
            if fold != current_fold:
                current_fold = fold
                scaler, entry = self._refit_calibrator(records, d, fold)
                calibration_log.append(entry)

            recs = self._recommend_date(d, features_by_date.get(pd.Timestamp(d)))
            raw = [r["raw_confidence"] for r in recs]
            calibrated = self._apply_scaler(scaler, raw)

            realised = forward.loc[d] if d in forward.index else None
            label_date = label_index.get(d, pd.NaT)
            for rec, conf in zip(recs, calibrated, strict=True):
                ticker = rec["ticker"]
                ret = float(realised.get(ticker, np.nan)) if realised is not None else np.nan
                rec["calibrated_confidence"] = float(conf)
                rec["realised_return"] = ret
                rec["label_date"] = label_date
                # An advisor failure is treated as HOLD for the portfolio but
                # carries no label: it is not a recommendation the advisor made.
                rec["correct"] = (
                    float("nan")
                    if rec["error"]
                    else float(
                        label_correctness([rec["action"]], [ret], self.hold_threshold).iloc[0]
                    )
                )
                rec["fold"] = fold
                rec["date"] = pd.Timestamp(d)
            records.extend(recs)

            buys = {r["ticker"] for r in recs if r["action"] == "BUY"}
            holds = {r["ticker"] for r in recs if r["action"] == "HOLD"}
            held = buys | (held & holds)
            row = np.zeros(len(tickers_all), dtype="float64")
            if held:
                for ticker in held:
                    row[col_index[ticker]] = 1.0 / len(held)
            weight_rows.append(row)

            self._report_progress(i, len(dates), d, calls_total_estimate, t_start)

        if not records:
            raise ValueError(
                "No recommendations were produced: no ticker had NaN-free features on "
                "any decision date (is the pre-window history long enough?)."
            )

        weights = pd.DataFrame(np.vstack(weight_rows), index=decision_index, columns=tickers_all)
        sim = simulate_weights(
            window, weights, cost_bps=self.cost_bps, initial_capital=self.initial_capital
        )
        recommendations = pd.DataFrame(records).reindex(columns=list(RECOMMENDATION_COLUMNS))
        recommendations["date"] = pd.to_datetime(recommendations["date"])
        recommendations["label_date"] = pd.to_datetime(recommendations["label_date"])

        summary = summarise(sim["equity_curve"], sim["daily_returns"], n_trials=1)
        calibration_metrics = evaluate_calibration(recommendations, n_bins=self.n_bins)
        per_period = self._per_period_metrics(
            sim["daily_returns"], sim["equity_curve"], recommendations
        )
        elapsed = time.monotonic() - t_start
        log.info(
            "Backtest done: %d decision dates, %d recommendations, %d advisor calls, "
            "%d errors, %.1fs",
            len(dates),
            len(recommendations),
            self.n_llm_calls,
            self.n_errors,
            elapsed,
        )
        return {
            "equity_curve": sim["equity_curve"],
            "daily_returns": sim["daily_returns"],
            "trades": sim["trades"],
            "turnover": sim["turnover"],
            "weights": sim["weights"],
            "holdings": sim["holdings"],
            "per_period_metrics": per_period,
            "recommendations": recommendations,
            "calibration": calibration_log,
            "calibration_metrics": calibration_metrics,
            "summary": summary,
            "n_llm_calls": int(self.n_llm_calls),
            "n_errors": int(self.n_errors),
            "decision_dates": dates,
            "config": self.config(),
            "elapsed_s": float(elapsed),
        }

    def config(self) -> dict[str, Any]:
        """Return the JSON-serialisable configuration of this backtest."""
        return {
            "train_window": self.train_window,
            "test_window": self.test_window,
            "step": self.step,
            "decision_freq": self.decision_freq,
            "horizon_days": self.horizon_days,
            "cost_bps": self.cost_bps,
            "calibrate": self.calibrate,
            "calibration_min_samples": self.calibration_min_samples,
            "start": self.start.date().isoformat(),
            "end": self.end.date().isoformat(),
            "tickers": self.tickers,
            "max_workers": self.max_workers,
            "hold_threshold": self.hold_threshold,
            "initial_capital": self.initial_capital,
            "price_column": self.price_column,
            "n_bins": self.n_bins,
        }

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _prepare_frame(self, prices_long: pd.DataFrame) -> pd.DataFrame:
        """Validate, filter and normalise the input price frame."""
        required = {"date", "ticker", "close", self.price_column}
        missing = required - set(prices_long.columns)
        if missing:
            raise ValueError(f"prices_long is missing columns {sorted(missing)!r}.")
        frame = prices_long.copy()
        frame["ticker"] = frame["ticker"].astype(str).str.upper()
        frame["date"] = pd.to_datetime(frame["date"])
        if self.tickers is not None:
            frame = frame[frame["ticker"].isin(self.tickers)]
        frame = frame.sort_values(["ticker", "date"]).reset_index(drop=True)
        if frame.empty:
            raise ValueError("prices_long has no rows for the requested tickers.")
        return frame

    def _assign_folds(self, dates: list[pd.Timestamp]) -> list[str]:
        """Map each decision date to a fold label (calendar month or block)."""
        if self.test_window is None:
            return [pd.Timestamp(d).strftime("%Y-%m") for d in dates]
        return [f"block-{i // self.test_window:03d}" for i in range(len(dates))]

    def _make_scaler(self) -> Any:
        """Instantiate a fresh calibrator (injected factory or TemperatureScaler)."""
        if self.scaler_factory is not None:
            return self.scaler_factory()
        from advisor.calibration.temperature import TemperatureScaler  # lazy import

        return TemperatureScaler()

    def _refit_calibrator(
        self,
        records: list[dict[str, Any]],
        current_date: pd.Timestamp,
        fold: str,
    ) -> tuple[Any, dict[str, Any]]:
        """Refit the calibrator on all labels realised strictly before ``current_date``.

        Args:
            records: Recommendations made so far.
            current_date: First decision date of the new fold.
            fold: Fold label.

        Returns:
            ``(scaler_or_None, log_entry)``.
        """
        entry: dict[str, Any] = {
            "fold": fold,
            "fitted_at": pd.Timestamp(current_date).date().isoformat(),
            "n_samples": 0,
            "temperature": None,
            "status": "disabled",
        }
        if not self.calibrate or self._calibration_disabled:
            return None, entry

        current = pd.Timestamp(current_date)
        labelled = [
            r
            for r in records
            if pd.notna(r.get("label_date"))
            and pd.Timestamp(r["label_date"]) < current
            and math.isfinite(float(r.get("raw_confidence", np.nan)))
            and math.isfinite(float(r.get("correct", np.nan)))
        ]
        if self.train_window is not None:
            labelled = labelled[-self.train_window :]
        entry["n_samples"] = len(labelled)
        if len(labelled) < self.calibration_min_samples:
            entry["status"] = "insufficient_samples"
            return None, entry

        confidences = [float(r["raw_confidence"]) for r in labelled]
        labels = [int(r["correct"]) for r in labelled]
        try:
            scaler = self._make_scaler()
            scaler.fit(confidences, labels)
        except (NotImplementedError, ImportError) as exc:
            log.warning("Calibration unavailable (%s); continuing with raw confidences.", exc)
            self._calibration_disabled = True
            entry["status"] = f"unavailable: {exc}"
            return None, entry
        except Exception as exc:
            log.warning("Calibrator fit failed on fold %s: %s", fold, exc)
            entry["status"] = f"failed: {exc}"
            return None, entry

        temperature = getattr(scaler, "temperature", None)
        entry["temperature"] = float(temperature) if temperature is not None else None
        entry["status"] = "fitted"
        log.info(
            "Fold %s: calibrator fitted on %d samples (T=%s).",
            fold,
            len(labelled),
            f"{entry['temperature']:.4f}" if entry["temperature"] is not None else "n/a",
        )
        return scaler, entry

    @staticmethod
    def _apply_scaler(scaler: Any, raw: list[float]) -> list[float]:
        """Apply a fitted calibrator to raw confidences (NaNs pass through)."""
        if scaler is None or not raw:
            return list(raw)
        finite_pos = [i for i, c in enumerate(raw) if math.isfinite(c)]
        if not finite_pos:
            return list(raw)
        try:
            transformed = list(scaler.transform([raw[i] for i in finite_pos]))
        except Exception as exc:
            log.warning("Calibrator transform failed (%s); using raw confidences.", exc)
            return list(raw)
        if len(transformed) != len(finite_pos):
            log.warning(
                "Calibrator returned %d values for %d inputs; using raw.",
                len(transformed),
                len(finite_pos),
            )
            return list(raw)
        out = list(raw)
        for i, value in zip(finite_pos, transformed, strict=True):
            out[i] = float(min(max(float(value), 0.0), 1.0))
        return out

    def _features_for(self, ticker: str, row: pd.Series) -> dict[str, float] | None:
        """Build the NaN-free float feature dict for one ticker, or ``None``."""
        feats: dict[str, float] = {}
        for key in FEATURE_KEYS:
            if key not in row.index:
                return None
            value = row[key]
            try:
                value = float(value)
            except (TypeError, ValueError):
                return None
            if not math.isfinite(value):
                return None
            feats[key] = value
        return feats

    def _recommend_one(self, ticker: str, feats: dict[str, float], as_of: date) -> dict[str, Any]:
        """Query the advisor for one ticker, converting failures into a HOLD."""
        try:
            rec = self.advisor.recommend(ticker, feats, as_of=as_of)
            action = str(getattr(rec.action, "value", rec.action)).strip().upper()
            if action not in _ACTIONS:
                raise ValueError(f"invalid action {action!r}")
            raw_conf = float(getattr(rec, "raw_confidence", getattr(rec, "confidence", np.nan)))
            if not math.isfinite(raw_conf) or not 0.0 <= raw_conf <= 1.0:
                raw_conf = float("nan")
            return {
                "ticker": ticker,
                "action": action,
                "raw_confidence": raw_conf,
                "reason": str(getattr(rec, "reason", "")),
                "counterfactual": str(getattr(rec, "counterfactual", "")),
                "error": False,
            }
        except Exception as exc:
            self.n_errors += 1
            log.warning("Advisor failed for %s on %s: %s", ticker, as_of, exc)
            return {
                "ticker": ticker,
                "action": "HOLD",
                "raw_confidence": float("nan"),
                "reason": f"advisor error: {exc}",
                "counterfactual": "",
                "error": True,
            }

    def _recommend_date(
        self, d: pd.Timestamp, feature_rows: pd.DataFrame | None
    ) -> list[dict[str, Any]]:
        """Collect recommendations for every ticker with usable features on ``d``."""
        if feature_rows is None or feature_rows.empty:
            return []
        as_of = pd.Timestamp(d).date()
        jobs: list[tuple[str, dict[str, float]]] = []
        for ticker in sorted(feature_rows.index):
            feats = self._features_for(str(ticker), feature_rows.loc[ticker])
            if feats is not None:
                jobs.append((str(ticker), feats))
        if not jobs:
            return []
        self.n_llm_calls += len(jobs)
        if self.max_workers > 1 and len(jobs) > 1:
            with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
                results = list(
                    pool.map(lambda job: self._recommend_one(job[0], job[1], as_of), jobs)
                )
        else:
            results = [self._recommend_one(ticker, feats, as_of) for ticker, feats in jobs]
        return results

    def _report_progress(
        self,
        i: int,
        n_dates: int,
        d: pd.Timestamp,
        calls_total_estimate: int,
        t_start: float,
    ) -> None:
        """Invoke the progress callback (if any) with an ETA estimate."""
        if self.progress_callback is None:
            return
        elapsed = time.monotonic() - t_start
        done_dates = i + 1
        remaining_dates = n_dates - done_dates
        eta = (elapsed / done_dates) * remaining_dates if done_dates > 0 else float("nan")
        try:
            self.progress_callback(
                {
                    "decision_index": i,
                    "n_decisions": n_dates,
                    "date": pd.Timestamp(d).date().isoformat(),
                    "calls_done": int(self.n_llm_calls),
                    "calls_total_estimate": int(calls_total_estimate),
                    "errors": int(self.n_errors),
                    "elapsed_s": float(elapsed),
                    "eta_s": float(eta),
                }
            )
        except Exception as exc:  # pragma: no cover - never let logging kill a run
            log.debug("progress_callback raised: %s", exc)

    @staticmethod
    def _per_period_metrics(
        daily_returns: pd.Series,
        equity_curve: pd.Series,
        recommendations: pd.DataFrame,
    ) -> pd.DataFrame:
        """Per-calendar-month performance and recommendation statistics."""
        rets = pd.Series(daily_returns, dtype="float64").dropna()
        rows: list[dict[str, Any]] = []
        if not rets.empty:
            periods = pd.DatetimeIndex(rets.index).to_period("M")
            for period, r in rets.groupby(periods):
                eq = equity_curve.reindex(r.index).dropna()
                rows.append(
                    {
                        "period": str(period),
                        "n_days": int(len(r)),
                        "return": float(np.prod(1.0 + r.to_numpy()) - 1.0),
                        "annualised_vol": (
                            float(r.std(ddof=1) * math.sqrt(252)) if len(r) > 1 else float("nan")
                        ),
                        "sharpe": float(sharpe_ratio(r)),
                        "max_drawdown": float(max_drawdown(eq)),
                    }
                )
        perf = pd.DataFrame(rows).set_index("period") if rows else pd.DataFrame()

        if recommendations is None or recommendations.empty:
            return perf
        recs = recommendations.copy()
        recs["period"] = pd.to_datetime(recs["date"]).dt.to_period("M").astype(str)
        grouped = recs.groupby("period")
        rec_stats = pd.DataFrame(
            {
                "n_recommendations": grouped.size(),
                "n_BUY": grouped["action"].apply(lambda s: int((s == "BUY").sum())),
                "n_HOLD": grouped["action"].apply(lambda s: int((s == "HOLD").sum())),
                "n_SELL": grouped["action"].apply(lambda s: int((s == "SELL").sum())),
                "hit_rate": grouped["correct"].mean(),
                "mean_raw_confidence": grouped["raw_confidence"].mean(),
                "mean_calibrated_confidence": grouped["calibrated_confidence"].mean(),
            }
        )
        out = perf.join(rec_stats, how="outer") if not perf.empty else rec_stats
        out.index.name = "period"
        return out.sort_index()
