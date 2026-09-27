"""Tests for ``advisor.evaluation.backtest`` -- the walk-forward engine.

A deterministic :class:`FakeAdvisor` (the momentum rule of the heuristic
ablation arm) records every call so the tests can prove the engine asks for
recommendations only on decision dates, hands the advisor exactly the
indicator row of that date (no look-ahead: truncating the future leaves every
earlier call untouched), labels recommendations with the forward horizon
return, refits the calibrator only once enough realised labels exist and
never lets calibration change an action or the portfolio, turns advisor
failures into HOLDs, and applies the documented held-set portfolio rule.
The :class:`FakeScaler` stands in for ``TemperatureScaler`` so the tests do
not depend on the calibration module. All offline and deterministic.
"""

from __future__ import annotations

import math
from datetime import date

import numpy as np
import pandas as pd
import pytest

from advisor.evaluation.backtest import (
    FEATURE_KEYS,
    RECOMMENDATION_COLUMNS,
    WalkForwardBacktest,
    ablation_table,
    evaluate_calibration,
    label_correctness,
)
from advisor.evaluation.metrics import expected_calibration_error
from advisor.evaluation.portfolio import decision_dates, pivot_prices, slice_window
from advisor.features.indicators import add_features_by_ticker
from advisor.recommender.schema import Action, Recommendation
from tests.test_portfolio import make_prices

START = date(2021, 9, 1)
END = date(2022, 6, 1)


class FakeAdvisor:
    """Deterministic momentum-rule advisor that records every call.

    Args:
        fail_tickers: Tickers for which ``recommend`` raises, to exercise the
            engine's error handling.
    """

    def __init__(self, fail_tickers: set[str] | None = None) -> None:
        """Start with an empty call log."""
        self.calls: list[tuple[str, date, dict[str, float]]] = []
        self.fail_tickers = set(fail_tickers or ())

    def recommend(self, ticker: str, features: dict, *, as_of: date) -> Recommendation:
        """Apply the rule ``BUY if mom>0 and close>sma_50; SELL if both negative``."""
        self.calls.append((ticker, as_of, dict(features)))
        if ticker in self.fail_tickers:
            raise RuntimeError("simulated advisor failure")
        mom, close, sma = features["mom_10"], features["close"], features["sma_50"]
        if mom > 0 and close > sma:
            action = Action.BUY
        elif mom < 0 and close < sma:
            action = Action.SELL
        else:
            action = Action.HOLD
        confidence = float(np.clip(0.5 + 4.0 * abs(mom), 0.5, 0.95))
        return Recommendation(
            ticker=ticker,
            action=action,
            confidence=confidence,
            raw_confidence=confidence,
            reason=f"mom={mom:+.4f}",
            counterfactual="I would change my mind if momentum flipped sign.",
            as_of=as_of,
        )


class FakeScaler:
    """Stand-in calibrator: shrinks confidences halfway towards 0.5."""

    def __init__(self) -> None:
        """Unfitted until ``fit`` is called."""
        self.temperature: float | None = None
        self.n_fit = 0

    def fit(self, confidences: list[float], labels: list[int]) -> FakeScaler:
        """Record the sample size and set a fixed temperature."""
        assert len(confidences) == len(labels)
        assert {int(label) for label in labels} <= {0, 1}
        assert all(0.0 <= float(c) <= 1.0 for c in confidences)
        self.n_fit = len(confidences)
        self.temperature = 2.0
        return self

    def transform(self, confidences: list[float]) -> list[float]:
        """Order- and side-preserving shrink towards 0.5."""
        assert self.temperature is not None
        return [0.5 + (float(c) - 0.5) / self.temperature for c in confidences]


class UnavailableScaler:
    """Calibrator whose ``fit`` is a stub (mimics an unimplemented module)."""

    temperature = None

    def fit(self, confidences: list[float], labels: list[int]) -> UnavailableScaler:
        """Always raise, like the stub-era ``TemperatureScaler``."""
        raise NotImplementedError("stub")

    def transform(self, confidences: list[float]) -> list[float]:  # pragma: no cover
        """Never reached."""
        return list(confidences)


@pytest.fixture
def prices() -> pd.DataFrame:
    """Three tickers, 400 business days from 2021-01-04 (window starts 2021-09-01)."""
    return make_prices(["AAA", "BBB", "CCC"], n=400, start=date(2021, 1, 4), seed=3)


def _run(prices: pd.DataFrame, advisor: FakeAdvisor | None = None, **overrides):
    """Run a backtest with test-friendly defaults; return ``(result, advisor)``."""
    advisor = advisor if advisor is not None else FakeAdvisor()
    kwargs = {
        "start": START,
        "end": END,
        "calibration_min_samples": 10,
        "scaler_factory": FakeScaler,
    }
    kwargs.update(overrides)
    return WalkForwardBacktest(advisor, **kwargs).run(prices), advisor


def _window_index(prices: pd.DataFrame) -> pd.DatetimeIndex:
    return slice_window(pivot_prices(prices), START, END).index


# --------------------------------------------------------------------------- #
# Contract and scheduling
# --------------------------------------------------------------------------- #
def test_run_contract(prices: pd.DataFrame) -> None:
    """The result dict has the documented keys and a well-formed log."""
    result, advisor = _run(prices)
    for key in (
        "equity_curve",
        "daily_returns",
        "trades",
        "per_period_metrics",
        "recommendations",
        "calibration",
        "calibration_metrics",
        "summary",
        "n_llm_calls",
        "n_errors",
        "decision_dates",
        "config",
    ):
        assert key in result
    assert result["equity_curve"].iloc[0] == 100_000.0
    assert result["equity_curve"].index.equals(_window_index(prices))
    recs = result["recommendations"]
    assert list(recs.columns) == list(RECOMMENDATION_COLUMNS)
    assert result["n_llm_calls"] == len(recs) == len(advisor.calls)
    assert result["n_errors"] == 0
    assert not recs.duplicated(subset=["date", "ticker"]).any()
    assert "deflated_sharpe" in result["summary"]
    assert all(len(p) == 7 for p in result["per_period_metrics"].index)  # "YYYY-MM"
    assert result["config"]["decision_freq"] == "W-FRI"


def test_recommendations_only_on_decision_dates(prices: pd.DataFrame) -> None:
    """Calls happen once per ticker per weekly decision date (plus the first day)."""
    result, advisor = _run(prices)
    window = _window_index(prices)
    expected = decision_dates(window, "W-FRI")
    if expected[0] != window[0]:
        expected.insert(0, window[0])
    assert result["decision_dates"] == expected
    assert set(pd.to_datetime(result["recommendations"]["date"])) == set(expected)
    assert {pd.Timestamp(as_of) for _, as_of, _ in advisor.calls} == set(expected)
    assert all(isinstance(as_of, date) for _, as_of, _ in advisor.calls)
    assert len(advisor.calls) == len(expected) * 3


def test_features_match_indicator_rows(prices: pd.DataFrame) -> None:
    """The feature dict equals the ``add_features_by_ticker`` row of that date."""
    _, advisor = _run(prices)
    feats = add_features_by_ticker(prices)
    feats["date"] = pd.to_datetime(feats["date"])
    indexed = feats.set_index(["ticker", "date"])
    for ticker, as_of, features in advisor.calls:
        assert set(features) == set(FEATURE_KEYS)
        row = indexed.loc[(ticker, pd.Timestamp(as_of))]
        for key in FEATURE_KEYS:
            assert features[key] == pytest.approx(float(row[key]))
            assert math.isfinite(features[key])


def test_no_lookahead_truncating_the_future_leaves_earlier_calls_unchanged(
    prices: pd.DataFrame,
) -> None:
    """Features and actions up to ``t`` do not depend on data after ``t``."""
    full, full_advisor = _run(prices)
    cutoff = full["decision_dates"][8]
    truncated = prices[pd.to_datetime(prices["date"]) <= cutoff]
    _, short_advisor = _run(truncated, end=cutoff + pd.Timedelta(days=1))

    full_calls = [(t, a, f) for t, a, f in full_advisor.calls if pd.Timestamp(a) <= cutoff]
    assert len(full_calls) == len(short_advisor.calls) > 0
    for (t1, a1, f1), (t2, a2, f2) in zip(full_calls, short_advisor.calls, strict=True):
        assert (t1, a1) == (t2, a2)
        assert f1 == f2


def test_tickers_filter_and_progress_callback(prices: pd.DataFrame) -> None:
    """``tickers`` restricts the universe; the callback fires once per decision."""
    seen: list[dict] = []
    result, _ = _run(prices, tickers=["AAA"], progress_callback=seen.append)
    assert set(result["recommendations"]["ticker"]) == {"AAA"}
    assert len(seen) == len(result["decision_dates"])
    assert seen[-1]["decision_index"] == len(result["decision_dates"]) - 1
    assert seen[-1]["calls_done"] == result["n_llm_calls"]
    assert math.isfinite(seen[0]["eta_s"])


# --------------------------------------------------------------------------- #
# Labels
# --------------------------------------------------------------------------- #
def test_label_correctness_rule() -> None:
    """BUY needs a rise, SELL a fall, HOLD a move inside the threshold."""
    actions = ["BUY", "BUY", "SELL", "SELL", "HOLD", "HOLD", "HOLD", "BUY"]
    returns = [0.02, -0.01, -0.02, 0.01, 0.005, -0.02, np.nan, np.nan]
    got = label_correctness(actions, returns, hold_threshold=0.01)
    assert got.tolist()[:6] == [1.0, 0.0, 1.0, 0.0, 1.0, 0.0]
    assert np.isnan(got.iloc[6]) and np.isnan(got.iloc[7])
    wide = label_correctness(actions, returns, hold_threshold=0.03)
    assert wide.tolist()[4:6] == [1.0, 1.0]
    series = pd.Series(actions, index=list("abcdefgh"))
    assert label_correctness(series, returns).index.equals(series.index)


def test_realised_return_and_correct_in_run(prices: pd.DataFrame) -> None:
    """Realised return is the forward 5-day adj_close return; correctness follows it."""
    result, _ = _run(prices, horizon_days=5)
    recs = result["recommendations"]
    wide = pivot_prices(prices)
    labelled = recs[recs["realised_return"].notna()]
    assert len(labelled) > 0
    for _, row in labelled.head(20).iterrows():
        pos = wide.index.get_loc(pd.Timestamp(row["date"]))
        expected = wide[row["ticker"]].iloc[pos + 5] / wide[row["ticker"]].iloc[pos] - 1.0
        assert row["realised_return"] == pytest.approx(expected)
        assert pd.Timestamp(row["label_date"]) == wide.index[pos + 5]
        rule = label_correctness([row["action"]], [row["realised_return"]]).iloc[0]
        assert row["correct"] == rule
    # Decisions in the last week of the history have no label yet.
    tail = recs[pd.to_datetime(recs["date"]) > wide.index[-6]]
    assert tail["realised_return"].isna().all()


# --------------------------------------------------------------------------- #
# Calibration
# --------------------------------------------------------------------------- #
def test_calibration_fires_after_min_samples_and_preserves_actions(
    prices: pd.DataFrame,
) -> None:
    """Early folds are raw; later folds are scaled; actions and equity are untouched."""
    calibrated, _ = _run(prices, calibration_min_samples=10)
    uncalibrated, _ = _run(prices, calibrate=False)

    log = calibrated["calibration"]
    assert log[0]["status"] == "insufficient_samples"
    fitted = [entry for entry in log if entry["status"] == "fitted"]
    assert fitted and all(entry["n_samples"] >= 10 for entry in fitted)
    assert all(entry["temperature"] == 2.0 for entry in fitted)
    assert all(entry["status"] == "disabled" for entry in uncalibrated["calibration"])

    recs = calibrated["recommendations"]
    fitted_folds = {entry["fold"] for entry in fitted}
    scaled = recs[recs["fold"].isin(fitted_folds)]
    raw_only = recs[~recs["fold"].isin(fitted_folds)]
    assert np.allclose(
        scaled["calibrated_confidence"], 0.5 + (scaled["raw_confidence"] - 0.5) / 2.0
    )
    assert np.allclose(raw_only["calibrated_confidence"], raw_only["raw_confidence"])
    assert (recs["calibrated_confidence"] > 0.5).equals(recs["raw_confidence"] > 0.5)

    pd.testing.assert_series_equal(
        recs["action"], uncalibrated["recommendations"]["action"], check_names=False
    )
    pd.testing.assert_series_equal(calibrated["equity_curve"], uncalibrated["equity_curve"])
    metrics = calibrated["calibration_metrics"]
    assert metrics["n"] > 0 and 0.0 <= metrics["ece_raw"] <= 1.0


def test_calibration_uses_only_labels_realised_before_the_fold(prices: pd.DataFrame) -> None:
    """The calibration set is exactly the earlier recommendations whose label date has passed."""
    result, _ = _run(prices, calibration_min_samples=10)
    recs = result["recommendations"]
    for entry in result["calibration"]:
        fitted_at = pd.Timestamp(entry["fitted_at"])
        available = recs[
            (pd.to_datetime(recs["label_date"]) < fitted_at)
            & recs["correct"].notna()
            & recs["raw_confidence"].notna()
        ]
        assert entry["n_samples"] == len(available)


def test_train_window_caps_the_calibration_set(prices: pd.DataFrame) -> None:
    """``train_window`` keeps only the most recent labelled samples."""
    result, _ = _run(prices, calibration_min_samples=5, train_window=12)
    fitted = [e for e in result["calibration"] if e["status"] == "fitted"]
    assert fitted and all(e["n_samples"] <= 12 for e in fitted)


def test_calibration_unavailable_falls_back_to_raw(prices: pd.DataFrame) -> None:
    """A stub calibrator disables calibration instead of aborting the run."""
    result, _ = _run(prices, scaler_factory=UnavailableScaler, calibration_min_samples=5)
    recs = result["recommendations"]
    assert np.allclose(recs["calibrated_confidence"], recs["raw_confidence"])
    statuses = [e["status"] for e in result["calibration"]]
    assert any(s.startswith("unavailable") for s in statuses)
    assert not any(s == "fitted" for s in statuses)


# --------------------------------------------------------------------------- #
# Errors, portfolio rule, costs, folds, threading
# --------------------------------------------------------------------------- #
def test_advisor_errors_become_hold(prices: pd.DataFrame) -> None:
    """Exceptions are counted and recorded as HOLD with unknown confidence."""
    result, advisor = _run(prices, advisor=FakeAdvisor(fail_tickers={"BBB"}))
    recs = result["recommendations"]
    failed = recs[recs["error"]]
    assert result["n_errors"] == len(failed) == sum(1 for t, _, _ in advisor.calls if t == "BBB")
    assert (failed["ticker"] == "BBB").all()
    assert (failed["action"] == "HOLD").all()
    assert failed["raw_confidence"].isna().all()
    assert failed["calibrated_confidence"].isna().all()
    assert failed["correct"].isna().all()
    assert np.isfinite(result["equity_curve"].to_numpy()).all()
    assert result["calibration_metrics"]["n"] == len(
        recs[~recs["error"]].dropna(subset=["correct"])
    )


def test_portfolio_rule_held_set(prices: pd.DataFrame) -> None:
    """Held set = BUY ∪ (previously held ∩ HOLD), equal-weighted; cash when empty."""
    result, _ = _run(prices)
    recs = result["recommendations"]
    weights = result["weights"]
    held: set[str] = set()
    for d, group in recs.groupby("date", sort=True):
        buys = set(group.loc[group["action"] == "BUY", "ticker"])
        holds = set(group.loc[group["action"] == "HOLD", "ticker"])
        held = buys | (held & holds)
        if pd.Timestamp(d) not in weights.index:  # final date carries no next mark
            continue
        row = weights.loc[pd.Timestamp(d)]
        assert set(row[row > 0].index) == held
        if held:
            assert np.allclose(row[row > 0].to_numpy(), 1.0 / len(held))
            assert row.sum() == pytest.approx(1.0)
        else:
            assert row.sum() == 0.0


def test_costs_reduce_returns(prices: pd.DataFrame) -> None:
    """The same recommendation stream ends lower under a higher cost."""
    free, _ = _run(prices, cost_bps=0.0)
    costly, _ = _run(prices, cost_bps=50.0)
    assert free["turnover"].sum() > 0
    assert free["equity_curve"].iloc[-1] > costly["equity_curve"].iloc[-1]
    pd.testing.assert_frame_equal(
        free["recommendations"].drop(columns=["reason"]),
        costly["recommendations"].drop(columns=["reason"]),
    )


def test_invalid_configuration_is_rejected() -> None:
    """Overlapping folds, zero horizon and zero workers are errors."""
    with pytest.raises(ValueError):
        WalkForwardBacktest(FakeAdvisor(), test_window=4, step=2)
    with pytest.raises(ValueError):
        WalkForwardBacktest(FakeAdvisor(), horizon_days=0)
    with pytest.raises(ValueError):
        WalkForwardBacktest(FakeAdvisor(), max_workers=0)
    with pytest.raises(ValueError):
        WalkForwardBacktest(FakeAdvisor(), train_window=0)


def test_block_folds_via_test_window(prices: pd.DataFrame) -> None:
    """``test_window`` groups decision dates into fixed-size blocks."""
    result, _ = _run(prices, test_window=4, step=4)
    folds = result["recommendations"]["fold"].unique().tolist()
    assert folds[0] == "block-000"
    assert len(result["calibration"]) == math.ceil(len(result["decision_dates"]) / 4)
    assert result["config"]["test_window"] == 4


def test_threaded_calls_give_identical_results(prices: pd.DataFrame) -> None:
    """``max_workers > 1`` changes nothing but wall-clock time."""
    serial, _ = _run(prices, max_workers=1)
    threaded, _ = _run(prices, max_workers=4)
    pd.testing.assert_frame_equal(serial["recommendations"], threaded["recommendations"])
    pd.testing.assert_series_equal(serial["equity_curve"], threaded["equity_curve"])


def test_run_rejects_bad_input(prices: pd.DataFrame) -> None:
    """Missing columns or an empty window raise ``ValueError``."""
    with pytest.raises(ValueError):
        WalkForwardBacktest(FakeAdvisor()).run(prices.drop(columns=["adj_close"]))
    with pytest.raises(ValueError):
        WalkForwardBacktest(FakeAdvisor(), start=date(2030, 1, 1), end=date(2031, 1, 1)).run(prices)


# --------------------------------------------------------------------------- #
# Pure evaluation helpers
# --------------------------------------------------------------------------- #
def test_evaluate_calibration_on_crafted_log() -> None:
    """ECE/Brier/hit rate agree with direct computation; per-action stats present."""
    frame = pd.DataFrame(
        {
            "action": ["BUY", "BUY", "SELL", "HOLD", "BUY", "SELL"],
            "raw_confidence": [0.9, 0.8, 0.7, 0.6, 0.55, np.nan],
            "calibrated_confidence": [0.7, 0.65, 0.6, 0.55, 0.52, np.nan],
            "correct": [1.0, 0.0, 1.0, 1.0, np.nan, 1.0],
        }
    )
    out = evaluate_calibration(frame, n_bins=5)
    assert out["n"] == 4 and out["n_total"] == 6
    assert out["hit_rate"] == pytest.approx(0.75)
    assert out["ece_raw"] == pytest.approx(
        expected_calibration_error([0.9, 0.8, 0.7, 0.6], [1, 0, 1, 1], n_bins=5)
    )
    assert out["ece_calibrated"] == pytest.approx(
        expected_calibration_error([0.7, 0.65, 0.6, 0.55], [1, 0, 1, 1], n_bins=5)
    )
    assert out["brier_raw"] == pytest.approx(
        np.mean((np.array([0.9, 0.8, 0.7, 0.6]) - np.array([1, 0, 1, 1])) ** 2)
    )
    assert out["n_BUY"] == 3 and out["n_SELL"] == 2 and out["n_HOLD"] == 1
    assert out["hit_rate_BUY"] == pytest.approx(0.5)
    assert out["ece_improvement"] == pytest.approx(out["ece_raw"] - out["ece_calibrated"])
    empty = evaluate_calibration(pd.DataFrame(columns=RECOMMENDATION_COLUMNS))
    assert empty["n"] == 0 and np.isnan(empty["ece_raw"])


def test_ablation_table(prices: pd.DataFrame) -> None:
    """Advisors are rows; relative columns reference the heuristic arm."""
    result, _ = _run(prices)
    table = ablation_table({"llm": result, "heuristic": result}, reference="heuristic")
    assert list(table.index) == ["llm", "heuristic"]
    for col in (
        "sharpe",
        "deflated_sharpe",
        "hit_rate",
        "ece_raw",
        "ece_calibrated",
        "n_llm_calls",
    ):
        assert col in table.columns
    assert table.loc["llm", "sharpe_pct_vs_reference"] == pytest.approx(0.0)
    assert table.loc["heuristic", "dsr_delta_vs_reference"] == pytest.approx(0.0)
    assert (table["reference"] == "heuristic").all()
    assert ablation_table({}).empty
