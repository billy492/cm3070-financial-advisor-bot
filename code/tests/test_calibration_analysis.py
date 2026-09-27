"""Tests for ``advisor.evaluation.calibration_analysis`` (post-hoc replay).

A synthetic recommendation log is written to a temporary results folder: a
HOLD-heavy advisor that states ~0.7 confidence but is right 30% of the time on
HOLD and 55% of the time on BUY/SELL. The replay must (i) keep raw
confidences until ``min_samples`` labels have realised, (ii) reproduce the
temperature-ceiling behaviour on the ``all`` subset, (iii) let Platt scaling
reach the sub-0.5 base rate, (iv) restrict the ``directional`` subset to
BUY/SELL, and (v) write the CSVs and figures the report will cite. Offline,
seeded, no LLM.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from advisor.evaluation import calibration_analysis as ca

TICKERS = [f"T{i:02d}" for i in range(12)]


def make_log(seed: int = 0, n_months: int = 10) -> pd.DataFrame:
    """Build a synthetic ``<tag>_recommendations.csv``-shaped frame."""
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    dates = pd.date_range("2025-01-03", periods=n_months * 4, freq="W-FRI")
    for d in dates:
        for t in TICKERS:
            action = rng.choice(["HOLD", "BUY", "SELL"], p=[0.7, 0.25, 0.05])
            hit = 0.30 if action == "HOLD" else 0.55
            correct = float(rng.uniform() < hit)
            rows.append(
                {
                    "date": d.date().isoformat(),
                    "ticker": t,
                    "action": action,
                    "raw_confidence": round(float(np.clip(rng.normal(0.72, 0.05), 0.5, 0.9)), 2),
                    "calibrated_confidence": np.nan,
                    "reason": "synthetic",
                    "counterfactual": "synthetic",
                    "realised_return": float(rng.normal(0.0, 0.03)),
                    "label_date": (d + pd.Timedelta(days=8)).date().isoformat(),
                    "correct": correct,
                    "fold": d.strftime("%Y-%m"),
                    "error": False,
                }
            )
    df = pd.DataFrame(rows)
    # Last two decision dates have no realised label yet, as in the real log.
    unlabelled = df["date"].isin([d.date().isoformat() for d in dates[-2:]])
    df.loc[unlabelled, ["correct", "realised_return"]] = np.nan
    df.loc[unlabelled, "label_date"] = np.nan
    return df


@pytest.fixture()
def results_dir(tmp_path: Path) -> Path:
    """A results folder holding one synthetic recommendation log."""
    make_log().to_csv(tmp_path / "fake_recommendations.csv", index=False)
    return tmp_path


# --------------------------------------------------------------------------- #
# Loading and subsetting
# --------------------------------------------------------------------------- #
def test_load_parses_types_and_sorts(results_dir: Path) -> None:
    """Dates become timestamps, numerics floats, actions upper-case, sorted."""
    df = ca.load_recommendations(results_dir / "fake_recommendations.csv")
    assert pd.api.types.is_datetime64_any_dtype(df["date"])
    assert pd.api.types.is_datetime64_any_dtype(df["label_date"])
    assert df["raw_confidence"].dtype == "float64"
    assert df["correct"].isna().sum() == 2 * len(TICKERS)
    assert df["date"].is_monotonic_increasing


def test_load_requires_columns(tmp_path: Path) -> None:
    """A log without the walk-forward columns is rejected."""
    pd.DataFrame({"date": ["2025-01-03"], "ticker": ["X"]}).to_csv(tmp_path / "x.csv", index=False)
    with pytest.raises(ValueError, match="missing columns"):
        ca.load_recommendations(tmp_path / "x.csv")


def test_directional_subset_keeps_only_buy_sell(results_dir: Path) -> None:
    """``directional`` drops HOLD; ``all`` keeps everything; unknown fails."""
    df = ca.load_recommendations(results_dir / "fake_recommendations.csv")
    directional = ca.select_subset(df, "directional")
    assert set(directional["action"]) <= {"BUY", "SELL"}
    assert len(directional) == int(df["action"].isin(["BUY", "SELL"]).sum())
    assert len(ca.select_subset(df, "all")) == len(df)
    with pytest.raises(ValueError, match="subset"):
        ca.select_subset(df, "nope")


# --------------------------------------------------------------------------- #
# Walk-forward replay
# --------------------------------------------------------------------------- #
def test_raw_method_is_identity(results_dir: Path) -> None:
    """``raw`` returns the raw confidences and an identity log per fold."""
    df = ca.load_recommendations(results_dir / "fake_recommendations.csv")
    cal, fold_log = ca.walk_forward_calibrate(df, "raw")
    np.testing.assert_array_equal(cal, df["raw_confidence"].to_numpy())
    assert {e["status"] for e in fold_log} == {"identity"}
    assert len(fold_log) == df["fold"].nunique()


@pytest.mark.parametrize("method", ["temperature", "platt", "base_rate"])
def test_early_folds_keep_raw_until_min_samples(results_dir: Path, method: str) -> None:
    """Folds fitted before ``min_samples`` labels exist keep raw confidences."""
    df = ca.load_recommendations(results_dir / "fake_recommendations.csv")
    cal, fold_log = ca.walk_forward_calibrate(df, method, min_samples=50)
    raw = df["raw_confidence"].to_numpy()
    folds = df["fold"].to_numpy()
    assert fold_log[0]["status"] == "insufficient_samples"
    assert fold_log[0]["n_samples"] == 0
    first = folds == fold_log[0]["fold"]
    np.testing.assert_array_equal(cal[first], raw[first])
    fitted = [e for e in fold_log if e["status"] == "fitted"]
    assert fitted, "later folds must be fitted"
    # Training sets only ever grow (expanding window) and use past labels only.
    samples = [e["n_samples"] for e in fold_log]
    assert samples == sorted(samples)
    last = folds == fitted[-1]["fold"]
    assert not np.array_equal(cal[last], raw[last])


def test_training_uses_only_labels_realised_before_fold_start(results_dir: Path) -> None:
    """The sample count at a fold equals the labels whose date precedes it."""
    df = ca.load_recommendations(results_dir / "fake_recommendations.csv")
    _, fold_log = ca.walk_forward_calibrate(df, "temperature", min_samples=1)
    for entry in fold_log[1:]:
        start = pd.Timestamp(entry["fitted_at"])
        expected = int(((df["label_date"] < start) & df["correct"].notna()).sum())
        assert entry["n_samples"] == expected


def test_platt_reaches_sub_half_base_rate_temperature_does_not(results_dir: Path) -> None:
    """The structural finding, reproduced on the synthetic log."""
    df = ca.load_recommendations(results_dir / "fake_recommendations.csv")
    raw = df["raw_confidence"].to_numpy()
    correct = df["correct"].to_numpy()
    rows = {}
    for method in ("raw", "temperature", "platt", "base_rate"):
        cal, fold_log = ca.walk_forward_calibrate(df, method, min_samples=50)
        rows[method] = ca.summarise("all", method, raw, cal, correct, fold_log)

    assert rows["raw"]["hit_rate"] < 0.5
    assert rows["temperature"]["temperature"] == pytest.approx(20.0, rel=1e-3)
    assert rows["temperature"]["frac_below_half"] == 0.0
    assert rows["platt"]["frac_below_half"] > 0.5
    assert rows["platt"]["frac_crossed_half"] > 0.5
    assert rows["platt"]["intercept"] < 0.0
    assert rows["platt"]["ece"] < rows["temperature"]["ece"] < rows["raw"]["ece"]
    assert rows["platt"]["brier"] < rows["temperature"]["brier"] < rows["raw"]["brier"]
    assert rows["base_rate"]["sharpness"] < rows["raw"]["sharpness"]
    assert rows["temperature"]["frac_crossed_half"] == 0.0


def test_unknown_method_rejected(results_dir: Path) -> None:
    """Only the four documented methods are accepted."""
    df = ca.load_recommendations(results_dir / "fake_recommendations.csv")
    with pytest.raises(ValueError, match="method"):
        ca.walk_forward_calibrate(df, "isotonic")


def test_summarise_on_empty_labels_returns_nans() -> None:
    """No labelled rows gives ``n = 0`` and NaN metrics rather than an error."""
    nan = np.array([np.nan, np.nan])
    row = ca.summarise("all", "raw", np.array([0.6, 0.7]), np.array([0.6, 0.7]), nan, [])
    assert row["n"] == 0
    assert np.isnan(row["ece"])


# --------------------------------------------------------------------------- #
# End to end
# --------------------------------------------------------------------------- #
def test_run_analysis_writes_tables_and_figures(results_dir: Path) -> None:
    """One row per (subset, method); per-fold log; reliability diagrams."""
    summary, folds = ca.run_analysis("fake", results_dir, min_samples=50)
    assert len(summary) == len(ca.SUBSETS) * len(ca.METHODS)
    assert set(summary["subset"]) == set(ca.SUBSETS)
    assert set(summary["method"]) == set(ca.METHODS)
    assert (summary["tag"] == "fake").all()
    directional_n = summary.loc[summary["subset"] == "directional", "n"].unique()
    all_n = summary.loc[summary["subset"] == "all", "n"].unique()
    assert len(directional_n) == 1 and len(all_n) == 1
    assert 0 < directional_n[0] < all_n[0]
    assert len(folds) == len(ca.SUBSETS) * len(ca.METHODS) * 10
    assert {"tag", "subset", "method", "fold", "status"} <= set(folds.columns)

    assert (results_dir / "calibration_analysis_fake.csv").exists()
    assert (results_dir / "calibration_analysis_fake_folds.csv").exists()
    for subset in ca.SUBSETS:
        for ext in ("png", "svg"):
            assert (results_dir / "figures" / f"reliability_analysis_fake_{subset}.{ext}").exists()

    written = pd.read_csv(results_dir / "calibration_analysis_fake.csv")
    assert list(written.columns) == list(summary.columns)


def test_main_cli_returns_zero_and_prints_table(
    results_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The CLI runs offline, skips figures on request and prints the summary."""
    code = ca.main(["--tags", "fake", "--results-dir", str(results_dir), "--no-figures", "--quiet"])
    assert code == 0
    out = capsys.readouterr().out
    assert "post-hoc" in out
    assert "platt" in out and "directional" in out
    assert not (results_dir / "figures").exists()


def test_main_missing_tag_returns_one(results_dir: Path) -> None:
    """A tag without a recommendation log is reported and counted as a failure."""
    code = ca.main(
        ["--tags", "ghost", "--results-dir", str(results_dir), "--no-figures", "--quiet"]
    )
    assert code == 1
