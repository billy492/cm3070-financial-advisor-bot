"""Tests for ``advisor.evaluation.run_evaluation`` -- the experiment runner.

The pure builders (ticker parsing, the summary table with its Deflated Sharpe
``n_trials`` and permutation p-values, the baseline/null tables, figure
rendering and the save/load round-trip) are tested on fake results. The CLI
is smoke-tested end to end, fully offline: synthetic parquet caches in a
temporary directory, a booby-trapped ``yfinance`` module so any network
attempt fails loudly, and an injected fake advisor. ``--resume`` must skip an
advisor whose results already exist.
"""

from __future__ import annotations

import json
import sys
import types
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from advisor.data.loader import _cache_path
from advisor.data.universe import UNIVERSE
from advisor.evaluation import run_evaluation as runner
from advisor.evaluation.backtest import WalkForwardBacktest
from advisor.evaluation.baselines import BASELINE_NAMES, run_all_baselines
from tests.test_backtest import FakeAdvisor, FakeScaler
from tests.test_portfolio import make_prices

HISTORY_START = date(2019, 6, 3)
START = date(2020, 7, 1)
END = date(2021, 3, 1)


class _ExplodingYFinance(types.ModuleType):
    """A ``yfinance`` stand-in whose every attribute access fails the test."""

    def __getattr__(self, name: str) -> object:
        raise AssertionError(f"yfinance.{name} accessed: the runner must stay offline.")


@pytest.fixture
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Install the exploding ``yfinance`` module for the duration of a test."""
    monkeypatch.setitem(sys.modules, "yfinance", _ExplodingYFinance("yfinance"))


def _write_cache(tickers: list[str], cache_dir: Path, *, seed: int = 21) -> pd.DataFrame:
    """Write synthetic per-ticker parquet caches the loader will read offline."""
    prices = make_prices(tickers, n=520, start=HISTORY_START, seed=seed)
    for ticker, group in prices.groupby("ticker"):
        path = _cache_path(str(ticker), cache_dir)
        path.parent.mkdir(parents=True, exist_ok=True)
        group.reset_index(drop=True).to_parquet(path, index=False)
    return prices


def _fake_curves(n: int = 120) -> dict[str, dict[str, pd.Series]]:
    """Synthetic equity curves for two advisors, four baselines and the null mean."""
    idx = pd.bdate_range("2025-01-01", periods=n)
    rng = np.random.default_rng(0)
    names = ["llama", "heuristic", *BASELINE_NAMES, "random_ensemble_mean"]
    out = {}
    for i, name in enumerate(names):
        rets = rng.normal(0.0004 * (i + 1), 0.01, n)
        curve = pd.Series(100_000.0 * np.exp(np.cumsum(rets)), index=idx, name=name)
        out[name] = {"equity_curve": curve, "daily_returns": curve.pct_change().dropna()}
    return out


# --------------------------------------------------------------------------- #
# Pure helpers
# --------------------------------------------------------------------------- #
def test_parse_tickers() -> None:
    """ALL / N / explicit lists resolve deterministically."""
    assert runner.parse_tickers("ALL") == list(UNIVERSE)
    assert runner.parse_tickers(None) == list(UNIVERSE)
    assert runner.parse_tickers("3") == list(UNIVERSE[:3])
    assert runner.parse_tickers("msft, aapl") == ["AAPL", "MSFT"]
    assert runner.parse_tickers(["nvda", "AAPL"]) == ["AAPL", "NVDA"]
    with pytest.raises(ValueError):
        runner.parse_tickers("0")
    with pytest.raises(ValueError):
        runner.parse_tickers("   ")


def test_build_summary_table_on_fake_results() -> None:
    """One row per strategy; DSR deflated by the number of candidate strategies."""
    strategies = _fake_curves()
    null = pd.Series(np.random.default_rng(1).normal(0.0, 1.0, 50))
    table = runner.build_summary_table(strategies, null_sharpes=null)
    assert list(table.index) == list(strategies)
    assert table.loc["llama", "kind"] == "advisor"
    assert table.loc["momentum_12_1", "kind"] == "baseline"
    assert table.loc["random_ensemble_mean", "kind"] == "null"
    assert (table["n_trials"] == 6).all()  # 2 advisors + 4 baselines
    assert ((table["p_value_vs_random"] > 0) & (table["p_value_vs_random"] <= 1)).all()
    assert ((table["deflated_sharpe"] >= 0) & (table["deflated_sharpe"] <= 1)).all()
    assert list(table.columns)[:2] == ["kind", "total_return"]

    override = runner.build_summary_table(strategies, n_trials=20)
    assert (override["n_trials"] == 20).all()
    assert override["p_value_vs_random"].isna().all()
    assert (override["deflated_sharpe"] <= table["deflated_sharpe"] + 1e-12).all()


def test_build_baselines_equity_and_null_table() -> None:
    """Baseline curves and the random band share one frame; the null table is per seed."""
    prices = make_prices(["AAA", "BBB", "CCC", "DDD"], n=520, start=HISTORY_START, seed=4)
    baselines = run_all_baselines(prices, start=START, end=END, n_seeds=3, seed=7)
    equity = runner.build_baselines_equity(baselines)
    expected = [
        *BASELINE_NAMES,
        "random_ensemble_mean",
        *(f"random_ensemble_p{p}" for p in ("05", "50", "95")),
    ]
    assert list(equity.columns) == expected
    assert equity.index.name == "date"
    null = runner.build_random_null_table(baselines["random_ensemble"])
    assert list(null["seed"]) == [7, 8, 9]
    assert {"sharpe", "sortino", "total_return", "max_drawdown"} <= set(null.columns)


def test_make_figures_writes_png_and_svg(tmp_path: Path) -> None:
    """Every figure is written in both formats with the Agg backend."""
    strategies = _fake_curves(n=80)
    curves = pd.DataFrame({k: v["equity_curve"] for k, v in strategies.items()})
    band = pd.DataFrame(
        {
            "p05": curves.min(axis=1) * 0.98,
            "p50": curves.median(axis=1),
            "p95": curves.max(axis=1) * 1.02,
        }
    )
    recs = pd.DataFrame(
        {
            "raw_confidence": np.linspace(0.5, 0.95, 40),
            "calibrated_confidence": np.linspace(0.5, 0.8, 40),
            "correct": np.tile([1.0, 0.0], 20),
        }
    )
    written = runner.make_figures(
        curves, tmp_path, band=band, recommendations_by_tag={"llama": recs}
    )
    names = {p.name for p in written}
    for stem in ("equity_curves", "drawdowns", "monthly_returns_heatmap", "reliability_llama"):
        assert f"{stem}.png" in names and f"{stem}.svg" in names
    assert all(p.exists() and p.stat().st_size > 0 for p in written)


def test_save_and_load_advisor_results_roundtrip(tmp_path: Path) -> None:
    """Saved CSV/JSON files rebuild an equivalent result dict."""
    prices = make_prices(["AAA", "BBB", "CCC"], n=400, start=date(2021, 1, 4), seed=3)
    result = WalkForwardBacktest(
        FakeAdvisor(),
        start=date(2021, 9, 1),
        end=date(2022, 6, 1),
        calibration_min_samples=10,
        scaler_factory=FakeScaler,
    ).run(prices)
    paths = runner.save_advisor_results("fake", result, tmp_path)
    assert all(p.exists() for p in paths.values())

    loaded = runner.load_advisor_results("fake", tmp_path)
    assert loaded is not None and loaded["resumed"] is True
    assert np.allclose(loaded["equity_curve"].to_numpy(), result["equity_curve"].to_numpy())
    assert loaded["equity_curve"].index.equals(result["equity_curve"].index)
    assert len(loaded["recommendations"]) == len(result["recommendations"])
    assert loaded["summary"]["sharpe"] == pytest.approx(result["summary"]["sharpe"])
    assert len(loaded["calibration"]) == len(result["calibration"])
    assert loaded["n_llm_calls"] == result["n_llm_calls"]
    assert loaded["calibration_metrics"]["ece_raw"] == pytest.approx(
        result["calibration_metrics"]["ece_raw"]
    )
    assert runner.load_advisor_results("missing", tmp_path) is None


def test_build_advisor_tags() -> None:
    """Known tags resolve; unknown tags raise with the list of valid ones."""
    heuristic = runner.build_advisor("heuristic")
    assert hasattr(heuristic, "recommend")
    with pytest.raises(ValueError, match="Unknown advisor tag"):
        runner.build_advisor("nope")


def test_manifest_and_versions(tmp_path: Path) -> None:
    """The manifest records config, seed, package versions and a git hash slot."""
    path = tmp_path / "run_manifest.json"
    manifest = runner.write_manifest(
        path, config={"seed": 7, "x": Path("y")}, extras={"tickers": ["A"]}
    )
    on_disk = json.loads(path.read_text())
    assert on_disk["seed"] == 7 and on_disk["tickers"] == ["A"]
    assert "git_hash" in on_disk and "timestamp" in on_disk
    assert "pandas" in on_disk["package_versions"] and "python" in on_disk["package_versions"]
    assert manifest["config"]["x"] == "y"


# --------------------------------------------------------------------------- #
# CLI smoke test (offline)
# --------------------------------------------------------------------------- #
def _cli_args(tmp_path: Path, *extra: str) -> list[str]:
    return [
        "--advisors",
        "heuristic",
        "--tickers",
        "3",
        "--cache-dir",
        str(tmp_path / "cache"),
        "--results-dir",
        str(tmp_path / "results"),
        "--history-start",
        HISTORY_START.isoformat(),
        "--start",
        START.isoformat(),
        "--end",
        END.isoformat(),
        "--seeds",
        "5",
        "--calibration-min-samples",
        "10",
        "--log-level",
        "WARNING",
        *extra,
    ]


def test_cli_smoke_offline(tmp_path: Path, no_network: None) -> None:
    """``--advisors heuristic --tickers 3`` runs end to end from parquet caches."""
    _write_cache([*UNIVERSE[:3], "SPY"], tmp_path / "cache")
    factory_calls: list[str] = []

    def factory(tag: str) -> FakeAdvisor:
        factory_calls.append(tag)
        return FakeAdvisor()

    assert runner.main(_cli_args(tmp_path), advisor_factory=factory) == 0
    assert factory_calls == ["heuristic"]

    results = tmp_path / "results"
    for name in (
        "heuristic_recommendations.csv",
        "heuristic_equity.csv",
        "heuristic_monthly.csv",
        "baselines_equity.csv",
        "summary_table.csv",
        "calibration_heuristic.json",
        "ablation.csv",
        "random_null.csv",
        "run_manifest.json",
        "figures/equity_curves.png",
        "figures/equity_curves.svg",
        "figures/drawdowns.png",
        "figures/reliability_heuristic.png",
    ):
        assert (results / name).exists(), name

    summary = pd.read_csv(results / "summary_table.csv", index_col="strategy")
    assert set(summary.index) == {"heuristic", *BASELINE_NAMES, "random_ensemble_mean"}
    assert (summary["n_trials"] == 5).all()
    assert summary["p_value_vs_random"].between(0, 1).all()
    assert len(pd.read_csv(results / "random_null.csv")) == 5

    recs = pd.read_csv(results / "heuristic_recommendations.csv")
    assert set(recs["ticker"]) == set(UNIVERSE[:3])
    assert recs["date"].min() >= START.isoformat() and recs["date"].max() < END.isoformat()

    calibration = json.loads((results / "calibration_heuristic.json").read_text())
    assert "ece_raw" in calibration["metrics"] and calibration["folds"]

    manifest = json.loads((results / "run_manifest.json").read_text())
    assert manifest["advisors_run"] == ["heuristic"] and manifest["advisors_resumed"] == []
    assert manifest["config"]["seed"] == 42 and manifest["tickers"] == list(UNIVERSE[:3])
    assert manifest["n_strategies"] == 6
    ablation = pd.read_csv(results / "ablation.csv", index_col="advisor")
    assert list(ablation.index) == ["heuristic"]


def test_cli_resume_skips_existing_advisor(tmp_path: Path, no_network: None) -> None:
    """With ``--resume`` an advisor with saved results is not re-run."""
    _write_cache([*UNIVERSE[:3], "SPY"], tmp_path / "cache")
    assert (
        runner.main(_cli_args(tmp_path, "--no-figures"), advisor_factory=lambda tag: FakeAdvisor())
        == 0
    )

    def must_not_build(tag: str) -> FakeAdvisor:
        raise AssertionError("advisor should have been resumed from disk")

    assert (
        runner.main(_cli_args(tmp_path, "--resume", "--no-figures"), advisor_factory=must_not_build)
        == 0
    )
    manifest = json.loads((tmp_path / "results" / "run_manifest.json").read_text())
    assert manifest["advisors_resumed"] == ["heuristic"] and manifest["advisors_run"] == []
    assert not (tmp_path / "results" / "figures" / "equity_curves.png").exists()
