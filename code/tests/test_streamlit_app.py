"""Tests for ``streamlit_app.py`` rendered headlessly with ``AppTest``.

No Streamlit server and no network: ``requests`` is booby-trapped, the price
loader is replaced by a synthetic generator, and the LLM/result directories
are redirected to a temporary folder. The rule-based advisor ("Simple rule
(no AI)") drives the full flow so the four output cards can be asserted.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import requests
import streamlit as st
from streamlit.testing.v1 import AppTest

import advisor.config as config
from advisor.data import loader

APP_PATH = Path(__file__).resolve().parent.parent / "streamlit_app.py"
RULE_LABEL = "Simple rule (no AI)"
CARD_TITLES = ("**Recommendation**", "**Confidence**", "**Why**", "**What would change my mind**")


def _synthetic_prices(ticker: str, n: int = 260) -> pd.DataFrame:
    """Deterministic long-form OHLCV history for any ticker (about one year)."""
    rng = np.random.default_rng(abs(hash(ticker)) % (2**32))
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0005, 0.01, size=n)))
    dates = pd.bdate_range(end=pd.Timestamp(date(2025, 6, 30)), periods=n)
    return pd.DataFrame(
        {
            "date": dates.date,
            "ticker": ticker,
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "adj_close": close,
            "volume": rng.integers(1_000_000, 5_000_000, size=n).astype("int64"),
        }
    )


def _no_network(*args: Any, **kwargs: Any) -> Any:
    raise requests.ConnectionError("network access is disabled in tests")


@pytest.fixture
def app(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> AppTest:
    """An ``AppTest`` for the page with all external I/O stubbed out."""
    monkeypatch.setattr(requests, "get", _no_network)
    monkeypatch.setattr(requests, "post", _no_network)

    def fake_load_prices(tickers: Any, start: date, end: date, **kw: Any) -> pd.DataFrame:
        symbols = [tickers] if isinstance(tickers, str) else list(tickers)
        return pd.concat([_synthetic_prices(t.upper()) for t in symbols], ignore_index=True)

    monkeypatch.setattr(loader, "load_prices", fake_load_prices)
    monkeypatch.setattr(config, "LLM_CACHE_DIR", tmp_path / "llm_cache")
    monkeypatch.setattr(config, "RESULTS_DIR", tmp_path / "results")
    st.cache_data.clear()
    st.cache_resource.clear()
    yield AppTest.from_file(str(APP_PATH), default_timeout=60)
    st.cache_data.clear()
    st.cache_resource.clear()


def _markdown_texts(at: AppTest) -> list[str]:
    return [str(m.value) for m in at.markdown]


def _use_rule_model(at: AppTest) -> AppTest:
    return at.sidebar.selectbox(key="model").select(RULE_LABEL).run()


# --------------------------------------------------------------------------- #
# Initial render
# --------------------------------------------------------------------------- #
def test_page_renders_controls_without_a_server(app: AppTest) -> None:
    """The page imports, renders its controls and draws no result until asked."""
    at = app.run()

    assert not at.exception
    assert at.title[0].value == "Financial Advisor Bot"
    assert at.sidebar.selectbox(key="model").value == "Llama 3.1 8B"
    assert at.sidebar.selectbox(key="model").options == [
        "Llama 3.1 8B",
        "Qwen3 8B",
        RULE_LABEL,
    ]
    assert at.sidebar.selectbox(key="sector").value == "All sectors"
    assert at.sidebar.selectbox(key="ticker").value == "AAPL"
    assert at.sidebar.date_input(key="as_of").value == date(2025, 6, 30)
    assert at.button(key="get_rec").label == "Get recommendation"
    assert not any(t in _markdown_texts(at) for t in CARD_TITLES)
    assert any("Press **Get recommendation**" in i.value for i in at.info)
    # Ollama is unreachable (requests is booby-trapped): the sidebar says so.
    assert any("not reachable" in w.value for w in at.sidebar.warning)
    # Permanent disclaimer footer.
    assert any("not financial advice" in c.value for c in at.caption)


def test_sector_filter_narrows_tickers(app: AppTest) -> None:
    """Choosing a sector restricts the company list to that sector."""
    at = app.run()
    at = at.sidebar.selectbox(key="sector").select("energy").run()
    assert not at.exception
    assert at.sidebar.selectbox(key="ticker").options == ["COP", "CVX", "EOG", "SLB", "XOM"]


# --------------------------------------------------------------------------- #
# Rule-based recommendation: the four cards
# --------------------------------------------------------------------------- #
def test_rule_based_recommendation_shows_four_cards(app: AppTest) -> None:
    """With the no-AI advisor, pressing the button renders all four output cards."""
    at = _use_rule_model(app.run())
    at = at.button(key="get_rec").click().run()

    assert not at.exception
    assert not at.error
    texts = _markdown_texts(at)
    for title in CARD_TITLES:
        assert title in texts, f"missing card: {title}"
    # Recommendation card: a coloured BUY / HOLD / SELL heading.
    assert any(
        t.startswith("## :") and t.split("[")[1].rstrip("]") in ("BUY", "HOLD", "SELL")
        for t in texts
    )
    # Confidence card: a percentage heading, unadjusted caption (no calibration file).
    assert any(t.startswith("## ") and t.endswith("%") for t in texts)
    assert any("Unadjusted" in c.value for c in at.caption)
    # Why / counterfactual sentences come from the heuristic advisor (st.write -> markdown).
    assert any("momentum" in t for t in texts)
    assert any("I would change my mind if" in t for t in texts)
    assert at.button(key="check_claim").label == "Run the check"


def test_result_persists_across_reruns_and_clears_on_new_query(app: AppTest) -> None:
    """The cards survive an unrelated rerun but disappear when the query changes."""
    at = _use_rule_model(app.run())
    at = at.button(key="get_rec").click().run()
    assert "**Recommendation**" in _markdown_texts(at)

    at = at.run()  # plain rerun (e.g. widget interaction elsewhere)
    assert "**Recommendation**" in _markdown_texts(at)

    at = at.sidebar.selectbox(key="ticker").select("MSFT").run()
    assert "**Recommendation**" not in _markdown_texts(at)
    assert any("Press **Get recommendation**" in i.value for i in at.info)


def test_check_claim_runs_counterfactual_search(app: AppTest) -> None:
    """The 'Check this claim' button produces a sentence without crashing."""
    at = _use_rule_model(app.run())
    at = at.button(key="get_rec").click().run()
    at = at.button(key="check_claim").click().run()

    assert not at.exception
    assert at.success, "expected the claim-check sentence in a success box"
    sentence = at.success[0].value
    assert isinstance(sentence, str) and len(sentence) > 10


# --------------------------------------------------------------------------- #
# Graceful degradation and the backtest tab
# --------------------------------------------------------------------------- #
def test_llm_unreachable_degrades_gracefully(app: AppTest) -> None:
    """With Ollama down, an AI advisor shows a friendly error and the rule fallback."""
    at = app.run()  # default advisor: Llama 3.1 8B
    at = at.button(key="get_rec").click().run()

    assert not at.exception
    assert at.error, "expected a friendly error box"
    assert "not reachable" in at.error[0].value
    assert any(RULE_LABEL in i.value for i in at.info)
    assert not any(t in _markdown_texts(at) for t in CARD_TITLES)


def test_history_tab_without_results_shows_hint(app: AppTest) -> None:
    """The backtest tab explains how to produce results when none exist."""
    at = app.run()
    assert not at.exception
    assert any("No backtest results yet" in i.value for i in at.info)


def test_history_tab_shows_summary_table_when_present(app: AppTest, tmp_path: Path) -> None:
    """A summary CSV under results/ is rendered as a table."""
    results = tmp_path / "results"
    results.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"strategy": ["llama", "heuristic"], "sharpe": [0.8, 0.5]}).to_csv(
        results / "summary_table.csv", index=False
    )
    at = app.run()
    assert not at.exception
    assert len(at.dataframe) == 1
    table = at.dataframe[0].value
    assert list(table.columns) == ["Strategy", "Sharpe ratio"]
    assert list(table["Strategy"]) == ["Llama 3.1 8B advisor", "Simple rule (no AI)"]
    assert list(table["Sharpe ratio"]) == ["0.80", "0.50"]
    assert not any("No backtest results yet" in i.value for i in at.info)


def test_confidence_card_shows_track_record_when_log_present(
    app: AppTest, tmp_path: Path
) -> None:
    """With a walk-forward log present, the confidence card adds the held-out hit rate."""
    results = tmp_path / "results"
    results.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "action": ["BUY", "BUY", "HOLD", "HOLD", "SELL", "SELL"],
            "correct": [1.0, 0.0, 1.0, 0.0, 1.0, None],
        }
    ).to_csv(results / "heuristic_recommendations.csv", index=False)
    at = _use_rule_model(app.run())
    at = at.button(key="get_rec").click().run()

    assert not at.exception
    lines = [c.value for c in at.caption if c.value.startswith("Track record:")]
    assert len(lines) == 1
    assert "right" in lines[0] and "calls" in lines[0]
    assert any(pct in lines[0] for pct in ("50%", "100%"))


def test_no_track_record_without_log(app: AppTest) -> None:
    """Without a walk-forward log the card stays as before (no track-record line)."""
    at = _use_rule_model(app.run())
    at = at.button(key="get_rec").click().run()
    assert not any(c.value.startswith("Track record:") for c in at.caption)


def test_history_tab_finds_equity_figure_in_figures_folder(app: AppTest, tmp_path: Path) -> None:
    """The runner saves figures under results/figures/; the tab must show them from there."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures = tmp_path / "results" / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(2, 1))
    ax.plot([0, 1], [0, 1])
    fig.savefig(figures / "equity_curves.png")
    plt.close(fig)
    at = app.run()
    assert not at.exception
    assert len(at.image) == 1


def test_own_claim_caption_uses_experiment_results(app: AppTest, tmp_path: Path) -> None:
    """The claim card says how often this advisor's own claims held up in testing."""
    import json

    results = tmp_path / "results"
    results.mkdir(parents=True, exist_ok=True)
    summary = {"faithfulness_sample": {"minimal_flips": 18, "n_tested": 40}}
    (results / "counterfactual_llm_heuristic.json").write_text(json.dumps(summary))
    at = _use_rule_model(app.run())
    at = at.button(key="get_rec").click().run()
    assert not at.exception
    assert any("held up 18 times out of 40" in c.value for c in at.caption)


def test_no_claim_caption_without_experiment_results(app: AppTest) -> None:
    """Without experiment results the page makes no claim about the claim."""
    at = _use_rule_model(app.run())
    at = at.button(key="get_rec").click().run()
    assert not any("held up" in c.value for c in at.caption)


def test_why_card_shows_the_numbers_the_advisor_saw(app: AppTest) -> None:
    """The "Why" card lets the user check the reason against the real indicator values."""
    at = _use_rule_model(app.run())
    at = at.button(key="get_rec").click().run()
    assert not at.exception
    assert any(e.label == "Numbers the advisor saw" for e in at.expander)
    indicators = at.table[0].value
    assert "14-day RSI" in list(indicators.index)
    assert any("overbought" in c.value for c in at.caption)
