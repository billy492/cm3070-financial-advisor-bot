"""Streamlit web interface for the Financial Advisor Bot (non-technical users).

Stage 6 of the design: for a chosen company the page shows a price chart, then
the advisor's four outputs as cards - the **Recommendation** (BUY / HOLD /
SELL), the **Confidence** (calibrated when a fitted temperature-scaling file
exists for the chosen model, otherwise the model's raw estimate), **Why**
(plain-English reason) and **What would change my mind** (the model's own
counterfactual sentence, plus an on-demand DiCE-style check that searches for
the smallest indicator change that actually flips the call). A second tab
shows how each advisor did in the walk-forward backtest, and a permanent footer
states the research-only scope.

Three advisors are selectable: two local LLMs served by Ollama (Llama 3.1 8B,
Qwen3 8B) and the transparent rule-based baseline ("Simple rule (no AI)"),
which needs no server. If Ollama is down the page explains what to do and
offers the rule-based model instead of crashing.

Run with::

    python scripts/run_app.py            # or: streamlit run streamlit_app.py

The page never places trades, never handles money, and never contacts anything
but the local Ollama server and (on a price-cache miss) Yahoo Finance.
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from typing import Any

import pandas as pd
import streamlit as st

from advisor.config import ADVISOR_MODELS, HISTORY_START, LLM_CACHE_DIR, RESULTS_DIR
from advisor.data import loader
from advisor.data.universe import SECTORS, UNIVERSE
from advisor.features.indicators import compute_features
from advisor.recommender.cache import CachedAdvisor
from advisor.recommender.heuristic import HeuristicAdvisor
from advisor.recommender.llm import OllamaAdvisor
from advisor.recommender.prompts import select_features
from advisor.recommender.schema import Action, Recommendation

# Sidebar label -> advisor tag. Tags are shared with the evaluation CLI
# (``--advisors llama qwen heuristic``) and with result file names.
MODEL_OPTIONS: dict[str, str] = {
    "Llama 3.1 8B": "llama",
    "Qwen3 8B": "qwen",
    "Simple rule (no AI)": "heuristic",
}
RULE_LABEL: str = "Simple rule (no AI)"
CHART_MONTHS: int = 6
# Feature columns that must be non-NaN before a day can be asked about.
REQUIRED_FEATURES: tuple[str, ...] = ("sma_10", "sma_50", "mom_10", "vol_20", "rsi_14", "ret_1d")
ACTION_COLOUR: dict[Action, str] = {Action.BUY: "green", Action.HOLD: "orange", Action.SELL: "red"}
SECTOR_OF: dict[str, str] = {t: s for s, tickers in SECTORS.items() for t in tickers}
DISCLAIMER: str = (
    "Research prototype built for a university final-year project (CM3070). "
    "It places no trades, handles no money and is not financial advice. "
    "Prices come from Yahoo Finance's free daily data and may be delayed or "
    "incomplete. Always do your own research before investing."
)


# --------------------------------------------------------------------------- #
# Cached data / resources
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner="Loading price history...")
def load_history(ticker: str) -> pd.DataFrame:
    """Return one ticker's full cached daily history with indicator columns.

    Args:
        ticker: Ticker symbol from the universe.

    Returns:
        Long-form price frame plus the six indicator columns, sorted by date
        (``date`` holds :class:`datetime.date`). Empty if nothing is cached
        and the fetch returned nothing.
    """
    prices = loader.load_prices([ticker], HISTORY_START, date.today() + timedelta(days=1))
    if prices.empty:
        return prices
    feats = compute_features(prices)
    feats["date"] = pd.to_datetime(feats["date"]).dt.date
    return feats


def _load_calibrator(tag: str) -> Any | None:
    """Load a fitted temperature scaler for ``tag`` if one has been saved.

    Looks for ``results/calibration_<tag>.json`` and rebuilds it with
    ``TemperatureScaler.from_dict``. Any problem (module missing, file
    malformed, unfitted scaler) silently yields ``None`` so the UI keeps
    working and shows the raw confidence instead.

    Args:
        tag: Advisor tag (``llama`` / ``qwen`` / ``heuristic``).

    Returns:
        A fitted calibrator exposing ``transform``, or ``None``.
    """
    path = RESULTS_DIR / f"calibration_{tag}.json"
    if not path.exists():
        return None
    try:
        from advisor.calibration.temperature import TemperatureScaler

        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and isinstance(payload.get("calibrator"), dict):
            payload = payload["calibrator"]
        scaler = TemperatureScaler.from_dict(payload)
        if getattr(scaler, "temperature", None) is None:
            return None
        scaler.transform([0.7])  # smoke-test the object before trusting it
        return scaler
    except Exception:  # noqa: BLE001 - degrade gracefully to raw confidence
        return None


@st.cache_data(show_spinner=False)
def load_track_record(tag: str) -> dict[str, tuple[float, int]]:
    """Held-out hit rate per action for ``tag``, from the walk-forward log.

    The evaluation found the advisors' stated confidence nearly constant and
    uninformative (report Section 5.4), so the page shows, beside it, how often
    calls of the same kind were actually right on data the model never saw.

    Args:
        tag: Advisor tag (``llama`` / ``qwen`` / ``heuristic``).

    Returns:
        ``{action: (hit_rate, n_labelled)}``; empty when no log exists.
    """
    path = RESULTS_DIR / f"{tag}_recommendations.csv"
    if not path.exists():
        return {}
    try:
        log = pd.read_csv(path, usecols=["action", "correct"]).dropna(subset=["correct"])
    except Exception:  # noqa: BLE001 - a broken log must not crash the page
        return {}
    return {
        str(action): (float(group["correct"].mean()), len(group))
        for action, group in log.groupby("action")
    }


@st.cache_data(show_spinner=False)
def load_claim_record(tag: str) -> tuple[int, int] | None:
    """How often this model's own "what would change my mind" claims held up.

    Read from the counterfactual experiment (report Section 5.5): the number
    of tested claims whose action really changed, and the number tested.
    Returns ``None`` when the experiment has not been run for this advisor.
    """
    path = RESULTS_DIR / f"counterfactual_llm_{tag}.json"
    try:
        sample = json.loads(path.read_text(encoding="utf-8"))["faithfulness_sample"]
        return int(sample["minimal_flips"]), int(sample["n_tested"])
    except (OSError, KeyError, TypeError, ValueError):
        return None


def indicator_rows(features: dict[str, Any]) -> pd.DataFrame:
    """The numbers the advisor was shown, with plain labels, for the "Why" card.

    Lets a user check the written reason against the real values (the
    evaluation found reasons that misstate the RSI, report Section 5.3).
    """
    def value(name: str, fmt: str) -> str:
        return fmt.format(features[name]) if name in features else "n/a"

    rows = [
        ("Latest closing price", value("close", "${:,.2f}")),
        ("10-day average price", value("sma_10", "${:,.2f}")),
        ("50-day average price", value("sma_50", "${:,.2f}")),
        ("10-day momentum", value("mom_10", "{:+.1%}")),
        ("14-day RSI", value("rsi_14", "{:.1f}")),
        ("20-day volatility (daily)", value("vol_20", "{:.1%}")),
    ]
    return pd.DataFrame(rows, columns=["Indicator", "Value"])


def track_record_sentence(action: Action, record: tuple[float, int]) -> str:
    """Plain-English track-record line for the confidence card."""
    rate, n = record
    rule = (
        " (a HOLD counts as right when the price moved less than 1% that week)"
        if action is Action.HOLD
        else ""
    )
    return (
        f"Track record: on past data it had never seen (2025-26), this advisor's "
        f"{action.value} calls were right {rate:.0%} of the time over {n:,} calls{rule}."
    )


@st.cache_resource(show_spinner=False)
def get_advisor(tag: str) -> CachedAdvisor:
    """Build (once per process) the advisor for a tag, wrapped in the JSONL cache.

    Args:
        tag: ``"llama"``, ``"qwen"`` or ``"heuristic"``.

    Returns:
        A :class:`CachedAdvisor` whose inner advisor carries the fitted
        calibrator when one exists.
    """
    inner: Any
    if tag == "heuristic":
        inner = HeuristicAdvisor()
    else:
        inner = OllamaAdvisor(model=ADVISOR_MODELS[tag])
    inner.calibrator = _load_calibrator(tag)
    return CachedAdvisor(inner, LLM_CACHE_DIR / f"{tag}.jsonl")


@st.cache_data(ttl=30, show_spinner=False)
def ollama_ready(model: str) -> bool:
    """Cheap, cached check that the Ollama server is up and ``model`` is installed."""
    return OllamaAdvisor(model=model, timeout=5).health()


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
def render_sidebar() -> tuple[str, str, str]:
    """Render the model / sector / company selectors.

    Returns:
        ``(tag, ticker, label)`` - the advisor tag, the chosen ticker and the
        human-readable advisor name.
    """
    with st.sidebar:
        st.header("Settings")
        label = st.selectbox(
            "Advisor",
            list(MODEL_OPTIONS),
            key="model",
            help=(
                "The two AI models run locally through Ollama. The simple rule "
                "needs no AI: it buys when the price is above its 50-day average "
                "and rising, sells when below and falling, and holds otherwise."
            ),
        )
        tag = MODEL_OPTIONS[label]
        sector = st.selectbox("Sector", ["All sectors", *sorted(SECTORS)], key="sector")
        tickers = list(UNIVERSE) if sector == "All sectors" else sorted(SECTORS[sector])
        ticker = st.selectbox("Company (ticker)", tickers, key="ticker")
        if tag != "heuristic":
            model = ADVISOR_MODELS[tag]
            if ollama_ready(model):
                st.caption(f"AI model ready ({model}).")
            else:
                st.warning(
                    f"The AI model ({model}) is not reachable right now. You can "
                    f"still use **{RULE_LABEL}**.",
                    icon="⚠️",
                )
    return tag, ticker, label


def pick_as_of(history: pd.DataFrame) -> date:
    """Render the "as of" date picker and return the trading day to use.

    Defaults to the last cached trading day. A chosen non-trading day snaps
    back to the last trading day on or before it, so every recommendation is
    tied to a real closing price.

    Args:
        history: Frame from :func:`load_history` (already filtered to rows
            whose indicators are available).

    Returns:
        The trading date the recommendation will be generated for.
    """
    dates = history["date"]
    first, last = dates.iloc[0], dates.iloc[-1]
    with st.sidebar:
        chosen = st.date_input(
            "As of",
            value=last,
            min_value=first,
            max_value=last,
            key="as_of",
            help="The advisor only sees information available up to this day.",
        )
    if isinstance(chosen, tuple | list):  # defensive: range mode never enabled
        chosen = chosen[0]
    eligible = dates[dates <= chosen]
    return eligible.iloc[-1] if not eligible.empty else first


# --------------------------------------------------------------------------- #
# Recommendation tab
# --------------------------------------------------------------------------- #
def render_chart(history: pd.DataFrame, as_of: date) -> None:
    """Plot the closing price and its 50-day average for the months before ``as_of``."""
    start = as_of - timedelta(days=int(CHART_MONTHS * 30.5))
    window = history[(history["date"] > start) & (history["date"] <= as_of)]
    chart = pd.DataFrame(
        {
            "Closing price": window["close"].to_numpy(),
            "50-day average": window["sma_50"].to_numpy(),
        },
        index=pd.to_datetime(window["date"]),
    )
    st.line_chart(chart, color=["#1f77b4", "#ff7f0e"])
    st.caption(
        f"Closing price (US$) and its 50-day average, {CHART_MONTHS} months to "
        f"{as_of:%d %b %Y}."
    )


def compute_recommendation(
    tag: str, ticker: str, history: pd.DataFrame, as_of: date
) -> dict[str, Any] | None:
    """Ask the chosen advisor about ``ticker`` on ``as_of``.

    Shows friendly errors (and the rule-based fallback hint) instead of
    raising when the AI model is down or answers unusably.

    Args:
        tag: Advisor tag.
        ticker: Ticker symbol.
        history: Frame from :func:`load_history`.
        as_of: Trading date.

    Returns:
        ``{"rec": Recommendation, "calibrated": bool}`` or ``None`` on failure.
    """
    rows = history[history["date"] == as_of]
    if rows.empty:
        st.error(f"No closing price for {ticker} on {as_of:%d %b %Y}.")
        return None
    try:
        features = select_features(rows.iloc[0])
    except ValueError as exc:
        st.error(f"Not enough price history before {as_of:%d %b %Y}: {exc}")
        return None

    advisor = get_advisor(tag)
    if tag != "heuristic" and not advisor.has(ticker, features, as_of=as_of):
        model = ADVISOR_MODELS[tag]
        if not advisor.health():
            st.error(
                f"The AI model ({model}) is not reachable, so no recommendation "
                "could be produced.",
                icon="🔌",
            )
            st.info(
                "Start Ollama (`ollama serve`) and make sure the model is installed "
                f"(`ollama pull {model}`), then try again - or switch the advisor "
                f"to **{RULE_LABEL}** in the sidebar, which works without AI."
            )
            return None

    with st.spinner("Thinking... an AI model can take about 10 seconds the first time."):
        try:
            rec = advisor.recommend(ticker, features, as_of=as_of)
        except ValueError as exc:
            st.error(f"The advisor could not produce a usable answer: {exc}", icon="⚠️")
            st.info(f"Try again, or switch the advisor to **{RULE_LABEL}** in the sidebar.")
            return None
    return {"rec": rec, "calibrated": advisor.calibrator is not None}


def run_claim_check(tag: str, rec: Recommendation) -> str:
    """Run the DiCE-style search for the smallest indicator change that flips the call.

    Args:
        tag: Advisor tag (the same advisor answers the search's what-if queries).
        rec: The recommendation being checked (its ``features`` are the instance).

    Returns:
        A plain-English sentence, or a short explanation if the check could
        not run.
    """
    if not rec.features:
        return "This recommendation has no stored indicators to test."
    try:
        from advisor.counterfactual.dice import generate_counterfactual
    except Exception as exc:  # noqa: BLE001 - module still being built
        return f"The claim checker is not available in this build ({exc})."
    advisor = get_advisor(tag)
    try:
        return generate_counterfactual(
            model=advisor,
            instance=rec.features,
            total_cfs=2,
            ticker=rec.ticker,
            as_of=rec.as_of,
        )
    except TypeError:
        return generate_counterfactual(model=advisor, instance=rec.features, total_cfs=2)
    except Exception as exc:  # noqa: BLE001 - never crash the page
        return f"The check could not be completed: {exc}"


def render_cards(
    rec: Recommendation, *, calibrated: bool, tag: str, label: str, result_key: str
) -> None:
    """Render the four output cards for a recommendation.

    Args:
        rec: The recommendation to display.
        calibrated: Whether ``rec.confidence`` went through a fitted calibrator.
        tag: Advisor tag (used for the claim check).
        label: Human-readable advisor name.
        result_key: Identifies the (advisor, ticker, date) the result belongs to.
    """
    colour = ACTION_COLOUR[rec.action]
    top_left, top_right = st.columns(2)
    with top_left, st.container(border=True):
        st.markdown("**Recommendation**")
        st.markdown(f"## :{colour}[{rec.action.value}]")
        st.caption(f"{rec.ticker} · as of {rec.as_of:%d %b %Y} · {label}")
    with top_right, st.container(border=True):
        st.markdown("**Confidence**")
        st.markdown(f"## {rec.confidence:.0%}")
        if calibrated:
            st.caption(
                "Adjusted so that 70% means right about 7 times in 10 "
                f"(the advisor's own estimate was {rec.raw_confidence:.0%})."
            )
        else:
            st.caption(
                "Unadjusted: the advisor's own estimate of being right over the "
                "next week or so. It can be over-confident."
            )
        record = load_track_record(tag).get(rec.action.value)
        if record is not None:
            st.caption(track_record_sentence(rec.action, record))

    bottom_left, bottom_right = st.columns(2)
    with bottom_left, st.container(border=True):
        st.markdown("**Why**")
        st.write(rec.reason or "The advisor gave no explanation.")
        if rec.features:
            with st.expander("Numbers the advisor saw"):
                st.table(indicator_rows(dict(rec.features)).set_index("Indicator"))
                st.caption("An RSI above 70 is usually read as overbought, below 30 as oversold.")
    with bottom_right, st.container(border=True):
        st.markdown("**What would change my mind**")
        st.write(rec.counterfactual or "The advisor gave no counterfactual.")
        claims = load_claim_record(tag)
        if claims is not None and rec.counterfactual:
            held, tested = claims
            st.caption(
                f"This is the model's own claim and has not been checked. In testing, "
                f"claims like this from this model held up {held} times out of {tested}."
            )
        with st.expander("Check this claim"):
            st.caption(
                "Searches for the smallest change in the indicators that actually "
                "flips the recommendation, by re-asking the advisor with slightly "
                "different numbers. With an AI model the first check can take a few "
                "minutes (up to 40 questions to the model); answers are remembered, so "
                "repeating it is instant."
            )
            if st.button("Run the check", key="check_claim"):
                with st.spinner("Checking... with an AI model this can take a few minutes."):
                    sentence = run_claim_check(tag, rec)
                st.session_state["cf_check"] = {"key": result_key, "sentence": sentence}
            stored = st.session_state.get("cf_check")
            if stored and stored.get("key") == result_key:
                st.success(stored["sentence"], icon="🔎")


def render_recommendation_tab(
    tag: str, label: str, ticker: str, history: pd.DataFrame, as_of: date
) -> None:
    """Compose the chart, the action button and the result cards."""
    st.subheader(f"{ticker} · {SECTOR_OF.get(ticker, 'benchmark').title()}")
    render_chart(history, as_of)

    result_key = f"{tag}|{ticker}|{as_of.isoformat()}"
    clicked = st.button("Get recommendation", type="primary", key="get_rec")
    if clicked:
        result = compute_recommendation(tag, ticker, history, as_of)
        if result is not None:
            st.session_state["result"] = {"key": result_key, **result}

    stored = st.session_state.get("result")
    if stored and stored.get("key") == result_key:
        render_cards(
            stored["rec"],
            calibrated=bool(stored["calibrated"]),
            tag=tag,
            label=label,
            result_key=result_key,
        )
    elif not clicked:
        st.info(
            f"Press **Get recommendation** to ask *{label}* about {ticker} as of "
            f"{as_of:%d %b %Y}."
        )


# --------------------------------------------------------------------------- #
# Backtest tab and footer
# --------------------------------------------------------------------------- #
#: Plain-English names for the strategies in ``summary_table.csv``.
STRATEGY_NAMES: dict[str, str] = {
    "llama": "Llama 3.1 8B advisor",
    "qwen": "Qwen3 8B advisor",
    "heuristic": "Simple rule (no AI)",
    "buy_and_hold": "S&P 500 buy-and-hold",
    "momentum_12_1": "12-1 momentum",
    "markowitz_mean_variance": "Markowitz mean-variance",
    "random_allocation": "Random portfolio (one seed)",
    "random_ensemble_mean": "Random portfolios (average of 100)",
}
#: (source column, display name, formatter) for the backtest table.
SUMMARY_COLUMNS: tuple[tuple[str, str, str], ...] = (
    ("total_return", "Total return", "{:+.1%}"),
    ("sharpe", "Sharpe ratio", "{:.2f}"),
    ("deflated_sharpe", "Deflated Sharpe", "{:.2f}"),
    ("max_drawdown", "Worst fall", "{:.1%}"),
)


def friendly_summary(summary: pd.DataFrame) -> pd.DataFrame:
    """Reduce the evaluation summary to a few plain-English, formatted columns.

    Args:
        summary: ``summary_table.csv`` as written by the evaluation runner.

    Returns:
        One row per strategy with a readable name and only the columns a
        non-expert needs; unknown strategies keep their raw name.
    """
    out = pd.DataFrame()
    if "strategy" in summary.columns:
        out["Strategy"] = [STRATEGY_NAMES.get(s, s) for s in summary["strategy"]]
    for column, name, fmt in SUMMARY_COLUMNS:
        if column in summary.columns:
            out[name] = [fmt.format(v) if pd.notna(v) else "" for v in summary[column]]
    return out


def render_history_tab() -> None:
    """Show the backtest summary table and equity-curve figure(s) when present."""
    st.markdown(
        "How each advisor - and the simple benchmarks it is compared against - "
        "would have done on past data it had never seen. Recommendations only: "
        "no money was ever traded."
    )
    summary = RESULTS_DIR / "summary_table.csv"
    figures = sorted(RESULTS_DIR.glob("*equity*.png")) + sorted(
        (RESULTS_DIR / "figures").glob("equity_curves.png")
    )
    if summary.exists():
        try:
            st.dataframe(friendly_summary(pd.read_csv(summary)), hide_index=True)
            st.caption(
                "Held-out test from Jan 2025 to May 2026, weekly decisions, trading costs "
                "included. Sharpe ratio: return per unit of risk (higher is better). "
                "Deflated Sharpe: the same, corrected for how many strategies were compared. "
                "Worst fall: the largest drop from a previous peak."
            )
        except Exception as exc:  # noqa: BLE001 - a broken CSV must not crash the page
            st.warning(f"Could not read {summary.name}: {exc}")
    for figure in figures:
        st.image(str(figure), caption="Growth of $100,000 for each strategy (held-out test)")
    if not summary.exists() and not figures:
        st.info(
            "No backtest results yet. Run "
            "`python -m advisor.evaluation.run_evaluation --advisors llama qwen heuristic` "
            "to produce them."
        )


def render_footer() -> None:
    """Permanent research-only disclaimer."""
    st.divider()
    st.caption(DISCLAIMER)


# --------------------------------------------------------------------------- #
# Page
# --------------------------------------------------------------------------- #
def main() -> None:
    """Compose and render the full page."""
    st.set_page_config(page_title="Financial Advisor Bot", page_icon="📈", layout="wide")
    tag, ticker, label = render_sidebar()

    st.title("Financial Advisor Bot")
    st.caption(
        "Ask for a BUY / HOLD / SELL view on a large US company, see why, and learn "
        "what would change the advisor's mind. Research prototype - not financial advice."
    )

    history = load_history(ticker)
    if not history.empty:
        history = history.dropna(subset=list(REQUIRED_FEATURES)).reset_index(drop=True)
    if history.empty:
        st.error(
            f"No price data is available for {ticker}. Warm the cache with "
            f"`python scripts/fetch_data.py --tickers {ticker}` and reload."
        )
        render_footer()
        st.stop()

    as_of = pick_as_of(history)
    tab_now, tab_past = st.tabs(["Recommendation", "How it did in the past"])
    with tab_now:
        render_recommendation_tab(tag, label, ticker, history, as_of)
    with tab_past:
        render_history_tab()
    render_footer()


# Streamlit runs this file top to bottom on every interaction (under both
# `streamlit run` and the test harness), so the page is drawn unconditionally.
main()
