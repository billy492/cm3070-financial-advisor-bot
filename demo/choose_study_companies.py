"""Suggest the user-study companies and pre-cache their answers.

Asks Llama 3.1 8B about every stock in the universe on the app's default date,
through the same code path as the web page (same features, same answer
cache), so the screenshots for the study are instant afterwards. Then applies
the study's selection rule: two companies with the same BUY or SELL call, confidences
within 10 points, and a concrete "what would change my mind" sentence, plus a
third company for the warm-up.

Usage (from the thesis folder; needs Ollama running):
    code/.venv/bin/python demo/choose_study_companies.py
"""

import sys
from datetime import date, timedelta
from pathlib import Path

CODE = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE))

from advisor.config import ADVISOR_MODELS, HISTORY_START, LLM_CACHE_DIR  # noqa: E402
from advisor.data import loader  # noqa: E402
from advisor.data.universe import SECTORS  # noqa: E402
from advisor.features.indicators import compute_features  # noqa: E402
from advisor.recommender.cache import CachedAdvisor  # noqa: E402
from advisor.recommender.llm import OllamaAdvisor  # noqa: E402
from advisor.recommender.prompts import select_features  # noqa: E402

REQUIRED = ["sma_10", "sma_50", "mom_10", "vol_20", "rsi_14", "ret_1d"]
SECTOR_OF = {ticker: sector for sector, tickers in SECTORS.items() for ticker in tickers}


def names_a_condition(sentence):
    """True if the sentence names something checkable (a number, or "turned positive/negative")."""
    if not sentence:
        return False
    return any(ch.isdigit() for ch in sentence) or "turned" in sentence


def ask_llama(advisor, ticker):
    """The recommendation the web page would show for ``ticker`` on its default date."""
    prices = loader.load_prices([ticker], HISTORY_START, date.today() + timedelta(days=1))
    history = compute_features(prices).dropna(subset=REQUIRED).reset_index(drop=True)
    last = history.iloc[-1]
    as_of = last["date"].date() if hasattr(last["date"], "date") else last["date"]
    return advisor.recommend(ticker, select_features(last), as_of=as_of)


def main():
    """Ask about every stock, print the answers and a suggested pick."""
    advisor = CachedAdvisor(OllamaAdvisor(model=ADVISOR_MODELS["llama"]),
                            LLM_CACHE_DIR / "llama.jsonl")
    answers = []
    for ticker in SECTOR_OF:
        rec = ask_llama(advisor, ticker)
        answers.append(rec)
        print(f"{ticker:<6} {SECTOR_OF[ticker]:<12} {rec.action.value:<5} "
              f"{rec.raw_confidence:.0%}  {rec.counterfactual}")

    concrete = [r for r in answers if names_a_condition(r.counterfactual)]
    pairs = []
    for first in concrete:
        for second in concrete:
            if (first.ticker < second.ticker
                    and first.action == second.action
                    and first.action.value in ("BUY", "SELL")
                    and abs(first.raw_confidence - second.raw_confidence) <= 0.10):
                same_sector = SECTOR_OF[first.ticker] == SECTOR_OF[second.ticker]
                pairs.append((not same_sector, first, second))
    if pairs:
        _, first, second = sorted(pairs, key=lambda p: p[0])[0]
        warm_up = next(r for r in answers if r.ticker not in (first.ticker, second.ticker))
        print(f"\nSuggested: T1 = {first.ticker}, T2 = {second.ticker} "
              f"(both {first.action.value}, {SECTOR_OF[first.ticker]} / "
              f"{SECTOR_OF[second.ticker]}); warm-up T0 = {warm_up.ticker}")
        return
    print("\nNo pair meets the selection rule on this date; pick an earlier date in the app.")


if __name__ == "__main__":
    main()
