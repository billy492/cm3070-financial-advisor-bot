r"""Counterfactual experiments on the two LLM advisors (report Section 5.5).

Two things are measured on a seeded sample of held-out BUY/SELL answers:

1. Faithfulness of the model's own sentence ("faith"). Every answer ends with
   "I would change my mind if <indicator> <rose above / fell below / turned
   positive / turned negative> <threshold>". We move that indicator just past
   the threshold (and, separately, one grid step past it), keep everything
   else the same, ask the model again and record whether the action changed.

2. Validity of the verified counterfactuals ("dice"). We run the same
   DiCE-style search the web page uses and re-check every counterfactual it
   returns with a fresh model call.

All calls go to the local Ollama server. Each finished row is appended to a
JSONL file straight away, so an interrupted run can simply be started again
and it continues where it stopped. The summary is written to
results/counterfactual_llm_<tag>.json.

Usage (from code/):
    .venv/bin/python scripts/cf_llm_experiment.py --tag llama --parts faith --n-faith 40
    .venv/bin/python scripts/cf_llm_experiment.py --tag llama --parts dice --n-dice 12 \
        --max-calls 40
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from collections.abc import Callable
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from advisor.counterfactual.dice import (  # noqa: E402
    apply_counterfactual,
    generate_counterfactuals,
)
from advisor.recommender.llm import OllamaAdvisor  # noqa: E402

log = logging.getLogger("cf_llm_experiment")

MODELS = {"llama": "llama3.1:8b", "qwen": "qwen3:8b"}
RESULTS = Path("results")

# Indicator names the prompt allows in the model's own sentence.
INDICATOR_NAMES = {
    "10-day momentum": "mom_10",
    "14-day rsi": "rsi_14",
    "20-day volatility": "vol_20",
    "50-day average price": "sma_50",
}
# How far past the stated threshold we move the indicator:
# "minimal" = just past it, "generous" = one more step of the search grid.
JUST_PAST = {"mom_10": 0.005, "rsi_14": 1.0, "vol_20": 0.001, "sma_50": 0.005}
GRID_STEP = {"mom_10": 0.02, "rsi_14": 5.0, "vol_20": 0.005, "sma_50": 0.02}


# --------------------------------------------------------------------------
# Reading the model's own counterfactual sentence
# --------------------------------------------------------------------------
def parse_sentence(sentence: str) -> dict[str, Any] | None:
    """Turn the model's sentence into (indicator, direction, threshold).

    Returns a dict like {"feature": "rsi_14", "direction": 1, "threshold": 70.0}
    (direction +1 means "rose above", -1 means "fell below"), or None if the
    sentence does not follow the one-indicator template.
    """
    text = str(sentence or "").lower()

    feature = None
    for words, name in INDICATOR_NAMES.items():
        if words in text:
            feature = name
            break
    if feature is None:
        return None

    number = r"(-?\d+(?:\.\d+)?)\s*(%?)"

    def as_number(match: re.Match[str]) -> float:
        value = float(match.group(1))
        return value / 100 if match.group(2) == "%" else value

    # "turned positive" / "turned negative", sometimes with a level ("..., above -0.01")
    if "turned positive" in text or "turned negative" in text:
        direction = 1 if "turned positive" in text else -1
        level = re.search(("above" if direction == 1 else "below") + r"\s+" + number, text)
        threshold = as_number(level) if level else 0.0
        return {"feature": feature, "direction": direction, "threshold": threshold}

    rising = r"(?:rose|rises|went|goes|climbed|increased)\s+above\s+"
    falling = r"(?:fell|falls|dropped|drops|went|goes|declined|decreased)\s+below\s+"
    for pattern, direction in [(rising, 1), (falling, -1)]:
        match = re.search(pattern + number, text)
        if match:
            return {"feature": feature, "direction": direction, "threshold": as_number(match)}
    return None


def already_true(features: dict[str, float], claim: dict[str, Any]) -> bool:
    """True if the stated condition already holds, so the sentence changes nothing."""
    if claim["feature"] == "sma_50":
        return False  # wording about the price level is too vague to judge
    value = float(features[claim["feature"]])
    if claim["direction"] == 1:
        return value > claim["threshold"]
    return value < claim["threshold"]


def move_past_threshold(
    features: dict[str, float], claim: dict[str, Any], distance: float
) -> dict[str, float]:
    """Copy of the features with the stated indicator moved past its threshold."""
    changed = dict(features)
    new_value = claim["threshold"] + claim["direction"] * distance
    if claim["feature"] == "rsi_14":
        new_value = min(max(new_value, 0.0), 100.0)
    changed[claim["feature"]] = new_value
    return changed


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def load_answers(tag: str) -> pd.DataFrame:
    """All held-out BUY/SELL answers for one model, with the features it saw."""
    answers = pd.read_csv(RESULTS / f"{tag}_recommendations.csv")

    rows = []
    with open(RESULTS / f"llm_cache_{tag}.jsonl", encoding="utf-8") as cache:
        for line in cache:
            rec = json.loads(line)["recommendation"]
            rows.append({"date": rec["as_of"], "ticker": rec["ticker"],
                         "features": rec["features"]})
    features = pd.DataFrame(rows).drop_duplicates(["date", "ticker"], keep="last")

    answers = answers.merge(features, on=["date", "ticker"], how="inner")
    return answers[answers["action"].isin(["BUY", "SELL"])].reset_index(drop=True)


def seeded_sample(tag: str, size: int, seed: int) -> pd.DataFrame:
    """The same random sample of BUY/SELL answers every time (fixed seed)."""
    answers = load_answers(tag)
    return answers.sample(n=min(size, len(answers)), random_state=seed).reset_index(drop=True)


def read_rows(path: Path) -> pd.DataFrame:
    """Rows already saved by an earlier (possibly interrupted) run."""
    if not path.exists():
        return pd.DataFrame()
    return pd.read_json(path, lines=True, convert_dates=False, dtype=False)


def save_row(path: Path, row: dict[str, Any]) -> None:
    """Append one finished row to the JSONL file."""
    with path.open("a", encoding="utf-8") as out:
        out.write(json.dumps(row, default=str) + "\n")


# --------------------------------------------------------------------------
# The two experiments
# --------------------------------------------------------------------------
class ModelCaller:
    """Asks the local model for an action and counts how many calls were made."""

    def __init__(self, model: str) -> None:
        """Connect to the Ollama model tag, e.g. "llama3.1:8b"."""
        self.advisor = OllamaAdvisor(model=model)
        self.calls = 0

    def action(self, ticker: str, features: dict[str, float], as_of: date) -> str:
        """Ask the model once and return "BUY", "HOLD" or "SELL"."""
        self.calls += 1
        return self.advisor.recommend(ticker, dict(features), as_of=as_of).action.value

    def predictor_for(self, ticker: str, as_of: date) -> Callable[[dict[str, float]], str]:
        """A features -> action function for the counterfactual search."""
        def predict(features: dict[str, float]) -> str:
            return self.action(ticker, features, as_of)
        return predict


def check_own_sentence(model: ModelCaller, answer: pd.Series) -> dict[str, Any]:
    """Experiment 1 for one answer: does the model's own sentence hold up?"""
    as_of = date.fromisoformat(answer["date"])
    features = {name: float(value) for name, value in answer["features"].items()}
    claim = parse_sentence(answer["counterfactual"])

    row = {
        "date": answer["date"],
        "ticker": answer["ticker"],
        "action": answer["action"],
        "self_cf": answer["counterfactual"],
        "parsed": claim is not None,
        # Asking the same question again shows whether decoding is repeatable.
        "reproduced": model.action(answer["ticker"], features, as_of) == answer["action"],
    }
    if claim is None:
        return row

    row["feature"] = claim["feature"]
    row["direction"] = claim["direction"]
    row["threshold"] = claim["threshold"]
    row["current"] = features.get(claim["feature"])
    row["vacuous"] = already_true(features, claim)
    if row["vacuous"]:
        return row

    distances = {
        "minimal": JUST_PAST[claim["feature"]],
        "generous": JUST_PAST[claim["feature"]] + GRID_STEP[claim["feature"]],
    }
    for label, distance in distances.items():
        new_action = model.action(answer["ticker"],
                                  move_past_threshold(features, claim, distance), as_of)
        row[f"{label}_action"] = new_action
        row[f"{label}_flip"] = new_action != answer["action"]
    return row


def check_verified_search(
    model: ModelCaller, answer: pd.Series, max_calls: int
) -> dict[str, Any]:
    """Experiment 2 for one answer: run the search, then re-check what it found."""
    as_of = date.fromisoformat(answer["date"])
    features = {name: float(value) for name, value in answer["features"].items()}

    calls_before = model.calls
    predict = model.predictor_for(answer["ticker"], as_of)
    found = generate_counterfactuals(predict, features, total_cfs=2, max_calls=max_calls)
    search_calls = model.calls - calls_before

    n_valid = 0
    for cf in found:
        # A fresh call (not the search's own memory) must reproduce the flip.
        if predict(apply_counterfactual(features, cf)) == cf.new_action:
            n_valid += 1

    return {
        "date": answer["date"],
        "ticker": answer["ticker"],
        "action": answer["action"],
        "self_cf": answer["counterfactual"],
        "n_cfs": len(found),
        "n_valid": n_valid,
        "search_calls": search_calls,
        "cf_features": ";".join(",".join(sorted(cf.changes())) for cf in found),
        "cf_new_actions": ";".join(cf.new_action for cf in found),
        "cf_sentences": " | ".join(cf.sentence for cf in found),
    }


def run_part(part: str, tag: str, sample: pd.DataFrame, max_calls: int) -> None:
    """Run one experiment over the sample, skipping rows saved by earlier runs."""
    if part == "faith":
        path = RESULTS / f"counterfactual_llm_{tag}_faith.jsonl"
    else:
        path = RESULTS / f"counterfactual_llm_{tag}_dice_b{max_calls}.jsonl"

    saved = read_rows(path)
    done = set()
    if not saved.empty:
        done = set(zip(saved["date"], saved["ticker"], strict=True))

    model = ModelCaller(MODELS[tag])
    if not model.advisor.health():
        raise SystemExit(f"Ollama model {MODELS[tag]} is not available.")

    for i, answer in sample.iterrows():
        if (answer["date"], answer["ticker"]) in done:
            continue
        if part == "faith":
            row = check_own_sentence(model, answer)
        else:
            row = check_verified_search(model, answer, max_calls)
        save_row(path, row)
        log.info("[%s] %s %d/%d, %d model calls so far", tag, part, i + 1, len(sample),
                 model.calls)


# --------------------------------------------------------------------------
# Summary
# --------------------------------------------------------------------------
def summarise(tag: str, max_calls: int) -> dict[str, Any]:
    """Build the summary JSON from every row saved so far."""
    summary = {"advisor": tag, "model": MODELS[tag]}

    # How many of ALL the BUY/SELL answers state a condition that is already true?
    answers = load_answers(tag)
    n_parsed = 0
    n_vacuous = 0
    for sentence, features in zip(answers["counterfactual"], answers["features"], strict=True):
        claim = parse_sentence(sentence)
        if claim is not None:
            n_parsed += 1
            if already_true(features, claim):
                n_vacuous += 1
    summary["all_buy_sell"] = {"n": len(answers), "n_parsed": n_parsed,
                               "n_vacuous": n_vacuous}

    faith = read_rows(RESULTS / f"counterfactual_llm_{tag}_faith.jsonl")
    if not faith.empty:
        vacuous = faith["vacuous"].fillna(False).astype(bool)
        tested = faith[faith["parsed"] & ~vacuous]
        summary["faithfulness_sample"] = {
            "n": len(faith),
            "n_parsed": int(faith["parsed"].sum()),
            "reproduced": int(faith["reproduced"].sum()),
            "n_vacuous": int(vacuous.sum()),
            "n_tested": len(tested),
            "minimal_flips": int(tested["minimal_flip"].astype(bool).sum()),
            "generous_flips": int(tested["generous_flip"].astype(bool).sum()),
        }

    dice = read_rows(RESULTS / f"counterfactual_llm_{tag}_dice_b{max_calls}.jsonl")
    if not dice.empty:
        summary[f"dice_b{max_calls}"] = {
            "n_rows": len(dice),
            "rows_with_cf": int((dice["n_cfs"] > 0).sum()),
            "n_counterfactuals": int(dice["n_cfs"].sum()),
            "n_valid_on_fresh_call": int(dice["n_valid"].sum()),
            "mean_search_calls": float(dice["search_calls"].mean()),
        }

    path = RESULTS / f"counterfactual_llm_{tag}.json"
    path.write_text(json.dumps(summary, indent=2))
    return summary


def compare_models() -> None:
    """Fisher's exact test: do the two models' own sentences hold up equally often?"""
    from scipy.stats import fisher_exact

    table = []
    for tag in MODELS:
        sample = json.loads((RESULTS / f"counterfactual_llm_{tag}.json").read_text())
        faith = sample["faithfulness_sample"]
        held = faith["minimal_flips"]
        table.append([held, faith["n_tested"] - held])
        print(f"{tag}: {held}/{faith['n_tested']} own sentences held up")
    result = fisher_exact(table)
    print(f"Fisher's exact test: p = {result.pvalue:.2g}")


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Counterfactual experiments on the LLM advisors.")
    parser.add_argument("--tag", choices=sorted(MODELS))
    parser.add_argument("--parts", nargs="*", default=["faith", "dice"],
                        choices=["faith", "dice"], help="experiments to run (none = summary only)")
    parser.add_argument("--n-faith", type=int, default=40)
    parser.add_argument("--n-dice", type=int, default=12)
    parser.add_argument("--max-calls", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--compare", action="store_true",
                        help="compare the two models' saved results and exit")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    if args.compare:
        compare_models()
        return
    if args.tag is None:
        parser.error("--tag is required unless --compare is given")

    # One seeded sample; the search uses its first n_dice rows.
    sample = seeded_sample(args.tag, max(args.n_faith, args.n_dice), args.seed)
    if "faith" in args.parts:
        run_part("faith", args.tag, sample.head(args.n_faith), args.max_calls)
    if "dice" in args.parts:
        run_part("dice", args.tag, sample.head(args.n_dice), args.max_calls)

    print(json.dumps(summarise(args.tag, args.max_calls), indent=2))


if __name__ == "__main__":
    main()
