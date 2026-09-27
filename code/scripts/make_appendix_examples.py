"""Write report Appendix D: worked examples of real recommendations.

For each language model, pick recommendations from the counterfactual
experiment and show, side by side, what the model said (action, confidence,
reason, its own "what would change my mind" sentence), whether that sentence
held up when tested, and what the verified search actually found.

Examples are chosen by a fixed rule, not by hand: for each model, the first
row (in sample order) whose own sentence failed the test and the first whose
own sentence held, among rows where the search found a counterfactual.

Usage (from code/):
    .venv/bin/python scripts/make_appendix_examples.py
"""

import json
from pathlib import Path
from typing import Any

import pandas as pd

RESULTS = Path(__file__).resolve().parents[1] / "results"
OUT = Path(__file__).resolve().parents[2] / "reports" / "final-report" / "11-appendix-d-examples.md"
MODELS = {"llama": "Llama 3.1 8B", "qwen": "Qwen3 8B"}


def read_jsonl(path: Path) -> pd.DataFrame:
    """All rows of a JSONL results file, or an empty frame if it is missing."""
    if not path.exists():
        return pd.DataFrame()
    return pd.read_json(path, lines=True, convert_dates=False, dtype=False)


def features_seen(tag: str) -> pd.DataFrame:
    """The RSI and momentum each answer was given, from the model's answer cache."""
    rows = []
    with open(RESULTS / f"llm_cache_{tag}.jsonl", encoding="utf-8") as cache:
        for line in cache:
            rec = json.loads(line)["recommendation"]
            rows.append({"date": rec["as_of"], "ticker": rec["ticker"],
                         "rsi": rec["features"]["rsi_14"], "momentum": rec["features"]["mom_10"]})
    return pd.DataFrame(rows).drop_duplicates(["date", "ticker"], keep="last")


def pick_examples(tag: str) -> pd.DataFrame:
    """Rows for one model, joined with the reason and confidence it gave."""
    dice = read_jsonl(RESULTS / f"counterfactual_llm_{tag}_dice_b40.jsonl")
    faith = read_jsonl(RESULTS / f"counterfactual_llm_{tag}_faith.jsonl")
    if dice.empty or faith.empty:
        return pd.DataFrame()
    answers = pd.read_csv(RESULTS / f"{tag}_recommendations.csv",
                          usecols=["date", "ticker", "raw_confidence", "reason"])
    answers = answers.merge(features_seen(tag), on=["date", "ticker"], how="left")
    rows = dice.merge(faith[["date", "ticker", "minimal_flip", "minimal_action"]],
                      on=["date", "ticker"], how="left")
    rows = rows.merge(answers, on=["date", "ticker"], how="left")
    rows = rows[rows["n_cfs"] > 0]

    chosen = []
    for held_up in (False, True):
        match = rows[rows["minimal_flip"].astype(bool) == held_up]
        if not match.empty:
            chosen.append(match.iloc[0])
    return pd.DataFrame(chosen)


def describe(tag: str, row: Any) -> str:
    """One example as a small Markdown table."""
    moved = "Moving the indicator just past the stated level"
    if row["minimal_flip"]:
        own_result = f"Held up. {moved} changed the call to {row['minimal_action']}."
    else:
        own_result = f"Did not hold up. {moved} left the call at {row['minimal_action']}."
    sentences = row["cf_sentences"].split(" | ")
    if len(sentences) > 1:
        verified = " ".join(f"({i}) {text}" for i, text in enumerate(sentences, start=1))
    else:
        verified = sentences[0]
    lines = [
        f"### {MODELS[tag]} — {row['ticker']}, {row['date']}",
        "",
        "| Item | Detail |",
        "|---|---|",
        f"| Recommendation | {row['action']} at {row['raw_confidence']:.0%} stated confidence |",
        f"| What it was shown | 14-day RSI {row['rsi']:.1f}, "
        f"10-day momentum {row['momentum']:+.1%} |",
        f"| Reason given | {row['reason']} |",
        f"| Model's own sentence | {row['self_cf']} |",
        f"| Own sentence tested | {own_result} |",
        f"| Verified counterfactual | {verified} ({row['n_valid']}/{row['n_cfs']} reproduced "
        f"on a fresh call) |",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    """Build the appendix file from the current experiment results."""
    parts = [
        "# Appendix D — Worked examples",
        "",
        "Real recommendations from the held-out window, taken from the counterfactual "
        "experiment in Section 5.5 by a fixed rule (for each model, the first sampled "
        "answer whose own sentence failed the test and the first whose sentence held). "
        "Model text is quoted exactly.",
        "",
    ]
    for tag in MODELS:
        examples = pick_examples(tag)
        for _, row in examples.iterrows():
            parts.append(describe(tag, row))
    OUT.write_text("\n".join(parts), encoding="utf-8")
    print("wrote", OUT)
    print(json.dumps({tag: len(pick_examples(tag)) for tag in MODELS}))


if __name__ == "__main__":
    main()
