"""Tests for ``advisor.evaluation.policy_adherence`` (post-hoc, offline).

The instructed policy must reproduce the prompt's four rules exactly, and the
runner must join a synthetic log with its cached features and report the
right adherence and on/off-policy BUY split.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from advisor.evaluation import policy_adherence as pa


def feats(close=100.0, sma_10=95.0, sma_50=90.0, mom_10=0.05, rsi_14=55.0):
    """Feature row in an uptrend by default."""
    return {"close": close, "sma_10": sma_10, "sma_50": sma_50,
            "mom_10": mom_10, "rsi_14": rsi_14, "ret_1d": 0.0, "vol_20": 0.01}


@pytest.mark.parametrize(
    ("row", "expected"),
    [
        (feats(), "BUY"),
        (feats(rsi_14=75.0), "HOLD"),  # RSI veto on buying
        (feats(mom_10=-0.01), "HOLD"),  # momentum does not confirm
        (feats(close=80.0), "HOLD"),  # below both averages but momentum still positive
        (feats(close=80.0, mom_10=-0.05), "SELL"),
        (feats(close=80.0, mom_10=-0.05, rsi_14=25.0), "HOLD"),  # RSI veto on selling
        (feats(close=92.0), "HOLD"),  # between the averages: no clear trend
    ],
)
def test_prompt_policy_action(row, expected):
    """Each prompt rule maps to the instructed action."""
    assert pa.prompt_policy_action(row) == expected


def test_prompt_policy_requires_features():
    """A missing indicator is an error, not a silent HOLD."""
    with pytest.raises(KeyError):
        pa.prompt_policy_action({"close": 1.0})


def write_log(root: Path, tag: str) -> None:
    """Write a four-row recommendation log and its feature cache."""
    rows = [
        ("2025-01-03", "AAA", "BUY", feats(), 0.02),                       # on-policy BUY
        ("2025-01-03", "BBB", "BUY", feats(close=80.0, mom_10=-0.05), -0.01),  # off-policy BUY
        ("2025-01-03", "CCC", "HOLD", feats(close=92.0), 0.0),             # on-policy HOLD
        ("2025-01-10", "AAA", "SELL", feats(), 0.03),                      # off-policy SELL
    ]
    reasons = [
        "Uptrend and positive momentum.",
        "The 14-day RSI is below 30, so it is oversold.",  # real RSI is 55: a false claim
        "No clear trend.",
        "The 14-day RSI is above 70.",  # real RSI is 55: a false claim
    ]
    recs = pd.DataFrame(
        [{"date": d, "ticker": t, "action": a, "realised_return": r, "reason": why}
         for (d, t, a, _f, r), why in zip(rows, reasons, strict=True)])
    recs.to_csv(root / f"{tag}_recommendations.csv", index=False)
    with open(root / f"llm_cache_{tag}.jsonl", "w", encoding="utf-8") as fh:
        for d, t, a, f, _r in rows:
            fh.write(json.dumps({"recommendation": {"as_of": d, "ticker": t, "action": a,
                                                    "features": f}}) + "\n")


def test_run_writes_summary_and_confusion(tmp_path):
    """Adherence, the on/off-policy BUY split and the CSVs are correct."""
    write_log(tmp_path, "toy")
    summary = pa.run(tmp_path, ["toy"])
    row = summary.iloc[0]
    assert row["n"] == 4
    assert row["adherence"] == pytest.approx(0.5)
    assert row["adherence_BUY"] == pytest.approx(0.5)
    assert row["buy_on_policy_n"] == 1 and row["buy_off_policy_n"] == 1
    assert row["buy_on_policy_up_rate"] == pytest.approx(1.0)
    assert row["buy_off_policy_up_rate"] == pytest.approx(0.0)
    confusion = pd.read_csv(tmp_path / "policy_adherence_toy.csv", index_col=0)
    assert confusion.loc["BUY", "SELL"] == 1
    assert confusion.loc["SELL", "BUY"] == 1
    assert (tmp_path / "policy_adherence_summary.csv").exists()
    assert row["rsi_below_30_claims"] == 1 and row["rsi_below_30_false"] == 1
    assert row["rsi_above_70_false_off_policy"] == 1


@pytest.mark.parametrize(
    ("reason", "rsi", "expected"),
    [
        ("The 14-day RSI is above 70, so it may be overbought.", 62.6, "above_70_false"),
        ("The 14-day RSI is above 70, so it may be overbought.", 71.0, "above_70_true"),
        ("The 14-day RSI is above 70.", 69.0, "above_70_true"),  # within the 2-point margin
        ("However, the RSI is below 30, suggesting it may bounce.", 39.1, "below_30_false"),
        ("The RSI is below 30.", 25.0, "below_30_true"),
        ("The RSI is not above 70.", 50.0, None),  # negated, so no claim
        ("The 14-day RSI is at 46.0, which is neutral.", 46.0, None),
    ],
)
def test_rsi_threshold_claim(reason, rsi, expected):
    """A reason's 'RSI above 70 / below 30' statement is checked against the real value."""
    assert pa.rsi_threshold_claim(reason, rsi) == expected
