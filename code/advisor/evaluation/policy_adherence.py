"""Post-hoc policy-adherence analysis of the LLM recommendation logs.

**Status: post-hoc.** Not part of the pre-registered hypotheses; every number
it produces must be labelled as exploratory in the report.

The advisor prompt (:data:`advisor.recommender.prompts.SYSTEM_PROMPT`) walks
the model through an explicit decision policy: trend (close versus both moving
averages), then momentum confirmation, then an RSI veto, and HOLD otherwise.
Because the policy is fully specified, the action it implies for any feature
row can be computed exactly. Comparing that action with what each model
actually answered measures *instruction adherence*: how often an 8B model
does what its own prompt tells it to do on the numbers it was shown. It also
splits the realised outcomes into on-policy and off-policy answers, which
shows where any trading edge (or loss) came from. Finally it checks each
written reason's statement that the RSI is "above 70" or "below 30" against
the real value, because those are the veto lines the models invoke.

Inputs (in ``--results-dir``): ``<tag>_recommendations.csv`` (actions and
realised labels) and ``llm_cache_<tag>.jsonl`` (the exact feature dict each
answer was given). Outputs:

* ``policy_adherence_<tag>.csv`` -- confusion counts, instructed action by
  answered action;
* ``policy_adherence_summary.csv`` -- one row per advisor: overall adherence,
  adherence per answered action, and hit rate / mean five-day return of
  on-policy versus off-policy BUYs.

No LLM calls are made.

Example:
    ``python -m advisor.evaluation.policy_adherence --tags llama qwen``
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

__all__ = [
    "prompt_policy_action",
    "rsi_threshold_claim",
    "load_log",
    "adherence_summary",
    "run",
    "main",
]

_log = logging.getLogger(__name__)

ACTIONS: tuple[str, ...] = ("BUY", "HOLD", "SELL")


def prompt_policy_action(features: Mapping[str, Any]) -> str:
    """Return the action the prompt's DECISION POLICY implies for one row.

    Mirrors steps 1-4 of the system prompt: uptrend = close above both
    ``sma_10`` and ``sma_50``; downtrend = below both; BUY only in an uptrend
    with positive ``mom_10`` and ``rsi_14`` not above 70; SELL only in a
    downtrend with negative ``mom_10`` and ``rsi_14`` not below 30; else HOLD.

    Raises:
        KeyError: If a required feature is missing.
    """
    close = float(features["close"])
    sma_10, sma_50 = float(features["sma_10"]), float(features["sma_50"])
    mom, rsi = float(features["mom_10"]), float(features["rsi_14"])
    if close > sma_10 and close > sma_50 and mom > 0 and rsi <= 70:
        return "BUY"
    if close < sma_10 and close < sma_50 and mom < 0 and rsi >= 30:
        return "SELL"
    return "HOLD"


#: "RSI is above 70" / "RSI is below 30" as a plain statement (a negated
#: sentence such as "RSI is not above 70" does not match).
_RSI_ABOVE_70 = re.compile(r"\brsi\b[^.]{0,15}?\b(?:is|was)\s+(?:above|over)\s+70", re.I)
_RSI_BELOW_30 = re.compile(r"\brsi\b[^.]{0,15}?\b(?:is|was)\s+(?:below|under)\s+30", re.I)


def rsi_threshold_claim(reason: str, rsi: float, margin: float = 2.0) -> str | None:
    """Check a written reason's RSI threshold statement against the real value.

    The prompt's RSI rule is a veto (above 70: do not buy; below 30: do not
    sell), so a reason that says the RSI crossed one of those lines is making
    a checkable claim. A claim counts as false only when the real RSI is more
    than ``margin`` points on the wrong side, so rounding is not punished.

    Returns:
        ``"above_70_true"``, ``"above_70_false"``, ``"below_30_true"``,
        ``"below_30_false"``, or ``None`` when the reason makes no such claim.
    """
    text = str(reason or "")
    if _RSI_ABOVE_70.search(text):
        return "above_70_false" if rsi < 70 - margin else "above_70_true"
    if _RSI_BELOW_30.search(text):
        return "below_30_false" if rsi > 30 + margin else "below_30_true"
    return None


def load_log(results_dir: str | Path, tag: str) -> pd.DataFrame:
    """Join a recommendation log with the features each answer was given."""
    root = Path(results_dir)
    recs = pd.read_csv(root / f"{tag}_recommendations.csv")
    rows: list[dict[str, Any]] = []
    with open(root / f"llm_cache_{tag}.jsonl", encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)["recommendation"]
            rows.append({"date": rec["as_of"], "ticker": rec["ticker"],
                         "features": rec["features"]})
    feats = pd.DataFrame(rows).drop_duplicates(["date", "ticker"], keep="last")
    df = recs.merge(feats, on=["date", "ticker"], how="inner")
    df["policy_action"] = [prompt_policy_action(f) for f in df["features"]]
    df["on_policy"] = df["policy_action"] == df["action"]
    df["rsi_claim"] = [
        rsi_threshold_claim(reason, features["rsi_14"])
        for reason, features in zip(df["reason"], df["features"], strict=True)
    ]
    return df


def adherence_summary(df: pd.DataFrame, tag: str) -> dict[str, Any]:
    """Summarise one advisor's adherence and on/off-policy BUY outcomes."""
    out: dict[str, Any] = {"advisor": tag, "n": len(df),
                           "adherence": float(df["on_policy"].mean())}
    for action in ACTIONS:
        sub = df[df["action"] == action]
        out[f"n_{action}"] = len(sub)
        out[f"adherence_{action}"] = float(sub["on_policy"].mean()) if len(sub) else None
        pol = df[df["policy_action"] == action]
        out[f"policy_n_{action}"] = len(pol)
    buys = df[df["action"] == "BUY"].dropna(subset=["realised_return"])
    for label, part in (("on", buys[buys["on_policy"]]), ("off", buys[~buys["on_policy"]])):
        out[f"buy_{label}_policy_n"] = len(part)
        out[f"buy_{label}_policy_up_rate"] = (
            float((part["realised_return"] > 0).mean()) if len(part) else None)
        out[f"buy_{label}_policy_mean_ret_5d"] = (
            float(part["realised_return"].mean()) if len(part) else None)
    # Reasons that state the RSI crossed a veto line when it had not.
    for claim in ("above_70", "below_30"):
        made = df["rsi_claim"].isin([f"{claim}_true", f"{claim}_false"])
        false = df["rsi_claim"] == f"{claim}_false"
        out[f"rsi_{claim}_claims"] = int(made.sum())
        out[f"rsi_{claim}_false"] = int(false.sum())
        out[f"rsi_{claim}_false_off_policy"] = int((false & ~df["on_policy"]).sum())
    return out


def run(results_dir: str | Path, tags: Sequence[str]) -> pd.DataFrame:
    """Write the per-advisor confusion tables and the summary; return the summary."""
    root = Path(results_dir)
    summaries = []
    for tag in tags:
        df = load_log(root, tag)
        confusion = pd.crosstab(df["policy_action"], df["action"]).reindex(
            index=list(ACTIONS), columns=list(ACTIONS), fill_value=0)
        confusion.index.name = "instructed"
        confusion.columns.name = "answered"
        confusion.to_csv(root / f"policy_adherence_{tag}.csv")
        summaries.append(adherence_summary(df, tag))
        _log.info("[%s] adherence %.3f on %d answers", tag, summaries[-1]["adherence"], len(df))
    summary = pd.DataFrame(summaries)
    summary.to_csv(root / "policy_adherence_summary.csv", index=False)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point; prints the summary table."""
    ap = argparse.ArgumentParser(
        prog="python -m advisor.evaluation.policy_adherence",
        description="POST-HOC: how often each LLM followed its own prompt's decision policy.")
    ap.add_argument("--tags", nargs="+", default=["llama", "qwen"])
    ap.add_argument("--results-dir", default="results")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    summary = run(args.results_dir, args.tags)
    print(summary.T.to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
