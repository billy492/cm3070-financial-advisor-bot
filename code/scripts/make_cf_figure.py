"""Draw the counterfactual line-search figure for the report (Section 4.5).

Runs the real :func:`advisor.counterfactual.dice.generate_counterfactuals` on
one illustrative instance against the prompt's own decision policy (so the
decision regions are exact and can be shaded), records every probe the search
makes, and plots them in the RSI x 10-day-momentum plane in the order they were
visited. No model calls. Output: ``results/figures/cf_line_search.{png,svg}``.

Usage (from ``code/``)::

    .venv/bin/python scripts/make_cf_figure.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from advisor.counterfactual.dice import (  # noqa: E402
    DEFAULT_FEATURE_SPECS,
    generate_counterfactuals,
)
from advisor.evaluation.policy_adherence import prompt_policy_action  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "results" / "figures" / "cf_line_search"
ORANGE, BLUE, GREEN, GREY = "#E69F00", "#0072B2", "#009E73", "#999999"

# An uptrend BUY: price above both averages, positive momentum, RSI 55.
INSTANCE = {"close": 104.0, "sma_10": 102.0, "sma_50": 100.0, "mom_10": 0.03,
            "rsi_14": 55.0, "vol_20": 0.012, "ret_1d": 0.004}


def main() -> None:
    """Record the search's probes and draw them over the policy's decision regions."""
    probes: list[tuple[float, float, str]] = []

    def predict(features):
        action = prompt_policy_action(features)
        probes.append((float(features["rsi_14"]), float(features["mom_10"]), action))
        return action

    specs = [s for s in DEFAULT_FEATURE_SPECS if s.name in {"rsi_14", "mom_10"}]
    cfs = generate_counterfactuals(predict, INSTANCE, specs=specs, total_cfs=3, max_calls=40)

    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=200)
    rsi = np.linspace(20, 90, 400)
    mom = np.linspace(-0.12, 0.17, 400)
    rr, mm = np.meshgrid(rsi, mom)
    buy = (mm > 0) & (rr <= 70)
    ax.contourf(rr, mm, buy.astype(float), levels=[-0.5, 0.5, 1.5],
                colors=["#F4F4F4", "#DCEFE7"])
    ax.text(30, 0.11, "policy says BUY", color=GREEN, fontsize=9)
    ax.text(75, 0.11, "HOLD (RSI veto)", color="#555555", fontsize=9)
    ax.text(28, -0.10, "HOLD (momentum does not confirm)", color="#555555", fontsize=9)

    for k, (r, m, action) in enumerate(probes[1:], start=1):
        flipped = action != "BUY"
        ax.scatter(r, m, s=46 if flipped else 26, color=ORANGE if flipped else BLUE,
                   edgecolor="black" if flipped else "none", linewidth=0.6, zorder=3)
        ax.annotate(str(k), (r, m), textcoords="offset points", xytext=(4, 4), fontsize=7,
                    color="#333333")
    ax.scatter(INSTANCE["rsi_14"], INSTANCE["mom_10"], marker="*", s=220, color="black",
               zorder=4, label="recommendation being explained (BUY)")
    ax.scatter([], [], s=26, color=BLUE, label="probe: action unchanged")
    ax.scatter([], [], s=46, color=ORANGE, edgecolor="black", linewidth=0.6,
               label="probe: action flipped (candidate counterfactual)")
    ax.axhline(0, color=GREY, linewidth=0.6)
    ax.axvline(70, color=GREY, linewidth=0.6)
    ax.set_xlabel("14-day RSI")
    ax.set_ylabel("10-day momentum")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_xlim(20, 90)
    ax.set_ylim(-0.12, 0.17)
    ax.legend(loc="lower right", fontsize=7.5, frameon=True)
    ax.set_title(f"{len(probes)} model calls, numbered in visiting order; "
                 f"{len(cfs)} verified flips found", fontsize=9)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(OUT.with_suffix(f".{ext}"))
    print(f"{len(probes)} calls; counterfactuals: {[cf.sentence for cf in cfs]}")


if __name__ == "__main__":
    main()
