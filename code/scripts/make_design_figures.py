"""Draw the two design figures used in report Chapter 3.

    Figure 1  results/figures/architecture_pipeline.{png,svg}
              the six-stage pipeline and the evaluation harness around it
    Figure 2  results/figures/walkforward_calibration_timeline.{png,svg}
              how the monthly calibration refit only uses labels that have
              already resolved

Pure matplotlib, no data needed. Usage (from code/):
    .venv/bin/python scripts/make_design_figures.py
"""

from pathlib import Path

import matplotlib
from matplotlib.axes import Axes
from matplotlib.figure import Figure

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

FIGURES = Path(__file__).resolve().parents[1] / "results" / "figures"

# Okabe-Ito colours, lightened for box fills.
BLUE, GREEN, ORANGE, PURPLE, GREY = "#DCEBF7", "#D9F0E7", "#FBE9CF", "#F1E1EC", "#EEEEEE"
EDGE = "#444444"


def box(ax: Axes, x: float, y: float, w: float, h: float, title: str, lines: list[str],
        fill: str) -> None:
    """A rounded box with a bold title and a few lines of smaller text."""
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                facecolor=fill, edgecolor=EDGE, linewidth=1.0))
    ax.text(x + w / 2, y + h - 0.2, title, ha="center", va="top", fontsize=9.5,
            fontweight="bold")
    ax.text(x + w / 2, y + h - 0.55, "\n".join(lines), ha="center", va="top", fontsize=7.6,
            linespacing=1.35, color="#222222")


def arrow(ax: Axes, start: tuple[float, float], end: tuple[float, float],
          text: str | None = None, style: str = "-|>") -> None:
    """An arrow between two points, with an optional small label."""
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle=style, mutation_scale=12,
                                 color=EDGE, linewidth=1.0))
    if text:
        ax.text((start[0] + end[0]) / 2, (start[1] + end[1]) / 2 + 0.08, text, ha="center",
                va="bottom", fontsize=7, color="#555555")


def architecture() -> None:
    """Figure 1: the six stages, the answer cache and the evaluation harness."""
    fig, ax = plt.subplots(figsize=(11, 4.6), dpi=200)
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 4.6)
    ax.axis("off")

    stages = [
        ("1  Data", ["yfinance daily prices", "51 US large caps + SPY", "parquet cache"], BLUE),
        ("2  Features",
         ["SMA 10 / 50, momentum,", "volatility, RSI, 1-day", "return (past data only)"], BLUE),
        ("3  Advisor",
         ["Llama 3.1 8B or Qwen3 8B", "(local, Ollama) or rule", "→ action, confidence,",
          "reason, own sentence"], ORANGE),
        ("4  Calibration", ["temperature scaling", "refit monthly on", "resolved labels only"],
         GREEN),
        ("5  Counterfactual",
         ["DiCE-style search", "≤ 40 model calls", "keeps observed flips only"], GREEN),
        ("6  Web page", ["Streamlit: four cards,", "track record,", "\"check this claim\""],
         PURPLE),
    ]
    width, height, gap, top = 1.6, 1.35, 0.2, 2.95
    for i, (title, lines, fill) in enumerate(stages):
        x = 0.2 + i * (width + gap)
        box(ax, x, top, width, height, title, lines, fill)
        if i:
            arrow(ax, (x - gap, top + height / 2), (x, top + height / 2))

    # Answer cache under the advisor.
    box(ax, 3.8, 1.55, 1.6, 0.95, "Answer cache", ["JSONL, one line per", "model answer"], GREY)
    arrow(ax, (4.6, top), (4.6, 2.5), style="<|-|>")

    # Evaluation harness along the bottom.
    box(ax, 0.2, 0.15, 10.6, 1.05, "Evaluation harness (offline, reproducible)",
        ["walk-forward backtest, weekly decisions, 10 bps costs  ·  baselines: S&P 500, "
         "random portfolios (100 seeds), 12-1 momentum, Markowitz",
         "Deflated Sharpe (7 trials)  ·  ECE, Brier, sharpness  ·  post-hoc: policy adherence, "
         "counterfactual faithfulness"], GREY)
    for x in (1.0, 2.8, 6.4):
        arrow(ax, (x, top), (x, 1.2))
    arrow(ax, (4.6, 1.55), (4.6, 1.2))

    ax.text(5.5, 4.55, "Recommendations only: no orders, no broker connection, no money",
            ha="center", va="top", fontsize=8.5, style="italic", color="#555555")
    save(fig, "architecture_pipeline")


def timeline() -> None:
    """Figure 2: weekly decisions, label delays and the monthly calibration refits."""
    fig, ax = plt.subplots(figsize=(11, 3.6), dpi=200)
    ax.set_xlim(-0.3, 20.5)
    ax.set_ylim(-1.6, 2.6)
    ax.axis("off")

    months = ["January", "February", "March", "April", "May"]
    weeks_per_month = 4
    for m, name in enumerate(months):
        start = m * weeks_per_month
        ax.axvspan(start, start + weeks_per_month, ymin=0.35, ymax=0.62,
                   color=[BLUE, GREEN][m % 2], zorder=0)
        ax.text(start + weeks_per_month / 2, 0.55, name, ha="center", va="center", fontsize=9)
        ax.plot([start, start], [-0.1, 1.9], color=EDGE, linewidth=1.0, zorder=1)
        if m == 0:
            label = "January: fewer than 50\nresolved labels, confidence\nshown raw"
        else:
            label = f"1 {name[:3]}: refit T on every\nlabel resolved before today"
        ax.text(start + 0.1, 2.0, label, ha="left", va="bottom", fontsize=7.2, color="#333333")

    # Weekly decisions and their five-trading-day labels.
    for week in range(len(months) * weeks_per_month):
        ax.plot(week + 0.5, 0.0, marker="o", markersize=5, color="#0072B2", zorder=3)
        ax.annotate("", xy=(week + 1.5, -0.55), xytext=(week + 0.5, -0.05),
                    arrowprops={"arrowstyle": "-|>", "color": "#999999", "linewidth": 0.8})
    ax.plot([], [], "o", color="#0072B2", label="weekly decision (Friday)")
    ax.plot([], [], color="#999999", label="its label resolves 5 trading days later")

    # Highlight what the March fit may use.
    ax.axvspan(0, 8, ymin=0.05, ymax=0.18, color="#D55E00", alpha=0.25, zorder=0)
    ax.text(4, -1.25, "data the March fit may use: labels resolved before 1 March "
            "(never the future)", ha="center", va="center", fontsize=8, color="#8A3A00")
    ax.legend(loc="upper right", bbox_to_anchor=(1.0, 0.18), fontsize=7.5, frameon=False)
    save(fig, "walkforward_calibration_timeline")


def save(fig: Figure, name: str) -> None:
    """Write the figure as PNG (for the report) and SVG (for slides)."""
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(FIGURES / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print("wrote", FIGURES / f"{name}.png")


if __name__ == "__main__":
    architecture()
    timeline()
