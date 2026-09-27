"""Make the plain slides used in the demo video.

Writes 1920x1080 PNGs to demo/slides/. Open them full screen in Preview while
recording. Re-run after changing the user-study size below.

Usage (from the thesis folder):
    code/.venv/bin/python demo/make_slides.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent / "slides"
USER_STUDY_N = "[N]"  # change to the real number of participants before recording

INK = "#1F2933"
MUTED = "#5B6770"
ACCENT = "#0072B2"


def new_slide():
    """A blank 16:9 canvas with coordinates from 0 to 1."""
    fig = plt.figure(figsize=(19.2, 10.8), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    return fig, ax


def save(fig, name):
    """Write one slide and report where it went."""
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}.png"
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    print("wrote", path)


def title_slide():
    """Beat 1: title card."""
    fig, ax = new_slide()
    ax.text(0.5, 0.62, "Financial Advisor Bot", ha="center", fontsize=72, color=INK,
            fontweight="bold")
    ax.text(0.5, 0.50, "Explained, calibration-aware stock advice for non-experts",
            ha="center", fontsize=34, color=MUTED)
    ax.text(0.5, 0.30, "Aly Thabet  ·  CM3070 Final Project  ·  CM3020 Project Idea 2",
            ha="center", fontsize=26, color=MUTED)
    save(fig, "01-title")


def bullet_slide(name, heading, lines):
    """A heading and up to four short lines."""
    fig, ax = new_slide()
    ax.text(0.08, 0.80, heading, fontsize=54, color=INK, fontweight="bold")
    for i, line in enumerate(lines):
        y = 0.60 - i * 0.14
        ax.add_patch(plt.Rectangle((0.08, y - 0.012), 0.012, 0.05, color=ACCENT))
        ax.text(0.11, y, line, fontsize=40, color=INK, va="bottom")
    save(fig, name)


def results_slide():
    """Beat 4: four numbers the narration reads out."""
    fig, ax = new_slide()
    ax.text(0.08, 0.84, "What the evaluation found", fontsize=54, color=INK, fontweight="bold")
    cards = [
        ("+25.5% vs +31.1%",
         "Qwen advisor vs S&P 500\n(neither model beats the index\n"
         "after deflating for 7 strategies)"),
        ("~70% said, ~27% right",
         "stated confidence vs actual hit rate;\ninside BUY/SELL calls the number\n"
         "carries no information"),
        ("47.6%",
         "how often Llama followed its own rules;\nthe search found its real one:\n"
         "buy when RSI is below about 40"),
        ("18 / 40  vs  39 / 40",
         "Llama vs Qwen: own \"what would\nchange my mind\" sentences\n"
         "that actually flip the call"),
    ]
    for i, (number, caption) in enumerate(cards):
        x = 0.08 + (i % 2) * 0.44
        y = 0.45 if i < 2 else 0.12
        ax.text(x, y + 0.22, number, fontsize=46, color=ACCENT, fontweight="bold")
        ax.text(x, y + 0.19, caption, fontsize=24, color=MUTED, va="top", linespacing=1.3)
    save(fig, "03-results")


def end_slide():
    """Beat 7: end card with the repository."""
    fig, ax = new_slide()
    ax.text(0.5, 0.58, "github.com/billy492/cm3070-financial-advisor-bot", ha="center",
            fontsize=40, color=ACCENT)
    ax.text(0.5, 0.44, "Research prototype — not financial advice", ha="center", fontsize=30,
            color=MUTED)
    save(fig, "05-end")


if __name__ == "__main__":
    title_slide()
    bullet_slide("02-problem", "The gap", [
        "Signals without reasons",
        "Reasons without honest uncertainty",
        "Nobody says what would flip the advice",
    ])
    results_slide()
    bullet_slide("04-limitations", "Limitations, honestly", [
        "One 17-month window, in a strong market",
        f"User study: n = {USER_STUDY_N}",
        "Counterfactuals move one indicator at a time",
    ])
    end_slide()
