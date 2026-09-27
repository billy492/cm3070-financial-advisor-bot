"""Summarise the user study (H4) exactly as planned before the first session.

Fill in results.csv (one row per participant) from the paper scoring sheets:

    participant  P1, P2, ...
    order        A or B (the condition seen first; A = WITH the counterfactual card)
    Q1_A..Q5_A   ratings 1-7 for the page WITH the card
    Q1_B..Q5_B   ratings 1-7 for the page WITHOUT the card
    probe_A/B    probe score 0-2 for each page
    sus          System Usability Scale total, 0-100

Then run:  python3 docs/user-study/analyse_user_study.py

It prints the results table for the Evaluation chapter, the medians and
ranges, the count of participants who rated WITH higher / equal / lower, the
order check, and the H4 verdict using the decision rule fixed before the
first session. No significance tests, on purpose (n is 3-5).
"""

import csv
import statistics
from pathlib import Path

RESULTS = Path(__file__).with_name("results.csv")
ITEMS = {
    "Q1": "Understand why",
    "Q2": "Could check",
    "Q3": "Trust right amount",
    "Q4": "Confidence helped",
    "Q5": "Wording clear",
    "probe": "Probe (0-2)",
}


def read_rows():
    """Participants from results.csv, skipping blank lines."""
    with RESULTS.open(newline="") as f:
        return [row for row in csv.DictReader(f) if row["participant"].strip()]


def score(row, item, condition):
    """One participant's score for an item under condition "A" or "B"."""
    return float(row[f"{item}_{condition}"])


def fmt(x):
    """Show whole numbers without a decimal point."""
    return str(int(x)) if x == int(x) else f"{x:.1f}"


def median_and_range(values):
    """Summarise values as "median (lowest-highest)", as the analysis plan asks."""
    return f"{fmt(statistics.median(values))} ({fmt(min(values))}-{fmt(max(values))})"


def main():
    """Print the table, the paired differences and the H4 verdict."""
    rows = read_rows()
    n = len(rows)
    if n == 0:
        print("results.csv has no participants yet.")
        return

    # Results table: WITH / WITHOUT (difference) for each participant.
    header = ["Participant", "Order"] + list(ITEMS.values()) + ["SUS"]
    print("| " + " | ".join(header) + " |")
    print("|" + "---|" * len(header))
    for row in rows:
        cells = [row["participant"], row["order"]]
        for item in ITEMS:
            a, b = score(row, item, "A"), score(row, item, "B")
            cells.append(f"{fmt(a)} / {fmt(b)} ({a - b:+g})")
        cells.append(row["sus"])
        print("| " + " | ".join(cells) + " |")

    medians = ["**Median (range)**", "—"]
    for item in ITEMS:
        a_values = [score(r, item, "A") for r in rows]
        b_values = [score(r, item, "B") for r in rows]
        medians.append(f"{median_and_range(a_values)} / {median_and_range(b_values)}")
    sus = [float(r["sus"]) for r in rows]
    medians.append(median_and_range(sus))
    print("| " + " | ".join(medians) + " |")

    # Paired differences: how many rated WITH higher, the same, or lower?
    print(f"\nPaired differences (WITH - WITHOUT), n = {n}:")
    meets_rule = {}
    for item in ITEMS:
        diffs = [score(r, item, "A") - score(r, item, "B") for r in rows]
        higher = sum(d > 0 for d in diffs)
        same = sum(d == 0 for d in diffs)
        lower = sum(d < 0 for d in diffs)
        median_a = statistics.median(score(r, item, "A") for r in rows)
        median_b = statistics.median(score(r, item, "B") for r in rows)
        print(f"  {ITEMS[item]:<20} higher {higher}, same {same}, lower {lower}; "
              f"median WITH {fmt(median_a)} vs WITHOUT {fmt(median_b)}")
        meets_rule[item] = median_a > median_b and higher >= lower

    # Decision rule fixed before session 1 (pre-registration, H4).
    if meets_rule["Q1"] and meets_rule["Q2"]:
        verdict = "supported"
    elif meets_rule["Q1"] or meets_rule["Q2"]:
        verdict = "partly supported"
    else:
        verdict = "not supported"
    print(f"\nH4 is {verdict} (rule: WITH median higher AND at least as many rate WITH "
          "higher as lower, for both Q1 and Q2).")

    # Order check: does whatever was seen second score higher, whatever the condition?
    first, second = [], []
    for row in rows:
        seen_first = row["order"].strip().upper()
        seen_second = "B" if seen_first == "A" else "A"
        first.append(statistics.mean(score(row, q, seen_first) for q in ["Q1", "Q2"]))
        second.append(statistics.mean(score(row, q, seen_second) for q in ["Q1", "Q2"]))
    print(f"Order check (mean of Q1-Q2): first-seen median {fmt(statistics.median(first))}, "
          f"second-seen median {fmt(statistics.median(second))}.")

    probe_twos_a = sum(score(r, "probe", "A") == 2 for r in rows)
    probe_twos_b = sum(score(r, "probe", "B") == 2 for r in rows)
    print(f"Probe: {probe_twos_a}/{n} scored 2 WITH the card, {probe_twos_b}/{n} WITHOUT.")
    print(f"SUS: {', '.join(r['sus'] for r in rows)} (median {fmt(statistics.median(sus))}; "
          "68 is the usual benchmark, for context only).")
    print(f"\nWith n = {n} these results are descriptive evidence only and do not generalise.")


if __name__ == "__main__":
    main()
