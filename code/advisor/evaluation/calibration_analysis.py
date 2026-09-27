"""Post-hoc calibration analysis of a labelled recommendation log.

**Status: post-hoc.** The pre-registered calibration result (H2 in
``docs/preregistration.md``) is the walk-forward temperature scaling run by
:mod:`advisor.evaluation.run_evaluation` and stored in ``calibration_<tag>.json``.
That result stands as reported. This script is a *secondary, exploratory*
analysis added after the primary result was known, and every number it
produces must be labelled as such in the report.

It exists to test one structural explanation of the primary finding: under the
thesis's correctness rule a HOLD is only "correct" when the five-day return
stays inside +/-1%, which happens about a fifth of the time, so a HOLD-heavy
advisor is correct well under half the time. Temperature scaling fixes 0.5 and
can only shrink towards it, so it cannot represent such a base rate; it lowers
ECE by collapsing sharpness and hits the temperature ceiling on every fold. The
analysis therefore re-runs the *same* monthly walk-forward protocol (refit at
the first decision date of each calendar month on every label realised strictly
before that date, ``--min-samples`` labels required before the first fit) with:

* ``raw`` -- the identity, for reference;
* ``base_rate`` -- the expanding-window hit rate as a constant forecast, the
  trivially calibrated, zero-sharpness reference (Brier ``r(1 - r)``);
* ``temperature`` -- :class:`~advisor.calibration.TemperatureScaler`, the
  pre-registered method, replayed from the log;
* ``platt`` -- :class:`~advisor.calibration.PlattScaler`, logistic regression
  with an intercept on the confidence logit, which *can* fit a base rate below
  one half at the cost of the action-preserving property;

on two subsets of the log:

* ``all`` -- every labelled recommendation (the primary protocol);
* ``directional`` -- BUY and SELL only, where "correct" means the sign of the
  realised return matched the call, so the HOLD labelling rule plays no part.

Outputs (in ``--results-dir``):

* ``calibration_analysis_<tag>.csv`` -- one row per (subset, method) with the
  sample size, hit rate, ECE (equal-width and equal-mass bins), Brier score,
  NLL, sharpness, mean confidence, the share of calibrated confidences below
  0.5 and the share that crossed 0.5, and the median fitted parameters;
* ``calibration_analysis_<tag>_folds.csv`` -- the per-fold fit log (sample
  size, status, fitted parameters) for every subset and method;
* ``figures/reliability_analysis_<tag>_<subset>.png/.svg`` -- reliability
  diagrams overlaying raw, temperature-scaled and Platt-scaled confidences.

No LLM calls are made; the script only re-reads ``<tag>_recommendations.csv``.

Example:
    ``python -m advisor.evaluation.calibration_analysis --tags llama qwen``
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from advisor.calibration import (
    PlattScaler,
    TemperatureScaler,
    plot_reliability_diagram,
    reliability_curve,
)
from advisor.evaluation.metrics import (
    adaptive_ece,
    brier_score,
    expected_calibration_error,
    sharpness,
)

__all__ = [
    "SUBSETS",
    "METHODS",
    "load_recommendations",
    "select_subset",
    "walk_forward_calibrate",
    "summarise",
    "run_analysis",
    "build_parser",
    "main",
]

log = logging.getLogger(__name__)

SUBSETS: dict[str, tuple[str, ...] | None] = {
    "all": None,
    "directional": ("BUY", "SELL"),
}
"""Subset name -> actions kept (``None`` keeps every action)."""

METHODS: tuple[str, ...] = ("raw", "base_rate", "temperature", "platt")
"""Calibration methods replayed, in output order."""

_METHOD_LABELS = {
    "raw": "Raw",
    "base_rate": "Base rate",
    "temperature": "Temperature-scaled",
    "platt": "Platt-scaled",
}

_EPS = 1e-6


# --------------------------------------------------------------------------- #
# Loading and subsetting
# --------------------------------------------------------------------------- #
#: Readable names used in figure titles.
_ADVISOR_NAMES: dict[str, str] = {
    "llama": "Llama 3.1 8B",
    "qwen": "Qwen3 8B",
    "heuristic": "Rule-based advisor",
}
_SUBSET_NAMES: dict[str, str] = {
    "all": "all recommendations",
    "directional": "BUY/SELL only",
}


def load_recommendations(path: str | Path) -> pd.DataFrame:
    """Read a ``<tag>_recommendations.csv`` written by the evaluation runner.

    Args:
        path: CSV path.

    Returns:
        The frame with ``date`` / ``label_date`` parsed to timestamps,
        ``raw_confidence`` / ``correct`` coerced to floats, ``action``
        upper-cased and rows sorted by ``date`` then ``ticker``.

    Raises:
        ValueError: If a required column is missing.
    """
    df = pd.read_csv(path)
    required = {"date", "ticker", "action", "raw_confidence", "correct", "label_date", "fold"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{path}: missing columns {missing}.")
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["label_date"] = pd.to_datetime(df["label_date"], errors="coerce")
    df["raw_confidence"] = pd.to_numeric(df["raw_confidence"], errors="coerce")
    df["correct"] = pd.to_numeric(df["correct"], errors="coerce")
    df["action"] = df["action"].astype(str).str.upper()
    df["fold"] = df["fold"].astype(str)
    return df.sort_values(["date", "ticker"], kind="stable").reset_index(drop=True)


def select_subset(df: pd.DataFrame, subset: str) -> pd.DataFrame:
    """Restrict a recommendation log to one of :data:`SUBSETS`.

    Args:
        df: Frame from :func:`load_recommendations`.
        subset: A key of :data:`SUBSETS`.

    Returns:
        The rows whose action belongs to the subset, index reset.

    Raises:
        ValueError: If ``subset`` is unknown.
    """
    if subset not in SUBSETS:
        raise ValueError(f"subset must be one of {sorted(SUBSETS)}; got {subset!r}.")
    actions = SUBSETS[subset]
    if actions is None:
        return df.reset_index(drop=True)
    return df[df["action"].isin(actions)].reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Walk-forward replay
# --------------------------------------------------------------------------- #
def _make_scaler(method: str) -> Any:
    """Return a fresh calibrator for ``method`` (``temperature`` or ``platt``)."""
    if method == "temperature":
        return TemperatureScaler()
    if method == "platt":
        return PlattScaler()
    raise ValueError(f"no scaler for method {method!r}.")


def _params_of(scaler: Any) -> dict[str, float | None]:
    """Extract the fitted parameters of a scaler as a flat dict."""
    return {
        "temperature": getattr(scaler, "temperature", None),
        "slope": getattr(scaler, "slope", None),
        "intercept": getattr(scaler, "intercept", None),
    }


def walk_forward_calibrate(
    df: pd.DataFrame,
    method: str,
    min_samples: int = 50,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Replay the monthly walk-forward calibration of the evaluation runner.

    At the first decision date of every fold the calibrator is refitted on all
    rows whose label realised strictly before that date (expanding window),
    provided at least ``min_samples`` such rows exist; otherwise the fold keeps
    its raw confidences. The fitted map is then applied to every row of the
    fold. This mirrors ``WalkForwardBacktest._refit_calibrator`` so the
    ``temperature`` replay reproduces the primary run.

    Args:
        df: A (possibly subsetted) frame from :func:`load_recommendations`.
        method: One of :data:`METHODS`.
        min_samples: Labelled samples required before the first fit.

    Returns:
        ``(calibrated, fold_log)`` where ``calibrated`` is a float array
        aligned with ``df`` (``NaN`` where the raw confidence is ``NaN``) and
        ``fold_log`` has one dict per fold with ``fold``, ``fitted_at``,
        ``n_samples``, ``status`` and the fitted parameters.

    Raises:
        ValueError: If ``method`` is unknown.
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}; got {method!r}.")

    raw = df["raw_confidence"].to_numpy(dtype="float64")
    correct = df["correct"].to_numpy(dtype="float64")
    label_dates = df["label_date"].to_numpy()
    folds = df["fold"].to_numpy()
    calibrated = raw.copy()
    fold_log: list[dict[str, Any]] = []

    for fold in sorted(pd.unique(folds)):
        in_fold = folds == fold
        current = df.loc[in_fold, "date"].min()
        entry: dict[str, Any] = {
            "method": method,
            "fold": str(fold),
            "fitted_at": pd.Timestamp(current).date().isoformat(),
            "n_samples": 0,
            "status": "identity",
            "temperature": None,
            "slope": None,
            "intercept": None,
            "base_rate": None,
        }
        if method == "raw":
            fold_log.append(entry)
            continue

        train = (
            pd.notna(label_dates)
            & (label_dates < np.datetime64(pd.Timestamp(current)))
            & np.isfinite(raw)
            & np.isfinite(correct)
        )
        n_train = int(train.sum())
        entry["n_samples"] = n_train
        if n_train < int(min_samples):
            entry["status"] = "insufficient_samples"
            fold_log.append(entry)
            continue

        apply = in_fold & np.isfinite(raw)
        if method == "base_rate":
            rate = float(np.mean(correct[train]))
            calibrated[apply] = rate
            entry["base_rate"] = rate
            entry["status"] = "fitted"
            fold_log.append(entry)
            continue

        try:
            scaler = _make_scaler(method).fit(raw[train], correct[train].astype(int))
        except Exception as exc:  # noqa: BLE001 - logged, fold keeps raw confidences
            log.warning("[%s] fold %s: fit failed (%s); keeping raw.", method, fold, exc)
            entry["status"] = f"failed: {exc}"
            fold_log.append(entry)
            continue
        if apply.any():
            calibrated[apply] = np.asarray(scaler.transform(raw[apply]), dtype="float64")
        entry.update(_params_of(scaler))
        entry["status"] = "fitted"
        fold_log.append(entry)

    return calibrated, fold_log


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def _nll(confidences: np.ndarray, labels: np.ndarray) -> float:
    """Mean binary negative log-likelihood with confidences clipped away from 0/1."""
    p = np.clip(confidences, _EPS, 1.0 - _EPS)
    return float(-np.mean(labels * np.log(p) + (1.0 - labels) * np.log1p(-p)))


def summarise(
    subset: str,
    method: str,
    raw: np.ndarray,
    calibrated: np.ndarray,
    correct: np.ndarray,
    fold_log: Sequence[dict[str, Any]],
    n_bins: int = 10,
) -> dict[str, Any]:
    """Score one (subset, method) pair on its labelled rows.

    Args:
        subset: Subset name (recorded in the row).
        method: Method name (recorded in the row).
        raw: Raw confidences aligned with ``correct``.
        calibrated: Calibrated confidences aligned with ``correct``.
        correct: Correctness labels (``NaN`` = unlabelled, excluded).
        fold_log: Per-fold log from :func:`walk_forward_calibrate`.
        n_bins: Bins for the ECE variants.

    Returns:
        A flat dict: ``subset``, ``method``, ``n``, ``n_folds``,
        ``n_folds_fitted``, ``hit_rate``, ``ece``, ``adaptive_ece``, ``brier``,
        ``nll``, ``sharpness``, ``mean_confidence``, ``frac_below_half``,
        ``frac_crossed_half`` and the median ``temperature`` / ``slope`` /
        ``intercept`` over fitted folds (``NaN`` when not applicable).
    """
    keep = np.isfinite(raw) & np.isfinite(calibrated) & np.isfinite(correct)
    r, c, y = raw[keep], calibrated[keep], correct[keep]
    nan = float("nan")
    fitted = [e for e in fold_log if e.get("status") == "fitted"]

    def _median(key: str) -> float:
        vals = [float(e[key]) for e in fitted if e.get(key) is not None]
        return float(np.median(vals)) if vals else nan

    row: dict[str, Any] = {
        "subset": subset,
        "method": method,
        "n": int(y.size),
        "n_folds": int(len(fold_log)),
        "n_folds_fitted": int(len(fitted)),
        "hit_rate": nan,
        "ece": nan,
        "adaptive_ece": nan,
        "brier": nan,
        "nll": nan,
        "sharpness": nan,
        "mean_confidence": nan,
        "frac_below_half": nan,
        "frac_crossed_half": nan,
        "temperature": _median("temperature"),
        "slope": _median("slope"),
        "intercept": _median("intercept"),
    }
    if y.size == 0:
        return row
    row.update(
        {
            "hit_rate": float(np.mean(y)),
            "ece": float(expected_calibration_error(c, y, n_bins=n_bins)),
            "adaptive_ece": float(adaptive_ece(c, y, n_bins=n_bins)),
            "brier": float(brier_score(c, y)),
            "nll": _nll(c, y),
            "sharpness": float(sharpness(c)),
            "mean_confidence": float(np.mean(c)),
            "frac_below_half": float(np.mean(c < 0.5)),
            "frac_crossed_half": float(np.mean((r > 0.5) != (c > 0.5))),
        }
    )
    return row


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run_analysis(
    tag: str,
    results_dir: str | Path,
    *,
    min_samples: int = 50,
    n_bins: int = 10,
    figures: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the full post-hoc analysis for one advisor tag and write its outputs.

    Args:
        tag: Advisor tag (``llama``, ``qwen``, ``heuristic``); reads
            ``<results_dir>/<tag>_recommendations.csv``.
        results_dir: Results folder of the evaluation runner.
        min_samples: Labelled samples required before the first calibrator fit.
        n_bins: Bins for the ECE variants and reliability diagrams.
        figures: Whether to render reliability diagrams.

    Returns:
        ``(summary, folds)`` -- the two tables that were written to disk.

    Raises:
        FileNotFoundError: If the recommendation log does not exist.
    """
    results_dir = Path(results_dir)
    rec_path = results_dir / f"{tag}_recommendations.csv"
    if not rec_path.exists():
        raise FileNotFoundError(f"No recommendation log at {rec_path}.")
    df = load_recommendations(rec_path)
    log.info("[%s] %d recommendations, %d folds", tag, len(df), df["fold"].nunique())

    summary_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    for subset in SUBSETS:
        part = select_subset(df, subset)
        if part.empty:
            log.warning("[%s] subset %s is empty; skipped.", tag, subset)
            continue
        raw = part["raw_confidence"].to_numpy(dtype="float64")
        correct = part["correct"].to_numpy(dtype="float64")
        curves: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        for method in METHODS:
            calibrated, fold_log = walk_forward_calibrate(part, method, min_samples=min_samples)
            summary_rows.append(
                summarise(subset, method, raw, calibrated, correct, fold_log, n_bins)
            )
            fold_rows.extend({"tag": tag, "subset": subset, **e} for e in fold_log)
            if method != "base_rate":
                keep = np.isfinite(calibrated) & np.isfinite(correct)
                if keep.any():
                    curves[_METHOD_LABELS[method]] = reliability_curve(
                        calibrated[keep], correct[keep], n_bins=n_bins, strategy="uniform"
                    )
        if figures and curves:
            fig_dir = results_dir / "figures"
            out = plot_reliability_diagram(
                curves,
                fig_dir / f"reliability_analysis_{tag}_{subset}",
                title=f"{_ADVISOR_NAMES.get(tag, tag)}: {_SUBSET_NAMES.get(subset, subset)} "
                "(post-hoc replay)",
            )
            log.info("[%s] wrote %s", tag, out[0])

    summary = pd.DataFrame(summary_rows)
    folds = pd.DataFrame(fold_rows)
    summary.insert(0, "tag", tag)
    summary_path = results_dir / f"calibration_analysis_{tag}.csv"
    folds_path = results_dir / f"calibration_analysis_{tag}_folds.csv"
    summary.to_csv(summary_path, index=False)
    folds.to_csv(folds_path, index=False)
    log.info("[%s] wrote %s and %s", tag, summary_path, folds_path)
    return summary, folds


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        prog="python -m advisor.evaluation.calibration_analysis",
        description=(
            "POST-HOC calibration analysis: replay monthly walk-forward temperature and "
            "Platt scaling on a recommendation log, on all recommendations and on BUY/SELL "
            "only. Secondary to the pre-registered result in calibration_<tag>.json."
        ),
    )
    parser.add_argument("--tags", nargs="+", default=["llama"], help="Advisor tags to analyse.")
    parser.add_argument(
        "--results-dir", type=Path, default=Path("results"), help="Evaluation results folder."
    )
    parser.add_argument(
        "--min-samples", type=int, default=50, help="Labels needed before the first fit."
    )
    parser.add_argument("--n-bins", type=int, default=10, help="Bins for ECE / diagrams.")
    parser.add_argument("--no-figures", action="store_true", help="Skip reliability diagrams.")
    parser.add_argument("--quiet", action="store_true", help="Only log warnings.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point.

    Args:
        argv: Arguments (defaults to ``sys.argv[1:]``).

    Returns:
        ``0`` on success, ``1`` if any tag failed.
    """
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    failed = 0
    for tag in args.tags:
        try:
            summary, _ = run_analysis(
                tag,
                args.results_dir,
                min_samples=args.min_samples,
                n_bins=args.n_bins,
                figures=not args.no_figures,
            )
        except FileNotFoundError as exc:
            log.error("%s", exc)
            failed += 1
            continue
        cols = [
            "subset", "method", "n", "hit_rate", "ece", "adaptive_ece", "brier", "nll",
            "sharpness", "mean_confidence", "frac_below_half", "frac_crossed_half",
            "temperature", "slope", "intercept",
        ]  # fmt: skip
        with pd.option_context("display.width", 200, "display.float_format", "{:.3f}".format):
            print(f"\n[{tag}] post-hoc calibration analysis (secondary, not pre-registered)")
            print(summary[cols].to_string(index=False))
    return 1 if failed else 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
