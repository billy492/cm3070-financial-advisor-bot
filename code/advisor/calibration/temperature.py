"""Temperature scaling for post-hoc confidence calibration.

This module implements :class:`TemperatureScaler`, the single-parameter post-hoc
calibration method of Guo et al. (2017), adapted to the *binary* setting used in
this thesis: the advisor LLM emits a verbal confidence ``p`` in ``[0, 1]`` that
its BUY/HOLD/SELL action is correct, and a held-out set of realised correctness
labels tells us how often it actually was.

Method (Chapter 4 of the preliminary report, ADR-0002):

1. Treat the verbal confidence ``p`` as a probability and map it to a logit
   ``z = log(p / (1 - p))``.
2. Rescale it as ``p' = sigmoid(z / T)`` for a single positive temperature ``T``.
   ``T > 1`` softens (reduces) confidence, ``T < 1`` sharpens it, ``T = 1`` is
   the identity.
3. Find ``T`` by minimising the negative log-likelihood (NLL) of the realised
   correctness labels. The objective is *reparameterised in the inverse
   temperature* ``beta = 1 / T``: ``NLL(beta) = mean(softplus(beta * z) - y *
   beta * z)``, which is convex in ``beta`` (softplus composed with a linear map),
   so a bounded one-dimensional search (:func:`scipy.optimize.minimize_scalar`,
   ``method="bounded"``, a golden-section/Brent scheme) finds the global optimum.
4. Because the sigmoid crosses one-half exactly where the logit crosses zero,
   scaling never changes which side of 0.5 a confidence lies on, nor the ranking
   of confidences: the recommended action is untouched, only how confident the
   bot *sounds* is adjusted. This is the accuracy-preserving property of
   temperature scaling stressed by Guo et al. (2017).

Calibration quality is measured with the Expected Calibration Error of Naeini,
Cooper & Hauskrecht (2015) (``advisor.evaluation.metrics``) and visualised with
reliability diagrams (:func:`reliability_curve`, :func:`plot_reliability_diagram`).

References:
    Guo, C., Pleiss, G., Sun, Y. & Weinberger, K. Q. (2017). On Calibration of
        Modern Neural Networks. *ICML 2017*, PMLR 70, 1321-1330.
    Naeini, M. P., Cooper, G. F. & Hauskrecht, M. (2015). Obtaining Well
        Calibrated Probabilities Using Bayesian Binning. *AAAI 2015*.
    Nixon, J., Dusenberry, M., Zhang, L., Jerfel, G. & Tran, D. (2019). Measuring
        Calibration in Deep Learning. *CVPR Workshops* (adaptive / equal-mass bins).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import minimize_scalar

__all__ = [
    "TemperatureScaler",
    "reliability_curve",
    "plot_reliability_diagram",
]

_STRATEGIES: tuple[str, ...] = ("uniform", "quantile")


# --------------------------------------------------------------------------- #
# Numerically stable primitives
# --------------------------------------------------------------------------- #
def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Logistic function ``1 / (1 + exp(-x))`` evaluated without overflow."""
    return np.exp(-np.logaddexp(0.0, -x))


def _logit(p: np.ndarray) -> np.ndarray:
    """Inverse logistic function ``log(p / (1 - p))`` for ``p`` in ``(0, 1)``."""
    return np.log(p) - np.log1p(-p)


def _nll(beta: float, z: np.ndarray, y: np.ndarray) -> float:
    """Mean binary negative log-likelihood of labels ``y`` under ``sigmoid(beta * z)``.

    Uses the identity ``-[y log s(x) + (1-y) log(1-s(x))] = softplus(x) - y x`` with
    ``softplus(x) = logaddexp(0, x)`` so no probability is ever formed explicitly.

    Args:
        beta: Inverse temperature ``1 / T``.
        z: Logits of the (clipped) raw confidences.
        y: Binary correctness labels aligned with ``z``.

    Returns:
        The mean NLL as a float; convex in ``beta``.
    """
    x = beta * z
    return float(np.mean(np.logaddexp(0.0, x) - y * x))


def _as_confidence_array(confidences: Sequence[float] | np.ndarray, eps: float) -> np.ndarray:
    """Validate confidences and clip them into ``[eps, 1 - eps]``.

    Args:
        confidences: Raw probabilities, nominally in ``[0, 1]``.
        eps: Clipping margin keeping logits finite.

    Returns:
        A 1-D ``float64`` array clipped to ``[eps, 1 - eps]``.

    Raises:
        ValueError: If any value is non-finite or lies outside ``[0, 1]`` (beyond
            a tiny floating-point tolerance).
    """
    p = np.asarray(confidences, dtype="float64").ravel()
    if p.size and not np.all(np.isfinite(p)):
        raise ValueError("confidences must be finite numbers.")
    if p.size and (np.any(p < -1e-9) or np.any(p > 1.0 + 1e-9)):
        raise ValueError(
            f"confidences must lie in [0, 1]; got min={p.min():.4g}, max={p.max():.4g}."
        )
    return np.clip(p, eps, 1.0 - eps)


def _as_label_array(labels: Sequence[int] | np.ndarray) -> np.ndarray:
    """Validate binary labels and return them as a ``float64`` array of 0/1.

    Args:
        labels: Correctness labels; ints, floats or bools.

    Returns:
        A 1-D ``float64`` array containing only ``0.0`` and ``1.0``.

    Raises:
        ValueError: If any label is not exactly 0 or 1 (booleans are accepted).
    """
    y = np.asarray(labels, dtype="float64").ravel()
    if y.size and not np.all(np.isin(y, (0.0, 1.0))):
        raise ValueError("labels must be binary (0/1 or False/True).")
    return y


# --------------------------------------------------------------------------- #
# The calibrator
# --------------------------------------------------------------------------- #
class TemperatureScaler:
    """Single-parameter temperature-scaling calibrator (Guo et al. 2017).

    Learns one positive scalar ``T`` on a held-out calibration split and applies
    ``p' = sigmoid(logit(p) / T)`` at inference time. The transform is
    order-preserving and leaves the ``p > 0.5`` decision untouched, so the
    advisor's action is never changed by calibration.

    Attributes:
        temperature: The learned (or pre-set) temperature ``T``; ``None`` until
            :meth:`fit` is called or a value is passed to the constructor.
            Values ``> 1`` soften confidence, values ``< 1`` sharpen it.
        t_min: Lower bound of the temperature search interval.
        t_max: Upper bound of the temperature search interval.
        eps: Confidences are clipped to ``[eps, 1 - eps]`` before the logit so
            the endpoints 0 and 1 stay finite.
        n_fit_: Number of examples the temperature was fitted on (``None`` if
            the temperature was set by hand).
        nll_before_: Mean NLL of the calibration split at ``T = 1`` (i.e. of the
            raw confidences) - the "before" score.
        nll_after_: Mean NLL of the calibration split at the fitted ``T`` - the
            "after" score. Not larger than ``nll_before_`` whenever ``T = 1`` lies
            inside ``[t_min, t_max]`` (the default), up to solver tolerance.

    Example:
        >>> scaler = TemperatureScaler().fit([0.9, 0.8, 0.95, 0.7], [1, 0, 1, 0])
        >>> calibrated = scaler.transform([0.9, 0.6])
    """

    def __init__(
        self,
        temperature: float | None = None,
        t_min: float = 0.05,
        t_max: float = 20.0,
        eps: float = 1e-6,
    ) -> None:
        """Initialise the calibrator, optionally with a pre-set temperature.

        Args:
            temperature: Optional fixed temperature. When given, :meth:`transform`
                works immediately without :meth:`fit`.
            t_min: Smallest temperature the optimiser may return (``> 0``).
            t_max: Largest temperature the optimiser may return (``> t_min``).
            eps: Clipping margin in ``(0, 0.5)`` applied to confidences before the
                logit transform.

        Raises:
            ValueError: If the bounds, ``eps`` or a pre-set ``temperature`` are
                not positive / not ordered.
        """
        if not (0.0 < t_min < t_max):
            raise ValueError(f"Require 0 < t_min < t_max; got t_min={t_min}, t_max={t_max}.")
        if not (0.0 < eps < 0.5):
            raise ValueError(f"eps must lie in (0, 0.5); got {eps}.")
        if temperature is not None and not (float(temperature) > 0.0):
            raise ValueError(f"temperature must be positive; got {temperature}.")

        self.temperature: float | None = None if temperature is None else float(temperature)
        self.t_min = float(t_min)
        self.t_max = float(t_max)
        self.eps = float(eps)
        self.n_fit_: int | None = None
        self.nll_before_: float | None = None
        self.nll_after_: float | None = None

    def __repr__(self) -> str:
        """Return a compact, informative representation."""
        t = "unfitted" if self.temperature is None else f"{self.temperature:.4f}"
        return f"TemperatureScaler(temperature={t}, n_fit={self.n_fit_})"

    # -- fitting -------------------------------------------------------------
    def fit(
        self,
        confidences: Sequence[float] | np.ndarray,
        labels: Sequence[int] | np.ndarray,
    ) -> TemperatureScaler:
        """Learn the temperature ``T`` from a held-out calibration split.

        Minimises the mean negative log-likelihood of ``labels`` under
        ``sigmoid(logit(p) / T)`` over the inverse temperature
        ``beta = 1 / T`` in ``[1 / t_max, 1 / t_min]`` using a bounded scalar
        search (:func:`scipy.optimize.minimize_scalar`, ``method="bounded"``).
        The objective is convex in ``beta``, so the bounded search converges to
        the global optimum (Guo et al. 2017, Section 4.2, restricted to one
        binary "correct / incorrect" outcome per recommendation).

        Args:
            confidences: Raw verbal confidences in ``[0, 1]``, one per example.
                Clipped to ``[eps, 1 - eps]`` before taking logits.
            labels: Binary correctness labels (1 = the action was correct)
                aligned with ``confidences``.

        Returns:
            ``self`` (fitted), to allow chaining.

        Raises:
            ValueError: If the inputs are empty, of different length, contain
                non-finite or out-of-range confidences, contain labels other than
                0/1, or contain fewer than two distinct labels (the NLL would then
                be minimised by an unbounded / degenerate temperature).
        """
        p = _as_confidence_array(confidences, self.eps)
        y = _as_label_array(labels)
        if p.size == 0 or y.size == 0:
            raise ValueError("fit requires at least one (confidence, label) pair.")
        if p.size != y.size:
            raise ValueError(
                f"confidences and labels must have the same length; got {p.size} and {y.size}."
            )
        if np.unique(y).size < 2:
            raise ValueError("labels must contain both classes (at least one 0 and one 1).")

        z = _logit(p)
        beta_lo, beta_hi = 1.0 / self.t_max, 1.0 / self.t_min
        result = minimize_scalar(
            _nll,
            bounds=(beta_lo, beta_hi),
            args=(z, y),
            method="bounded",
            options={"xatol": 1e-9, "maxiter": 500},
        )
        beta = float(np.clip(result.x, beta_lo, beta_hi))

        self.temperature = 1.0 / beta
        self.n_fit_ = int(p.size)
        self.nll_before_ = _nll(1.0, z, y)
        self.nll_after_ = _nll(beta, z, y)
        return self

    # -- inference -----------------------------------------------------------
    def transform(self, confidences: Sequence[float] | np.ndarray) -> list[float]:
        """Apply the learned temperature: ``sigmoid(logit(p) / T)``.

        Args:
            confidences: Raw confidences in ``[0, 1]``.

        Returns:
            Calibrated confidences in ``(0, 1)``, one per input, as plain floats.
            The mapping is strictly increasing and fixes 0.5, so ``p > 0.5`` iff
            the calibrated value ``> 0.5``: the recommended action is preserved.

        Raises:
            RuntimeError: If no temperature is available (call :meth:`fit`
                first, or construct with ``temperature=...``).
            ValueError: If a confidence is non-finite or outside ``[0, 1]``.
        """
        if self.temperature is None:
            raise RuntimeError("TemperatureScaler is not fitted; call fit() first.")
        p = _as_confidence_array(confidences, self.eps)
        if p.size == 0:
            return []
        calibrated = _sigmoid(_logit(p) / self.temperature)
        return [float(v) for v in calibrated]

    def fit_transform(
        self,
        confidences: Sequence[float] | np.ndarray,
        labels: Sequence[int] | np.ndarray,
    ) -> list[float]:
        """Fit on ``(confidences, labels)`` and return the calibrated confidences.

        Args:
            confidences: Raw confidences in ``[0, 1]``.
            labels: Binary correctness labels aligned with ``confidences``.

        Returns:
            The calibrated confidences of the *same* examples used for fitting
            (in-sample; use a held-out split for reported metrics).
        """
        return self.fit(confidences, labels).transform(confidences)

    # -- persistence ---------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        """Serialise the calibrator to a JSON-safe dictionary.

        Returns:
            A dict with the temperature, search bounds, clipping margin and the
            fit diagnostics (``n_fit_``, ``nll_before_``, ``nll_after_``).
        """
        return {
            "temperature": self.temperature,
            "t_min": self.t_min,
            "t_max": self.t_max,
            "eps": self.eps,
            "n_fit_": self.n_fit_,
            "nll_before_": self.nll_before_,
            "nll_after_": self.nll_after_,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> TemperatureScaler:
        """Restore a calibrator from :meth:`to_dict` output.

        Args:
            payload: Mapping produced by :meth:`to_dict` (e.g. loaded from JSON).

        Returns:
            A :class:`TemperatureScaler` with the stored temperature and
            diagnostics, ready for :meth:`transform`.

        Raises:
            ValueError: If the stored bounds / temperature are invalid.
        """
        scaler = cls(
            temperature=payload.get("temperature"),
            t_min=float(payload.get("t_min", 0.05)),
            t_max=float(payload.get("t_max", 20.0)),
            eps=float(payload.get("eps", 1e-6)),
        )
        n_fit = payload.get("n_fit_")
        scaler.n_fit_ = None if n_fit is None else int(n_fit)
        for key in ("nll_before_", "nll_after_"):
            value = payload.get(key)
            setattr(scaler, key, None if value is None else float(value))
        return scaler


# --------------------------------------------------------------------------- #
# Reliability diagrams
# --------------------------------------------------------------------------- #
def reliability_curve(
    confidences: Sequence[float] | np.ndarray,
    labels: Sequence[int] | np.ndarray,
    n_bins: int = 10,
    strategy: str = "uniform",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-bin statistics for a reliability diagram (Naeini et al. 2015).

    Predictions are grouped into ``n_bins`` bins and, for every *non-empty* bin,
    the mean confidence, the empirical accuracy and the number of predictions
    are returned. Plotting ``mean_acc`` against ``mean_conf`` gives the
    reliability diagram of Guo et al. (2017); the count-weighted mean of
    ``|mean_acc - mean_conf|`` is the Expected Calibration Error.

    Args:
        confidences: Predicted confidences in ``[0, 1]``.
        labels: Binary correctness labels aligned with ``confidences``.
        n_bins: Number of bins (``>= 1``).
        strategy: ``"uniform"`` for equal-width bins over ``[0, 1]`` (the same
            binning as ``expected_calibration_error``), or ``"quantile"`` for
            equal-mass bins (sorted confidences split into ``n_bins`` contiguous
            groups, as in the adaptive ECE of Nixon et al. 2019).

    Returns:
        A tuple ``(mean_conf, mean_acc, count)`` of equal-length 1-D arrays, one
        entry per non-empty bin in increasing confidence order. ``count`` sums
        to the number of valid observations. All three are empty when there are
        no valid observations.

    Raises:
        ValueError: If ``strategy`` is unknown, ``n_bins < 1``, or the inputs
            have different lengths.
    """
    if strategy not in _STRATEGIES:
        raise ValueError(f"strategy must be one of {_STRATEGIES}; got {strategy!r}.")
    if n_bins < 1:
        raise ValueError(f"n_bins must be >= 1; got {n_bins}.")

    conf = np.asarray(confidences, dtype="float64").ravel()
    corr = np.asarray(labels, dtype="float64").ravel()
    if conf.size != corr.size:
        raise ValueError(
            f"confidences and labels must have the same length; got {conf.size} and {corr.size}."
        )
    keep = np.isfinite(conf) & np.isfinite(corr)
    conf, corr = np.clip(conf[keep], 0.0, 1.0), corr[keep]
    empty = np.empty(0, dtype="float64")
    if conf.size == 0:
        return empty, empty, empty.astype("int64")

    if strategy == "uniform":
        edges = np.linspace(0.0, 1.0, n_bins + 1)
        bin_ids = np.digitize(conf, edges[1:-1], right=False)
        groups = [np.flatnonzero(bin_ids == b) for b in range(n_bins)]
    else:
        order = np.argsort(conf, kind="stable")
        groups = np.array_split(order, n_bins)

    mean_conf, mean_acc, count = [], [], []
    for idx in groups:
        if idx.size == 0:
            continue
        mean_conf.append(float(np.mean(conf[idx])))
        mean_acc.append(float(np.mean(corr[idx])))
        count.append(int(idx.size))
    return (
        np.asarray(mean_conf, dtype="float64"),
        np.asarray(mean_acc, dtype="float64"),
        np.asarray(count, dtype="int64"),
    )


def plot_reliability_diagram(
    curves: Mapping[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    path: str | Path,
    *,
    title: str = "Reliability diagram",
) -> tuple[Path, Path]:
    """Save a reliability diagram as PNG and SVG (matplotlib, Agg backend).

    Each entry of ``curves`` is drawn as a line with markers whose sizes grow
    with the bin count, coloured from the perceptually uniform, colour-blind-safe
    ``viridis`` palette; the dashed diagonal marks perfect calibration. The
    figure is rendered with an explicit Agg canvas, so no GUI backend is touched
    and the function is safe on headless machines and inside Streamlit.

    Args:
        curves: Mapping from legend label (e.g. ``"raw"``, ``"calibrated"``) to a
            ``(mean_conf, mean_acc, count)`` tuple from :func:`reliability_curve`.
        path: Output location. The suffix (``.png`` / ``.svg``) is ignored and
            both ``<stem>.png`` and ``<stem>.svg`` are written; parent folders
            are created.
        title: Axes title.

    Returns:
        ``(png_path, svg_path)`` of the written files.

    Raises:
        ValueError: If ``curves`` is empty.
    """
    if not curves:
        raise ValueError("plot_reliability_diagram needs at least one curve.")

    import matplotlib
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    base = Path(path)
    if base.suffix.lower() in {".png", ".svg"}:
        base = base.with_suffix("")
    base.parent.mkdir(parents=True, exist_ok=True)
    png_path, svg_path = base.with_suffix(".png"), base.with_suffix(".svg")

    fig = Figure(figsize=(5.5, 5.0), dpi=150)
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(1, 1, 1)
    ax.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", color="0.45", label="Perfect calibration")

    cmap = matplotlib.colormaps["viridis"]
    shades = np.linspace(0.1, 0.85, max(len(curves), 2))
    for shade, (label, (mean_conf, mean_acc, count)) in zip(shades, curves.items(), strict=False):
        mean_conf = np.asarray(mean_conf, dtype="float64")
        mean_acc = np.asarray(mean_acc, dtype="float64")
        count = np.asarray(count, dtype="float64")
        total = float(count.sum()) if count.size else 0.0
        sizes = 20.0 + 180.0 * (count / total) if total > 0 else np.full(mean_conf.shape, 40.0)
        colour = cmap(float(shade))
        ax.plot(mean_conf, mean_acc, color=colour, linewidth=1.5, zorder=2)
        ax.scatter(mean_conf, mean_acc, s=sizes, color=colour, label=label, zorder=3)

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Mean stated confidence")
    ax.set_ylabel("Empirical accuracy")
    ax.set_title(title)
    ax.grid(True, linewidth=0.4, alpha=0.5)
    ax.legend(loc="upper left", frameon=False)
    ax.set_aspect("equal", adjustable="box")

    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    return png_path, svg_path
