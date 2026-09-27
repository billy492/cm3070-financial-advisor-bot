"""Platt scaling for post-hoc confidence calibration (post-hoc analysis).

This module implements :class:`PlattScaler`, the two-parameter logistic
calibrator of Platt (1999), applied - as in Guo et al. (2017), Section 4.2 - to
the *logit* of the advisor's verbal confidence rather than to a raw classifier
score. It exists to answer a question raised by the pre-registered temperature
scaling result (``docs/preregistration.md``, H2): temperature scaling is a
one-parameter *shrink-or-sharpen* map that fixes 0.5, so it can never move the
average calibrated confidence below one half. When the advisor is correct less
than half of the time - as happens under the HOLD-correctness rule of this
thesis - the best a temperature can do is push every confidence towards 0.5,
collapsing sharpness without ever reaching the empirical hit rate. Platt scaling
adds an intercept and can therefore fit a base rate below 0.5. The price is
that it is no longer action-preserving: a confidence may cross 0.5 after
calibration, so the map must be interpreted as a *confidence report*, not as a
decision rule.

Method:

1. Map the verbal confidence ``p`` to a logit ``z = log(p / (1 - p))``.
2. Rescale it affinely: ``p' = sigmoid(a * z + b)`` with slope ``a`` and
   intercept ``b``. ``a = 1, b = 0`` is the identity; ``b = 0`` recovers
   temperature scaling with ``T = 1 / a``.
3. Choose ``(a, b)`` by minimising the mean binary negative log-likelihood of
   the realised correctness labels. The NLL of a logistic model is convex in
   its parameters, so a quasi-Newton search (:func:`scipy.optimize.minimize`,
   L-BFGS-B) from the identity converges to the global optimum. A small ridge
   penalty on ``(a - 1, b)`` keeps the fit finite when the labels are
   perfectly separable, as Platt's own target-smoothing does.

References:
    Platt, J. C. (1999). Probabilistic Outputs for Support Vector Machines and
        Comparisons to Regularized Likelihood Methods. In *Advances in Large
        Margin Classifiers*, MIT Press, 61-74.
    Guo, C., Pleiss, G., Sun, Y. & Weinberger, K. Q. (2017). On Calibration of
        Modern Neural Networks. *ICML 2017*, PMLR 70, 1321-1330.
    Niculescu-Mizil, A. & Caruana, R. (2005). Predicting Good Probabilities
        with Supervised Learning. *ICML 2005*, 625-632.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from scipy.optimize import minimize

from .temperature import _as_confidence_array, _as_label_array, _logit, _sigmoid

__all__ = ["PlattScaler"]


def _nll_affine(params: np.ndarray, z: np.ndarray, y: np.ndarray, ridge: float) -> float:
    """Mean NLL of labels ``y`` under ``sigmoid(a z + b)`` plus a ridge penalty.

    Uses ``-[y log s(x) + (1-y) log(1-s(x))] = softplus(x) - y x`` so that no
    probability is ever formed explicitly and the objective stays finite.

    Args:
        params: ``(a, b)`` slope and intercept.
        z: Logits of the (clipped) raw confidences.
        y: Binary correctness labels aligned with ``z``.
        ridge: L2 penalty weight on ``(a - 1, b)``; ``0`` is the pure MLE.

    Returns:
        The penalised mean NLL as a float.
    """
    a, b = float(params[0]), float(params[1])
    x = a * z + b
    nll = float(np.mean(np.logaddexp(0.0, x) - y * x))
    return nll + ridge * ((a - 1.0) ** 2 + b**2)


def _grad_affine(params: np.ndarray, z: np.ndarray, y: np.ndarray, ridge: float) -> np.ndarray:
    """Gradient of :func:`_nll_affine` with respect to ``(a, b)``."""
    a, b = float(params[0]), float(params[1])
    resid = _sigmoid(a * z + b) - y
    grad_a = float(np.mean(resid * z)) + 2.0 * ridge * (a - 1.0)
    grad_b = float(np.mean(resid)) + 2.0 * ridge * b
    return np.asarray([grad_a, grad_b], dtype="float64")


class PlattScaler:
    """Two-parameter logistic calibrator on the confidence logit (Platt 1999).

    Learns a slope ``a`` and an intercept ``b`` on a held-out calibration split
    and applies ``p' = sigmoid(a * logit(p) + b)`` at inference time. Unlike
    :class:`~advisor.calibration.temperature.TemperatureScaler` the map can
    shift confidences across 0.5, so it can represent a correctness rate below
    one half, at the cost of the action-preserving guarantee.

    Attributes:
        slope: The learned (or pre-set) slope ``a``; ``None`` until fitted.
        intercept: The learned (or pre-set) intercept ``b``; ``None`` until
            fitted.
        ridge: L2 penalty on ``(a - 1, b)`` used during fitting.
        eps: Confidences are clipped to ``[eps, 1 - eps]`` before the logit.
        n_fit_: Number of examples the parameters were fitted on.
        nll_before_: Mean NLL of the calibration split under the identity map.
        nll_after_: Mean NLL of the calibration split under the fitted map.

    Example:
        >>> scaler = PlattScaler().fit([0.9, 0.8, 0.95, 0.7], [1, 0, 1, 0])
        >>> calibrated = scaler.transform([0.9, 0.6])
    """

    def __init__(
        self,
        slope: float | None = None,
        intercept: float | None = None,
        ridge: float = 1e-4,
        eps: float = 1e-6,
    ) -> None:
        """Initialise the calibrator, optionally with pre-set parameters.

        Args:
            slope: Optional fixed slope ``a``. Must be given together with
                ``intercept`` for :meth:`transform` to work without :meth:`fit`.
            intercept: Optional fixed intercept ``b``.
            ridge: Non-negative L2 penalty weight on ``(a - 1, b)``.
            eps: Clipping margin in ``(0, 0.5)`` applied to confidences before
                the logit transform.

        Raises:
            ValueError: If ``ridge`` is negative, ``eps`` is out of range, or
                only one of ``slope`` / ``intercept`` is given.
        """
        if ridge < 0.0:
            raise ValueError(f"ridge must be non-negative; got {ridge}.")
        if not (0.0 < eps < 0.5):
            raise ValueError(f"eps must lie in (0, 0.5); got {eps}.")
        if (slope is None) != (intercept is None):
            raise ValueError("slope and intercept must be given together (or neither).")
        if slope is not None and not (np.isfinite(slope) and np.isfinite(intercept)):
            raise ValueError("slope and intercept must be finite.")

        self.slope: float | None = None if slope is None else float(slope)
        self.intercept: float | None = None if intercept is None else float(intercept)
        self.ridge = float(ridge)
        self.eps = float(eps)
        self.n_fit_: int | None = None
        self.nll_before_: float | None = None
        self.nll_after_: float | None = None

    def __repr__(self) -> str:
        """Return a compact, informative representation."""
        if self.slope is None:
            return f"PlattScaler(unfitted, n_fit={self.n_fit_})"
        return (
            f"PlattScaler(slope={self.slope:.4f}, intercept={self.intercept:.4f}, "
            f"n_fit={self.n_fit_})"
        )

    # -- fitting -------------------------------------------------------------
    def fit(
        self,
        confidences: Sequence[float] | np.ndarray,
        labels: Sequence[int] | np.ndarray,
    ) -> PlattScaler:
        """Learn ``(a, b)`` from a held-out calibration split.

        Minimises the (lightly ridge-penalised) mean negative log-likelihood of
        ``labels`` under ``sigmoid(a * logit(p) + b)`` with L-BFGS-B, starting
        from the identity ``(1, 0)``. The objective is convex, so the solution
        is the global optimum.

        Args:
            confidences: Raw verbal confidences in ``[0, 1]``, one per example.
            labels: Binary correctness labels aligned with ``confidences``.

        Returns:
            ``self`` (fitted), to allow chaining.

        Raises:
            ValueError: If the inputs are empty, of different length, contain
                non-finite or out-of-range confidences, contain labels other
                than 0/1, or contain fewer than two distinct labels.
            RuntimeError: If the optimiser fails to converge.
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
        result = minimize(
            _nll_affine,
            x0=np.asarray([1.0, 0.0], dtype="float64"),
            args=(z, y, self.ridge),
            jac=_grad_affine,
            method="L-BFGS-B",
            options={"maxiter": 1000, "gtol": 1e-9},
        )
        if not result.success and not np.all(np.isfinite(result.x)):
            raise RuntimeError(f"Platt scaling failed to converge: {result.message}")

        self.slope, self.intercept = float(result.x[0]), float(result.x[1])
        self.n_fit_ = int(p.size)
        self.nll_before_ = _nll_affine(np.asarray([1.0, 0.0]), z, y, 0.0)
        self.nll_after_ = _nll_affine(result.x, z, y, 0.0)
        return self

    # -- inference -----------------------------------------------------------
    def transform(self, confidences: Sequence[float] | np.ndarray) -> list[float]:
        """Apply the learned map ``sigmoid(a * logit(p) + b)``.

        Args:
            confidences: Raw confidences in ``[0, 1]``.

        Returns:
            Calibrated confidences in ``(0, 1)``, one per input, as plain floats.
            The map is monotone when ``a > 0`` but does *not* fix 0.5, so an
            input above one half may come out below it.

        Raises:
            RuntimeError: If the scaler is unfitted.
            ValueError: If a confidence is non-finite or outside ``[0, 1]``.
        """
        if self.slope is None or self.intercept is None:
            raise RuntimeError("PlattScaler is not fitted; call fit() first.")
        p = _as_confidence_array(confidences, self.eps)
        if p.size == 0:
            return []
        calibrated = _sigmoid(self.slope * _logit(p) + self.intercept)
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
            In-sample calibrated confidences of the fitting examples.
        """
        return self.fit(confidences, labels).transform(confidences)

    # -- persistence ---------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        """Serialise the calibrator to a JSON-safe dictionary.

        Returns:
            A dict with the slope, intercept, ridge, clipping margin and the fit
            diagnostics (``n_fit_``, ``nll_before_``, ``nll_after_``).
        """
        return {
            "slope": self.slope,
            "intercept": self.intercept,
            "ridge": self.ridge,
            "eps": self.eps,
            "n_fit_": self.n_fit_,
            "nll_before_": self.nll_before_,
            "nll_after_": self.nll_after_,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> PlattScaler:
        """Restore a calibrator from :meth:`to_dict` output.

        Args:
            payload: Mapping produced by :meth:`to_dict`.

        Returns:
            A :class:`PlattScaler` ready for :meth:`transform`.

        Raises:
            ValueError: If the stored parameters are invalid.
        """
        scaler = cls(
            slope=payload.get("slope"),
            intercept=payload.get("intercept"),
            ridge=float(payload.get("ridge", 1e-4)),
            eps=float(payload.get("eps", 1e-6)),
        )
        n_fit = payload.get("n_fit_")
        scaler.n_fit_ = None if n_fit is None else int(n_fit)
        for key in ("nll_before_", "nll_after_"):
            value = payload.get(key)
            setattr(scaler, key, None if value is None else float(value))
        return scaler
