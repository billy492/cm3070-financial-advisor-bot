"""Calibration layer: post-hoc confidence calibration.

Implements temperature scaling (Guo et al. 2017) to map the LLM's raw verbal
confidence onto well-calibrated probabilities, evaluated via Expected
Calibration Error (Naeini et al. 2015) and reliability diagrams. Platt scaling
(Platt 1999) is provided for the post-hoc analysis of the pre-registered
result: it adds an intercept so a correctness rate below one half can be fitted.
"""

from __future__ import annotations

from .platt import PlattScaler
from .temperature import (
    TemperatureScaler,
    plot_reliability_diagram,
    reliability_curve,
)

__all__ = [
    "TemperatureScaler",
    "PlattScaler",
    "reliability_curve",
    "plot_reliability_diagram",
]
