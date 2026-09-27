"""Counterfactual layer: contrastive 'I would change my mind if ...' reasons.

Generates counterfactual explanations (DiCE, Mothilal et al. 2020; framed after
Wachter et al. 2018) describing the minimal feasible feature change that would
flip the advisor's recommendation, verified against the black-box model.
"""

from __future__ import annotations

from .dice import (
    DEFAULT_FEATURE_SPECS,
    NO_COUNTERFACTUAL_SENTENCE,
    Counterfactual,
    FeatureSpec,
    apply_counterfactual,
    counterfactual_validity,
    feature_specs_from_data,
    generate_counterfactual,
    generate_counterfactuals,
)

__all__ = [
    "Counterfactual",
    "FeatureSpec",
    "DEFAULT_FEATURE_SPECS",
    "NO_COUNTERFACTUAL_SENTENCE",
    "apply_counterfactual",
    "counterfactual_validity",
    "feature_specs_from_data",
    "generate_counterfactual",
    "generate_counterfactuals",
]
