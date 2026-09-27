"""Tests for ``advisor.counterfactual.dice``.

The black-box advisor is stood in for by small deterministic rules over the
engineered features, so every expected counterfactual can be derived by hand:
the default rule says BUY when ``rsi_14 < 70`` and ``mom_10 > 0``, SELL when
``rsi_14 > 70``, otherwise HOLD. From the BUY instance the search must find the
momentum flip (``mom_10 -> 0``) and the RSI flip (``rsi_14 -> 70``), render
them in plain English, respect the call budget, prefer different features,
pass the validity re-check, and degrade gracefully when nothing flips.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import date
from typing import Any

import numpy as np
import pandas as pd
import pytest

from advisor.counterfactual import (
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
from advisor.counterfactual.dice import (
    _CachedPredictor,
    apply_change,
    current_value,
    format_integer,
    format_percent,
)
from advisor.recommender.schema import Action, Recommendation


class CountingRule:
    """Deterministic advisor rule that counts how often it is queried."""

    def __init__(self) -> None:
        """Start with an empty call counter."""
        self.calls = 0

    def __call__(self, features: Mapping[str, Any]) -> str:
        """Return BUY / HOLD / SELL for ``features`` and count the call."""
        self.calls += 1
        if features["rsi_14"] > 70.0:
            return "SELL"
        if features["rsi_14"] < 70.0 and features["mom_10"] > 0.0:
            return "BUY"
        return "HOLD"


@pytest.fixture
def rule() -> CountingRule:
    """A fresh counting rule predictor."""
    return CountingRule()


@pytest.fixture
def buy_instance() -> dict[str, float]:
    """Feature snapshot the rule classifies as BUY (RSI 57.3, momentum +3.4 %)."""
    return {
        "close": 101.5,
        "sma_10": 100.1,
        "sma_50": 98.2,
        "ret_1d": 0.012,
        "mom_10": 0.034,
        "vol_20": 0.018,
        "rsi_14": 57.3,
    }


# --------------------------------------------------------------------------- #
# Single-feature search
# --------------------------------------------------------------------------- #
def test_finds_momentum_and_rsi_flips(rule: CountingRule, buy_instance: dict) -> None:
    """Both hand-derived single-feature flips are found with the right values and actions."""
    found = generate_counterfactuals(rule, buy_instance, total_cfs=3, max_calls=200)
    by_feature = {cf.changed_features: cf for cf in found}
    assert set(by_feature) == {("mom_10",), ("rsi_14",)}
    assert by_feature[("mom_10",)].new_values == (0.0,)
    assert by_feature[("rsi_14",)].new_values == (70.0,)
    for cf in found:
        assert cf.original_action == "BUY"
        assert cf.new_action == "HOLD"
        assert cf.original_values == (buy_instance[cf.changed_features[0]],)


def test_sentences_are_plain_english(rule: CountingRule, buy_instance: dict) -> None:
    """Sentences use human labels, integer RSI and percentage momentum."""
    sentences = {
        cf.changed_features[0]: cf.sentence
        for cf in generate_counterfactuals(rule, buy_instance, max_calls=200)
    }
    assert sentences["mom_10"] == (
        "I would change my mind from BUY to HOLD if the 10-day momentum fell below 0%."
    )
    assert sentences["rsi_14"] == (
        "I would change my mind from BUY to HOLD if the 14-day RSI rose above 70."
    )


def test_results_are_sorted_by_proximity(rule: CountingRule, buy_instance: dict) -> None:
    """The closest change comes first; proximity is the range-normalised L1 distance."""
    found = generate_counterfactuals(rule, buy_instance, max_calls=200)
    assert [cf.changed_features[0] for cf in found] == ["mom_10", "rsi_14"]
    assert found[0].proximity == pytest.approx(0.034 / 0.60)
    assert found[1].proximity == pytest.approx((70.0 - 57.3) / 100.0)
    assert generate_counterfactual(rule, buy_instance) == found[0].sentence


def test_respects_max_calls(rule: CountingRule, buy_instance: dict) -> None:
    """The predictor is never queried more than ``max_calls`` times, whatever is found."""
    found = generate_counterfactuals(rule, buy_instance, max_calls=7)
    assert rule.calls <= 7
    assert found == []
    rule.calls = 0
    assert generate_counterfactuals(rule, buy_instance, max_calls=1) == []
    assert rule.calls == 1  # only the original action was established


def test_diversity_prefers_different_features() -> None:
    """With two RSI flips closer than the momentum flip, the second pick is still momentum."""

    def band(features: Mapping[str, Any]) -> str:
        inside = 55.0 <= features["rsi_14"] <= 60.0 and features["mom_10"] > -0.10
        return "HOLD" if inside else "SELL"

    instance = {"rsi_14": 57.3, "mom_10": 0.034}
    two = generate_counterfactuals(band, instance, total_cfs=2, max_calls=200)
    assert {cf.changed_features for cf in two} == {("rsi_14",), ("mom_10",)}
    three = generate_counterfactuals(band, instance, total_cfs=3, max_calls=200)
    assert [cf.changed_features[0] for cf in three] == ["rsi_14", "rsi_14", "mom_10"]
    assert three[0].proximity <= three[1].proximity <= three[2].proximity


# --------------------------------------------------------------------------- #
# Pairwise search, validity, graceful failure
# --------------------------------------------------------------------------- #
def test_pairwise_stage_finds_joint_change() -> None:
    """When no single feature flips the action, the two most sensitive ones move together."""

    def both(features: Mapping[str, Any]) -> str:
        return "BUY" if features["rsi_14"] < 70.0 and features["mom_10"] > 0.0 else "HOLD"

    instance = {"rsi_14": 75.0, "mom_10": -0.01}
    found = generate_counterfactuals(
        both, instance, specs=DEFAULT_FEATURE_SPECS[:2], total_cfs=2, max_calls=500
    )
    assert len(found) == 1
    cf = found[0]
    assert cf.changed_features == ("rsi_14", "mom_10")
    assert cf.new_values == (65.0, 0.02)
    assert cf.proximity == pytest.approx(10.0 / 100.0 + 0.03 / 0.60)
    assert cf.sentence == (
        "I would change my mind from HOLD to BUY if the 14-day RSI fell below 65 "
        "and the 10-day momentum rose above 2%."
    )
    assert counterfactual_validity(both, instance, cf, strict=True)


def test_validity_check(rule: CountingRule, buy_instance: dict) -> None:
    """Every generated counterfactual re-validates; a fabricated one does not."""
    for cf in generate_counterfactuals(rule, buy_instance, max_calls=200):
        assert counterfactual_validity(rule, buy_instance, cf)
        assert counterfactual_validity(rule, buy_instance, cf, strict=True)
    bogus = Counterfactual(("rsi_14",), (57.3,), (60.0,), "BUY", "HOLD", 0.027, "bogus")
    assert not counterfactual_validity(rule, buy_instance, bogus)


def test_no_flip_is_handled_gracefully(buy_instance: dict) -> None:
    """A model that never changes its mind yields no counterfactuals and a stock sentence."""
    assert generate_counterfactuals(lambda _f: "HOLD", buy_instance, max_calls=300) == []
    assert generate_counterfactual(lambda _f: "HOLD", buy_instance) == NO_COUNTERFACTUAL_SENTENCE


# --------------------------------------------------------------------------- #
# Model adapters and argument validation
# --------------------------------------------------------------------------- #
def test_predict_method_adapter_accepts_lowercase_actions(
    rule: CountingRule, buy_instance: dict
) -> None:
    """An object with ``.predict`` works and case-insensitive actions are normalised."""

    class Model:
        def predict(self, features: Mapping[str, Any]) -> str:
            return rule(features).lower()

    assert generate_counterfactual(Model(), buy_instance).startswith("I would change my mind")


def test_recommend_adapter_uses_ticker_and_as_of(rule: CountingRule, buy_instance: dict) -> None:
    """A thesis advisor exposing ``.recommend`` is probed through ``Recommendation.action``."""
    seen: list[tuple[str, date]] = []

    class Advisor:
        def recommend(self, ticker: str, features: Mapping[str, Any], *, as_of: date):
            seen.append((ticker, as_of))
            return Recommendation(
                ticker, Action(rule(features)), 0.6, 0.6, "r", "c", as_of, dict(features)
            )

    sentence = generate_counterfactual(
        Advisor(), buy_instance, ticker="AAA", as_of=date(2025, 3, 14), max_calls=200
    )
    assert sentence.startswith("I would change my mind from BUY to HOLD")
    assert seen and all(item == ("AAA", date(2025, 3, 14)) for item in seen)
    with pytest.raises(ValueError):
        generate_counterfactual(Advisor(), buy_instance)


def test_invalid_predictor_output_and_arguments(buy_instance: dict) -> None:
    """Unknown actions, bad budgets/counts, non-mapping instances and odd models raise."""
    with pytest.raises(ValueError):
        generate_counterfactuals(lambda _f: "MAYBE", buy_instance)
    with pytest.raises(ValueError):
        generate_counterfactuals(lambda _f: "BUY", buy_instance, total_cfs=0)
    with pytest.raises(ValueError):
        generate_counterfactuals(lambda _f: "BUY", buy_instance, max_calls=0)
    with pytest.raises(TypeError):
        generate_counterfactuals(lambda _f: "BUY", [1, 2, 3])  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        generate_counterfactual(object(), buy_instance)


def test_cache_avoids_repeat_calls(rule: CountingRule, buy_instance: dict) -> None:
    """Equal feature dicts (even in different key order) hit the cache."""
    cached = _CachedPredictor(rule, max_calls=5)
    assert cached(buy_instance) == "BUY"
    assert cached(dict(reversed(list(buy_instance.items())))) == "BUY"
    assert rule.calls == 1
    assert cached.calls == 1


# --------------------------------------------------------------------------- #
# Derived feature (sma_gap), data-driven ranges, formatting
# --------------------------------------------------------------------------- #
def test_sma_gap_is_derived_and_rewrites_close(buy_instance: dict) -> None:
    """``sma_gap`` is read as ``close / sma_50 - 1`` and written back through ``close``."""
    assert current_value(buy_instance, "sma_gap") == pytest.approx(101.5 / 98.2 - 1.0)
    moved = apply_change(buy_instance, "sma_gap", -0.10)
    assert moved["close"] == pytest.approx(98.2 * 0.90)
    assert moved["sma_50"] == 98.2
    assert "sma_gap" not in moved  # no key that the model has not seen before
    assert buy_instance["close"] == 101.5  # input untouched

    def trend(features: Mapping[str, Any]) -> str:
        return "BUY" if features["close"] > features["sma_50"] else "SELL"

    found = generate_counterfactuals(trend, buy_instance, total_cfs=1, max_calls=200)
    assert found[0].changed_features == ("sma_gap",)
    assert found[0].sentence == (
        "I would change my mind from BUY to SELL if the price fell back to its 50-day average."
    )
    assert apply_counterfactual(buy_instance, found[0])["close"] == pytest.approx(98.2)
    assert counterfactual_validity(trend, buy_instance, found[0])


def test_missing_features_are_skipped() -> None:
    """Specs whose feature is absent (and not derivable) are ignored, not errors."""

    def rsi_only(features: Mapping[str, Any]) -> str:
        return "SELL" if features["rsi_14"] > 70.0 else "HOLD"

    found = generate_counterfactuals(rsi_only, {"rsi_14": 62.0}, max_calls=100)
    assert [cf.changed_features for cf in found] == [("rsi_14",)]
    assert found[0].new_values == (75.0,)


def test_feature_specs_from_data_uses_quantiles_rounded_to_grid() -> None:
    """Ranges follow data quantiles rounded outward to the step; absent columns keep defaults."""
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(
        {
            "rsi_14": rng.uniform(20.0, 80.0, size=5000),
            "close": rng.uniform(95.0, 105.0, size=5000),
            "sma_50": np.full(5000, 100.0),
        }
    )
    specs = {spec.name: spec for spec in feature_specs_from_data(frame, q=(0.01, 0.99))}
    assert 15.0 <= specs["rsi_14"].low <= 25.0 and specs["rsi_14"].low % 5.0 == 0.0
    assert 75.0 <= specs["rsi_14"].high <= 85.0 and specs["rsi_14"].high % 5.0 == 0.0
    assert specs["rsi_14"].label == "14-day RSI"
    assert specs["mom_10"] == DEFAULT_FEATURE_SPECS[1]  # column absent: unchanged
    assert specs["sma_gap"].low == pytest.approx(-0.06)  # derived from close / sma_50 - 1
    assert specs["sma_gap"].high == pytest.approx(0.06)
    with pytest.raises(ValueError):
        feature_specs_from_data(frame, q=(0.99, 0.01))


def test_feature_spec_validation_and_wording() -> None:
    """Bad ranges/steps are rejected; describe() words the direction and formats the value."""
    with pytest.raises(ValueError):
        FeatureSpec("x", "x", 1.0, 0.0, 0.1)
    with pytest.raises(ValueError):
        FeatureSpec("x", "x", 0.0, 1.0, 0.0)
    mom = DEFAULT_FEATURE_SPECS[1]
    assert mom.describe(0.02, 0.034) == "fell below 2%"
    assert mom.describe(0.06, 0.034) == "rose above 6%"
    assert mom.ray(0.034, -1)[:3] == [0.02, 0.0, -0.02]
    assert format_percent(0.005) == "0.5%"
    assert format_percent(-0.10) == "-10%"
    assert format_percent(0.0) == "0%"
    assert format_integer(57.3) == "57"


def test_counterfactual_to_dict_is_json_safe(rule: CountingRule, buy_instance: dict) -> None:
    """``to_dict`` round-trips through ``json`` and ``changes`` maps feature to new value."""
    cf = generate_counterfactuals(rule, buy_instance, max_calls=200)[0]
    payload = json.loads(json.dumps(cf.to_dict()))
    assert payload["changed_features"] == ["mom_10"]
    assert payload["new_action"] == "HOLD"
    assert cf.changes() == {"mom_10": 0.0}
