"""DiCE-style counterfactual explanations for a black-box advisor.

This module answers the question a non-technical user actually asks of a
recommendation: *"what would have to change for you to change your mind?"*
It searches the feature space around one recommendation for the **smallest,
feasible** change in a handful of actionable technical indicators that flips
the advisor's BUY/HOLD/SELL action, and renders the result as a plain-English
sentence such as ``"I would change my mind from BUY to HOLD if the 14-day RSI
fell below 30."``

The framing is that of Wachter, Mittelstadt & Russell (2018): a counterfactual
is the nearest point on the other side of the decision boundary. The search
objective follows DiCE (Mothilal, Sharma & Tan 2020): minimise *proximity*
(normalised L1 distance to the original instance), maximise *diversity* among
the returned counterfactuals, and respect *feasibility* (each feature may only
move within a plausible range, on a human-friendly grid). Because the advisor
may be a large language model, the model is treated as a **black box** that
only answers ``predict(features) -> action``; there are no gradients, so the
search is a budgeted, nearest-first line search:

1. **One-feature line search.** For every actionable feature, move its value
   along the grid away from the current value in both directions, nearest
   candidate (across *all* features) first, until the action flips; the first
   flip per direction is recorded.
2. **Pairwise search.** If fewer than ``total_cfs`` counterfactuals were found
   within the call budget, the two most sensitive features (nearest single
   flip, else spec order) are moved jointly, again nearest first.
3. **Ranking and diversity.** Candidates are ranked by proximity and selected
   greedily so that different counterfactuals change different features
   (a feature-disjoint approximation of DiCE's determinantal diversity term).

Every prediction is counted against ``max_calls`` (an LLM call is slow) and
cached, so repeated feature vectors are never re-queried within one search.

Feature dependence: indicators are perturbed one at a time (*ceteris
paribus*), the standard DiCE independence assumption. The single exception is
the derived feature ``sma_gap = close / sma_50 - 1``: perturbing it rewrites
``close`` so that the pair stays mutually consistent (the slow 50-day average
is held fixed, so the counterfactual reads as a *price* move). Other
price-derived features (``mom_10``, ``ret_1d``, ``sma_10``) are held at their
observed values; fully dependency-aware counterfactuals would need the full
price history and are a documented limitation.

References:
    Mothilal, R. K., Sharma, A. & Tan, C. (2020). Explaining Machine Learning
        Classifiers through Diverse Counterfactual Explanations. *FAccT 2020*.
    Wachter, S., Mittelstadt, B. & Russell, C. (2018). Counterfactual
        Explanations without Opening the Black Box: Automated Decisions and the
        GDPR. *Harvard Journal of Law & Technology*, 31(2), 841-887.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

__all__ = [
    "ACTIONS",
    "NO_COUNTERFACTUAL_SENTENCE",
    "FeatureSpec",
    "Counterfactual",
    "DEFAULT_FEATURE_SPECS",
    "feature_specs_from_data",
    "format_integer",
    "format_percent",
    "format_plain",
    "current_value",
    "apply_change",
    "apply_counterfactual",
    "generate_counterfactuals",
    "generate_counterfactual",
    "counterfactual_validity",
]

#: The advisor's action vocabulary (mirrors ``advisor.recommender.schema.Action``).
ACTIONS: tuple[str, ...] = ("BUY", "HOLD", "SELL")

#: Sentence returned when no feasible perturbation flips the recommendation.
NO_COUNTERFACTUAL_SENTENCE: str = (
    "No small change in the indicators would flip this recommendation within the searched range."
)

#: A black-box predictor: features in, action (``"BUY"``/``"HOLD"``/``"SELL"``) out.
Predict = Callable[[Mapping[str, Any]], Any]

_GRID_TOL: float = 1e-9


# --------------------------------------------------------------------------- #
# Plain-English value formatting
# --------------------------------------------------------------------------- #
def format_integer(value: float) -> str:
    """Render a value as a whole number (``57.3 -> "57"``), for RSI-like scales."""
    return f"{float(value):.0f}"


def format_percent(value: float) -> str:
    """Render a fraction as a percentage with at most one decimal (``0.05 -> "5%"``)."""
    text = f"{float(value) * 100.0:.1f}"
    if text.endswith(".0"):
        text = text[:-2]
    if text in {"-0", "-0.0"}:
        text = "0"
    return f"{text}%"


def format_plain(value: float) -> str:
    """Render a value with four significant figures (generic fallback)."""
    return f"{float(value):.4g}"


# --------------------------------------------------------------------------- #
# Feature specifications (actionability / feasibility constraints)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class FeatureSpec:
    """Feasibility and wording of one actionable feature.

    Attributes:
        name: Feature key as it appears in the feature dict (``"rsi_14"``).
        label: Human label used in sentences (``"14-day RSI"``).
        low: Smallest feasible value the search may propose.
        high: Largest feasible value the search may propose.
        step: Grid spacing; candidate values are ``low + k * step``, which keeps
            the quoted thresholds round and human-friendly.
        fmt: Renders a value for the sentence (integer, percentage, ...).
        describer: Optional override of :meth:`describe` taking
            ``(new_value, original_value)`` and returning the predicate clause.
    """

    name: str
    label: str
    low: float
    high: float
    step: float
    fmt: Callable[[float], str] = format_plain
    describer: Callable[[float, float], str] | None = None

    def __post_init__(self) -> None:
        """Validate the range and grid step."""
        if not self.name:
            raise ValueError("FeatureSpec.name must be a non-empty string.")
        if not (self.low < self.high):
            raise ValueError(f"{self.name}: require low < high; got {self.low} >= {self.high}.")
        if not (0.0 < self.step <= self.high - self.low):
            raise ValueError(
                f"{self.name}: step must lie in (0, high - low]; got {self.step}."
            )

    @property
    def span(self) -> float:
        """Width of the feasible range, used to normalise distances."""
        return self.high - self.low

    def grid(self) -> np.ndarray:
        """All feasible grid values ``low, low + step, ..., <= high`` (ascending)."""
        n_steps = int(math.floor(self.span / self.step + _GRID_TOL)) + 1
        return np.round(self.low + self.step * np.arange(n_steps), 10)

    def ray(self, current: float, direction: int) -> list[float]:
        """Grid values strictly beyond ``current`` in one direction, nearest first.

        Args:
            current: The feature's observed value.
            direction: ``-1`` to move down (lower values), ``+1`` to move up.

        Returns:
            Candidate values ordered by increasing distance from ``current``.
        """
        grid = self.grid()
        if direction < 0:
            return [float(v) for v in grid[grid < current - _GRID_TOL][::-1]]
        return [float(v) for v in grid[grid > current + _GRID_TOL]]

    def distance(self, a: float, b: float) -> float:
        """Range-normalised absolute distance ``|a - b| / (high - low)``."""
        return abs(float(a) - float(b)) / self.span

    def describe(self, value: float, original: float) -> str:
        """Predicate clause describing the move from ``original`` to ``value``.

        The default wording is ``"fell below <value>"`` / ``"rose above
        <value>"``; ``value`` is the nearest grid point at which the model's
        action was observed to change, so the threshold should be read
        inclusively (the recommendation changes once the feature reaches it).

        Args:
            value: The counterfactual value.
            original: The observed value.

        Returns:
            A clause to follow ``"if the <label> ..."``.
        """
        if self.describer is not None:
            return self.describer(value, original)
        verb = "fell below" if value < original else "rose above"
        return f"{verb} {self.fmt(value)}"

    def with_range(self, low: float, high: float) -> FeatureSpec:
        """Return a copy with a new feasible range (same label, step and wording)."""
        return replace(self, low=float(low), high=float(high))


def _describe_sma_gap(value: float, original: float) -> str:
    """Phrase a ``sma_gap`` change as a price move relative to the 50-day average."""
    verb = "fell" if value < original else "rose"
    if abs(value) < _GRID_TOL:
        return f"{verb} back to its 50-day average"
    side = "below" if value < 0 else "above"
    return f"{verb} to {format_percent(abs(value))} {side} its 50-day average"


#: Default actionable features (ranges are typical for US large caps; override
#: from data with :func:`feature_specs_from_data`).
DEFAULT_FEATURE_SPECS: tuple[FeatureSpec, ...] = (
    FeatureSpec("rsi_14", "14-day RSI", 0.0, 100.0, 5.0, fmt=format_integer),
    FeatureSpec("mom_10", "10-day momentum", -0.30, 0.30, 0.02, fmt=format_percent),
    FeatureSpec("vol_20", "20-day volatility", 0.0, 0.08, 0.005, fmt=format_percent),
    FeatureSpec(
        "sma_gap", "price", -0.30, 0.30, 0.02, fmt=format_percent, describer=_describe_sma_gap
    ),
)


# --------------------------------------------------------------------------- #
# Reading and writing (possibly derived) feature values
# --------------------------------------------------------------------------- #
def _finite(value: Any) -> float | None:
    """Return ``value`` as a finite float, or ``None`` if it is missing/non-numeric."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _derive_sma_gap(features: Mapping[str, Any]) -> float | None:
    """``close / sma_50 - 1`` when both inputs are present and usable."""
    close, sma = _finite(features.get("close")), _finite(features.get("sma_50"))
    if close is None or sma is None or sma <= 0.0:
        return None
    return close / sma - 1.0


_DERIVED: dict[str, Callable[[Mapping[str, Any]], float | None]] = {"sma_gap": _derive_sma_gap}


def current_value(features: Mapping[str, Any], name: str) -> float | None:
    """Read a feature, deriving it (e.g. ``sma_gap``) when it is not stored.

    Args:
        features: The instance's feature dict.
        name: Feature key.

    Returns:
        The finite value, or ``None`` when the feature is absent, non-numeric,
        non-finite or not derivable.
    """
    if name in features:
        value = _finite(features[name])
        if value is not None:
            return value
    derive = _DERIVED.get(name)
    return derive(features) if derive is not None else None


def apply_change(features: Mapping[str, Any], name: str, value: float) -> dict[str, Any]:
    """Return a copy of ``features`` with ``name`` set to ``value``.

    Setting ``sma_gap`` keeps the feature dict consistent: ``close`` is rewritten
    as ``sma_50 * (1 + value)`` (with ``sma_50`` held fixed) and an explicit
    ``sma_gap`` key, if present, is updated too. No other key is added, so the
    dict handed to the model differs from the original only in the perturbed
    feature(s).

    Args:
        features: The instance's feature dict (not modified).
        name: Feature key to change.
        value: New value.

    Returns:
        A new dict with the change applied.
    """
    out = dict(features)
    if name == "sma_gap":
        if "sma_gap" in out:
            out["sma_gap"] = float(value)
        sma = _finite(out.get("sma_50"))
        if sma is not None and sma > 0.0:
            out["close"] = sma * (1.0 + float(value))
        return out
    out[name] = float(value)
    return out


# --------------------------------------------------------------------------- #
# The counterfactual record
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Counterfactual:
    """One verified counterfactual: the change(s) that flip the recommendation.

    Attributes:
        changed_features: Feature keys that were changed (one or two).
        original_values: Their observed values, aligned with ``changed_features``.
        new_values: Their counterfactual values.
        original_action: The advisor's action on the original instance.
        new_action: The action observed after applying the change.
        proximity: Range-normalised L1 distance to the original instance (sum
            over changed features of ``|new - original| / (high - low)``); lower
            is closer, i.e. a smaller change.
        sentence: The plain-English rendering.
    """

    changed_features: tuple[str, ...]
    original_values: tuple[float, ...]
    new_values: tuple[float, ...]
    original_action: str
    new_action: str
    proximity: float
    sentence: str

    def changes(self) -> dict[str, float]:
        """Return the counterfactual as a ``{feature: new_value}`` mapping."""
        return dict(zip(self.changed_features, self.new_values, strict=True))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict representation."""
        return {
            "changed_features": list(self.changed_features),
            "original_values": [float(v) for v in self.original_values],
            "new_values": [float(v) for v in self.new_values],
            "original_action": self.original_action,
            "new_action": self.new_action,
            "proximity": float(self.proximity),
            "sentence": self.sentence,
        }


def apply_counterfactual(instance: Mapping[str, Any], cf: Counterfactual) -> dict[str, Any]:
    """Apply every change of ``cf`` to ``instance`` (see :func:`apply_change`).

    Args:
        instance: The original feature dict (not modified).
        cf: A counterfactual produced by :func:`generate_counterfactuals`.

    Returns:
        The counterfactual feature dict.
    """
    out: dict[str, Any] = dict(instance)
    for name, value in cf.changes().items():
        out = apply_change(out, name, value)
    return out


# --------------------------------------------------------------------------- #
# Data-driven feasible ranges
# --------------------------------------------------------------------------- #
def feature_specs_from_data(
    features_df: pd.DataFrame,
    q: tuple[float, float] = (0.01, 0.99),
    specs: Sequence[FeatureSpec] = DEFAULT_FEATURE_SPECS,
) -> tuple[FeatureSpec, ...]:
    """Override the feasible range of each spec with empirical quantiles.

    DiCE's feasibility constraint asks that counterfactuals stay within the
    data manifold. For every spec whose feature is a column of ``features_df``
    (or derivable from it, e.g. ``sma_gap`` from ``close`` and ``sma_50``) the
    range becomes ``[quantile(q[0]), quantile(q[1])]``, rounded *outward* to the
    spec's grid step so thresholds stay round. Specs whose feature is missing or
    all-NaN are returned unchanged.

    Args:
        features_df: Frame of engineered features (e.g. the output of
            ``advisor.features.indicators.add_features_by_ticker``).
        q: Lower and upper quantiles in ``[0, 1]`` with ``q[0] < q[1]``.
        specs: Specs to adapt (default :data:`DEFAULT_FEATURE_SPECS`).

    Returns:
        A tuple of specs in the same order.

    Raises:
        ValueError: If ``q`` is not an ordered pair inside ``[0, 1]``.
    """
    q_lo, q_hi = float(q[0]), float(q[1])
    if not (0.0 <= q_lo < q_hi <= 1.0):
        raise ValueError(f"q must satisfy 0 <= q[0] < q[1] <= 1; got {q}.")

    adapted: list[FeatureSpec] = []
    for spec in specs:
        if spec.name in features_df.columns:
            values = pd.to_numeric(features_df[spec.name], errors="coerce").to_numpy("float64")
        elif spec.name in _DERIVED and {"close", "sma_50"} <= set(features_df.columns):
            close = pd.to_numeric(features_df["close"], errors="coerce").to_numpy("float64")
            sma = pd.to_numeric(features_df["sma_50"], errors="coerce").to_numpy("float64")
            with np.errstate(divide="ignore", invalid="ignore"):
                values = np.where(sma > 0.0, close / sma - 1.0, np.nan)
        else:
            adapted.append(spec)
            continue
        values = values[np.isfinite(values)]
        if values.size == 0:
            adapted.append(spec)
            continue
        lo, hi = np.quantile(values, [q_lo, q_hi])
        low = math.floor(lo / spec.step + _GRID_TOL) * spec.step
        high = math.ceil(hi / spec.step - _GRID_TOL) * spec.step
        if high <= low:
            high = low + spec.step
        adapted.append(spec.with_range(round(low, 10), round(high, 10)))
    return tuple(adapted)


# --------------------------------------------------------------------------- #
# Black-box access: action normalisation, call budget and cache
# --------------------------------------------------------------------------- #
def _normalise_action(raw: Any) -> str:
    """Coerce a predictor's answer (str or ``Action`` enum) to ``"BUY"/"HOLD"/"SELL"``.

    Raises:
        ValueError: If the answer is not one of :data:`ACTIONS` (case-insensitive).
    """
    action = str(getattr(raw, "value", raw)).strip().upper()
    if action not in ACTIONS:
        raise ValueError(f"Predictor returned {raw!r}; expected one of {ACTIONS}.")
    return action


def _cache_key(features: Mapping[str, Any]) -> tuple[tuple[str, Any], ...]:
    """Hashable, order-independent key for a feature dict."""
    return tuple(
        (key, value if isinstance(value, int | float | str | bool | None) else repr(value))
        for key, value in sorted(features.items())
    )


class _CachedPredictor:
    """Wrap a predictor with an exact-match cache and a hard call budget.

    Attributes:
        calls: Number of *underlying* predictions made so far (cache hits are free).
        max_calls: Budget; :meth:`__call__` refuses uncached queries beyond it.
    """

    def __init__(self, predict: Predict, max_calls: int) -> None:
        self._predict = predict
        self.max_calls = int(max_calls)
        self.calls = 0
        self._cache: dict[tuple[tuple[str, Any], ...], str] = {}

    def can_call(self, features: Mapping[str, Any]) -> bool:
        """``True`` if ``features`` is cached or budget remains."""
        return _cache_key(features) in self._cache or self.calls < self.max_calls

    def __call__(self, features: Mapping[str, Any]) -> str:
        """Return the (normalised) action for ``features``, querying at most once."""
        key = _cache_key(features)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        if self.calls >= self.max_calls:
            raise RuntimeError(f"Prediction budget of {self.max_calls} calls exhausted.")
        self.calls += 1
        action = _normalise_action(self._predict(dict(features)))
        self._cache[key] = action
        return action


# --------------------------------------------------------------------------- #
# Search
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class _Move:
    """One feature moved from ``original`` to ``value``."""

    spec: FeatureSpec
    original: float
    value: float

    @property
    def distance(self) -> float:
        """Range-normalised size of the move."""
        return self.spec.distance(self.value, self.original)


def _render_sentence(moves: Sequence[_Move], original_action: str, new_action: str) -> str:
    """Render ``"I would change my mind from X to Y if the ... and the ..."``."""
    clauses = [f"the {m.spec.label} {m.spec.describe(m.value, m.original)}" for m in moves]
    condition = " and ".join(clauses)
    return f"I would change my mind from {original_action} to {new_action} if {condition}."


def _make_counterfactual(
    moves: Sequence[_Move], original_action: str, new_action: str
) -> Counterfactual:
    """Package verified moves as a :class:`Counterfactual`."""
    return Counterfactual(
        changed_features=tuple(m.spec.name for m in moves),
        original_values=tuple(m.original for m in moves),
        new_values=tuple(m.value for m in moves),
        original_action=original_action,
        new_action=new_action,
        proximity=float(sum(m.distance for m in moves)),
        sentence=_render_sentence(moves, original_action, new_action),
    )


def _select_diverse(found: Sequence[Counterfactual], total_cfs: int) -> list[Counterfactual]:
    """Pick up to ``total_cfs`` counterfactuals: closest first, distinct features preferred.

    Greedy feature-disjoint selection in proximity order approximates DiCE's
    diversity term; if that yields fewer than ``total_cfs``, the closest
    remaining (feature-overlapping) counterfactuals fill the gap. The result is
    returned in proximity order.
    """
    ranked = sorted(found, key=lambda cf: (cf.proximity, cf.changed_features))
    chosen: list[Counterfactual] = []
    used: set[str] = set()
    for cf in ranked:
        if len(chosen) >= total_cfs:
            break
        if used.isdisjoint(cf.changed_features):
            chosen.append(cf)
            used.update(cf.changed_features)
    for cf in ranked:
        if len(chosen) >= total_cfs:
            break
        if cf not in chosen:
            chosen.append(cf)
    return sorted(chosen, key=lambda cf: (cf.proximity, cf.changed_features))


def generate_counterfactuals(
    predict: Predict,
    instance: Mapping[str, Any],
    *,
    specs: Sequence[FeatureSpec] = DEFAULT_FEATURE_SPECS,
    total_cfs: int = 3,
    max_calls: int = 40,
) -> list[Counterfactual]:
    """Search for up to ``total_cfs`` diverse, verified counterfactuals.

    Implements the budgeted black-box DiCE-style search described in the
    module docstring: a nearest-first one-feature line search over every
    actionable feature, then (if needed) a pairwise search over the two most
    sensitive features, then proximity ranking with feature-disjoint diversity.
    Each returned counterfactual was *observed* to flip the action, so the
    sentence is verifiable rather than hallucinated.

    Args:
        predict: Black-box ``predict(features) -> action``; the action may be a
            string (any case) or an ``Action`` enum member. Exceptions raised by
            ``predict`` propagate.
        instance: Feature dict of the recommendation being explained. Specs
            whose feature is absent (and not derivable) are skipped.
        specs: Actionable features with feasible ranges and wording.
        total_cfs: Maximum number of counterfactuals to return (``>= 1``).
        max_calls: Hard budget of ``predict`` calls, *including* the one that
            establishes the original action. Cache hits are free.

    Returns:
        Counterfactuals in proximity order (closest first); empty if nothing
        flipped within the budget and the feasible ranges.

    Raises:
        ValueError: If ``total_cfs`` or ``max_calls`` is below 1, or the
            predictor returns something other than BUY/HOLD/SELL.
        TypeError: If ``instance`` is not a mapping.
    """
    if total_cfs < 1:
        raise ValueError(f"total_cfs must be >= 1; got {total_cfs}.")
    if max_calls < 1:
        raise ValueError(f"max_calls must be >= 1; got {max_calls}.")
    if not isinstance(instance, Mapping):
        raise TypeError(f"instance must be a mapping of feature -> value; got {type(instance)}.")

    base: dict[str, Any] = dict(instance)
    predictor = _CachedPredictor(predict, max_calls)
    original_action = predictor(base)

    active: list[tuple[FeatureSpec, float]] = []
    seen: set[str] = set()
    for spec in specs:
        current = current_value(base, spec.name)
        if current is not None and spec.name not in seen:
            active.append((spec, current))
            seen.add(spec.name)

    found: list[Counterfactual] = []

    # Stage 1: one-feature line search, nearest candidate (across features) first.
    first_flip: dict[tuple[int, int], _Move] = {}
    singles: list[tuple[float, int, int, float]] = []
    for i, (spec, current) in enumerate(active):
        for direction in (-1, 1):
            for value in spec.ray(current, direction):
                singles.append((spec.distance(value, current), i, direction, value))
    singles.sort()
    for _dist, i, direction, value in singles:
        if (i, direction) in first_flip:
            continue
        spec, current = active[i]
        candidate = apply_change(base, spec.name, value)
        if not predictor.can_call(candidate):
            break
        action = predictor(candidate)
        if action != original_action:
            move = _Move(spec, current, value)
            first_flip[(i, direction)] = move
            found.append(_make_counterfactual([move], original_action, action))

    # Stage 2: joint moves of the two most sensitive features.
    if len(found) < total_cfs and len(active) >= 2:

        def nearest_flip(i: int) -> float:
            return min(
                (m.distance for (j, _d), m in first_flip.items() if j == i), default=math.inf
            )

        def segment(i: int, direction: int) -> list[float]:
            """Ray values strictly closer than the feature's own flip (if any)."""
            spec, current = active[i]
            values = spec.ray(current, direction)
            flip = first_flip.get((i, direction))
            if flip is not None:
                values = [v for v in values if spec.distance(v, current) < flip.distance]
            return values

        a, b = sorted(range(len(active)), key=lambda i: (nearest_flip(i), i))[:2]
        spec_a, cur_a = active[a]
        spec_b, cur_b = active[b]
        pairs: list[tuple[float, int, int, float, float]] = []
        for dir_a in (-1, 1):
            for dir_b in (-1, 1):
                for va in segment(a, dir_a):
                    for vb in segment(b, dir_b):
                        dist = spec_a.distance(va, cur_a) + spec_b.distance(vb, cur_b)
                        pairs.append((dist, dir_a, dir_b, va, vb))
        pairs.sort()
        done_quadrants: set[tuple[int, int]] = set()
        for _dist, dir_a, dir_b, va, vb in pairs:
            if len(found) >= total_cfs:
                break
            if (dir_a, dir_b) in done_quadrants:
                continue
            candidate = apply_change(apply_change(base, spec_a.name, va), spec_b.name, vb)
            if not predictor.can_call(candidate):
                break
            action = predictor(candidate)
            if action != original_action:
                done_quadrants.add((dir_a, dir_b))
                moves = [_Move(spec_a, cur_a, va), _Move(spec_b, cur_b, vb)]
                found.append(_make_counterfactual(moves, original_action, action))

    return _select_diverse(found, total_cfs)


# --------------------------------------------------------------------------- #
# Public convenience API
# --------------------------------------------------------------------------- #
def _as_predict(model: Any, *, ticker: str | None = None, as_of: date | None = None) -> Predict:
    """Adapt a callable / ``.predict`` object / ``.recommend`` advisor to ``Predict``.

    Detection order: an object exposing ``recommend(ticker, features, as_of=...)``
    (the thesis advisor interface, whose ``Recommendation.action`` is used), then
    ``predict(features) -> action``, then a plain callable.

    Raises:
        ValueError: If a ``.recommend`` model is given without ``ticker``.
        TypeError: If ``model`` matches none of the interfaces.
    """
    recommend = getattr(model, "recommend", None)
    if callable(recommend):
        if not ticker:
            raise ValueError("ticker= is required for a model exposing .recommend(...).")
        stamp = as_of if as_of is not None else date.today()

        def predict(features: Mapping[str, Any]) -> Any:
            return recommend(ticker, features, as_of=stamp).action

        return predict
    predict_method = getattr(model, "predict", None)
    if callable(predict_method):
        return predict_method
    if callable(model):
        return model
    raise TypeError(
        "model must be callable, or expose .predict(features) or "
        ".recommend(ticker, features, as_of=...)."
    )


def generate_counterfactual(
    model: Any,
    instance: Mapping[str, Any],
    *,
    total_cfs: int = 1,
    specs: Sequence[FeatureSpec] = DEFAULT_FEATURE_SPECS,
    max_calls: int = 40,
    ticker: str | None = None,
    as_of: date | None = None,
) -> str:
    """Generate the plain-English counterfactual sentence for one recommendation.

    Runs :func:`generate_counterfactuals` and returns the closest sentence,
    e.g. ``"I would change my mind from BUY to HOLD if the 14-day RSI fell
    below 30."`` This is the field shown to the user next to each
    recommendation (Wachter et al. 2018 framing; DiCE search per Mothilal et
    al. 2020).

    Args:
        model: A callable ``predict(features) -> action``, an object with
            ``.predict(features) -> action``, or an advisor with
            ``.recommend(ticker, features, as_of=...) -> Recommendation``.
        instance: Feature dict of the recommendation being explained.
        total_cfs: How many diverse counterfactuals to search for; the closest
            one is rendered.
        specs: Actionable features (default :data:`DEFAULT_FEATURE_SPECS`).
        max_calls: Budget of model calls for the whole search.
        ticker: Required when ``model`` exposes ``.recommend``.
        as_of: Date stamped on ``.recommend`` calls (default: today).

    Returns:
        The counterfactual sentence, or :data:`NO_COUNTERFACTUAL_SENTENCE` when
        no feasible change flips the recommendation within the budget.
    """
    predict = _as_predict(model, ticker=ticker, as_of=as_of)
    found = generate_counterfactuals(
        predict, instance, specs=specs, total_cfs=total_cfs, max_calls=max_calls
    )
    return found[0].sentence if found else NO_COUNTERFACTUAL_SENTENCE


def counterfactual_validity(
    predict: Any,
    instance: Mapping[str, Any],
    cf: Counterfactual,
    *,
    strict: bool = False,
    ticker: str | None = None,
    as_of: date | None = None,
) -> bool:
    """Re-check that applying ``cf`` to ``instance`` yields ``cf.new_action``.

    Used in the evaluation to measure the *validity* rate of generated
    counterfactuals (Mothilal et al. 2020, Section 4): the fraction whose
    claimed flip is reproduced by the model.

    Args:
        predict: Any model accepted by :func:`generate_counterfactual`.
        instance: The original feature dict.
        cf: The counterfactual to verify.
        strict: Also re-query the original instance and require its action to
            equal ``cf.original_action`` (one extra model call).
        ticker: Required when ``predict`` exposes ``.recommend``.
        as_of: Date stamped on ``.recommend`` calls (default: today).

    Returns:
        ``True`` if the counterfactual is valid under the model.
    """
    fn = _as_predict(predict, ticker=ticker, as_of=as_of)
    if _normalise_action(fn(apply_counterfactual(instance, cf))) != cf.new_action:
        return False
    if strict:
        return _normalise_action(fn(dict(instance))) == cf.original_action
    return True
