"""Tests for ``advisor.evaluation.sharpe_difference`` (post-hoc, offline).

The bootstrap must be reproducible, must centre on the observed difference,
must find no difference between a series and itself, and must detect a
large, real difference in Sharpe ratio.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from advisor.evaluation.sharpe_difference import (
    block_bootstrap_indices,
    sharpe_by_year,
    sharpe_difference_test,
)


def test_block_indices_have_the_right_length_and_range():
    """A resample is n valid row indices built from consecutive blocks."""
    rng = np.random.default_rng(0)
    idx = block_bootstrap_indices(23, 5, rng)
    assert len(idx) == 23
    assert idx.min() >= 0 and idx.max() < 23
    # Inside a block, indices step by one (wrapping at the end).
    assert ((idx[1:5] - idx[0:4]) % 23 == 1).all()


def test_block_indices_reject_bad_arguments():
    """Zero-length samples or blocks are an error."""
    with pytest.raises(ValueError):
        block_bootstrap_indices(0, 5, np.random.default_rng(0))


def test_identical_series_show_no_difference():
    """A strategy compared with itself has a zero difference and p = 1."""
    returns = np.random.default_rng(1).normal(0.0005, 0.01, size=300)
    result = sharpe_difference_test(returns, returns, n_resamples=200)
    assert result["difference"] == pytest.approx(0.0)
    assert result["ci_low"] == pytest.approx(0.0) and result["ci_high"] == pytest.approx(0.0)
    assert result["p_value"] == pytest.approx(1.0)


def test_large_real_difference_is_detected():
    """A clearly better strategy on the same days gives a CI above zero."""
    rng = np.random.default_rng(2)
    noise = rng.normal(0.0, 0.01, size=750)
    better = noise + 0.002
    worse = noise - 0.001
    result = sharpe_difference_test(better, worse, n_resamples=500)
    assert result["difference"] > 0
    assert result["ci_low"] > 0
    assert result["p_value"] < 0.05


def test_result_is_reproducible_with_a_seed():
    """The same seed gives the same interval."""
    rng = np.random.default_rng(3)
    a, b = rng.normal(0.001, 0.01, 200), rng.normal(0.0, 0.01, 200)
    first = sharpe_difference_test(a, b, n_resamples=300, seed=7)
    second = sharpe_difference_test(a, b, n_resamples=300, seed=7)
    assert first == second


def test_series_must_cover_the_same_days():
    """Different lengths mean the series are not paired."""
    with pytest.raises(ValueError):
        sharpe_difference_test(np.zeros(10), np.zeros(11), n_resamples=10)


def test_sharpe_by_year_has_one_column_per_year():
    """Each calendar year gets its own Sharpe ratio for every strategy."""
    days = pd.bdate_range("2025-01-01", "2026-06-30")
    rng = np.random.default_rng(4)
    returns = pd.DataFrame({"a": rng.normal(0.001, 0.01, len(days)),
                            "b": rng.normal(0.0, 0.01, len(days))}, index=days)
    table = sharpe_by_year(returns)
    assert list(table.columns) == [2025, 2026]
    assert list(table.index) == ["a", "b"]
