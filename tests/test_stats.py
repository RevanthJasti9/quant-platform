"""src.stats -- the Wilson interval (Accuracy page hit-rate) and IC
significance t-test (backtest calibration) that make small-sample results
show their real uncertainty instead of a bare, falsely-precise percentage.
"""
from __future__ import annotations

import pytest

from src.stats import ic_significance, wilson_interval


def test_wilson_interval_none_when_no_data():
    assert wilson_interval(0, 0) is None


def test_wilson_interval_bounds_always_within_zero_and_one():
    for successes, n in [(0, 3), (3, 3), (1, 3), (50, 100), (99, 100)]:
        lo, hi = wilson_interval(successes, n)
        assert 0.0 <= lo <= hi <= 1.0


def test_wilson_interval_is_wide_for_a_small_sample_even_at_100_percent():
    # A naive normal-approximation interval would collapse to [1.0, 1.0] here --
    # exactly the false-precision this function exists to avoid ("3 for 3" is not
    # proof of a real edge). Wilson's interval must show real remaining uncertainty.
    lo, hi = wilson_interval(3, 3)
    assert lo < 0.5
    assert hi == pytest.approx(1.0, abs=0.02)


def test_wilson_interval_narrows_as_sample_size_grows():
    lo_small, hi_small = wilson_interval(5, 10)
    lo_large, hi_large = wilson_interval(500, 1000)
    assert (hi_large - lo_large) < (hi_small - lo_small)


def test_wilson_interval_centered_near_observed_rate_for_a_decent_sample():
    lo, hi = wilson_interval(70, 100)
    assert lo < 0.70 < hi
    assert (hi - lo) < 0.25


def test_ic_significance_none_for_too_few_points():
    assert ic_significance(0.5, 2) is None


def test_ic_significance_none_for_degenerate_perfect_correlation():
    assert ic_significance(1.0, 10) is None
    assert ic_significance(-1.0, 10) is None


def test_ic_significance_near_one_when_ic_is_zero():
    # No correlation at all should never look "significant" regardless of n.
    assert ic_significance(0.0, 500) == pytest.approx(1.0)


def test_ic_significance_small_pvalue_for_strong_ic_and_large_n():
    p = ic_significance(0.15, 2000)
    assert p < 0.01


def test_ic_significance_large_pvalue_for_weak_ic_and_small_n():
    p = ic_significance(0.05, 20)
    assert p > 0.5


def test_ic_significance_symmetric_in_sign():
    assert ic_significance(0.08, 300) == pytest.approx(ic_significance(-0.08, 300))
