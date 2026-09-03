"""Regime classification and per-regime breakdown (src/backtesting/metrics.py)
-- the point of these is to stop a strong aggregate Sharpe/alpha from hiding
a strategy that only worked in one regime or one lucky stretch.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtesting.metrics import (
    classify_regimes,
    compute_all_metrics,
    quantile_returns,
    quantile_spread,
    regime_breakdown_metrics,
)


def _price_series(closes, start="2023-01-02"):
    dates = pd.date_range(start, periods=len(closes), freq="B")
    return pd.Series(closes, index=dates)


def test_classify_regimes_labels_bull_and_bear_by_trailing_trend():
    # 80 days flat-ish, then a sustained rally -- the rally's own tail
    # should read as "bull" once the 60d trailing window is inside it.
    flat = [100.0] * 80
    rally = [100.0 + i * 2 for i in range(1, 41)]
    prices = _price_series(flat + rally)

    regimes = classify_regimes(prices, trend_window=60, vol_window=20)
    # last date: 60 trading days back is well inside the rally, trailing return should be positive
    assert regimes["regime_trend"].iloc[-1] == "bull"


def test_classify_regimes_labels_bear_on_a_sustained_decline():
    flat = [100.0] * 80
    decline = [100.0 - i * 2 for i in range(1, 41)]
    prices = _price_series(flat + decline)

    regimes = classify_regimes(prices, trend_window=60, vol_window=20)
    assert regimes["regime_trend"].iloc[-1] == "bear"


def test_classify_regimes_marks_insufficient_warmup_as_unknown():
    prices = _price_series([100.0 + i for i in range(10)])  # far short of the 60d trend window
    regimes = classify_regimes(prices, trend_window=60, vol_window=20)
    assert (regimes["regime_trend"].iloc[:10] == "unknown").all()


def test_classify_regimes_flags_a_sustained_vol_spike_as_high_vol():
    # A calm stretch followed by a much choppier one -- once the expanding
    # median has enough of the choppy period behind it to have shifted,
    # the tail should read high_vol relative to its own trailing history.
    # (An early point in the calm stretch has no way to "know" a choppier
    # period is coming, so it isn't guaranteed to read low_vol here --
    # only its own trailing window and expanding median decide that.)
    rng = np.random.default_rng(0)
    calm = 100 + np.cumsum(rng.normal(0, 0.2, 60))
    choppy = calm[-1] + np.cumsum(rng.normal(0, 3.0, 60))
    prices = _price_series(np.concatenate([calm, choppy]))

    regimes = classify_regimes(prices, trend_window=60, vol_window=20)
    assert regimes["regime_vol"].iloc[-1] == "high_vol"


def test_regime_label_at_date_t_unaffected_by_prices_after_t():
    """Even though this is reporting-only (not a model feature), a date's
    regime label must still only reflect its own trailing history --
    otherwise "what regime was day T in" would depend on what the
    strategy did after day T, which defeats the point of the breakdown.
    """
    rng = np.random.default_rng(1)
    base = list(100 + np.cumsum(rng.normal(0, 1, 90)))
    prices_short = _price_series(base)
    prices_long = _price_series(base + [500.0] * 20)  # wildly different future added

    early_short = classify_regimes(prices_short).iloc[50]
    early_long = classify_regimes(prices_long).iloc[50]
    assert early_short["regime_trend"] == early_long["regime_trend"]
    assert early_short["regime_vol"] == early_long["regime_vol"]


def test_regime_breakdown_metrics_computes_per_regime_stats():
    # 10 "bull" days all positive, 10 "bear" days all negative -- hand-verifiable.
    port = pd.Series([0.01] * 10 + [-0.01] * 10)
    bench = pd.Series([0.005] * 10 + [-0.02] * 10)
    labels = pd.Series(["bull"] * 10 + ["bear"] * 10)

    breakdown = regime_breakdown_metrics(port, bench, labels)
    assert set(breakdown) == {"bull", "bear"}
    assert breakdown["bull"]["n_days"] == 10
    assert breakdown["bull"]["win_rate"] == 1.0
    assert breakdown["bear"]["win_rate"] == 0.0
    # bear days: portfolio -1%/day vs benchmark -2%/day -- positive alpha
    assert breakdown["bear"]["alpha"] > 0


def test_regime_breakdown_metrics_excludes_regimes_with_too_few_days():
    port = pd.Series([0.01] * 20 + [0.02] * 3)
    bench = pd.Series([0.005] * 23)
    labels = pd.Series(["bull"] * 20 + ["bear"] * 3)  # bear has only 3 days -- below the 5-day floor

    breakdown = regime_breakdown_metrics(port, bench, labels)
    assert "bull" in breakdown
    assert "bear" not in breakdown


def test_regime_breakdown_metrics_never_reports_unknown():
    port = pd.Series([0.01] * 10)
    bench = pd.Series([0.005] * 10)
    labels = pd.Series(["unknown"] * 10)
    assert regime_breakdown_metrics(port, bench, labels) == {}


def _scored_frame(n_dates=40, n_tickers=10, seed=0, predictive=True):
    """Synthetic (date, ticker, score, forward_relative_return) panel.
    predictive=True: score is actual_return plus a little noise, so ranking
    by score should genuinely separate winners from losers.
    predictive=False: score is independent random noise, uncorrelated with
    the outcome -- ranking by it shouldn't reveal a real pattern.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2023-01-02", periods=n_dates, freq="B")
    rows = []
    for d in dates:
        actual = rng.normal(0, 0.02, n_tickers)
        score = actual + rng.normal(0, 0.002, n_tickers) if predictive else rng.normal(0, 0.02, n_tickers)
        for i in range(n_tickers):
            rows.append({"date": d, "ticker": f"T{i}", "score": score[i], "forward_relative_return": actual[i]})
    return pd.DataFrame(rows)


def test_quantile_returns_is_monotonic_for_a_genuinely_predictive_score():
    scored = _scored_frame(predictive=True)
    result = quantile_returns(scored, n_quantiles=5)

    assert list(result["quantile"]) == [1, 2, 3, 4, 5]
    returns = result["mean_return"].tolist()
    assert returns == sorted(returns)  # strictly non-decreasing quantile 1 -> 5
    assert quantile_spread(result) > 0.01  # a real, sizeable top-minus-bottom spread


def test_quantile_returns_is_near_flat_for_a_random_uninformative_score():
    scored = _scored_frame(predictive=False, n_dates=200)  # large sample so noise averages out
    result = quantile_returns(scored, n_quantiles=5)

    spread = quantile_spread(result)
    # with no real signal, the top-minus-bottom spread should be small
    # relative to the predictive case's (which cleared 0.01 above)
    assert abs(spread) < 0.005


def test_quantile_returns_buckets_within_each_date_not_across_dates():
    """A ticker with a below-average score on a strong day and an
    above-average score on a weak day must land in different quantiles on
    each date -- bucketing globally (ignoring date) would put both of a
    stock's appearances in the same bucket and miss this entirely.
    """
    scored = pd.DataFrame(
        [
            # strong day: everyone's return is high; T0 is relatively worst
            {"date": "2023-01-02", "ticker": "T0", "score": 0.01, "forward_relative_return": 0.05},
            {"date": "2023-01-02", "ticker": "T1", "score": 0.05, "forward_relative_return": 0.09},
            # weak day: everyone's return is low; T0 is relatively best
            {"date": "2023-01-03", "ticker": "T0", "score": 0.03, "forward_relative_return": -0.01},
            {"date": "2023-01-03", "ticker": "T1", "score": -0.02, "forward_relative_return": -0.05},
        ]
    )
    result = quantile_returns(scored, n_quantiles=2)
    assert set(result["quantile"]) == {1, 2}
    # quantile 2 (top score each day) should show the better of that day's two returns on average
    top_mean = result.loc[result["quantile"] == 2, "mean_return"].iloc[0]
    bottom_mean = result.loc[result["quantile"] == 1, "mean_return"].iloc[0]
    assert top_mean > bottom_mean


def test_quantile_returns_empty_input():
    empty = pd.DataFrame(columns=["date", "score", "forward_relative_return"])
    result = quantile_returns(empty)
    assert result.empty
    assert quantile_spread(result) is None


def test_quantile_spread_is_top_minus_bottom():
    quantiles = pd.DataFrame({"quantile": [1, 2, 3], "mean_return": [-0.02, 0.0, 0.03], "n": [10, 10, 10]})
    assert quantile_spread(quantiles) == pytest.approx(0.05)


def test_compute_all_metrics_includes_calibration_pvalue():
    rng = np.random.default_rng(0)
    n = 200
    portfolio_returns = pd.Series(rng.normal(0.001, 0.01, n))
    benchmark_returns = pd.Series(rng.normal(0.0005, 0.01, n))
    turnover = pd.Series(rng.uniform(0, 1, n))
    predicted = pd.Series(rng.normal(size=n))
    # actual correlated with predicted -- a real (if modest) relationship, not pure noise
    actual = predicted * 0.1 + pd.Series(rng.normal(size=n))

    metrics = compute_all_metrics(portfolio_returns, benchmark_returns, turnover, predicted, actual)

    assert "calibration_pvalue" in metrics
    assert metrics["calibration_pvalue"] is not None
    assert 0.0 <= metrics["calibration_pvalue"] <= 1.0
