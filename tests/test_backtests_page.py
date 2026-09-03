"""UI-adapter functions for the Strategy Performance page (app/api/backtests.py).
Regression coverage for a real bug: _regime_rows() broke with an
AttributeError the moment quantile_spread/quantiles were added to the same
JSON blob, because it wasn't told to skip the new "quantiles" key (a list,
not a regime->stats dict like the others) -- caught only by loading the
live page, not by any test, since every prior test exercised the metrics
computation in isolation and never the combined shape these functions
actually receive from a real backtest run.
"""
from __future__ import annotations

from app.api.backtests import _fold_rows, _format_calibration, _metric_rows, _quantile_chart_data, _regime_rows

# The exact combined shape engine.py's regimes_by_date actually produces --
# by_trend/by_volatility/by_fold all regime->stats dicts, quantiles a list.
REALISTIC_BREAKDOWN = {
    "by_trend": {"bull": {"n_days": 200, "sharpe": 1.5, "cagr": 0.3, "alpha": 0.1, "win_rate": 0.6}},
    "by_volatility": {"low_vol": {"n_days": 150, "sharpe": 1.2, "cagr": 0.2, "alpha": 0.08, "win_rate": 0.55}},
    "by_fold": {
        "fold_1": {"n_days": 126, "sharpe": 1.7, "cagr": 0.4, "alpha": 0.4, "win_rate": 0.6},
        "fold_2": {"n_days": 126, "sharpe": -0.6, "cagr": -0.1, "alpha": -0.08, "win_rate": 0.48},
    },
    "quantiles": [
        {"quantile": 1, "mean_return": -0.004, "n": 500},
        {"quantile": 2, "mean_return": 0.001, "n": 500},
        {"quantile": 3, "mean_return": 0.01, "n": 500},
    ],
}


def test_regime_rows_handles_the_full_realistic_breakdown_without_crashing():
    """This is the exact scenario that broke in production: a breakdown
    dict containing by_trend/by_volatility/by_fold/quantiles together.
    """
    rows = _regime_rows(REALISTIC_BREAKDOWN)
    labels = {r["label"] for r in rows}
    assert labels == {"Bull", "Low volatility"}  # by_fold and quantiles excluded


def test_regime_rows_sorted_by_days_descending():
    rows = _regime_rows(REALISTIC_BREAKDOWN)
    assert [r["n_days"] for r in rows] == sorted((r["n_days"] for r in rows), reverse=True)


def test_fold_rows_kept_in_chronological_order():
    rows = _fold_rows(REALISTIC_BREAKDOWN)
    assert [r["label"] for r in rows] == ["Fold 1", "Fold 2"]


def test_fold_rows_empty_when_no_fold_data():
    assert _fold_rows({"by_trend": {}}) == []


def test_quantile_chart_data_shapes_labels_and_values():
    chart = _quantile_chart_data(REALISTIC_BREAKDOWN)
    assert chart["labels"] == ["Q1", "Q2", "Q3"]
    assert chart["values"] == [-0.4, 0.1, 1.0]  # mean_return * 100, rounded


def test_quantile_chart_data_empty_when_no_quantiles():
    assert _quantile_chart_data({}) == {"labels": [], "values": []}


def test_metric_rows_skips_none_values_for_older_backtests():
    """Backtests recorded before calibration/quantile_spread existed have
    those columns as NULL -- must be skipped, not rendered as "None%".
    """
    run = {
        "sharpe": 1.1, "cagr": 0.2, "max_drawdown": -0.1, "alpha": 0.05,
        "win_rate": 0.5, "volatility": 0.2, "turnover": 0.3,
        "calibration": None, "quantile_spread": None,
    }
    rows = _metric_rows(run)
    labels = {r["label"] for r in rows}
    assert "Ranking Skill (IC)" not in labels
    assert "Top vs. Bottom Picks" not in labels
    assert len(rows) == 7


def test_metric_rows_includes_calibration_and_spread_when_present():
    run = {
        "sharpe": 1.1, "cagr": 0.2, "max_drawdown": -0.1, "alpha": 0.05,
        "win_rate": 0.5, "volatility": 0.2, "turnover": 0.3,
        "calibration": 0.011, "quantile_spread": 0.014,
    }
    rows = {r["label"]: r["value"] for r in _metric_rows(run)}
    assert rows["Ranking Skill (IC)"] == "0.011"
    assert rows["Top vs. Bottom Picks"] == "1.4%"


def test_metric_rows_shows_significance_verdict_when_pvalue_present():
    run = {
        "sharpe": 1.1, "cagr": 0.2, "max_drawdown": -0.1, "alpha": 0.05,
        "win_rate": 0.5, "volatility": 0.2, "turnover": 0.3,
        "calibration": 0.09, "calibration_pvalue": 0.01, "quantile_spread": None,
    }
    rows = {r["label"]: r["value"] for r in _metric_rows(run)}
    assert rows["Ranking Skill (IC)"] == "0.090 (significant, p=0.01)"


def test_format_calibration_no_pvalue_shows_plain_ic():
    assert _format_calibration(0.045, None) == "0.045"


def test_format_calibration_significant():
    assert _format_calibration(0.09, 0.02) == "0.090 (significant, p=0.02)"


def test_format_calibration_not_significant():
    assert _format_calibration(0.01, 0.83) == "0.010 (not significant, p=0.83)"
