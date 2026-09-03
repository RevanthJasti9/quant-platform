"""Tests the prediction journal's "was this right or wrong" logic
(src/journal/evaluate.py:compute_evaluations) -- the mechanism that answers
whether a forecast actually panned out. Uses synthetic prices with a known,
independently-computed answer rather than trusting the code under test to
grade itself.
"""
import numpy as np
import pandas as pd

from src.journal.evaluate import compute_evaluations

SETTINGS = {"benchmark": "BENCH", "models": {"horizons_days": [5, 20]}}


def _synthetic_prices(n=40):
    dates = pd.date_range("2023-01-01", periods=n, freq="B").date
    frames = []
    for ticker, base, seed in [("AAA", 100, 1), ("BENCH", 50, 2)]:
        rng = np.random.default_rng(seed)
        close = base + np.cumsum(rng.normal(0, 1, n))
        frames.append(pd.DataFrame({"ticker": ticker, "date": dates, "close": close}))
    return pd.concat(frames, ignore_index=True)


def test_prediction_gets_evaluated_once_horizon_elapses():
    prices = _synthetic_prices(n=40)
    px = prices.pivot(index="date", columns="ticker", values="close").sort_index()
    idx, horizon = 10, 5
    prediction_date, outcome_date = px.index[idx], px.index[idx + horizon]

    # Ground truth, computed independently of the code under test.
    expected_actual = (px["AAA"].loc[outcome_date] / px["AAA"].loc[prediction_date] - 1) - (
        px["BENCH"].loc[outcome_date] / px["BENCH"].loc[prediction_date] - 1
    )

    pending = pd.DataFrame(
        {
            "ticker": ["AAA"],
            "prediction_date": [prediction_date],
            "horizon_days": [horizon],
            "expected_relative_return": [0.02],
        }
    )

    result = compute_evaluations(pending, prices, SETTINGS)

    assert len(result) == 1
    row = result.iloc[0]
    assert abs(row["actual_relative_return"] - expected_actual) < 1e-9
    assert abs(row["error"] - (expected_actual - 0.02)) < 1e-9
    assert pd.notna(row["evaluated_at"])


def test_prediction_not_evaluated_before_horizon_elapses():
    """A prediction whose horizon hasn't happened yet -- no future price
    data exists for it -- must stay pending, not get scored early.
    """
    prices = _synthetic_prices(n=40)
    px = prices.pivot(index="date", columns="ticker", values="close").sort_index()
    prediction_date = px.index[-2]  # only 1 day of future data exists; horizon is 20

    pending = pd.DataFrame(
        {
            "ticker": ["AAA"],
            "prediction_date": [prediction_date],
            "horizon_days": [20],
            "expected_relative_return": [0.02],
        }
    )

    result = compute_evaluations(pending, prices, SETTINGS)
    assert result.empty


def test_mixed_batch_evaluates_only_the_ones_that_are_due():
    prices = _synthetic_prices(n=40)
    px = prices.pivot(index="date", columns="ticker", values="close").sort_index()

    pending = pd.DataFrame(
        {
            "ticker": ["AAA", "AAA"],
            "prediction_date": [px.index[10], px.index[-2]],
            "horizon_days": [5, 20],
            "expected_relative_return": [0.02, -0.01],
        }
    )

    result = compute_evaluations(pending, prices, SETTINGS)
    assert len(result) == 1
    assert result.iloc[0]["prediction_date"] == px.index[10]


def test_empty_pending_returns_empty():
    prices = _synthetic_prices(n=40)
    pending = pd.DataFrame(columns=["ticker", "prediction_date", "horizon_days", "expected_relative_return"])
    assert compute_evaluations(pending, prices, SETTINGS).empty
