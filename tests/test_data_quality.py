"""Data quality checks (src/data/quality.py) -- each is a pure function so
these run against synthetic DataFrames with a known right answer, no
database or network involved.
"""
from __future__ import annotations

import pandas as pd

from src.data.quality import (
    has_blocking_failure,
    check_abnormal_price_jumps,
    check_expected_ticker_coverage,
    check_feature_null_rates,
    check_missing_benchmark,
    check_missing_holdings_prices,
    check_stale_prices,
)


def _prices(rows):
    df = pd.DataFrame(rows, columns=["ticker", "date", "close"])
    df["date"] = pd.to_datetime(df["date"])
    return df


def test_stale_prices_passes_when_every_ticker_matches_the_latest_date():
    prices = _prices([("AAA", "2024-01-10", 100), ("BBB", "2024-01-10", 50)])
    assert check_stale_prices(prices)["status"] == "pass"


def test_stale_prices_flags_a_ticker_lagging_behind_the_rest():
    prices = _prices([("AAA", "2024-01-10", 100), ("BBB", "2024-01-01", 50)])
    result = check_stale_prices(prices, threshold_days=5)
    assert result["status"] == "warn"
    assert "BBB" in result["detail"]


def test_stale_prices_fails_on_no_data():
    assert check_stale_prices(pd.DataFrame(columns=["ticker", "date", "close"]))["status"] == "fail"


def test_missing_benchmark_fails_when_benchmark_has_no_rows():
    prices = _prices([("AAA", "2024-01-10", 100)])
    result = check_missing_benchmark(prices, "SPY")
    assert result["status"] == "fail"
    assert "SPY" in result["detail"]


def test_missing_benchmark_fails_when_benchmark_is_stale():
    prices = _prices([("AAA", "2024-01-10", 100), ("SPY", "2024-01-01", 400)])
    result = check_missing_benchmark(prices, "SPY", threshold_days=5)
    assert result["status"] == "fail"


def test_missing_benchmark_passes_when_current():
    prices = _prices([("AAA", "2024-01-10", 100), ("SPY", "2024-01-10", 400)])
    assert check_missing_benchmark(prices, "SPY")["status"] == "pass"


def test_missing_holdings_prices_passes_with_no_holdings():
    assert check_missing_holdings_prices(pd.DataFrame(), [])["status"] == "pass"


def test_missing_holdings_prices_fails_when_a_held_ticker_has_no_data():
    prices = _prices([("AAA", "2024-01-10", 100)])
    result = check_missing_holdings_prices(prices, ["AAA", "ZZZ"])
    assert result["status"] == "fail"
    assert "ZZZ" in result["detail"]


def test_missing_holdings_prices_fails_when_a_held_ticker_is_stale():
    """This is the acceptance check from the product brief: a stale holdings
    price must not silently feed a confident-looking 'current' total.
    """
    prices = _prices([("AAA", "2024-01-10", 100), ("HELD", "2024-01-01", 50)])
    result = check_missing_holdings_prices(prices, ["HELD"], threshold_days=5)
    assert result["status"] == "fail"
    assert "HELD" in result["detail"]


def test_missing_holdings_prices_passes_when_all_current():
    prices = _prices([("AAA", "2024-01-10", 100), ("HELD", "2024-01-10", 50)])
    assert check_missing_holdings_prices(prices, ["HELD"])["status"] == "pass"


def test_expected_ticker_coverage_warns_on_a_missing_ticker():
    prices = _prices([("AAA", "2024-01-10", 100)])
    result = check_expected_ticker_coverage(prices, ["AAA", "BBB"])
    assert result["status"] == "warn"
    assert "BBB" in result["detail"]


def test_expected_ticker_coverage_passes_when_complete():
    prices = _prices([("AAA", "2024-01-10", 100), ("BBB", "2024-01-10", 50)])
    assert check_expected_ticker_coverage(prices, ["AAA", "BBB"])["status"] == "pass"


def test_abnormal_price_jump_flagged():
    prices = _prices(
        [("AAA", "2024-01-09", 100), ("AAA", "2024-01-10", 200)]  # +100% in one day
    )
    result = check_abnormal_price_jumps(prices)
    assert result["status"] == "warn"
    assert "AAA" in result["detail"]


def test_normal_price_move_not_flagged():
    prices = _prices([("AAA", "2024-01-09", 100), ("AAA", "2024-01-10", 102)])
    assert check_abnormal_price_jumps(prices)["status"] == "pass"


def test_feature_null_rate_fails_when_a_core_column_is_entirely_null():
    features = pd.DataFrame(
        {
            "ticker": ["AAA", "BBB"],
            "date": pd.to_datetime(["2024-01-10", "2024-01-10"]),
            "momentum_5d": [0.01, 0.02],
            "rsi": [None, None],
        }
    )
    result = check_feature_null_rates(features)
    assert result["status"] == "fail"
    assert "rsi" in result["detail"]


def test_feature_null_rate_passes_when_core_columns_populated():
    features = pd.DataFrame(
        {
            "ticker": ["AAA", "BBB"],
            "date": pd.to_datetime(["2024-01-10", "2024-01-10"]),
            "momentum_5d": [0.01, 0.02],
            "rsi": [55.0, 60.0],
        }
    )
    assert check_feature_null_rates(features)["status"] == "pass"


def test_blocking_failure_only_blocks_failed_checks():
    assert has_blocking_failure([{"status": "pass"}, {"status": "warn"}]) is False
    assert has_blocking_failure([{"status": "fail"}]) is True
