"""Data quality checks: unlike source_runs (which only knows whether a
fetch raised an exception), these look at the *content* of what landed --
stale prices, missing benchmark/holdings coverage, tickers that never got
any data, single-day price moves that look like bad data rather than a real
market move, and features that came out unexpectedly empty. Each check is a
pure function (DataFrame in, result dict out) so it's unit-testable without
a database; run_data_quality_checks() is the thin I/O wrapper that feeds
them real data and records the results.

Warnings are recorded and surfaced on the dashboard. Failed checks are
pipeline blockers: issuing a prediction without a current benchmark, prices,
or core features would make the result materially misleading.
"""
from __future__ import annotations

import logging

import pandas as pd

from src.config import get_settings
from src.data.db import get_connection, upsert_wide

logger = logging.getLogger(__name__)

# How many calendar days a ticker's latest price can lag the freshest price
# anywhere in the universe before it's flagged. Wide enough to absorb a
# normal 3-day weekend plus one buffer day without false-positiving.
STALE_PRICE_THRESHOLD_DAYS = 5
ABNORMAL_JUMP_THRESHOLD = 0.5
_CORE_FEATURE_COLUMNS = ["momentum_5d", "rsi", "macd", "volatility", "volume_zscore"]


def check_stale_prices(prices: pd.DataFrame, threshold_days: int = STALE_PRICE_THRESHOLD_DAYS) -> dict:
    if prices.empty:
        return {"check_name": "stale_prices", "status": "fail", "detail": "No price data at all"}
    latest_by_ticker = prices.groupby("ticker")["date"].max()
    universe_latest = latest_by_ticker.max()
    stale = latest_by_ticker[(universe_latest - latest_by_ticker).dt.days > threshold_days]
    if stale.empty:
        return {"check_name": "stale_prices", "status": "pass", "detail": f"All tickers current as of {universe_latest.date()}"}
    return {
        "check_name": "stale_prices",
        "status": "warn",
        "detail": f"{len(stale)} ticker(s) lagging the latest date ({universe_latest.date()}): {', '.join(stale.index[:10])}",
    }


def check_missing_benchmark(prices: pd.DataFrame, benchmark: str, threshold_days: int = STALE_PRICE_THRESHOLD_DAYS) -> dict:
    bench = prices[prices["ticker"] == benchmark]
    if bench.empty:
        return {"check_name": "missing_benchmark", "status": "fail", "detail": f"No price data for benchmark {benchmark}"}
    universe_latest = prices["date"].max()
    bench_latest = bench["date"].max()
    if (universe_latest - bench_latest).days > threshold_days:
        return {
            "check_name": "missing_benchmark",
            "status": "fail",
            "detail": f"Benchmark {benchmark} stale: latest {bench_latest.date()} vs universe {universe_latest.date()}",
        }
    return {"check_name": "missing_benchmark", "status": "pass", "detail": f"{benchmark} current as of {bench_latest.date()}"}


def check_missing_holdings_prices(
    prices: pd.DataFrame, holdings_tickers: list[str], threshold_days: int = STALE_PRICE_THRESHOLD_DAYS
) -> dict:
    """The acceptance check this exists for: a stale holdings price
    shouldn't be silently folded into a confident-looking portfolio total.
    """
    if not holdings_tickers:
        return {"check_name": "missing_holdings_prices", "status": "pass", "detail": "No holdings to check"}
    if prices.empty:
        return {
            "check_name": "missing_holdings_prices",
            "status": "fail",
            "detail": f"No price data at all; can't value {len(holdings_tickers)} held ticker(s)",
        }
    universe_latest = prices["date"].max()
    latest_by_ticker = prices.groupby("ticker")["date"].max()
    problems = []
    for t in holdings_tickers:
        if t not in latest_by_ticker.index:
            problems.append(f"{t} (no price data)")
        elif (universe_latest - latest_by_ticker[t]).days > threshold_days:
            problems.append(f"{t} (stale: {latest_by_ticker[t].date()})")
    if problems:
        return {
            "check_name": "missing_holdings_prices",
            "status": "fail",
            "detail": f"{len(problems)} held ticker(s) with missing/stale prices: {', '.join(problems)}",
        }
    return {"check_name": "missing_holdings_prices", "status": "pass", "detail": "All held tickers current"}


def check_expected_ticker_coverage(prices: pd.DataFrame, expected_tickers: list[str]) -> dict:
    have = set(prices["ticker"].unique()) if not prices.empty else set()
    missing = [t for t in expected_tickers if t not in have]
    if missing:
        return {
            "check_name": "expected_ticker_coverage",
            "status": "warn",
            "detail": f"{len(missing)} expected ticker(s) never got price data: {', '.join(missing)}",
        }
    return {
        "check_name": "expected_ticker_coverage",
        "status": "pass",
        "detail": f"All {len(expected_tickers)} expected tickers present",
    }


def check_abnormal_price_jumps(prices: pd.DataFrame, threshold: float = ABNORMAL_JUMP_THRESHOLD) -> dict:
    """Flags a >50% single-day move on the latest date in the adjusted
    research-price series. A jump here is more likely a bad tick than a stock
    split, which adjusted prices intentionally remove.
    """
    if prices.empty:
        return {"check_name": "abnormal_price_jumps", "status": "pass", "detail": "No price data to check"}
    px = prices.sort_values(["ticker", "date"]).copy()
    px["prev_close"] = px.groupby("ticker")["close"].shift(1)
    px["ret"] = px["close"] / px["prev_close"] - 1
    latest = px.loc[px.groupby("ticker")["date"].idxmax()]
    jumps = latest[latest["ret"].abs() > threshold]
    if jumps.empty:
        return {"check_name": "abnormal_price_jumps", "status": "pass", "detail": "No abnormal single-day moves on the latest date"}
    detail = ", ".join(f"{r.ticker} {r.ret:+.0%}" for r in jumps.itertuples())
    return {
        "check_name": "abnormal_price_jumps",
        "status": "warn",
        "detail": f"Possible bad data (unadjusted split, bad tick): {detail}",
    }


def check_feature_null_rates(features: pd.DataFrame) -> dict:
    """A handful of technical features that should exist for essentially
    every ticker once there's enough price history -- if one comes back
    100% null on the latest date, something upstream broke (a renamed
    column, a bad merge), not a data-coverage gap like the event features.
    """
    if features.empty:
        return {"check_name": "null_rate_features", "status": "warn", "detail": "No feature data at all"}
    latest_date = features["date"].max()
    latest = features[features["date"] == latest_date]
    core_cols = [c for c in _CORE_FEATURE_COLUMNS if c in latest.columns]
    broken = [c for c in core_cols if latest[c].isna().all()]
    if broken:
        return {
            "check_name": "null_rate_features",
            "status": "fail",
            "detail": f"Core feature(s) entirely null on {latest_date.date()}: {', '.join(broken)}",
        }
    return {"check_name": "null_rate_features", "status": "pass", "detail": f"Core features populated as of {latest_date.date()}"}


def run_data_quality_checks(run_id: str) -> list[dict]:
    settings = get_settings()
    con = get_connection()
    prices = con.execute(
        "SELECT ticker, date, COALESCE(adj_close, close) AS close FROM prices ORDER BY ticker, date"
    ).fetchdf()
    features = con.execute("SELECT * FROM features ORDER BY ticker, date").fetchdf()
    con.close()

    # Local import: src.data's __init__ eagerly imports this module, and src.holdings imports
    # src.data.db, so a top-level import here would be circular.
    from src.holdings import get_holdings_tickers

    holdings_tickers = get_holdings_tickers()
    expected_tickers = list(dict.fromkeys([*settings["universe"], *holdings_tickers, settings["benchmark"]]))

    results = [
        check_stale_prices(prices),
        check_missing_benchmark(prices, settings["benchmark"]),
        check_missing_holdings_prices(prices, holdings_tickers),
        check_expected_ticker_coverage(prices, expected_tickers),
        check_abnormal_price_jumps(prices),
        check_feature_null_rates(features),
    ]

    checked_at = pd.Timestamp.now("UTC")
    rows_df = pd.DataFrame([{**r, "run_id": run_id, "checked_at": checked_at} for r in results])

    con = get_connection()
    upsert_wide(con, "data_quality_results", rows_df, ("run_id", "check_name"))
    con.close()

    failed = [r["check_name"] for r in results if r["status"] == "fail"]
    if failed:
        logger.error("Data quality check(s) failed for run %s: %s", run_id, failed)
    warned = [r["check_name"] for r in results if r["status"] == "warn"]
    if warned:
        logger.warning("Data quality check(s) warned for run %s: %s", run_id, warned)

    return results


def has_blocking_failure(results: list[dict]) -> bool:
    """Return whether any recorded quality result makes prediction unsafe."""
    return any(result["status"] == "fail" for result in results)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print(run_data_quality_checks("manual-check"))
