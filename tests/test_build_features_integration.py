"""End-to-end test of the feature build pipeline (src/features/build.py)
against a real (temporary) DuckDB -- not just the isolated pure functions
covered elsewhere. Proves the actual SQL selects + merges wire together
correctly, and that a bug in one event-feature source can't take down the
whole build (technical/fundamental features still have to come out).
"""
from __future__ import annotations

import duckdb
import pandas as pd
import pytest

from src.data.db import bootstrap_schema, upsert_wide
from src.features import build as build_module

SETTINGS = {
    "benchmark": "BENCH",
    "features": {
        "momentum_windows": [5, 10],
        "rsi_window": 14,
        "macd_fast": 12,
        "macd_slow": 26,
        "macd_signal": 9,
        "ma_windows": [10, 20],
        "volatility_window": 20,
        "volume_zscore_window": 20,
    },
}


@pytest.fixture
def seeded_db(tmp_path, monkeypatch):
    db_path = tmp_path / "test.duckdb"
    con = duckdb.connect(str(db_path))
    bootstrap_schema(con)

    dates = pd.date_range("2024-01-02", periods=40, freq="B")
    prices = pd.concat(
        [
            pd.DataFrame(
                {"ticker": "AAA", "date": dates, "close": [100.0 + i * 0.5 for i in range(len(dates))], "volume": 1_000_000}
            ),
            pd.DataFrame(
                {"ticker": "BENCH", "date": dates, "close": [50.0 + i * 0.2 for i in range(len(dates))], "volume": 2_000_000}
            ),
        ],
        ignore_index=True,
    )
    upsert_wide(con, "prices", prices, ("ticker", "date"))

    # Deliberately early in the price grid, not centered -- this is what
    # exercises the through_date fix (see test_latest_date_still_has_event_
    # signal_after_a_sparse_early_event below). A centered event date would
    # never have caught that bug, since the range only needs to extend a
    # little past it either way.
    event_date = dates[5]

    news = pd.DataFrame(
        [
            {
                "ticker": "AAA",
                "url": "https://example.com/story",
                "received_at": event_date,
                "event_type": "earnings_guidance",
                "reliability_score": 0.6,
                "duplicate_group": "story-1",
            }
        ]
    )
    upsert_wide(con, "news_events", news, ("ticker", "url"))

    insider = pd.DataFrame(
        [
            {
                "ticker": "AAA",
                "insider_name": "Jane Doe",
                # traded a couple days before the Form 4 was actually filed --
                # the feature must key off the later, public filing_date.
                "transaction_date": dates[3].date(),
                "filing_date": event_date.date(),
                "transaction_code": "P",
                "shares": 1000.0,
                "role": "Officer",
                "price": 100.0,
                "value": 100000.0,
                "shares_owned_after": 5000.0,
                "filing_url": "https://example.com/form4",
            }
        ]
    )
    upsert_wide(
        con, "insider_transactions", insider, ("ticker", "insider_name", "transaction_date", "transaction_code", "shares")
    )

    filings = pd.DataFrame(
        [
            {
                "accession_number": "0001",
                "ticker": "AAA",
                "cik": "123",
                "filing_type": "8-K",
                "filing_date": event_date.date(),
                "url": "https://example.com/8k",
            }
        ]
    )
    upsert_wide(con, "sec_filings", filings, ("accession_number",))
    con.close()

    def _get_connection():
        c = duckdb.connect(str(db_path))
        bootstrap_schema(c)
        return c

    monkeypatch.setattr(build_module, "get_connection", _get_connection)
    monkeypatch.setattr(build_module, "get_settings", lambda: SETTINGS)
    return db_path, dates, event_date


def test_build_features_end_to_end_populates_event_columns(seeded_db):
    db_path, _dates, event_date = seeded_db

    n = build_module.build_features()
    assert n > 0

    con = duckdb.connect(str(db_path))
    row = con.execute(
        "SELECT news_event_count_30d, insider_buy_count_90d, insider_days_since_last_txn, sec_8k_count_90d "
        "FROM features WHERE ticker = 'AAA' AND date = ?",
        [event_date.date()],
    ).fetchone()
    con.close()

    assert row == (1, 1, 0, 1)


def test_build_features_uses_adjusted_close_for_return_features(seeded_db):
    """A raw-price split must not enter momentum or volatility features."""
    db_path, dates, _event_date = seeded_db
    split_date = dates[25].date()

    con = duckdb.connect(str(db_path))
    con.execute("UPDATE prices SET adj_close = close")
    con.execute("UPDATE prices SET close = close * 0.5 WHERE ticker = 'AAA' AND date >= ?", [split_date])
    con.close()

    build_module.build_features()

    con = duckdb.connect(str(db_path))
    momentum = con.execute(
        "SELECT momentum_5d FROM features WHERE ticker = 'AAA' AND date = ?", [split_date]
    ).fetchone()[0]
    con.close()

    # The adjusted series rises from 110 to 112.5 across the five-day window.
    assert momentum == pytest.approx(112.5 / 110.0 - 1)


def test_latest_date_still_has_event_signal_after_a_sparse_early_event(seeded_db):
    """Regression test for a real bug: event features only computed a
    calendar range up through each ticker's own last event date, so a
    ticker whose only qualifying event was early (like AAPL's actual 8-K
    history, or any lightly-covered ticker) went back to entirely missing
    (not zero) event columns on every later date -- including today.
    """
    db_path, dates, event_date = seeded_db
    latest_date = dates[-1]
    days_elapsed = (latest_date - event_date).days

    n = build_module.build_features()
    assert n > 0

    con = duckdb.connect(str(db_path))
    row = con.execute(
        "SELECT sec_8k_count_90d, sec_days_since_last_8k, insider_buy_count_90d, insider_days_since_last_txn "
        "FROM features WHERE ticker = 'AAA' AND date = ?",
        [latest_date.date()],
    ).fetchone()
    con.close()

    assert row is not None
    sec_count, sec_days_since, insider_count, insider_days_since = row
    assert sec_count == 1  # the one 8-K, still inside its 90d window this far out
    assert sec_days_since == days_elapsed
    assert insider_count == 1
    assert insider_days_since == days_elapsed


def test_build_features_survives_a_broken_event_source(seeded_db, monkeypatch):
    db_path, _dates, event_date = seeded_db

    def _boom(*args, **kwargs):
        raise RuntimeError("simulated bug in event feature code")

    monkeypatch.setattr(build_module, "compute_news_event_features", _boom)

    n = build_module.build_features()
    assert n > 0  # technical/fundamental features still built despite the broken source

    con = duckdb.connect(str(db_path))
    cols = {
        r[0]
        for r in con.execute("SELECT column_name FROM information_schema.columns WHERE table_name = 'features'").fetchall()
    }
    assert "news_event_count_30d" not in cols  # the broken source contributed nothing, not garbage

    row = con.execute(
        "SELECT momentum_5d, insider_buy_count_90d FROM features WHERE ticker = 'AAA' AND date = ?",
        [event_date.date()],
    ).fetchone()
    con.close()
    assert row[0] is not None  # technical features unaffected by the news failure
    assert row[1] == 1  # insider features unaffected by the news failure
