"""app.deps.get_company_names -- batched ticker -> company_name lookup for
the holdings list's avatar/primary-label display. company_name is a
dynamically-added fundamentals column (upsert_wide adds columns on demand,
it's not in the static schema), so a fresh DB or a ticker with no
fundamentals ingested yet must degrade gracefully, not error.
"""
from __future__ import annotations

import duckdb
import pandas as pd
import pytest

from app.deps import get_company_names
from src.data.db import bootstrap_schema, upsert_wide


@pytest.fixture
def con():
    connection = duckdb.connect(":memory:")
    bootstrap_schema(connection)
    yield connection
    connection.close()


def test_empty_tickers_returns_empty_dict(con):
    assert get_company_names(con, []) == {}


def test_no_fundamentals_column_yet_returns_empty_dict(con):
    # Nothing has ever been ingested -- company_name doesn't exist as a column yet.
    assert get_company_names(con, ["AAPL"]) == {}


def test_returns_names_for_tickers_with_fundamentals(con):
    df = pd.DataFrame(
        [
            {"ticker": "AAPL", "as_of": pd.Timestamp("2024-01-01").date(), "company_name": "Apple Inc."},
            {"ticker": "MSFT", "as_of": pd.Timestamp("2024-01-01").date(), "company_name": "Microsoft Corporation"},
        ]
    )
    upsert_wide(con, "fundamentals", df, ("ticker", "as_of"))

    result = get_company_names(con, ["AAPL", "MSFT", "UNKNOWN"])

    assert result == {"AAPL": "Apple Inc.", "MSFT": "Microsoft Corporation"}


def test_uses_the_most_recent_as_of_per_ticker(con):
    df = pd.DataFrame(
        [
            {"ticker": "AAPL", "as_of": pd.Timestamp("2024-01-01").date(), "company_name": "Old Name Inc."},
            {"ticker": "AAPL", "as_of": pd.Timestamp("2024-06-01").date(), "company_name": "Apple Inc."},
        ]
    )
    upsert_wide(con, "fundamentals", df, ("ticker", "as_of"))

    assert get_company_names(con, ["AAPL"]) == {"AAPL": "Apple Inc."}
