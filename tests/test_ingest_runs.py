"""Tests the ingest run-tracking mechanism (src/data/ingest.py): every
source attempted gets a source_runs record (success or failure), every
run_ingest() call gets an ingest_runs record, and a failure in a source
listed under settings.data_quality.critical_sources is flagged on the
returned result so callers can refuse to rebuild features/predictions on
missing or broken data. Uses fake in-process DataSources rather than
yfinance/EDGAR -- that's exercised live by scripts/run_pipeline.py.
"""
from __future__ import annotations

import duckdb
import pandas as pd
import pytest

from src.data import ingest as ingest_module
from src.data.base import DataSource
from src.data.db import bootstrap_schema
from src.data.ingest import _is_critical_failure


class _OkSource(DataSource):
    table = "prices"
    key_cols = ("ticker", "date")

    def fetch(self, tickers, settings, env):
        return pd.DataFrame(
            {
                "ticker": tickers,
                "date": [pd.Timestamp("2024-01-02").date()] * len(tickers),
                "close": [1.0] * len(tickers),
            }
        )


class _FailingSource(DataSource):
    table = "news_events"
    key_cols = ("ticker", "url")

    def fetch(self, tickers, settings, env):
        raise RuntimeError("simulated network failure")


@pytest.fixture
def fake_ingest(tmp_path, monkeypatch):
    """Points run_ingest() at a throwaway DuckDB file and fixed settings,
    with no holdings/network lookups involved.
    """
    db_path = tmp_path / "test.duckdb"

    def _get_connection():
        con = duckdb.connect(str(db_path))
        bootstrap_schema(con)
        return con

    monkeypatch.setattr(ingest_module, "get_connection", _get_connection)
    monkeypatch.setattr(ingest_module, "get_settings", lambda: {"data_quality": {"critical_sources": ["prices"]}})
    monkeypatch.setattr(ingest_module, "get_env", lambda: None)
    return db_path


def test_failing_source_creates_a_visible_failed_run_record(fake_ingest, monkeypatch):
    monkeypatch.setattr(ingest_module.base, "iter_sources", lambda: {"prices": _OkSource, "news_events": _FailingSource})

    result = ingest_module.run_ingest(tickers=["AAA"])

    con = duckdb.connect(str(fake_ingest))
    status, error = con.execute(
        "SELECT status, error FROM source_runs WHERE run_id = ? AND source = 'news_events'", [result.run_id]
    ).fetchone()
    con.close()

    assert status == "failed"
    assert "simulated network failure" in error
    assert result.failed_sources == ["news_events"]


def test_critical_source_failure_blocks_and_is_recorded(fake_ingest, monkeypatch):
    monkeypatch.setattr(ingest_module.base, "iter_sources", lambda: {"prices": _FailingSource})

    result = ingest_module.run_ingest(tickers=["AAA"])

    assert result.critical_failure is True

    con = duckdb.connect(str(fake_ingest))
    status = con.execute("SELECT status FROM ingest_runs WHERE run_id = ?", [result.run_id]).fetchone()[0]
    con.close()
    assert status == "failed"


def test_noncritical_failure_does_not_block(fake_ingest, monkeypatch):
    monkeypatch.setattr(ingest_module.base, "iter_sources", lambda: {"prices": _OkSource, "news_events": _FailingSource})

    result = ingest_module.run_ingest(tickers=["AAA"])

    assert result.critical_failure is False
    assert result.results.get("prices") == 1


def test_successful_run_recorded_with_row_counts(fake_ingest, monkeypatch):
    monkeypatch.setattr(ingest_module.base, "iter_sources", lambda: {"prices": _OkSource})

    result = ingest_module.run_ingest(tickers=["AAA", "BBB"])

    assert result.critical_failure is False
    assert result.results["prices"] == 2

    con = duckdb.connect(str(fake_ingest))
    row_count, status = con.execute(
        "SELECT row_count, status FROM source_runs WHERE run_id = ? AND source = 'prices'", [result.run_id]
    ).fetchone()
    ingest_status = con.execute("SELECT status FROM ingest_runs WHERE run_id = ?", [result.run_id]).fetchone()[0]
    con.close()
    assert (row_count, status, ingest_status) == (2, "success", "success")


# Pure decision-logic tests -- no DB, no network.


def test_is_critical_failure_true_when_critical_source_errors():
    assert _is_critical_failure({"prices"}, {"prices", "news_events"}, {"prices"}, {"news_events": 10}) is True


def test_is_critical_failure_true_when_critical_source_returns_zero_rows():
    assert _is_critical_failure({"prices"}, {"prices"}, set(), {"prices": 0}) is True


def test_is_critical_failure_false_for_noncritical_failure():
    assert _is_critical_failure({"prices"}, {"prices", "news_events"}, {"news_events"}, {"prices": 100}) is False


def test_is_critical_failure_false_when_critical_source_not_attempted_this_run():
    assert _is_critical_failure({"prices"}, {"news_events"}, set(), {"news_events": 10}) is False
