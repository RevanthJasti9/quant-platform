"""app.deps.get_db_stats -- the dashboard's Database stat card and Storage
breakdown. Uses a throwaway DuckDB file with the real schema bootstrapped,
not the live app database.
"""
from __future__ import annotations

import duckdb
import pytest

from app.deps import get_db_stats
from src.data.db import bootstrap_schema


@pytest.fixture
def con(tmp_path, monkeypatch):
    db_path = tmp_path / "test.duckdb"
    connection = duckdb.connect(str(db_path))
    bootstrap_schema(connection)
    monkeypatch.setattr("app.deps.DB_PATH", db_path)
    yield connection
    connection.close()


def test_file_size_matches_the_real_file_on_disk(con):
    stats = get_db_stats(con)
    assert stats["file_size_bytes"] > 0


def test_total_rows_is_zero_for_a_freshly_bootstrapped_db(con):
    stats = get_db_stats(con)
    assert stats["total_rows"] == 0
    assert all(t["row_count"] == 0 for t in stats["tables"])


def test_row_counts_reflect_real_inserted_rows(con):
    con.execute("INSERT INTO holdings (ticker, shares, cost_basis, added_at) VALUES ('AAPL', 1.0, 100.0, now())")
    con.execute("INSERT INTO holdings (ticker, shares, cost_basis, added_at) VALUES ('MSFT', 1.0, 200.0, now())")

    stats = get_db_stats(con)

    by_table = {t["table_name"]: t["row_count"] for t in stats["tables"]}
    assert by_table["holdings"] == 2
    assert stats["total_rows"] == 2


def test_tables_are_sorted_by_row_count_descending(con):
    con.execute("INSERT INTO holdings (ticker, shares, cost_basis, added_at) VALUES ('AAPL', 1.0, 100.0, now())")

    stats = get_db_stats(con)

    counts = [t["row_count"] for t in stats["tables"]]
    assert counts == sorted(counts, reverse=True)
    assert stats["tables"][0]["table_name"] == "holdings"
