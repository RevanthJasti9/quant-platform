"""app.copy.format_bytes / table_label -- the dashboard's Database stat and
per-table storage breakdown need a human-readable size (not raw byte counts)
and a readable table name (not a raw snake_case column value).
"""
from __future__ import annotations

from app.copy import format_bytes, table_label


def test_format_bytes_picks_the_largest_sensible_unit():
    assert format_bytes(500) == "500 B"
    assert format_bytes(2048) == "2.0 KB"
    assert format_bytes(5 * 1024**2) == "5.0 MB"
    assert format_bytes(1.5 * 1024**3) == "1.5 GB"


def test_format_bytes_zero():
    assert format_bytes(0) == "0 B"


def test_table_label_has_a_readable_override_for_acronym_tables():
    assert table_label("sec_filings") == "SEC filings"
    assert table_label("news_digests") == "News digests (LLM)"


def test_table_label_falls_back_to_title_case_for_unlisted_tables():
    assert table_label("prices") == "Prices"
    assert table_label("backtests") == "Backtests"
