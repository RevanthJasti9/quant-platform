"""Tests the extensibility mechanism in src/data/db.py -- new columns (a new
feature, a new fundamentals field) get added on the fly, and upserts replace
rather than duplicate. Deliberately does not hit yfinance/EDGAR over the
network; that's exercised by scripts/run_pipeline.py against live data.
"""
import duckdb
import pandas as pd

from src.data.db import bootstrap_schema, upsert_wide


def test_upsert_wide_adds_missing_columns_and_dedupes(tmp_path):
    con = duckdb.connect(str(tmp_path / "test.duckdb"))
    bootstrap_schema(con)

    df1 = pd.DataFrame({"ticker": ["AAA"], "date": [pd.Timestamp("2024-01-02").date()], "close": [100.0]})
    assert upsert_wide(con, "prices", df1, ("ticker", "date")) == 1

    df2 = pd.DataFrame({"ticker": ["AAA"], "date": [pd.Timestamp("2024-01-02").date()], "new_feature": [1.23]})
    upsert_wide(con, "features", df2, ("ticker", "date"))
    cols = {
        r[0]
        for r in con.execute(
            "SELECT column_name FROM information_schema.columns WHERE table_name = 'features'"
        ).fetchall()
    }
    assert "new_feature" in cols

    df3 = pd.DataFrame({"ticker": ["AAA"], "date": [pd.Timestamp("2024-01-02").date()], "new_feature": [9.99]})
    upsert_wide(con, "features", df3, ("ticker", "date"))
    result = con.execute("SELECT new_feature FROM features WHERE ticker = 'AAA'").fetchall()
    assert result == [(9.99,)]

    con.close()


def test_upsert_wide_noop_on_empty_dataframe(tmp_path):
    con = duckdb.connect(str(tmp_path / "test2.duckdb"))
    bootstrap_schema(con)
    empty = pd.DataFrame(columns=["ticker", "date", "close"])
    assert upsert_wide(con, "prices", empty, ("ticker", "date")) == 0
    con.close()


def test_upsert_wide_new_text_column_is_varchar_not_double(tmp_path):
    """Regression test: pandas >= 3.0 infers a native "str" dtype for
    string columns instead of the legacy "object" dtype. A dtype map that
    only recognizes "object" silently falls through to its numeric
    fallback, which then rejects every row of real text data outright.
    """
    con = duckdb.connect(str(tmp_path / "test3.duckdb"))
    bootstrap_schema(con)

    df = pd.DataFrame(
        {
            "ticker": ["AAA"],
            "prediction_date": [pd.Timestamp("2024-01-02").date()],
            "horizon_days": [5],
            "reasons_json": ['[{"feature": "momentum_5d", "shap": 0.01}]'],
        }
    )
    assert upsert_wide(con, "predictions", df, ("ticker", "prediction_date", "horizon_days")) == 1

    col_type = con.execute(
        "SELECT data_type FROM information_schema.columns WHERE table_name = 'predictions' AND column_name = 'reasons_json'"
    ).fetchone()[0]
    assert col_type == "VARCHAR"

    stored = con.execute("SELECT reasons_json FROM predictions WHERE ticker = 'AAA'").fetchone()[0]
    assert "momentum_5d" in stored

    con.close()
