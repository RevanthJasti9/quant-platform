"""DuckDB connection + schema bootstrap + a generic upsert helper.

The upsert helper (`upsert_wide`) adds any DataFrame column that doesn't
exist on the target table yet (ALTER TABLE ... ADD COLUMN IF NOT EXISTS),
then does a delete-by-key + insert. That's what lets new features or new
fundamentals fields show up just by being present in a DataFrame — no
migration step, no schema file to edit for a scalar column.
"""
from __future__ import annotations

import logging
from pathlib import Path

import duckdb
import pandas as pd

from src.config import DB_PATH

logger = logging.getLogger(__name__)

SCHEMA_PATH = Path(__file__).parent / "schema.sql"

_DTYPE_MAP = {
    "int64": "BIGINT",
    "Int64": "BIGINT",
    "float64": "DOUBLE",
    "object": "VARCHAR",
    "str": "VARCHAR",  # pandas >= 3.0's default inferred string dtype (not "object")
    "string": "VARCHAR",
    "bool": "BOOLEAN",
    "datetime64[ns]": "TIMESTAMP",
}


def _duckdb_type(dtype) -> str:
    # VARCHAR, not DOUBLE, is the safe fallback for a dtype string we don't
    # recognize -- DuckDB will happily store numbers as text, but a real
    # string column typed DOUBLE rejects every row outright (see the pandas
    # 3.0 string-dtype case above, which is exactly how this was found).
    return _DTYPE_MAP.get(str(dtype), "VARCHAR")


def get_connection() -> duckdb.DuckDBPyConnection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(DB_PATH))
    bootstrap_schema(con)
    return con


def bootstrap_schema(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(SCHEMA_PATH.read_text())


def upsert_wide(
    con: duckdb.DuckDBPyConnection,
    table: str,
    df: pd.DataFrame,
    key_cols: tuple[str, ...],
) -> int:
    """Upsert `df` into `table`, adding any missing columns first.

    Returns the number of rows written. No-op on an empty DataFrame.
    """
    if df.empty:
        return 0

    existing_cols = {
        row[0]
        for row in con.execute(
            "SELECT column_name FROM information_schema.columns WHERE table_name = ?",
            [table],
        ).fetchall()
    }
    for col in df.columns:
        if col not in existing_cols:
            con.execute(f'ALTER TABLE {table} ADD COLUMN IF NOT EXISTS "{col}" {_duckdb_type(df[col].dtype)}')

    con.register("_upsert_df", df)
    key_predicate = " AND ".join(f'{table}."{k}" = _upsert_df."{k}"' for k in key_cols)
    con.execute(f"DELETE FROM {table} USING _upsert_df WHERE {key_predicate}")
    cols = ", ".join(f'"{c}"' for c in df.columns)
    con.execute(f"INSERT INTO {table} ({cols}) SELECT {cols} FROM _upsert_df")
    con.unregister("_upsert_df")
    return len(df)
