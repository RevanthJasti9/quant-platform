from pathlib import Path

import pandas as pd
import psutil
from fastapi.templating import Jinja2Templates

from src.config import DB_PATH

APP_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(APP_DIR / "templates"))


def _static_version() -> int:
    """CSS/JS in /static have no cache-busting -- FastAPI's StaticFiles sets
    only Last-Modified/ETag (no Cache-Control), and browsers can still hold
    onto an old cached body indefinitely under that policy. Called fresh on
    every render (not just once at startup) so an edit shows up on the very
    next page load, including across a --reload restart, with no manual
    version bump required. See base.html's <link>/<script> tags.
    """
    try:
        return int((APP_DIR / "static" / "style.css").stat().st_mtime)
    except OSError:
        return 0


templates.env.globals["static_version"] = _static_version

# Every table in schema.sql -- kept as an explicit list (rather than querying
# information_schema) so a table that's dropped or renamed shows up as a
# clear query error immediately instead of just silently vanishing from the
# breakdown.
_STATS_TABLES = [
    "prices", "features", "fundamentals", "news_events", "news_digests", "news_sentiment",
    "sec_filings", "insider_transactions", "predictions", "backtests", "model_versions",
    "ingest_runs", "source_runs", "data_quality_results", "holdings", "broker_portfolio_snapshots",
]


def get_system_stats() -> dict:
    """Live CPU/RAM snapshot for the dashboard's System stat -- this machine
    only has 8GB total, so keeping an eye on both (especially while the LLM
    or a pipeline run is active) is worth surfacing, not just RAM alone.
    interval=0.2 blocks briefly to get a real (not stale/zero) CPU reading;
    cheap enough not to be felt on a page that's otherwise ~50ms.
    """
    cpu_percent = psutil.cpu_percent(interval=0.2)
    mem = psutil.virtual_memory()
    gib = 1024**3  # matches how macOS itself labels RAM (e.g. "8 GB" is 8 GiB) --
    # dividing by 1e9 instead would show this 8GB Mac as "9 GB", which reads as a bug.
    return {
        "cpu_percent": cpu_percent,
        "mem_percent": mem.percent,
        "mem_used_gb": mem.used / gib,
        "mem_total_gb": mem.total / gib,
    }


def get_db_stats(con) -> dict:
    """How much data this local install has actually accumulated -- file
    size straight from disk (a plain stat(), so it's cheap and doesn't
    contend with whatever connection the caller already holds) plus a
    per-table row-count breakdown in one round trip.
    """
    file_size_bytes = DB_PATH.stat().st_size if DB_PATH.exists() else 0
    union_sql = " UNION ALL ".join(f"SELECT '{t}' AS table_name, COUNT(*) AS row_count FROM {t}" for t in _STATS_TABLES)
    tables = records(con.execute(f"{union_sql} ORDER BY row_count DESC").fetchdf())
    return {
        "file_size_bytes": file_size_bytes,
        "total_rows": sum(t["row_count"] for t in tables),
        "tables": tables,
    }


def records(df: pd.DataFrame) -> list[dict]:
    """DataFrame -> list of dicts with NaN/NaT replaced by None, so Jinja's
    `is not none` checks behave -- pandas gives back NaN for SQL NULLs, not
    None, and `{{ value is not none }}` doesn't catch NaN.
    """
    return df.astype(object).where(pd.notnull(df), None).to_dict("records")


def get_next_eval_date(con):
    """The date the earliest still-pending prediction becomes gradeable
    (prediction_date + its horizon, in trading days) -- used to show "first
    check {date}" instead of a bare dash before anything's been evaluated
    yet. Returns None once at least one prediction has already been
    evaluated (callers only need this for the "nothing graded yet" state).
    """
    pending = con.execute(
        "SELECT DISTINCT prediction_date, horizon_days FROM predictions WHERE actual_relative_return IS NULL"
    ).fetchall()
    if not pending:
        return None
    ready_dates = [pd.Timestamp(d) + pd.tseries.offsets.BDay(int(h)) for d, h in pending]
    return min(ready_dates).date()


def get_daily_changes(con, tickers: list[str]) -> dict[str, dict]:
    """Latest close + today's $ / % move for each ticker, in one batched
    query -- ticker -> {"price", "change_dollar", "change_pct"}. Used
    anywhere multiple tickers need this (holdings list, stock header) so
    nothing loops per-ticker doing its own round trip.
    """
    if not tickers:
        return {}
    placeholders = ", ".join("?" for _ in tickers)
    df = con.execute(
        f"""
        WITH ranked AS (
            SELECT ticker, close,
                   ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY date DESC) AS rn
            FROM prices
            WHERE ticker IN ({placeholders})
        )
        SELECT ticker,
               MAX(CASE WHEN rn = 1 THEN close END) AS price,
               MAX(CASE WHEN rn = 2 THEN close END) AS prior_close
        FROM ranked
        WHERE rn <= 2
        GROUP BY ticker
        """,
        tickers,
    ).fetchdf()

    out = {}
    for row in df.itertuples():
        change_dollar = (row.price - row.prior_close) if pd.notna(row.prior_close) else None
        change_pct = (change_dollar / row.prior_close) if change_dollar is not None and row.prior_close else None
        out[row.ticker] = {
            "price": row.price if pd.notna(row.price) else None,
            "change_dollar": change_dollar,
            "change_pct": change_pct,
        }
    return out


def get_company_names(con, tickers: list[str]) -> dict[str, str]:
    """ticker -> most recent company_name on file, in one batched query.
    company_name is a dynamically-added fundamentals column (see
    upsert_wide), not part of the static schema, so a fresh DB or a ticker
    with no fundamentals ingested yet legitimately has none -- callers get
    an empty dict/missing key rather than an error.
    """
    if not tickers:
        return {}
    has_column = con.execute(
        "SELECT 1 FROM information_schema.columns WHERE table_name = 'fundamentals' AND column_name = 'company_name'"
    ).fetchone()
    if not has_column:
        return {}
    placeholders = ", ".join("?" for _ in tickers)
    df = con.execute(
        f"""
        SELECT ticker, company_name FROM (
            SELECT ticker, company_name, ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY as_of DESC) AS rn
            FROM fundamentals WHERE ticker IN ({placeholders})
        ) WHERE rn = 1 AND company_name IS NOT NULL
        """,
        tickers,
    ).fetchdf()
    return dict(zip(df["ticker"], df["company_name"]))
