"""Runs every registered DataSource and upserts its rows. This is the one
place `run_pipeline.py` and the scheduler call into — it never needs to know
what sources exist, so adding a new one (see base.py) never touches this file.

Every call also records what happened: one `ingest_runs` row for the whole
call, one `source_runs` row per source attempted (success or failure, with
row count and error). A failure in a source listed under
`data_quality.critical_sources` in settings.yaml (prices by default —
nothing downstream works without it) is flagged on the returned result so
callers can refuse to rebuild features/predictions on data that's missing
or didn't actually refresh, instead of silently serving a stale or wrong
"current" value.
"""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field

import pandas as pd

from src.config import get_env, get_settings
from src.data import base
from src.data.db import get_connection, upsert_wide
from src.holdings import get_holdings_tickers

logger = logging.getLogger(__name__)


@dataclass
class IngestRunResult:
    run_id: str
    results: dict[str, int] = field(default_factory=dict)
    failed_sources: list[str] = field(default_factory=list)
    critical_failure: bool = False


def _is_critical_failure(
    critical_sources: set[str],
    attempted_sources: set[str],
    failed_sources: set[str],
    results: dict[str, int],
) -> bool:
    """A source only counts against this run if it was actually attempted --
    a scoped call like run_ingest(sources=["prices"]) shouldn't be flagged
    critical because some *other* critical source wasn't requested this
    time. Within what was attempted, either an exception or a suspicious
    zero-row result counts: e.g. the prices source re-fetches its full
    history window every call, so 0 rows back from a real provider only
    happens when the fetch genuinely failed, not from "nothing new today."
    """
    for name in critical_sources:
        if name not in attempted_sources:
            continue
        if name in failed_sources or results.get(name, 0) == 0:
            return True
    return False


def run_ingest(sources: list[str] | None = None, tickers: list[str] | None = None) -> IngestRunResult:
    settings = get_settings()
    env = get_env()
    if tickers is None:
        # Anything the user actually holds is folded in automatically --
        # add it in the UI and it starts getting real data with no config edit.
        tickers = list(dict.fromkeys([*settings["universe"], *get_holdings_tickers(), settings["benchmark"]]))

    registry = base.iter_sources()
    if sources:
        registry = {k: v for k, v in registry.items() if k in sources}
    critical_sources = set(settings.get("data_quality", {}).get("critical_sources", ["prices"]))

    run_id = f"{pd.Timestamp.now('UTC'):%Y%m%d%H%M%S}_{uuid.uuid4().hex[:8]}"
    started_at = pd.Timestamp.now("UTC")

    results: dict[str, int] = {}
    failed_sources: list[str] = []
    source_run_rows: list[dict] = []

    for name, cls in registry.items():
        source = cls()
        source_started = pd.Timestamp.now("UTC")
        try:
            logger.info("Fetching %s data for %d ticker(s)", name, len(tickers))
            # fetch() is the slow, paced, network-bound part (a full run across all
            # sources can take 10+ minutes) -- only open a connection for the brief
            # write after it, not for the whole loop. DuckDB is single-writer, so
            # holding one connection open for the entire run would lock every other
            # process (including the dashboard itself) out of the database until
            # ingest finished.
            df = source.fetch(tickers, settings, env)
            con = get_connection()
            try:
                n = upsert_wide(con, source.table, df, source.key_cols)
            finally:
                con.close()
        except Exception as e:
            logger.exception("Source %s failed, skipping", name)
            failed_sources.append(name)
            source_run_rows.append(
                {
                    "run_id": run_id,
                    "source": name,
                    "status": "failed",
                    "row_count": 0,
                    "started_at": source_started,
                    "finished_at": pd.Timestamp.now("UTC"),
                    "error": str(e)[:500],
                }
            )
            continue
        results[name] = n
        source_run_rows.append(
            {
                "run_id": run_id,
                "source": name,
                "status": "success",
                "row_count": n,
                "started_at": source_started,
                "finished_at": pd.Timestamp.now("UTC"),
                "error": None,
            }
        )
        logger.info("Ingested %d rows from %s -> %s", n, name, source.table)

    critical_failure = _is_critical_failure(critical_sources, set(registry), set(failed_sources), results)
    status = "failed" if critical_failure else ("partial" if failed_sources else "success")

    con = get_connection()
    try:
        if source_run_rows:
            upsert_wide(con, "source_runs", pd.DataFrame(source_run_rows), ("run_id", "source"))
        con.execute(
            """
            INSERT INTO ingest_runs
                (run_id, started_at, finished_at, status, sources_requested, sources_succeeded, sources_failed, critical_failure)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [run_id, started_at, pd.Timestamp.now("UTC"), status, len(registry), len(results), len(failed_sources), critical_failure],
        )
    finally:
        con.close()

    if critical_failure:
        logger.error(
            "CRITICAL: ingest run %s failed a critical source %s -- downstream feature/prediction "
            "builds should not run on this data.",
            run_id,
            failed_sources,
        )

    return IngestRunResult(run_id=run_id, results=results, failed_sources=failed_sources, critical_failure=critical_failure)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print(run_ingest())
