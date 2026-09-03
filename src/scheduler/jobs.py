"""APScheduler wiring for the after-close pipeline: refresh data, rebuild
features, run inference with the current model, record predictions, and
evaluate predictions whose horizon has elapsed. Weekly retraining is exposed
as `retrain_job` but is NOT auto-scheduled in V1 — run `src/models/train.py`
by hand until the V2 experiment/comparison engine exists to gate promotion.
"""
from __future__ import annotations

import logging
import threading

import pandas as pd
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from src.cloud.scheduler import default_job_definitions
from src.config import get_settings
from src.data.db import get_connection
from src.data.ingest import run_ingest
from src.data.quality import has_blocking_failure, run_data_quality_checks
from src.features.build import build_features
from src.journal.evaluate import evaluate_predictions
from src.llm.client import unload as unload_llm
from src.models.news_digest import build_news_digests
from src.models.predict import run_predictions
from src.observability.runtime import cleanup_idle_resources, track_task

logger = logging.getLogger(__name__)

CATCH_UP_RECHECK_MINUTES = 20

# The pipeline takes 10-15+ minutes and only writes its ingest_runs row at
# the very end -- _catch_up_if_missed's periodic re-check would otherwise
# see "nothing recorded yet" and queue a second overlapping run while the
# first is still in flight. This is checked from a different thread than it's
# set on (APScheduler runs jobs on its own worker threads), hence Event
# rather than a plain bool.
_after_close_job_running = threading.Event()


def cloud_job_plan() -> list[dict]:
    """Expose the cloud-first job graph without replacing the working local scheduler."""
    return [
        {
            "job_id": job.job_id,
            "job_type": job.job_type,
            "priority": int(job.priority),
            "dependencies": list(job.dependencies),
            "allow_partial_publish": job.allow_partial_publish,
        }
        for job in default_job_definitions()
    ]


def after_close_job() -> None:
    if _after_close_job_running.is_set():
        logger.warning("after_close_job() triggered while a previous run is still in progress -- skipping")
        return
    _after_close_job_running.set()
    try:
        with track_task("after-close data refresh"):
            logger.info("Running after-close pipeline")
            ingest_result = run_ingest()
            if ingest_result.critical_failure:
                logger.error(
                    "After-close pipeline stopped: critical source(s) failed (%s). Not rebuilding "
                    "features/predictions on incomplete data -- will retry at the next scheduled run.",
                    ingest_result.failed_sources,
                )
                return
            # Generate today's LLM-derived sentiment before feature construction,
            # so inference never consumes yesterday's digest as if it were current.
            build_news_digests()
            build_features()
            quality_results = run_data_quality_checks(ingest_result.run_id)
            for r in quality_results:
                if r["status"] != "pass":
                    logger.warning("Data quality [%s] %s: %s", r["status"], r["check_name"], r["detail"])
            if has_blocking_failure(quality_results):
                logger.error("After-close pipeline stopped: blocking data-quality check failed")
                return
            run_predictions(run_id=ingest_result.run_id)
            evaluate_predictions()
            logger.info("After-close pipeline complete")
    finally:
        # Run even after a data-quality failure or an exception, so an optional
        # local LLM never remains in memory merely because the run ended early.
        unload_llm(get_settings())
        _after_close_job_running.clear()


def _catch_up_if_missed(
    scheduler: BackgroundScheduler,
    timezone: str,
    day_of_week: str,
    hour: int,
    minute: int,
    now: pd.Timestamp | None = None,
) -> None:
    """A cron trigger only fires while the process happens to be running at
    that exact minute -- a dev server that's down at run time (a reload, a
    crash, the laptop asleep) silently loses that day's run with no retry,
    and the dashboard just goes stale until someone notices. If today's
    scheduled run-time has already passed and nothing has run yet today,
    run once immediately instead of waiting for the next cron fire.

    `now` is injectable so tests can pick a fixed instant instead of
    depending on the real wall clock and today's real day-of-week. Called
    both once at startup and periodically thereafter (see
    CATCH_UP_RECHECK_MINUTES) so a startup attempt that hits a transient DB
    lock isn't the only chance today's data gets to catch up.
    """
    if _after_close_job_running.is_set():
        return
    now = now if now is not None else pd.Timestamp.now(tz=timezone)
    trigger = CronTrigger(day_of_week=day_of_week, hour=hour, minute=minute, timezone=timezone)
    next_fire = trigger.get_next_fire_time(None, now.normalize())
    if next_fire is None or next_fire > now:
        return

    today_utc_start = now.normalize().tz_convert("UTC").tz_localize(None)
    try:
        con = get_connection()
        try:
            already_ran = con.execute(
                "SELECT 1 FROM ingest_runs WHERE started_at >= ? LIMIT 1", [today_utc_start]
            ).fetchone()
        finally:
            con.close()
    except Exception:
        # DuckDB is single-writer -- if another run (e.g. one kicked off by hand) already
        # holds the file lock at the exact moment the server restarts, this must not take
        # the whole app down. Skip catch-up for this startup; it'll be checked again next time.
        logger.warning("Skipped missed-run catch-up check: database unavailable (likely locked by another run)", exc_info=True)
        return
    if already_ran:
        return

    logger.info("Missed today's %02d:%02d run (server wasn't running at the time) -- catching up now", hour, minute)
    scheduler.add_job(after_close_job)


def start_scheduler() -> BackgroundScheduler:
    cfg = get_settings().get("scheduler", {})
    timezone = cfg.get("timezone", "America/New_York")
    day_of_week = cfg.get("day_of_week", "mon-fri")
    hour = cfg.get("hour", 16)
    minute = cfg.get("minute", 30)

    # A missed cron event is coalesced into one catch-up instead of accumulating
    # jobs after sleep/restarts. Only one instance of each job may run at once.
    scheduler = BackgroundScheduler(
        timezone=timezone,
        job_defaults={"coalesce": True, "max_instances": 1, "misfire_grace_time": 300},
    )
    scheduler.add_job(after_close_job, CronTrigger(day_of_week=day_of_week, hour=hour, minute=minute))
    scheduler.start()
    logger.info(
        "Scheduler started: after-close pipeline runs %s at %02d:%02d %s", day_of_week, hour, minute, timezone
    )
    _catch_up_if_missed(scheduler, timezone, day_of_week, hour, minute)
    # Re-check periodically too, not just once at startup -- e.g. the startup
    # attempt colliding with a manually-run script's DB lock would otherwise
    # be the only chance today gets, silently leaving the dashboard stale
    # for the rest of the day with no further retry.
    scheduler.add_job(
        lambda: _catch_up_if_missed(scheduler, timezone, day_of_week, hour, minute),
        IntervalTrigger(minutes=CATCH_UP_RECHECK_MINUTES),
    )
    # The server remains online for the dashboard, but unused optional model
    # memory is released regularly while no data/model task is active.
    scheduler.add_job(
        lambda: cleanup_idle_resources(get_settings()),
        IntervalTrigger(minutes=5),
        id="idle-resource-cleanup",
    )
    return scheduler
