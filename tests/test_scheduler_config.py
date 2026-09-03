"""The after-close pipeline's cadence (src/scheduler/jobs.py:start_scheduler)
reads from settings.yaml's `scheduler` section rather than being hardcoded,
so changing when it runs never needs a code change. Verifies the actual
CronTrigger APScheduler builds, not just that settings are read.
"""
from __future__ import annotations

from apscheduler.triggers.cron import CronTrigger

from src.scheduler import jobs as jobs_module


def _trigger_fields(scheduler) -> dict[str, str]:
    # start_scheduler() registers two jobs -- the cron-scheduled after_close_job
    # and a periodic catch-up re-check (IntervalTrigger) -- and get_jobs() sorts
    # by next run time, not insertion order, so index 0 isn't reliably the cron
    # job. Find it by trigger type instead.
    cron_jobs = [j for j in scheduler.get_jobs() if isinstance(j.trigger, CronTrigger)]
    assert len(cron_jobs) == 1, f"expected exactly one cron-triggered job, found {len(cron_jobs)}"
    return {f.name: str(f) for f in cron_jobs[0].trigger.fields}


def test_start_scheduler_uses_settings_for_cron_trigger(monkeypatch):
    monkeypatch.setattr(
        jobs_module,
        "get_settings",
        lambda: {"scheduler": {"timezone": "America/New_York", "day_of_week": "mon,wed,fri", "hour": 9, "minute": 15}},
    )
    # This test is only about the CronTrigger fields -- without this, catch-up would run
    # for real against the live DB (and possibly kick off the real pipeline) depending on
    # what time/day the suite happens to run. See test_scheduler_catchup.py for that logic.
    monkeypatch.setattr(jobs_module, "_catch_up_if_missed", lambda *a, **k: None)
    scheduler = jobs_module.start_scheduler()
    try:
        fields = _trigger_fields(scheduler)
        assert fields["day_of_week"] == "mon,wed,fri"
        assert fields["hour"] == "9"
        assert fields["minute"] == "15"
    finally:
        scheduler.shutdown(wait=False)


def test_start_scheduler_defaults_when_scheduler_config_is_absent(monkeypatch):
    """An older settings.yaml without a `scheduler` section (or a fresh one
    missing individual keys) must still fall back to the documented default
    -- 4:30pm ET weekdays -- not crash or silently schedule nothing.
    """
    monkeypatch.setattr(jobs_module, "get_settings", lambda: {})
    monkeypatch.setattr(jobs_module, "_catch_up_if_missed", lambda *a, **k: None)
    scheduler = jobs_module.start_scheduler()
    try:
        fields = _trigger_fields(scheduler)
        assert fields["day_of_week"] == "mon-fri"
        assert fields["hour"] == "16"
        assert fields["minute"] == "30"
    finally:
        scheduler.shutdown(wait=False)


def test_start_scheduler_also_registers_a_periodic_catchup_recheck(monkeypatch):
    """Without this second job, a missed catch-up attempt (e.g. one that hit
    a transient DB lock at startup -- see test_scheduler_catchup.py) only
    ever gets retried on the next full server restart, which during an
    active dev session could be hours away or never.
    """
    from apscheduler.triggers.interval import IntervalTrigger

    monkeypatch.setattr(jobs_module, "get_settings", lambda: {})
    monkeypatch.setattr(jobs_module, "_catch_up_if_missed", lambda *a, **k: None)
    scheduler = jobs_module.start_scheduler()
    try:
        interval_jobs = [j for j in scheduler.get_jobs() if isinstance(j.trigger, IntervalTrigger)]
        assert len(interval_jobs) == 2
        intervals = {job.trigger.interval.total_seconds() for job in interval_jobs}
        assert jobs_module.CATCH_UP_RECHECK_MINUTES * 60 in intervals
        assert 5 * 60 in intervals
    finally:
        scheduler.shutdown(wait=False)
