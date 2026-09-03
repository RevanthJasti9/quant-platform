"""src.scheduler.jobs._catch_up_if_missed: a CronTrigger only fires while the
process happens to be running at that exact minute, so a dev server that's
down at 16:30 (a reload cycle, a crash, the laptop asleep) silently loses
that day's run with no retry -- the dashboard just goes stale until someone
notices. This covers the fix: on startup, run once immediately if today's
scheduled time has already passed and nothing has run yet today.
"""
from __future__ import annotations

import pandas as pd
import pytest

from src.scheduler import jobs as jobs_module

TZ = "America/New_York"


@pytest.fixture(autouse=True)
def _reset_running_flag():
    """_after_close_job_running is module-level (read by both
    _catch_up_if_missed and after_close_job itself) -- must not leak
    between tests.
    """
    jobs_module._after_close_job_running.clear()
    yield
    jobs_module._after_close_job_running.clear()


class _FakeScheduler:
    def __init__(self):
        self.added_jobs = []

    def add_job(self, func):
        self.added_jobs.append(func)


class _FakeConnection:
    def __init__(self, has_run_today: bool):
        self._has_run_today = has_run_today

    def execute(self, *args, **kwargs):
        return self

    def fetchone(self):
        return (1,) if self._has_run_today else None

    def close(self):
        pass


def _patch_connection(monkeypatch, has_run_today: bool):
    monkeypatch.setattr(jobs_module, "get_connection", lambda: _FakeConnection(has_run_today))


def test_no_catchup_when_scheduled_time_has_not_arrived_yet(monkeypatch):
    # Monday 9:00am, scheduled for 16:30 -- nothing to catch up on yet today.
    _patch_connection(monkeypatch, has_run_today=False)
    scheduler = _FakeScheduler()
    now = pd.Timestamp("2026-08-31 09:00", tz=TZ)  # Monday
    jobs_module._catch_up_if_missed(scheduler, TZ, "mon-fri", 16, 30, now=now)
    assert scheduler.added_jobs == []


def test_no_catchup_on_a_non_scheduled_day(monkeypatch):
    # Saturday at 6pm -- well past 16:30, but mon-fri never fires on a Saturday.
    _patch_connection(monkeypatch, has_run_today=False)
    scheduler = _FakeScheduler()
    now = pd.Timestamp("2026-08-29 18:00", tz=TZ)  # Saturday
    jobs_module._catch_up_if_missed(scheduler, TZ, "mon-fri", 16, 30, now=now)
    assert scheduler.added_jobs == []


def test_no_catchup_when_todays_run_already_happened(monkeypatch):
    _patch_connection(monkeypatch, has_run_today=True)
    scheduler = _FakeScheduler()
    now = pd.Timestamp("2026-08-31 18:00", tz=TZ)  # Monday evening
    jobs_module._catch_up_if_missed(scheduler, TZ, "mon-fri", 16, 30, now=now)
    assert scheduler.added_jobs == []


def test_catchup_runs_once_when_scheduled_time_passed_with_no_run_today(monkeypatch):
    _patch_connection(monkeypatch, has_run_today=False)
    scheduler = _FakeScheduler()
    now = pd.Timestamp("2026-08-31 18:00", tz=TZ)  # Monday evening, 16:30 already passed
    jobs_module._catch_up_if_missed(scheduler, TZ, "mon-fri", 16, 30, now=now)
    assert scheduler.added_jobs == [jobs_module.after_close_job]


def test_catchup_uses_the_configured_day_of_week_not_just_weekday(monkeypatch):
    # Tuesday is a real weekday but isn't in "mon,wed,fri" -- must not fire.
    _patch_connection(monkeypatch, has_run_today=False)
    scheduler = _FakeScheduler()
    now = pd.Timestamp("2026-09-01 18:00", tz=TZ)  # Tuesday evening
    jobs_module._catch_up_if_missed(scheduler, TZ, "mon,wed,fri", 16, 30, now=now)
    assert scheduler.added_jobs == []


def test_no_catchup_queued_while_a_run_is_already_in_progress(monkeypatch):
    """Regression case: the periodic re-check (CATCH_UP_RECHECK_MINUTES)
    calls this while a long-running after_close_job may still be mid-flight
    -- it hasn't written its ingest_runs row yet (that only happens at the
    very end), so the naive DB check alone would queue a second, overlapping
    run. Must not even touch the DB in that case.
    """
    monkeypatch.setattr(
        jobs_module, "get_connection", lambda: pytest.fail("must not query the DB while a run is in progress")
    )
    jobs_module._after_close_job_running.set()
    scheduler = _FakeScheduler()
    now = pd.Timestamp("2026-08-31 18:00", tz=TZ)  # Monday evening, well past 16:30
    jobs_module._catch_up_if_missed(scheduler, TZ, "mon-fri", 16, 30, now=now)
    assert scheduler.added_jobs == []


def test_after_close_job_skips_when_already_running(monkeypatch):
    monkeypatch.setattr(jobs_module, "run_ingest", lambda: pytest.fail("must not start a second overlapping run"))
    jobs_module._after_close_job_running.set()

    jobs_module.after_close_job()  # should log a warning and return, not raise

    assert jobs_module._after_close_job_running.is_set()  # unchanged -- the in-progress run still owns it


def test_after_close_job_clears_the_flag_even_if_ingest_raises(monkeypatch):
    def _boom():
        raise RuntimeError("simulated failure")

    monkeypatch.setattr(jobs_module, "run_ingest", _boom)

    with pytest.raises(RuntimeError):
        jobs_module.after_close_job()

    assert jobs_module._after_close_job_running.is_set() is False
