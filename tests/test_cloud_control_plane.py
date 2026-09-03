from __future__ import annotations

from datetime import datetime, timedelta

from src.cloud.control_plane import ControlPlane
from src.cloud.contracts import JobDefinition, JobPriority, JobState
from src.cloud.model_registry import default_model_specs
from src.cloud.scheduler import default_job_definitions
from src.scheduler.jobs import cloud_job_plan


def test_control_plane_orders_ready_jobs_by_priority_then_deadline():
    now = datetime(2026, 9, 3, 16, 30)
    control_plane = ControlPlane()
    control_plane.submit(
        JobDefinition("experiment", "backtest", JobPriority.EXPERIMENT, now, deadline_at=now + timedelta(hours=3))
    )
    control_plane.submit(JobDefinition("critical", "predict", JobPriority.CRITICAL, now, deadline_at=now + timedelta(hours=2)))
    control_plane.submit(JobDefinition("high", "news", JobPriority.HIGH, now, deadline_at=now + timedelta(hours=1)))

    ready = control_plane.ready_jobs(now)

    assert [job.job_id for job in ready] == ["critical", "high", "experiment"]


def test_control_plane_blocks_jobs_until_dependencies_finish():
    now = datetime(2026, 9, 3, 16, 30)
    control_plane = ControlPlane()
    control_plane.submit(JobDefinition("features", "feature_build", JobPriority.CRITICAL, now))
    control_plane.submit(
        JobDefinition("predict", "specialist_inference", JobPriority.CRITICAL, now, dependencies=("features",))
    )

    assert [job.job_id for job in control_plane.ready_jobs(now)] == ["features"]
    control_plane.mark_running("features")
    control_plane.mark_succeeded("features")

    assert [job.job_id for job in control_plane.ready_jobs(now)] == ["predict"]


def test_control_plane_requeues_failed_job_until_retry_budget_is_spent():
    now = datetime(2026, 9, 3, 16, 30)
    control_plane = ControlPlane()
    control_plane.submit(JobDefinition("gpu-job", "specialist_inference", JobPriority.HIGH, now, max_retries=1))

    control_plane.mark_running("gpu-job")
    requeued = control_plane.mark_failed("gpu-job", "quota exhausted", now)
    assert requeued.state == JobState.QUEUED

    control_plane.mark_running("gpu-job")
    failed = control_plane.mark_failed("gpu-job", "quota exhausted", now)
    assert failed.state == JobState.FAILED
    assert failed.last_error == "quota exhausted"


def test_default_job_definitions_put_predictions_ahead_of_experiments():
    jobs = {job.job_id: job for job in default_job_definitions(datetime(2026, 9, 3, 16, 30))}

    assert jobs["predict-specialists"].priority == JobPriority.CRITICAL
    assert jobs["rank-predictions"].priority == JobPriority.CRITICAL
    assert jobs["retrain-models"].priority == JobPriority.EXPERIMENT
    assert jobs["model-arena"].priority == JobPriority.EXPERIMENT


def test_cloud_job_plan_exposes_partial_publish_path():
    plan = {job["job_id"]: job for job in cloud_job_plan()}

    assert plan["predict-specialists"]["allow_partial_publish"] is True
    assert "build-features" in plan["predict-specialists"]["dependencies"]


def test_default_model_specs_cover_current_specialist_targets():
    names = {spec.name for spec in default_model_specs()}

    assert {"xgboost", "lightgbm", "catboost", "chronos-2", "timesfm-2.5", "chronos-bolt", "finbert"} <= names
