from __future__ import annotations

from datetime import UTC, datetime, timedelta

from src.cloud.contracts import JobDefinition, JobPriority


def default_job_definitions(now: datetime | None = None) -> list[JobDefinition]:
    now = now or datetime.now(UTC)
    eod_deadline = now + timedelta(minutes=45)
    train_deadline = now + timedelta(days=1)
    return [
        JobDefinition("collect-news", "news_capture", JobPriority.HIGH, now, deadline_at=now + timedelta(minutes=15)),
        JobDefinition("collect-sec", "sec_capture", JobPriority.HIGH, now, deadline_at=now + timedelta(minutes=20)),
        JobDefinition("collect-events", "event_capture", JobPriority.HIGH, now, deadline_at=now + timedelta(minutes=20)),
        JobDefinition("collect-eod-prices", "eod_prices", JobPriority.CRITICAL, now, deadline_at=now + timedelta(minutes=25)),
        JobDefinition(
            "build-features",
            "feature_build",
            JobPriority.CRITICAL,
            now,
            deadline_at=now + timedelta(minutes=30),
            dependencies=("collect-eod-prices", "collect-news", "collect-sec", "collect-events"),
        ),
        JobDefinition(
            "predict-specialists",
            "specialist_inference",
            JobPriority.CRITICAL,
            now,
            deadline_at=eod_deadline,
            dependencies=("build-features",),
            allow_partial_publish=True,
            refresh_job_types=("refresh-predictions",),
        ),
        JobDefinition(
            "rank-predictions",
            "ranking",
            JobPriority.CRITICAL,
            now,
            deadline_at=eod_deadline,
            dependencies=("predict-specialists",),
        ),
        JobDefinition(
            "journal-update",
            "prediction_journal",
            JobPriority.HIGH,
            now,
            deadline_at=eod_deadline,
            dependencies=("rank-predictions",),
        ),
        JobDefinition(
            "retrain-models",
            "retraining",
            JobPriority.EXPERIMENT,
            now,
            deadline_at=train_deadline,
            dependencies=("build-features",),
            max_retries=1,
        ),
        JobDefinition(
            "model-arena",
            "model_arena_backtest",
            JobPriority.EXPERIMENT,
            now,
            deadline_at=train_deadline,
            dependencies=("retrain-models",),
        ),
    ]
