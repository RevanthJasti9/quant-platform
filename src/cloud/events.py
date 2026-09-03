"""Event-driven job creation for material news, filings, and insider signals."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from src.cloud.contracts import JobDefinition, JobPriority

_CRITICAL_TYPES = {"earnings", "guidance", "sec_8k", "sec_10q", "sec_10k"}


@dataclass(frozen=True)
class EventSignal:
    event_id: str
    ticker: str
    event_type: str
    novelty: float
    materiality: float
    confidence: float
    received_at: datetime


def event_priority(signal: EventSignal) -> JobPriority:
    score = signal.novelty * 0.3 + signal.materiality * 0.5 + signal.confidence * 0.2
    if signal.event_type in _CRITICAL_TYPES and score >= 0.65:
        return JobPriority.CRITICAL
    if score >= 0.55:
        return JobPriority.HIGH
    return JobPriority.NORMAL


def event_job_graph(signal: EventSignal, now: datetime | None = None) -> list[JobDefinition]:
    """Create a baseline refresh plus an optional enrichment/refresh chain."""
    now = now or datetime.now(UTC)
    priority = event_priority(signal)
    prefix = f"event-{signal.ticker.lower()}-{signal.event_id}"
    baseline = JobDefinition(
        job_id=f"{prefix}-baseline",
        job_type="targeted_prediction",
        priority=priority,
        created_at=now,
        deadline_at=now + timedelta(minutes=10),
        payload={"ticker": signal.ticker, "event_id": signal.event_id, "mode": "baseline"},
        allow_partial_publish=True,
    )
    event_analysis = JobDefinition(
        job_id=f"{prefix}-analysis",
        job_type="event_nlp",
        priority=priority,
        created_at=now,
        deadline_at=now + timedelta(minutes=20),
        max_retries=1,
        payload={"ticker": signal.ticker, "event_id": signal.event_id, "event_type": signal.event_type},
    )
    enriched = JobDefinition(
        job_id=f"{prefix}-enriched-refresh",
        job_type="targeted_prediction",
        priority=priority,
        created_at=now,
        deadline_at=now + timedelta(minutes=30),
        dependencies=(event_analysis.job_id,),
        payload={"ticker": signal.ticker, "event_id": signal.event_id, "mode": "event_enriched"},
        allow_partial_publish=True,
    )
    return [baseline, event_analysis, enriched]
