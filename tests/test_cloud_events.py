from datetime import UTC, datetime

from src.cloud.contracts import JobPriority
from src.cloud.events import EventSignal, event_job_graph, event_priority


def _signal(event_type: str, novelty: float, materiality: float, confidence: float) -> EventSignal:
    return EventSignal("evt-1", "AMZN", event_type, novelty, materiality, confidence, datetime(2026, 9, 3, 20, 0, tzinfo=UTC))


def test_material_earnings_event_is_critical():
    assert event_priority(_signal("earnings", 0.9, 0.9, 0.9)) == JobPriority.CRITICAL


def test_routine_event_does_not_consume_critical_compute():
    assert event_priority(_signal("routine_news", 0.1, 0.1, 0.8)) == JobPriority.NORMAL


def test_event_graph_publishes_baseline_before_waiting_for_event_analysis():
    signal = _signal("sec_8k", 0.9, 0.8, 0.9)
    jobs = {job.job_id: job for job in event_job_graph(signal, signal.received_at)}

    baseline = jobs["event-amzn-evt-1-baseline"]
    analysis = jobs["event-amzn-evt-1-analysis"]
    enriched = jobs["event-amzn-evt-1-enriched-refresh"]
    assert baseline.dependencies == ()
    assert baseline.allow_partial_publish is True
    assert analysis.priority == JobPriority.CRITICAL
    assert enriched.dependencies == (analysis.job_id,)
    assert enriched.payload["mode"] == "event_enriched"
