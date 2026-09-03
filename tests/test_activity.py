from __future__ import annotations

import logging

from src.observability.activity import configure_activity_logging, recent_activity


def test_activity_feed_captures_normal_application_logs():
    configure_activity_logging()
    logger = logging.getLogger("src.pipeline_test")
    logger.warning("Fetching market data")

    event = recent_activity(1)[0]
    assert event["level"] == "warning"
    assert event["source"] == "pipeline_test"
    assert event["message"] == "Fetching market data"
