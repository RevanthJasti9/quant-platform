"""In-memory activity feed for the local dashboard.

This deliberately mirrors normal application logging instead of creating a
second status system. Anything the pipeline logs about ingesting data,
building features, training, backtesting, or prediction appears here too.
The buffer is intentionally ephemeral: logs survive while the app is open,
but are not mixed into the research database or retained as user data.
"""
from __future__ import annotations

import logging
import threading
from collections import deque
from datetime import datetime, timezone

MAX_ACTIVITY_EVENTS = 300


class _ActivityHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self._events: deque[dict[str, str]] = deque(maxlen=MAX_ACTIVITY_EVENTS)
        self._lock = threading.Lock()

    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = record.getMessage()
        except Exception:
            message = "Unable to format activity message"
        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname.lower(),
            "source": record.name.removeprefix("src.").removeprefix("app."),
            "message": message,
        }
        with self._lock:
            self._events.append(event)

    def recent(self, limit: int) -> list[dict[str, str]]:
        with self._lock:
            return list(self._events)[-limit:]


_handler = _ActivityHandler()


def configure_activity_logging() -> None:
    """Attach the feed once, even when FastAPI reloads the application."""
    root = logging.getLogger()
    if _handler not in root.handlers:
        root.addHandler(_handler)


def recent_activity(limit: int = 100) -> list[dict[str, str]]:
    """Return the most recent events in chronological order."""
    return _handler.recent(max(1, min(limit, MAX_ACTIVITY_EVENTS)))
