"""Small, process-local guard for work that must not overlap.

The dashboard server intentionally stays up so the UI remains available. This
module tracks only active pipeline work, letting idle cleanup release optional
resources without ever stopping a database write halfway through.
"""
from __future__ import annotations

import gc
import logging
import threading
from contextlib import contextmanager
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_active_tasks: dict[str, str] = {}


@contextmanager
def track_task(name: str):
    """Mark a unit of background work active for its complete lifetime."""
    started_at = datetime.now(timezone.utc).isoformat()
    with _lock:
        _active_tasks[name] = started_at
    logger.info("Started task: %s", name)
    try:
        yield
    finally:
        with _lock:
            _active_tasks.pop(name, None)
        logger.info("Finished task: %s", name)


def active_tasks() -> list[dict[str, str]]:
    """Return a snapshot suitable for the local dashboard."""
    with _lock:
        return [{"name": name, "started_at": started_at} for name, started_at in _active_tasks.items()]


def cleanup_idle_resources(settings: dict) -> bool:
    """Release optional local resources only when no pipeline task is active.

    Python's allocator may retain some memory for reuse, but unloading the
    optional local model is the meaningful multi-gigabyte win on this machine.
    """
    with _lock:
        if _active_tasks:
            return False

    from src.llm.client import unload

    unload(settings)
    reclaimed = gc.collect()
    logger.debug("Idle cleanup completed; collected %d unreachable objects", reclaimed)
    return True
