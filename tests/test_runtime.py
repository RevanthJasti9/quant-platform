from __future__ import annotations

from src.observability import runtime


def test_idle_cleanup_skips_an_active_task(monkeypatch):
    called = []
    monkeypatch.setattr("src.llm.client.unload", lambda settings: called.append(settings))

    with runtime.track_task("test run"):
        assert runtime.cleanup_idle_resources({"llm": {"enabled": True}}) is False
        assert runtime.active_tasks()[0]["name"] == "test run"

    assert runtime.cleanup_idle_resources({"llm": {"enabled": True}}) is True
    assert called == [{"llm": {"enabled": True}}]
