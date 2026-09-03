"""Finnhub and Polygon's free tiers cap requests per minute (60/min and
5/min respectively) -- a plain unthrottled loop over ~30 tickers would blow
through Polygon's limit almost immediately, after which every later ticker
silently gets nothing (a 429 caught by the per-ticker try/except, logged as
a generic "fetch failed" warning). These tests verify the fetch loop
actually paces itself rather than bursting, without real network calls or
actually sleeping for minutes.
"""
from __future__ import annotations

from types import SimpleNamespace

import httpx
import pytest

from src.data import finnhub_news, polygon_news


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


class _RecordingClient:
    """Stands in for httpx.Client: records every call's ticker param and
    returns an empty result, no real network I/O.
    """

    def __init__(self, *args, **kwargs):
        self.tickers_called: list[str] = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get(self, url, params=None):
        self.tickers_called.append((params or {}).get("symbol") or (params or {}).get("ticker"))
        return _FakeResponse({"results": []} if "polygon" in url else [])


@pytest.fixture
def fake_client(monkeypatch):
    client = _RecordingClient()
    monkeypatch.setattr(httpx, "Client", lambda *a, **kw: client)
    return client


@pytest.fixture
def fake_sleep(monkeypatch):
    calls = []
    monkeypatch.setattr(finnhub_news.time, "sleep", lambda s: calls.append(s))
    return calls


def test_finnhub_paces_between_calls_not_before_first_or_after_last(fake_client, fake_sleep):
    env = SimpleNamespace(finnhub_api_key="fake-key")
    finnhub_news.FinnhubNewsSource().fetch(["AAA", "BBB", "CCC"], {"news": {}}, env)

    assert fake_client.tickers_called == ["AAA", "BBB", "CCC"]
    assert len(fake_sleep) == 2  # paced between the 3 calls, not before/after
    assert all(s == pytest.approx(finnhub_news._SECONDS_PER_CALL) for s in fake_sleep)


def test_finnhub_skips_entirely_without_an_api_key(fake_client, fake_sleep):
    env = SimpleNamespace(finnhub_api_key="")
    result = finnhub_news.FinnhubNewsSource().fetch(["AAA", "BBB"], {"news": {}}, env)

    assert result.empty
    assert fake_client.tickers_called == []  # never even tried
    assert fake_sleep == []


def test_polygon_paces_between_calls_at_the_5_per_minute_rate(fake_client, monkeypatch):
    calls = []
    monkeypatch.setattr(polygon_news.time, "sleep", lambda s: calls.append(s))
    env = SimpleNamespace(polygon_api_key="fake-key")

    polygon_news.PolygonNewsSource().fetch(["AAA", "BBB", "CCC"], {"news": {}}, env)

    assert fake_client.tickers_called == ["AAA", "BBB", "CCC"]
    assert len(calls) == 2
    assert all(s == pytest.approx(polygon_news._SECONDS_PER_CALL) for s in calls)
    # Polygon's interval must respect its far tighter 5/min limit -- 12s+,
    # not accidentally reusing Finnhub's ~1s pacing.
    assert polygon_news._SECONDS_PER_CALL > 10


def test_polygon_skips_entirely_without_an_api_key(fake_client, monkeypatch):
    calls = []
    monkeypatch.setattr(polygon_news.time, "sleep", lambda s: calls.append(s))
    env = SimpleNamespace(polygon_api_key="")

    result = polygon_news.PolygonNewsSource().fetch(["AAA", "BBB"], {"news": {}}, env)

    assert result.empty
    assert fake_client.tickers_called == []
    assert calls == []
