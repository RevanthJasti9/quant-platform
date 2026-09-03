"""src.llm.client -- Groq (cloud, free tier) as primary, local Ollama as a
fallback that's started on demand and stopped again afterward if this run
is what started it. Groq/Ollama I/O is mocked throughout; nothing here
makes a real network call or spawns a real process.
"""
from __future__ import annotations

import subprocess
import time

import pytest

from src.llm import client as llm

SETTINGS = {"llm": {"enabled": True, "model": "llama3.2:3b", "groq": {"model": "llama-3.3-70b-versatile"}}}
DISABLED_SETTINGS = {"llm": {"enabled": False}}


@pytest.fixture(autouse=True)
def _reset_module_state(monkeypatch):
    """_started_ollama_process/_ollama_used_this_run/_groq_rate_limiter are
    module-level (a batch run makes many generate() calls and should only
    start the server once, unload() reads _ollama_used_this_run at the end
    of the run, and the rate limiter is lazily created once) -- must not
    leak between tests.
    """
    monkeypatch.setattr(llm, "_started_ollama_process", None)
    monkeypatch.setattr(llm, "_ollama_used_this_run", False)
    monkeypatch.setattr(llm, "_groq_rate_limiter", None)
    yield


class _FakeProcess:
    def __init__(self):
        self.terminated = False
        self.killed = False

    def terminate(self):
        self.terminated = True

    def wait(self, timeout=None):
        pass

    def kill(self):
        self.killed = True


# ---------- is_available ----------


def test_not_available_when_disabled(monkeypatch):
    monkeypatch.setattr(llm, "_groq_key", lambda: "")
    assert llm.is_available(DISABLED_SETTINGS) is False


def test_available_when_groq_key_present_regardless_of_ollama(monkeypatch):
    # Groq-key path short-circuits before ever checking Ollama -- fail loudly if that stops being true.
    monkeypatch.setattr(llm, "_groq_key", lambda: "gsk_fake")
    monkeypatch.setattr(llm, "_ollama_reachable", lambda settings: pytest.fail("should not check Ollama"))
    monkeypatch.setattr(llm.shutil, "which", lambda name: pytest.fail("should not check Ollama"))
    assert llm.is_available(SETTINGS) is True


def test_available_when_no_groq_but_ollama_already_reachable(monkeypatch):
    monkeypatch.setattr(llm, "_groq_key", lambda: "")
    monkeypatch.setattr(llm, "_ollama_reachable", lambda settings: True)
    assert llm.is_available(SETTINGS) is True


def test_available_when_no_groq_and_ollama_unreachable_but_installed(monkeypatch):
    monkeypatch.setattr(llm, "_groq_key", lambda: "")
    monkeypatch.setattr(llm, "_ollama_reachable", lambda settings: False)
    monkeypatch.setattr(llm.shutil, "which", lambda name: "/usr/local/bin/ollama")
    assert llm.is_available(SETTINGS) is True


def test_not_available_when_no_groq_and_no_ollama_at_all(monkeypatch):
    monkeypatch.setattr(llm, "_groq_key", lambda: "")
    monkeypatch.setattr(llm, "_ollama_reachable", lambda settings: False)
    monkeypatch.setattr(llm.shutil, "which", lambda name: None)
    assert llm.is_available(SETTINGS) is False


# ---------- generate: routing between Groq and the Ollama fallback ----------


def test_generate_returns_none_when_disabled(monkeypatch):
    monkeypatch.setattr(llm, "_groq_generate", lambda *a: pytest.fail("should not be called"))
    assert llm.generate("prompt", DISABLED_SETTINGS) is None


def test_generate_uses_groq_when_key_present_and_it_succeeds(monkeypatch):
    monkeypatch.setattr(llm, "_groq_key", lambda: "gsk_fake")
    monkeypatch.setattr(llm, "_groq_generate", lambda prompt, system, settings: "groq says hi")
    monkeypatch.setattr(llm, "_ensure_ollama_running", lambda settings: pytest.fail("should not fall back"))
    assert llm.generate("prompt", SETTINGS) == "groq says hi"


def test_generate_falls_back_to_ollama_when_groq_call_fails(monkeypatch):
    monkeypatch.setattr(llm, "_groq_key", lambda: "gsk_fake")
    monkeypatch.setattr(llm, "_groq_generate", lambda prompt, system, settings: None)
    monkeypatch.setattr(llm, "_ensure_ollama_running", lambda settings: True)
    monkeypatch.setattr(llm, "_ollama_generate", lambda prompt, system, settings: "ollama fallback")
    assert llm.generate("prompt", SETTINGS) == "ollama fallback"


def test_generate_goes_straight_to_ollama_when_no_groq_key(monkeypatch):
    monkeypatch.setattr(llm, "_groq_key", lambda: "")
    monkeypatch.setattr(llm, "_groq_generate", lambda *a: pytest.fail("no key -- must not call Groq"))
    monkeypatch.setattr(llm, "_ensure_ollama_running", lambda settings: True)
    monkeypatch.setattr(llm, "_ollama_generate", lambda prompt, system, settings: "local only")
    assert llm.generate("prompt", SETTINGS) == "local only"


def test_generate_returns_none_when_fallback_cannot_start(monkeypatch):
    monkeypatch.setattr(llm, "_groq_key", lambda: "")
    monkeypatch.setattr(llm, "_ensure_ollama_running", lambda settings: False)
    assert llm.generate("prompt", SETTINGS) is None


# ---------- _ensure_ollama_running: the actual start-on-demand logic ----------


def test_ensure_ollama_running_skips_start_if_already_reachable(monkeypatch):
    monkeypatch.setattr(llm, "_ollama_reachable", lambda settings: True)
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: pytest.fail("should not spawn a process"))
    assert llm._ensure_ollama_running(SETTINGS) is True
    assert llm._started_ollama_process is None


def test_ensure_ollama_running_starts_it_when_not_reachable(monkeypatch):
    calls = {"reachable": 0}

    def fake_reachable(settings):
        calls["reachable"] += 1
        return calls["reachable"] > 1  # unreachable on the first check, reachable after "starting"

    monkeypatch.setattr(llm, "_ollama_reachable", fake_reachable)
    monkeypatch.setattr(llm.shutil, "which", lambda name: "/usr/local/bin/ollama")
    monkeypatch.setattr(llm.time, "sleep", lambda s: None)
    fake_proc = _FakeProcess()
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: fake_proc)

    assert llm._ensure_ollama_running(SETTINGS) is True
    assert llm._started_ollama_process is fake_proc


def test_ensure_ollama_running_fails_when_binary_missing(monkeypatch):
    monkeypatch.setattr(llm, "_ollama_reachable", lambda settings: False)
    monkeypatch.setattr(llm.shutil, "which", lambda name: None)
    assert llm._ensure_ollama_running(SETTINGS) is False
    assert llm._started_ollama_process is None


# ---------- unload: stop what we started, only unload-in-place what we didn't ----------


def test_unload_stops_the_process_this_run_started(monkeypatch):
    fake_proc = _FakeProcess()
    monkeypatch.setattr(llm, "_started_ollama_process", fake_proc)
    monkeypatch.setattr(llm, "_ollama_used_this_run", True)
    monkeypatch.setattr(
        llm.httpx, "post", lambda *a, **k: pytest.fail("must not also send the keep_alive call")
    )

    llm.unload(SETTINGS)

    assert fake_proc.terminated is True
    assert llm._started_ollama_process is None
    assert llm._ollama_used_this_run is False


def test_unload_just_unloads_model_when_ollama_was_used_but_not_started_by_this_run(monkeypatch):
    # e.g. Groq failed partway through and generate() fell back to an Ollama that was already running.
    monkeypatch.setattr(llm, "_ollama_used_this_run", True)
    calls = []
    monkeypatch.setattr(llm.httpx, "post", lambda url, **kwargs: calls.append((url, kwargs)))

    llm.unload(SETTINGS)

    assert len(calls) == 1
    assert "keep_alive" in calls[0][1]["json"]
    assert llm._ollama_used_this_run is False


def test_unload_does_nothing_when_ollama_was_never_used_this_run(monkeypatch):
    """Regression test: unload() used to unconditionally ping Ollama's
    unload endpoint any time the LLM feature was enabled, even on a run
    where Groq handled every call and Ollama was never touched -- caught by
    a live run where the log showed an Ollama POST despite Groq succeeding
    for the only generate() call made. That would unload a model this run
    never loaded, e.g. one the user has resident for unrelated use.
    """
    monkeypatch.setattr(llm.httpx, "post", lambda *a, **k: pytest.fail("must not touch Ollama at all"))
    llm.unload(SETTINGS)


def test_unload_does_nothing_when_disabled(monkeypatch):
    monkeypatch.setattr(llm.httpx, "post", lambda *a, **k: pytest.fail("should not be called"))
    llm.unload(DISABLED_SETTINGS)


# ---------- _SlidingWindowRateLimiter ----------


class _FakeClock:
    """Deterministic stand-in for time.monotonic/time.sleep -- lets the
    rate limiter tests exercise real 60-second-window logic without an
    actual 60-second test.
    """

    def __init__(self):
        self.now = 0.0
        self.sleep_calls: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleep_calls.append(seconds)
        self.now += seconds


def test_rate_limiter_allows_up_to_max_without_sleeping(monkeypatch):
    clock = _FakeClock()
    monkeypatch.setattr(llm.time, "monotonic", clock.monotonic)
    monkeypatch.setattr(llm.time, "sleep", clock.sleep)
    limiter = llm._SlidingWindowRateLimiter(max_per_minute=3)

    limiter.acquire()
    limiter.acquire()
    limiter.acquire()

    assert clock.sleep_calls == []


def test_rate_limiter_blocks_once_at_capacity(monkeypatch):
    clock = _FakeClock()
    monkeypatch.setattr(llm.time, "monotonic", clock.monotonic)
    monkeypatch.setattr(llm.time, "sleep", clock.sleep)
    limiter = llm._SlidingWindowRateLimiter(max_per_minute=2)

    limiter.acquire()
    limiter.acquire()
    limiter.acquire()  # 3rd call at capacity -- must wait for the window to free up

    assert clock.sleep_calls  # slept at least once
    assert clock.now >= 60  # only freed up once the oldest timestamp fell outside the 60s window


def test_rate_limiter_does_not_block_after_the_window_passes(monkeypatch):
    clock = _FakeClock()
    monkeypatch.setattr(llm.time, "monotonic", clock.monotonic)
    monkeypatch.setattr(llm.time, "sleep", clock.sleep)
    limiter = llm._SlidingWindowRateLimiter(max_per_minute=2)

    limiter.acquire()
    limiter.acquire()
    clock.now += 61  # simulate real time passing without going through acquire()
    limiter.acquire()  # should see the window has already cleared and not need to sleep

    assert clock.sleep_calls == []


def test_get_groq_rate_limiter_uses_configured_limit_and_is_a_singleton():
    settings = {"llm": {"groq": {"max_requests_per_minute": 7}}}
    a = llm._get_groq_rate_limiter(settings)
    b = llm._get_groq_rate_limiter(settings)
    assert a is b
    assert a._max == 7


def test_get_groq_rate_limiter_defaults_when_unconfigured():
    limiter = llm._get_groq_rate_limiter(SETTINGS)
    assert limiter._max == llm.DEFAULT_GROQ_MAX_REQUESTS_PER_MINUTE


# ---------- generate_many ----------


def test_generate_many_empty_list_returns_empty():
    assert llm.generate_many([], SETTINGS) == []


def test_generate_many_falls_back_to_sequential_without_groq_key(monkeypatch):
    """No Groq key means every call would go through local Ollama, which is
    CPU/RAM-bound on this machine -- concurrency there would cause
    contention, not a speedup, so this must not spin up a thread pool.
    """
    monkeypatch.setattr(llm, "_groq_key", lambda: "")
    calls = []

    def fake_generate(prompt, settings, system=None):
        calls.append(prompt)
        return f"response to {prompt}"

    monkeypatch.setattr(llm, "generate", fake_generate)
    items = [("p1", "s1"), ("p2", "s2"), ("p3", None)]

    results = llm.generate_many(items, SETTINGS)

    assert results == ["response to p1", "response to p2", "response to p3"]
    assert calls == ["p1", "p2", "p3"]  # sequential, in submission order


def test_generate_many_runs_concurrently_and_preserves_input_order(monkeypatch):
    monkeypatch.setattr(llm, "_groq_key", lambda: "gsk_fake")

    delays = {"p1": 0.06, "p2": 0.01, "p3": 0.03}

    def fake_generate(prompt, settings, system=None):
        time.sleep(delays[prompt])
        return f"response to {prompt}"

    monkeypatch.setattr(llm, "generate", fake_generate)
    items = [("p1", None), ("p2", None), ("p3", None)]

    start = time.monotonic()
    results = llm.generate_many(items, SETTINGS, max_workers=3)
    elapsed = time.monotonic() - start

    # Order matches input order, not completion order (p2 finishes first, p1 last).
    assert results == ["response to p1", "response to p2", "response to p3"]
    # Genuinely concurrent: close to the slowest single call, not the sum of all three.
    assert elapsed < sum(delays.values())


def test_generate_many_one_failure_does_not_break_the_batch(monkeypatch):
    monkeypatch.setattr(llm, "_groq_key", lambda: "gsk_fake")

    def fake_generate(prompt, settings, system=None):
        if prompt == "p2":
            raise RuntimeError("simulated failure")
        return f"response to {prompt}"

    monkeypatch.setattr(llm, "generate", fake_generate)
    items = [("p1", None), ("p2", None), ("p3", None)]

    results = llm.generate_many(items, SETTINGS, max_workers=3)

    assert results == ["response to p1", None, "response to p3"]
