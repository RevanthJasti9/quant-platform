"""LLM client with Groq (cloud, free tier) as primary and local Ollama as
fallback. Entirely optional: nothing in this app requires an LLM to
function -- it's only used to turn data the pipeline already computed into
plain English. Every call here fails soft (returns None / False) rather
than raising, so a missing key and a stopped Ollama install together just
degrade the app to its structured (chip/number) display.

Groq is tried first when GROQ_API_KEY is set: it's free (see the daily-limit
math in the session that added this -- Groq's no-card tier comfortably
covers this app's real call volume, unlike OpenRouter's 50/day default),
costs no local RAM, and its hosted models are larger than the local 3B
fallback. Local Ollama is used only when Groq isn't configured or a call to
it fails, and even then only started on demand: generate() starts the local
Ollama server itself if it isn't already running, and unload() stops it
again afterward IF this run is what started it -- an Ollama instance the
user already had running for their own reasons is never touched.
"""
from __future__ import annotations

import concurrent.futures
import logging
import shutil
import subprocess
import threading
import time

import httpx

from src.config import get_env

logger = logging.getLogger(__name__)

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
OLLAMA_STARTUP_TIMEOUT_SECONDS = 10
OLLAMA_STARTUP_POLL_SECONDS = 0.5
DEFAULT_GROQ_MAX_REQUESTS_PER_MINUTE = 25  # Groq's free tier is 30 RPM; a
# small safety margin absorbs clock-skew at the edge of the sliding window.
DEFAULT_MAX_CONCURRENT_LLM_CALLS = 8

_started_ollama_process: subprocess.Popen | None = None
_ollama_used_this_run = False  # set by _ollama_generate; read by unload() so a
# run where Groq handled everything never pings Ollama's unload endpoint --
# doing so unconditionally would unload a model this run never loaded, e.g.
# one the user has resident for their own unrelated use of Ollama.


def _cfg(settings: dict) -> dict:
    return settings.get("llm", {})


def _groq_cfg(settings: dict) -> dict:
    return _cfg(settings).get("groq", {})


# ---------- Groq (primary) ----------


def _groq_key() -> str:
    return get_env().groq_api_key


class _SlidingWindowRateLimiter:
    """Thread-safe: blocks the calling thread until issuing another call
    keeps the count within max_per_minute over any trailing 60s window.
    Needed because generate_many() runs multiple Groq calls concurrently --
    without this, a thread pool would submit far faster than one request
    per ~2s and blow through Groq's real quota in seconds, turning a speedup
    into a wall of 429s.
    """

    def __init__(self, max_per_minute: int):
        self._max = max_per_minute
        self._lock = threading.Lock()
        self._timestamps: list[float] = []

    def acquire(self) -> None:
        while True:
            with self._lock:
                now = time.monotonic()
                self._timestamps = [t for t in self._timestamps if now - t < 60]
                if len(self._timestamps) < self._max:
                    self._timestamps.append(now)
                    return
                wait_for = 60 - (now - self._timestamps[0])
            time.sleep(max(wait_for, 0.05))


_groq_rate_limiter: _SlidingWindowRateLimiter | None = None
_groq_rate_limiter_lock = threading.Lock()


def _get_groq_rate_limiter(settings: dict) -> _SlidingWindowRateLimiter:
    global _groq_rate_limiter
    with _groq_rate_limiter_lock:
        if _groq_rate_limiter is None:
            max_per_minute = _groq_cfg(settings).get("max_requests_per_minute", DEFAULT_GROQ_MAX_REQUESTS_PER_MINUTE)
            _groq_rate_limiter = _SlidingWindowRateLimiter(max_per_minute)
        return _groq_rate_limiter


def _groq_generate(prompt: str, system: str | None, settings: dict) -> str | None:
    cfg = _groq_cfg(settings)
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    _get_groq_rate_limiter(settings).acquire()
    try:
        resp = httpx.post(
            GROQ_URL,
            headers={"Authorization": f"Bearer {_groq_key()}"},
            json={"model": cfg.get("model", "llama-3.3-70b-versatile"), "messages": messages, "temperature": 0.3},
            timeout=cfg.get("timeout_seconds", 20),
        )
        resp.raise_for_status()
        text = resp.json()["choices"][0]["message"]["content"].strip()
        return text or None
    except Exception:
        logger.warning("Groq generate() failed, falling back to local Ollama", exc_info=True)
        return None


# ---------- local Ollama (fallback, started/stopped on demand) ----------


def _ollama_reachable(settings: dict) -> bool:
    cfg = _cfg(settings)
    try:
        resp = httpx.get(f"{cfg.get('base_url', 'http://localhost:11434')}/api/version", timeout=3)
        return resp.status_code == 200
    except Exception:
        return False


def _ensure_ollama_running(settings: dict) -> bool:
    """Starts the local Ollama server if it isn't already reachable.
    Returns whether it's reachable after this call. Records whether THIS
    call is what started it (module-level, since a batch run makes many
    generate() calls and should only start the server once) so unload()
    knows whether it's safe to stop it afterward.
    """
    global _started_ollama_process
    if _ollama_reachable(settings):
        return True
    if shutil.which("ollama") is None:
        logger.warning("Ollama isn't installed -- no local fallback available")
        return False
    try:
        _started_ollama_process = subprocess.Popen(
            ["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
    except OSError:
        logger.warning("Failed to start local Ollama server", exc_info=True)
        return False
    deadline = time.monotonic() + OLLAMA_STARTUP_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        if _ollama_reachable(settings):
            logger.info("Started local Ollama server as a fallback")
            return True
        time.sleep(OLLAMA_STARTUP_POLL_SECONDS)
    logger.warning("Started Ollama but it didn't come up within %ss", OLLAMA_STARTUP_TIMEOUT_SECONDS)
    return False


def _ollama_generate(prompt: str, system: str | None, settings: dict) -> str | None:
    global _ollama_used_this_run
    _ollama_used_this_run = True
    cfg = _cfg(settings)
    payload = {"model": cfg.get("model", "llama3.2:3b"), "prompt": prompt, "stream": False}
    if system:
        payload["system"] = system
    try:
        resp = httpx.post(
            f"{cfg.get('base_url', 'http://localhost:11434')}/api/generate",
            json=payload,
            timeout=cfg.get("timeout_seconds", 30),
        )
        resp.raise_for_status()
        text = resp.json().get("response", "").strip()
        return text or None
    except Exception:
        logger.warning("Local Ollama generate() failed too, continuing without it", exc_info=True)
        return None


# ---------- public interface ----------


def is_available(settings: dict) -> bool:
    """True if there's some way to generate text right now: a Groq key is
    configured, Ollama is already running, or Ollama could be started (the
    binary exists) -- checked cheaply (no process spawned) so callers can
    gate work on this before building any prompts.
    """
    cfg = _cfg(settings)
    if not cfg.get("enabled", False):
        return False
    if _groq_key():
        return True
    return _ollama_reachable(settings) or shutil.which("ollama") is not None


def generate(prompt: str, settings: dict, system: str | None = None) -> str | None:
    cfg = _cfg(settings)
    if not cfg.get("enabled", False):
        return None
    if _groq_key():
        text = _groq_generate(prompt, system, settings)
        if text is not None:
            return text
        # fall through to local Ollama below
    if not _ensure_ollama_running(settings):
        return None
    return _ollama_generate(prompt, system, settings)


def generate_many(
    items: list[tuple[str, str | None]],
    settings: dict,
    max_workers: int = DEFAULT_MAX_CONCURRENT_LLM_CALLS,
) -> list[str | None]:
    """Runs generate() for each (prompt, system) pair concurrently instead
    of one at a time -- worth doing here specifically because these are
    independent, unrelated calls (one per ticker) that spend most of their
    wall-clock time waiting on network I/O, not local CPU. Results come
    back in the same order as `items`, regardless of completion order.

    Safe to call with many items at once: _groq_generate's rate limiter is
    shared across every worker thread, so this can't exceed Groq's real
    per-minute quota just because more requests are in flight -- it only
    changes how many are queued up waiting for their turn, not how fast
    they're actually allowed to fire.

    Falls back to plain sequential execution (no thread pool at all) when
    no Groq key is configured. In that case every call would go through
    local Ollama, which is CPU/RAM-bound on this machine, not I/O-bound --
    running several inference calls at once would cause contention and
    could easily be slower than one at a time, not faster.
    """
    if not items:
        return []
    if not _groq_key():
        return [generate(prompt, settings, system=system) for prompt, system in items]

    results: list[str | None] = [None] * len(items)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_to_index = {
            pool.submit(generate, prompt, settings, system): i for i, (prompt, system) in enumerate(items)
        }
        for future in concurrent.futures.as_completed(future_to_index):
            i = future_to_index[future]
            try:
                results[i] = future.result()
            except Exception:
                logger.warning("generate_many: item %d raised", i, exc_info=True)
                results[i] = None
    return results


def unload(settings: dict) -> None:
    """Releases whatever local LLM resources this run actually used -- and
    nothing it didn't. Three cases:
      1. This run started the local Ollama server (see _ensure_ollama_running):
         fully stop it. No reason to leave a freshly-started ~2.5GB process
         resident once the run is done.
      2. This run used an Ollama that was already running (didn't start it,
         e.g. Groq failed partway through and it fell back): just unload the
         model from memory (Ollama's own keep_alive=0) -- that Ollama
         instance isn't ours to shut down, matching the original behavior.
      3. This run never touched Ollama at all (Groq handled everything, or
         the LLM feature never triggered): do nothing. Pinging Ollama's
         unload endpoint here would unload a model this run never loaded --
         e.g. one the user has resident for their own unrelated use.
    """
    global _started_ollama_process, _ollama_used_this_run
    if _started_ollama_process is not None:
        proc, _started_ollama_process = _started_ollama_process, None
        _ollama_used_this_run = False
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        logger.info("Stopped the local Ollama server this run started")
        return

    if not _ollama_used_this_run:
        return
    _ollama_used_this_run = False

    cfg = _cfg(settings)
    if not cfg.get("enabled", False):
        return
    try:
        httpx.post(
            f"{cfg.get('base_url', 'http://localhost:11434')}/api/generate",
            json={"model": cfg.get("model", "llama3.2:3b"), "keep_alive": 0},
            timeout=10,
        )
        logger.info("Unloaded LLM (%s) from memory", cfg.get("model", "llama3.2:3b"))
    except Exception:
        logger.debug("LLM unload call failed (likely already unloaded)", exc_info=True)
