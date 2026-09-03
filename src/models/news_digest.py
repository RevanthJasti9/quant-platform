"""Turns each ticker's recent headlines (already ingested by src/data/news.py
and friends but never otherwise used) into a short "what's happening" blurb
plus a sentiment score, via a local LLM, when available. Both come from one
prompt/response per ticker -- summarizes and scores only the headlines
already stored, never fetches full articles or adds outside facts.

The sentiment score is a genuinely different kind of thing than everything
else this app feeds into the model: it's the LLM's own read of the
headlines, not a deterministic, auditable calculation like the SHAP reasons
or the rule-based event features. Calibrated by hand against a few
known-positive/negative/neutral cases (see the system prompt below) but not
validated against a labeled dataset. Treat it as a candidate signal, not a
given one -- src/features/events.py:compute_news_sentiment_features()'s
docstring has the point-in-time details, and whether it actually earns a
place in the model is a feature-importance question, checked after
training, not assumed up front.
"""
from __future__ import annotations

import logging
import re

import pandas as pd

from src.config import get_settings
from src.data.db import get_connection, upsert_wide
from src.llm.client import generate_many, is_available

logger = logging.getLogger(__name__)

LOOKBACK_DAYS = 14
MAX_HEADLINES = 8

_SYSTEM = (
    "You analyze recent news headlines about a company for a non-expert investor. "
    "Respond in EXACTLY this format, nothing else:\n"
    "SENTIMENT: <a number from -1.0 (very negative) to 1.0 (very positive)>\n"
    "SUMMARY: <one or two plain sentences summarizing what has been happening>\n"
    "Score strictly: 0 means routine, procedural, or purely informational news with no "
    "clear positive or negative implication (a scheduled event, an office opening, a routine "
    "filing). Only move away from 0 for headlines that state an actual positive or negative "
    "outcome (beat/missed earnings, upgrade/downgrade, lawsuit, layoffs, a real business win "
    "or setback). Only use the headlines given -- never invent facts, numbers, or events not "
    "present in them."
)

_SENTIMENT_RE = re.compile(r"SENTIMENT:\s*(-?\d+(?:\.\d+)?)", re.IGNORECASE)
_SUMMARY_RE = re.compile(r"SUMMARY:\s*(.+)", re.IGNORECASE | re.DOTALL)


def parse_response(text: str) -> tuple[float | None, str | None]:
    """Pulls a clamped sentiment score and a summary out of the LLM's raw
    response. If the model didn't follow the SENTIMENT:/SUMMARY: format --
    small local models aren't perfectly reliable about it -- falls back to
    treating the whole response as the summary rather than losing it, so a
    formatting miss degrades to "no sentiment" instead of "no digest at all".
    """
    sentiment = None
    m = _SENTIMENT_RE.search(text)
    if m:
        try:
            sentiment = max(-1.0, min(1.0, float(m.group(1))))
        except ValueError:
            sentiment = None

    m = _SUMMARY_RE.search(text)
    summary = m.group(1).strip() if m else (text.strip() or None)
    return sentiment, summary


def _build_prompt(ticker: str, headlines: list[str]) -> str:
    bullets = "\n".join(f"- {h}" for h in headlines[:MAX_HEADLINES])
    return f"Recent headlines about {ticker}:\n{bullets}"


def build_news_digests(settings: dict | None = None) -> int:
    settings = settings or get_settings()
    if not is_available(settings):
        logger.info("No LLM available (Groq unconfigured and Ollama unreachable) — skipping news digests")
        return 0

    con = get_connection()
    cutoff = pd.Timestamp.now("UTC") - pd.Timedelta(days=LOOKBACK_DAYS)
    df = con.execute(
        """
        WITH ranked AS (
            SELECT ticker, headline, published_at,
                   ROW_NUMBER() OVER (
                       PARTITION BY ticker, COALESCE(duplicate_group, url)
                       ORDER BY reliability_score DESC NULLS LAST, published_at DESC
                   ) AS duplicate_rank
            FROM news_events
            WHERE published_at >= ?
        )
        SELECT ticker, headline
        FROM ranked
        WHERE duplicate_rank = 1
        ORDER BY ticker, published_at DESC
        """,
        [cutoff],
    ).fetchdf()
    con.close()  # closed before the slow LLM loop -- see predict.py for why:
    # holding DuckDB's single writer lock across many multi-second LLM calls
    # would block the live dashboard's own queries for the whole run.

    # Build every prompt first (fast, local, no I/O), then run them all through
    # generate_many() in one batch -- concurrent when Groq is configured, so
    # e.g. 31 tickers no longer means 31 sequential ~1s round trips.
    tickers: list[str] = []
    headline_counts: list[int] = []
    prompts: list[tuple[str, str | None]] = []
    for ticker, group in df.groupby("ticker"):
        headlines = group["headline"].tolist()
        tickers.append(ticker)
        headline_counts.append(len(headlines))
        prompts.append((_build_prompt(ticker, headlines), _SYSTEM))

    responses = generate_many(prompts, settings)

    digest_rows = []
    sentiment_rows = []
    for ticker, headline_count, response in zip(tickers, headline_counts, responses):
        sentiment, summary = parse_response(response) if response else (None, None)
        generated_at = pd.Timestamp.now("UTC")
        if summary:
            digest_rows.append(
                {
                    "ticker": ticker,
                    "generated_at": generated_at,
                    "summary": summary,
                    "headline_count": headline_count,
                    "sentiment_score": sentiment,
                }
            )
        if sentiment is not None:
            sentiment_rows.append(
                {
                    "ticker": ticker,
                    "as_of": generated_at.date(),
                    "sentiment_score": sentiment,
                    "headline_count": headline_count,
                    "generated_at": generated_at,
                }
            )

    if not digest_rows:
        return 0
    con = get_connection()  # reopened just for this write
    n = upsert_wide(con, "news_digests", pd.DataFrame(digest_rows), ("ticker",))
    if sentiment_rows:
        upsert_wide(con, "news_sentiment", pd.DataFrame(sentiment_rows), ("ticker", "as_of"))
    con.close()
    logger.info("Generated %d news digests (%d with a sentiment score)", n, len(sentiment_rows))
    return n


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print(build_news_digests())
