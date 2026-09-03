"""Shared normalization for all company-news providers.

Provider APIs disagree about field names, timestamps, and publisher metadata.
This module turns them into one auditable event shape before DuckDB sees them.
The classifier is deliberately rule-based for now: its output is transparent,
testable, and safe to improve later with a trained event model.
"""
from __future__ import annotations

import hashlib
import re

import pandas as pd

NEWS_COLUMNS = [
    "ticker",
    "url",
    "published_at",
    "headline",
    "source",
    "provider",
    "received_at",
    "event_type",
    "reliability_score",
    "duplicate_group",
]

_PROVIDER_RELIABILITY = {
    "polygon": 0.85,
    "finnhub": 0.80,
    "sec": 1.00,
    "company_ir": 0.95,
    "yahoo": 0.55,
}

_EVENT_PATTERNS = (
    ("insider_transaction", ("form 4", "insider sale", "insider buy", "ceo sells", "director sells")),
    ("earnings_guidance", ("earnings", "eps", "revenue", "guidance", "outlook", "forecast")),
    ("merger_acquisition", ("acquire", "acquisition", "merger", "takeover", "buyout")),
    ("regulatory", ("sec probe", "doj", "antitrust", "regulator", "regulatory", "investigation")),
    ("lawsuit", ("lawsuit", "sued", "litigation", "settlement")),
    ("product", ("launches", "launch", "unveils", "announces new", "product")),
    ("contract", ("contract", "partnership", "deal with", "award")),
)


def classify_event(headline: str) -> str:
    """Return one explainable broad event category for a headline."""
    text = headline.lower()
    for event_type, patterns in _EVENT_PATTERNS:
        if any(pattern in text for pattern in patterns):
            return event_type
    return "general_news"


def duplicate_group(headline: str) -> str:
    """Group syndicated retellings without discarding their source records."""
    normalized = re.sub(r"[^a-z0-9]+", " ", headline.lower()).strip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


def normalize_news_event(
    *,
    ticker: str,
    headline: str,
    url: str,
    published_at,
    source: str | None,
    provider: str,
) -> dict:
    """Create one storage-ready, provider-neutral news event."""
    provider = provider.lower()
    if isinstance(published_at, (int, float)):
        published_at = pd.to_datetime(published_at, unit="s", utc=True, errors="coerce")
    else:
        published_at = pd.to_datetime(published_at, utc=True, errors="coerce")
    return {
        "ticker": ticker.upper(),
        "url": url,
        "published_at": published_at,
        "headline": headline.strip(),
        "source": source or provider,
        "provider": provider,
        "received_at": pd.Timestamp.now(tz="UTC"),
        "event_type": classify_event(headline),
        "reliability_score": _PROVIDER_RELIABILITY.get(provider, 0.50),
        "duplicate_group": duplicate_group(headline),
    }
