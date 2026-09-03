"""Secondary headline ingestion via yfinance's ``Ticker.news``.

Yahoo is intentionally kept as a coverage and cross-check feed, not the
authoritative event feed. Every article is normalized with provider metadata,
event classification, and a duplicate group before storage.

yfinance's news payload shape has changed across releases (sometimes
top-level keys, sometimes nested under "content"), so extraction below is
defensive and skips anything it can't confidently parse rather than raising.
"""
from __future__ import annotations

import logging

import pandas as pd
import yfinance as yf

from src.data.base import DataSource, register_source
from src.data.news_intelligence import NEWS_COLUMNS, normalize_news_event

logger = logging.getLogger(__name__)


def _extract(item: dict) -> dict | None:
    content = item.get("content") if isinstance(item.get("content"), dict) else item
    headline = content.get("title")

    url = None
    canonical = content.get("canonicalUrl")
    if isinstance(canonical, dict):
        url = canonical.get("url")
    url = url or content.get("link") or item.get("link")

    if not headline or not url:
        return None

    pub = content.get("pubDate") or content.get("providerPublishTime") or item.get("providerPublishTime")
    if isinstance(pub, (int, float)):
        published_at = pd.to_datetime(pub, unit="s", utc=True, errors="coerce")
    else:
        published_at = pd.to_datetime(pub, utc=True, errors="coerce")

    source = None
    provider = content.get("provider")
    if isinstance(provider, dict):
        source = provider.get("displayName")
    source = source or content.get("publisher")

    return {"headline": headline, "url": url, "published_at": published_at, "source": source}


@register_source("news")
class NewsSource(DataSource):
    table = "news_events"
    key_cols = ("ticker", "url")

    def fetch(self, tickers: list[str], settings: dict, env) -> pd.DataFrame:
        rows = []
        for t in tickers:
            try:
                items = yf.Ticker(t).news or []
            except Exception:
                logger.warning("News fetch failed for %s", t, exc_info=True)
                continue
            for item in items:
                parsed = _extract(item)
                if parsed is None:
                    continue
                rows.append(normalize_news_event(ticker=t, provider="yahoo", **parsed))
        if not rows:
            return pd.DataFrame(columns=NEWS_COLUMNS)
        # The same article can legitimately show up under multiple tickers'
        # feeds (a syndicated piece mentioning several names) -- the key is
        # (ticker, url), not url alone, so each ticker keeps its own row.
        return pd.DataFrame(rows).drop_duplicates(subset=["ticker", "url"], keep="first")
