"""Polygon ticker-news provider (optional; enabled by POLYGON_API_KEY).

Free tier is capped at 5 requests/minute (a Polygon.io/Massive.com account
limit -- Polygon rebranded to massive.com in 2026, keys and API are
unchanged -- not something this app controls). Calls are paced evenly at
that rate so a full-universe run actually gets through every ticker instead
of the first ~5 succeeding and every request after that silently failing
with a 429 (caught by the per-ticker try/except below, so it would look
like "no news today" rather than an obvious error).
"""
from __future__ import annotations

import logging
import time
from datetime import date, timedelta

import httpx
import pandas as pd

from src.data.base import DataSource, register_source
from src.data.news_intelligence import NEWS_COLUMNS, normalize_news_event

logger = logging.getLogger(__name__)
_BASE_URL = "https://api.polygon.io/v2/reference/news"
# 60s / 5 requests-per-minute, plus a small margin since each request's own
# network time also counts against the limit.
_SECONDS_PER_CALL = 60 / 5 + 0.5


@register_source("polygon_news")
class PolygonNewsSource(DataSource):
    table = "news_events"
    key_cols = ("ticker", "url")

    def fetch(self, tickers: list[str], settings: dict, env) -> pd.DataFrame:
        if not env.polygon_api_key:
            logger.info("Skipping Polygon news: POLYGON_API_KEY is not configured")
            return pd.DataFrame(columns=NEWS_COLUMNS)

        config = settings.get("news", {})
        start = date.today() - timedelta(days=int(config.get("lookback_days", 7)))
        limit = int(config.get("max_articles_per_ticker", 50))
        rows = []
        with httpx.Client(timeout=20) as client:
            for i, ticker in enumerate(tickers):
                if i > 0:
                    time.sleep(_SECONDS_PER_CALL)
                try:
                    response = client.get(
                        _BASE_URL,
                        params={
                            "ticker": ticker,
                            "published_utc.gte": start.isoformat(),
                            "order": "desc",
                            "limit": limit,
                            "apiKey": env.polygon_api_key,
                        },
                    )
                    response.raise_for_status()
                    articles = response.json().get("results", [])
                except Exception:
                    logger.warning("Polygon news fetch failed for %s", ticker, exc_info=True)
                    continue
                for article in articles:
                    headline, url = article.get("title"), article.get("article_url")
                    publisher = article.get("publisher") or {}
                    if headline and url:
                        rows.append(
                            normalize_news_event(
                                ticker=ticker,
                                headline=headline,
                                url=url,
                                published_at=article.get("published_utc"),
                                source=publisher.get("name"),
                                provider="polygon",
                            )
                        )
        return pd.DataFrame(rows, columns=NEWS_COLUMNS).drop_duplicates(subset=["ticker", "url"], keep="first")
