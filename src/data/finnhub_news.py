"""Finnhub company-news provider (optional; enabled by FINNHUB_API_KEY).

Free tier is 60 requests/minute, comfortably above this app's universe size
today, but calls are still paced defensively -- a plain synchronous loop
would otherwise fire every request back-to-back, which is fine at 30
tickers but stops being fine the moment the universe grows past ~60.
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
_BASE_URL = "https://finnhub.io/api/v1/company-news"
# 60s / 60 requests-per-minute -- real network latency per call already
# pushes the achieved rate comfortably under the limit.
_SECONDS_PER_CALL = 60 / 60


@register_source("finnhub_news")
class FinnhubNewsSource(DataSource):
    table = "news_events"
    key_cols = ("ticker", "url")

    def fetch(self, tickers: list[str], settings: dict, env) -> pd.DataFrame:
        if not env.finnhub_api_key:
            logger.info("Skipping Finnhub news: FINNHUB_API_KEY is not configured")
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
                        params={"symbol": ticker, "from": start.isoformat(), "to": date.today().isoformat(), "token": env.finnhub_api_key},
                    )
                    response.raise_for_status()
                    articles = response.json()[:limit]
                except Exception:
                    logger.warning("Finnhub news fetch failed for %s", ticker, exc_info=True)
                    continue
                for article in articles:
                    headline, url = article.get("headline"), article.get("url")
                    if headline and url:
                        rows.append(
                            normalize_news_event(
                                ticker=ticker,
                                headline=headline,
                                url=url,
                                published_at=article.get("datetime"),
                                source=article.get("source"),
                                provider="finnhub",
                            )
                        )
        return pd.DataFrame(rows, columns=NEWS_COLUMNS).drop_duplicates(subset=["ticker", "url"], keep="first")
