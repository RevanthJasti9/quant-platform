"""Official company investor-relations RSS/Atom feeds, configured per ticker."""
from __future__ import annotations

import logging
import xml.etree.ElementTree as ET

import httpx
import pandas as pd

from src.data.base import DataSource, register_source
from src.data.news_intelligence import NEWS_COLUMNS, normalize_news_event

logger = logging.getLogger(__name__)


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1].lower()


def _child_text(entry: ET.Element, name: str) -> str | None:
    for child in entry:
        if _local_name(child.tag) == name and child.text:
            return child.text.strip()
    return None


def _entry_url(entry: ET.Element) -> str | None:
    for child in entry:
        if _local_name(child.tag) != "link":
            continue
        return child.attrib.get("href") or (child.text.strip() if child.text else None)
    return None


def parse_press_release_feed(xml_text: str, ticker: str) -> list[dict]:
    """Parse the small common subset shared by RSS and Atom feeds."""
    root = ET.fromstring(xml_text)
    rows = []
    for entry in root.iter():
        if _local_name(entry.tag) not in {"item", "entry"}:
            continue
        headline = _child_text(entry, "title")
        url = _entry_url(entry)
        published_at = _child_text(entry, "pubdate") or _child_text(entry, "published") or _child_text(entry, "updated")
        if headline and url:
            rows.append(
                normalize_news_event(
                    ticker=ticker,
                    headline=headline,
                    url=url,
                    published_at=published_at,
                    source="Company investor relations",
                    provider="company_ir",
                )
            )
    return rows


@register_source("company_ir_news")
class CompanyInvestorRelationsNewsSource(DataSource):
    table = "news_events"
    key_cols = ("ticker", "url")

    def fetch(self, tickers: list[str], settings: dict, env) -> pd.DataFrame:
        feeds = settings.get("news", {}).get("company_press_release_feeds", {})
        rows = []
        with httpx.Client(timeout=20, follow_redirects=True) as client:
            for ticker in tickers:
                feed_url = feeds.get(ticker.upper())
                if not feed_url:
                    continue
                try:
                    response = client.get(feed_url, headers={"User-Agent": env.sec_edgar_user_agent})
                    response.raise_for_status()
                    rows.extend(parse_press_release_feed(response.text, ticker))
                except Exception:
                    logger.warning("Company IR feed fetch failed for %s", ticker, exc_info=True)
        return pd.DataFrame(rows, columns=NEWS_COLUMNS).drop_duplicates(subset=["ticker", "url"], keep="first")
