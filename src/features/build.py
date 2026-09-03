"""Orchestrates the feature engine: reads prices + fundamentals from
DuckDB, runs technical/fundamental/sector-relative feature functions, joins
them into one wide row per (ticker, date), and upserts into `features`.
"""
from __future__ import annotations

import logging

import pandas as pd

from src.config import get_settings
from src.data.db import get_connection, upsert_wide
from src.features.events import (
    compute_filing_event_features,
    compute_insider_event_features,
    compute_news_event_features,
    compute_news_sentiment_features,
)
from src.features.fundamental import compute_fundamental_features
from src.features.sector_relative import compute_sector_relative_features
from src.features.technical import compute_technical_features

logger = logging.getLogger(__name__)


def _safe_event_features(label: str, fn, *args) -> pd.DataFrame:
    """Event features are supplementary, not load-bearing like prices --
    a bug in one shouldn't take down technical/fundamental features (which
    predictions can't run without at all). Falls back to no signal from
    that source rather than aborting the whole feature build.
    """
    try:
        return fn(*args)
    except Exception:
        logger.exception("%s feature computation failed, continuing without it", label)
        return pd.DataFrame(columns=["ticker", "date"])


def build_features() -> int:
    settings = get_settings()
    con = get_connection()
    # Keep raw close in storage for audit/display, but derive every return-based
    # feature from split/dividend-adjusted prices. Older rows may predate the
    # adj_close column, so safely fall back to close rather than dropping them.
    prices = con.execute(
        "SELECT ticker, date, COALESCE(adj_close, close) AS close, volume FROM prices ORDER BY ticker, date"
    ).fetchdf()
    fundamentals = con.execute("SELECT * FROM fundamentals").fetchdf()
    news = con.execute(
        "SELECT ticker, url, received_at, event_type, reliability_score, duplicate_group FROM news_events"
    ).fetchdf()
    insider = con.execute(
        "SELECT ticker, transaction_date, filing_date, transaction_code, shares, price, value FROM insider_transactions"
    ).fetchdf()
    filings = con.execute("SELECT ticker, filing_type, filing_date FROM sec_filings").fetchdf()
    sentiment = con.execute("SELECT ticker, as_of, sentiment_score FROM news_sentiment").fetchdf()
    con.close()

    if prices.empty:
        logger.warning("No price data found — run ingest before building features")
        return 0

    technical = compute_technical_features(prices, settings)
    fund_feat = compute_fundamental_features(prices, fundamentals)
    # Extends each event feature's output through the latest price date even
    # for tickers whose most recent event is much older -- otherwise a
    # ticker with a single 8-K three months ago would silently stop getting
    # any rolling-count/recency values the day after that filing.
    latest_price_date = prices["date"].max()
    news_feat = _safe_event_features("news", compute_news_event_features, news, latest_price_date)
    insider_feat = _safe_event_features("insider", compute_insider_event_features, insider, latest_price_date)
    filing_feat = _safe_event_features("SEC filing", compute_filing_event_features, filings, latest_price_date)
    sentiment_feat = _safe_event_features("news sentiment", compute_news_sentiment_features, sentiment, prices)

    sector_map = (
        fundamentals.sort_values("as_of").groupby("ticker")["sector"].last().to_dict()
        if not fundamentals.empty
        else {}
    )
    benchmark = settings["benchmark"]
    sector_feat = compute_sector_relative_features(
        prices, sector_map, benchmark, settings["features"]["momentum_windows"][:2]
    )

    merged = technical.merge(fund_feat, on=["ticker", "date"], how="left")
    merged = merged.merge(sector_feat, on=["ticker", "date"], how="left")
    merged = merged.merge(news_feat, on=["ticker", "date"], how="left")
    merged = merged.merge(insider_feat, on=["ticker", "date"], how="left")
    merged = merged.merge(filing_feat, on=["ticker", "date"], how="left")
    merged = merged.merge(sentiment_feat, on=["ticker", "date"], how="left")
    merged = merged[merged["ticker"] != benchmark].reset_index(drop=True)

    con = get_connection()
    n = upsert_wide(con, "features", merged, ("ticker", "date"))
    con.close()
    logger.info("Built %d feature rows", n)
    return n


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print(build_features())
