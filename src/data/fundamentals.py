"""Fundamentals snapshot via yfinance's `Ticker.info`. One row per
ticker per ingest run (as_of = today) — the feature engine derives
point-in-time-safe features from whatever's latest as of each date."""
from __future__ import annotations

import logging

import pandas as pd
import yfinance as yf

from src.data.base import DataSource, register_source

logger = logging.getLogger(__name__)

# yfinance `.info` key -> our column name
_FIELDS = {
    "shortName": "company_name",
    "sector": "sector",
    "industry": "industry",
    "marketCap": "market_cap",
    "trailingPE": "pe_ratio",
    "forwardPE": "forward_pe",
    "priceToBook": "price_to_book",
    "revenueGrowth": "revenue_growth",
    "earningsGrowth": "earnings_growth",
    "grossMargins": "gross_margin",
    "operatingMargins": "operating_margin",
    "profitMargins": "profit_margin",
    "freeCashflow": "free_cash_flow",
    "totalDebt": "total_debt",
    "debtToEquity": "debt_to_equity",
    "returnOnEquity": "return_on_equity",
    "dividendYield": "dividend_yield",
}


@register_source("fundamentals")
class FundamentalsSource(DataSource):
    table = "fundamentals"
    key_cols = ("ticker", "as_of")

    def fetch(self, tickers: list[str], settings: dict, env) -> pd.DataFrame:
        today = pd.Timestamp.now("UTC").date()
        rows = []
        for t in tickers:
            try:
                info = yf.Ticker(t).info
            except Exception:
                logger.warning("Fundamentals fetch failed for %s", t, exc_info=True)
                continue
            if not info:
                continue
            row = {"ticker": t, "as_of": today}
            for src_key, col in _FIELDS.items():
                row[col] = info.get(src_key)
            rows.append(row)
        return pd.DataFrame(rows)
