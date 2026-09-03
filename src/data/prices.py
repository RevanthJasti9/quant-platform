"""OHLCV price ingestion via yfinance. See base.py for how to add a
different price feed alongside/instead of this one."""
from __future__ import annotations

import logging

import pandas as pd
import yfinance as yf

from src.data.base import DataSource, register_source

logger = logging.getLogger(__name__)

_COLS = ["ticker", "date", "open", "high", "low", "close", "adj_close", "volume"]


@register_source("prices")
class PriceSource(DataSource):
    table = "prices"
    key_cols = ("ticker", "date")

    def fetch(self, tickers: list[str], settings: dict, env) -> pd.DataFrame:
        years = settings["data"]["price_history_years"]
        raw = yf.download(
            tickers,
            period=f"{years}y",
            auto_adjust=False,
            group_by="ticker",
            threads=True,
            progress=False,
        )

        frames = []
        for t in tickers:
            try:
                sub = raw[t] if len(tickers) > 1 else raw
            except (KeyError, TypeError):
                logger.warning("No price data returned for %s", t)
                continue
            sub = sub.dropna(how="all")
            if sub.empty or "Close" not in sub.columns:
                continue
            df = sub.reset_index()
            df = df.rename(
                columns={
                    "Date": "date",
                    "Open": "open",
                    "High": "high",
                    "Low": "low",
                    "Close": "close",
                    "Adj Close": "adj_close",
                    "Volume": "volume",
                }
            )
            df["ticker"] = t
            df["date"] = pd.to_datetime(df["date"]).dt.date
            frames.append(df[_COLS])

        if not frames:
            return pd.DataFrame(columns=_COLS)
        return pd.concat(frames, ignore_index=True).dropna(subset=["close"])
