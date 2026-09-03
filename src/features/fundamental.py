"""Fundamentals-derived features, joined onto price dates with a strict
as-of (backward) merge: a date T only ever sees the most recent fundamentals
snapshot with as_of <= T, never a later one. V1 only ingests the latest
snapshot (no fundamentals history), so backtest periods before the first
ingest date will show NaN here — expected, and fine, since XGBoost/LightGBM
both handle missing values natively.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

_PASSTHROUGH = [
    "pe_ratio", "forward_pe", "price_to_book", "revenue_growth", "earnings_growth",
    "gross_margin", "operating_margin", "profit_margin", "debt_to_equity",
    "return_on_equity", "dividend_yield",
]


def compute_fundamental_features(prices: pd.DataFrame, fundamentals: pd.DataFrame) -> pd.DataFrame:
    if fundamentals.empty:
        return pd.DataFrame(columns=["ticker", "date"])

    px = prices[["ticker", "date"]].copy()
    px["date"] = pd.to_datetime(px["date"])
    fnd = fundamentals.copy()
    fnd["as_of"] = pd.to_datetime(fnd["as_of"])

    cols = [c for c in _PASSTHROUGH + ["market_cap", "free_cash_flow"] if c in fnd.columns]

    frames = []
    for ticker, g in px.groupby("ticker"):
        fdata = fnd[fnd["ticker"] == ticker].sort_values("as_of")
        if fdata.empty:
            continue
        g = g.sort_values("date")
        merged = pd.merge_asof(g, fdata[["as_of"] + cols], left_on="date", right_on="as_of", direction="backward")
        merged = merged.drop(columns=["as_of"])

        if "market_cap" in merged.columns and "free_cash_flow" in merged.columns:
            merged["fcf_yield"] = merged["free_cash_flow"] / merged["market_cap"]
        if "market_cap" in merged.columns:
            merged["log_market_cap"] = np.log(merged["market_cap"].clip(lower=1))

        frames.append(merged)

    if not frames:
        return pd.DataFrame(columns=["ticker", "date"])
    return pd.concat(frames, ignore_index=True)
