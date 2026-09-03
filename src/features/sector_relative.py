"""Relative-strength features: how a ticker's trailing return compares to
the benchmark (SPY) and to its sector peers over the same window. Comparing
a stock only to its own history is weaker than knowing how it's doing
relative to its sector and the broader market.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_sector_relative_features(
    prices: pd.DataFrame,
    sector_map: dict[str, str],
    benchmark: str,
    windows: list[int],
) -> pd.DataFrame:
    px = prices.pivot(index="date", columns="ticker", values="close").sort_index()
    returns = {w: px.pct_change(w) for w in windows}

    tickers = [t for t in px.columns if t != benchmark]
    frames = []
    for t in tickers:
        sector = sector_map.get(t)
        peers = [p for p in tickers if p != t and sector_map.get(p) == sector] if sector else []

        feat = pd.DataFrame({"date": px.index, "ticker": t})
        for w in windows:
            r = returns[w]
            feat[f"rel_return_{w}d_vs_benchmark"] = (r[t] - r[benchmark]).values if benchmark in r.columns else np.nan
            feat[f"rel_return_{w}d_vs_sector"] = (r[t] - r[peers].mean(axis=1)).values if peers else np.nan
        frames.append(feat)

    if not frames:
        return pd.DataFrame(columns=["ticker", "date"])
    return pd.concat(frames, ignore_index=True)
