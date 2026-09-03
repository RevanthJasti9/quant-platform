"""Label construction: forward relative return vs the benchmark, per
horizon. These are labels, not features — never join this output into the
`features` table. Rows whose forward window runs past the available price
history are dropped (their label would need data that doesn't exist yet).
"""
from __future__ import annotations

import pandas as pd


def build_targets(prices: pd.DataFrame, benchmark: str, horizons: list[int]) -> pd.DataFrame:
    """`prices` needs columns: ticker, date, close."""
    px = prices.pivot(index="date", columns="ticker", values="close").sort_index()
    bench = px[benchmark] if benchmark in px.columns else None

    frames = []
    for h in horizons:
        fwd = px.shift(-h) / px - 1
        bench_fwd = (bench.shift(-h) / bench - 1) if bench is not None else None
        for t in px.columns:
            if t == benchmark:
                continue
            rel = fwd[t] - bench_fwd if bench_fwd is not None else fwd[t]
            frames.append(
                pd.DataFrame(
                    {
                        "date": px.index,
                        "ticker": t,
                        "horizon_days": h,
                        "forward_relative_return": rel.values,
                    }
                )
            )

    out = pd.concat(frames, ignore_index=True).dropna(subset=["forward_relative_return"])
    out["outperform"] = (out["forward_relative_return"] > 0).astype(int)
    return out.reset_index(drop=True)
