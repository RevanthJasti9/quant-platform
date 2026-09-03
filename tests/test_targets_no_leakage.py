import numpy as np
import pandas as pd

from src.models.targets import build_targets


def _synthetic_prices(n=60):
    dates = pd.date_range("2023-01-01", periods=n, freq="B").date
    frames = []
    for ticker, base, seed in [("AAA", 100, 1), ("BENCH", 50, 2)]:
        rng = np.random.default_rng(seed)
        close = base + np.cumsum(rng.normal(0, 1, n))
        frames.append(pd.DataFrame({"ticker": ticker, "date": dates, "close": close}))
    return pd.concat(frames, ignore_index=True)


def test_targets_drop_rows_without_enough_future_data():
    prices = _synthetic_prices(n=60)
    horizon = 10
    targets = build_targets(prices, benchmark="BENCH", horizons=[horizon])
    max_price_date = prices["date"].max()
    # the trailing `horizon` trading days can't have a label yet -- no future price exists
    assert targets["date"].max() < max_price_date


def test_target_matches_manual_forward_return():
    prices = _synthetic_prices(n=60)
    horizon = 5
    targets = build_targets(prices, benchmark="BENCH", horizons=[horizon])

    px = prices.pivot(index="date", columns="ticker", values="close").sort_index()
    idx = 10
    d0, d1 = px.index[idx], px.index[idx + horizon]
    expected = (px["AAA"].loc[d1] / px["AAA"].loc[d0] - 1) - (px["BENCH"].loc[d1] / px["BENCH"].loc[d0] - 1)

    row = targets[(targets["ticker"] == "AAA") & (targets["date"] == d0)]
    assert not row.empty
    assert abs(row["forward_relative_return"].iloc[0] - expected) < 1e-9
