import numpy as np
import pandas as pd

from src.features.technical import compute_technical_features

SETTINGS = {
    "features": {
        "momentum_windows": [5, 10],
        "rsi_window": 14,
        "macd_fast": 12,
        "macd_slow": 26,
        "macd_signal": 9,
        "ma_windows": [10, 20],
        "volatility_window": 20,
        "volume_zscore_window": 20,
    }
}


def _synthetic_prices(n=80, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2023-01-01", periods=n, freq="B").date
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    volume = rng.integers(1_000_000, 5_000_000, n)
    return pd.DataFrame({"ticker": "TEST", "date": dates, "close": close, "volume": volume})


def test_features_are_finite_after_warmup():
    df = _synthetic_prices()
    feat = compute_technical_features(df, SETTINGS)
    tail = feat.iloc[30:]
    assert tail.drop(columns=["ticker", "date"]).notna().all().all()


def test_features_do_not_use_future_data():
    """Truncating the price series should not change any already-computed
    feature value at an earlier date -- if it did, that feature would be
    peeking at data from beyond its own date.
    """
    df = _synthetic_prices()
    feat_full = compute_technical_features(df, SETTINGS)

    truncated = df.iloc[:50].copy()
    feat_truncated = compute_technical_features(truncated, SETTINGS)

    shared = feat_truncated.merge(feat_full, on=["ticker", "date"], suffixes=("_trunc", "_full"))
    assert len(shared) == len(feat_truncated)
    for col in [c for c in feat_full.columns if c not in ("ticker", "date")]:
        pd.testing.assert_series_equal(
            shared[f"{col}_trunc"], shared[f"{col}_full"], check_names=False, atol=1e-9
        )


def test_adjusted_prices_remove_false_split_momentum():
    """The data boundary must pass adjusted close into technical features."""
    df = _synthetic_prices(n=30)
    df["adj_close"] = df["close"]
    df.loc[df.index >= 20, "close"] *= 0.5

    raw_features = compute_technical_features(df[["ticker", "date", "close", "volume"]], SETTINGS)
    adjusted_features = compute_technical_features(
        df.assign(close=df["adj_close"])[["ticker", "date", "close", "volume"]], SETTINGS
    )
    assert raw_features.loc[20, "momentum_5d"] != adjusted_features.loc[20, "momentum_5d"]
