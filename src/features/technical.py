"""Price-derived technical features: momentum, RSI, MACD, moving averages,
volatility, volume z-score. Every function here is strictly backward-looking
(rolling/ewm/pct_change/diff), so features at date T never see data from
after T — that's what makes the backtester's no-leakage guarantee hold.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _rsi(close: pd.Series, window: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _macd(close: pd.Series, fast: int, slow: int, signal: int):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line, macd_line - signal_line


def compute_technical_features(prices: pd.DataFrame, settings: dict) -> pd.DataFrame:
    """`prices` needs columns: ticker, date, close, volume (sorted by date is not required)."""
    cfg = settings["features"]
    frames = []
    for ticker, g in prices.sort_values("date").groupby("ticker"):
        g = g.sort_values("date").reset_index(drop=True)
        close = g["close"].astype(float)

        feat = pd.DataFrame({"ticker": ticker, "date": g["date"]})
        for w in cfg["momentum_windows"]:
            feat[f"momentum_{w}d"] = close.pct_change(w)

        feat["rsi"] = _rsi(close, cfg["rsi_window"])
        macd_line, signal_line, hist = _macd(close, cfg["macd_fast"], cfg["macd_slow"], cfg["macd_signal"])
        feat["macd"] = macd_line
        feat["macd_signal"] = signal_line
        feat["macd_hist"] = hist

        for w in cfg["ma_windows"]:
            ma = close.rolling(w).mean()
            feat[f"close_to_ma_{w}"] = close / ma - 1

        daily_ret = close.pct_change()
        feat["volatility"] = daily_ret.rolling(cfg["volatility_window"]).std() * np.sqrt(252)

        volume = g["volume"].astype(float)
        vol_mean = volume.rolling(cfg["volume_zscore_window"]).mean()
        vol_std = volume.rolling(cfg["volume_zscore_window"]).std()
        feat["volume_zscore"] = (volume - vol_mean) / vol_std.replace(0, np.nan)

        frames.append(feat)

    if not frames:
        return pd.DataFrame(columns=["ticker", "date"])
    return pd.concat(frames, ignore_index=True)
