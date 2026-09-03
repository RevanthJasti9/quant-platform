"""Backtest performance metrics: Sharpe, CAGR, max drawdown, win rate,
turnover, alpha vs benchmark, volatility, a calibration proxy (the
correlation between predicted score and realized forward return, i.e. the
information coefficient), and a market-regime breakdown so a strong
aggregate number can't hide a strategy that only worked in one regime.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.stats import ic_significance


def sharpe_ratio(daily_returns: pd.Series, periods_per_year: int = 252) -> float:
    if daily_returns.empty or daily_returns.std() == 0:
        return 0.0
    return float(daily_returns.mean() / daily_returns.std() * np.sqrt(periods_per_year))


def cagr(equity_curve: pd.Series, periods_per_year: int = 252) -> float:
    if len(equity_curve) < 2:
        return 0.0
    total_return = equity_curve.iloc[-1] / equity_curve.iloc[0]
    years = len(equity_curve) / periods_per_year
    if years <= 0 or total_return <= 0:
        return 0.0
    return float(total_return ** (1 / years) - 1)


def max_drawdown(equity_curve: pd.Series) -> float:
    running_max = equity_curve.cummax()
    drawdown = equity_curve / running_max - 1
    return float(drawdown.min())


def win_rate(daily_returns: pd.Series) -> float:
    if daily_returns.empty:
        return 0.0
    return float((daily_returns > 0).mean())


def volatility(daily_returns: pd.Series, periods_per_year: int = 252) -> float:
    return float(daily_returns.std() * np.sqrt(periods_per_year))


def information_coefficient(predicted: pd.Series, actual: pd.Series) -> float:
    if len(predicted) < 2:
        return 0.0
    corr = pd.Series(predicted).reset_index(drop=True).corr(pd.Series(actual).reset_index(drop=True))
    return float(corr) if pd.notna(corr) else 0.0


def _qcut_safe(scores: pd.Series, n_quantiles: int) -> pd.Series:
    try:
        return pd.qcut(scores, n_quantiles, labels=False, duplicates="drop")
    except ValueError:
        # Fewer distinct scores than quantiles requested (a genuinely flat
        # day) -- contributes nothing to the breakdown rather than raising.
        return pd.Series(np.nan, index=scores.index)


def quantile_returns(scored: pd.DataFrame, n_quantiles: int = 5) -> pd.DataFrame:
    """Buckets every (date, ticker) prediction into `n_quantiles` by score,
    computed WITHIN each date (cross-sectionally, comparing stocks against
    each other on the same day -- never across different market conditions),
    then averages realized forward return per bucket across all dates.

    This is a more direct, visual test of ranking value than a single IC
    number: a model with real skill should show mean return increasing
    roughly monotonically from quantile 1 (lowest score) to quantile N
    (highest). Flat or out-of-order buckets mean the model isn't actually
    distinguishing better stocks from worse ones on a given day, whatever a
    single correlation coefficient says. `scored` needs columns: date,
    score, forward_relative_return.
    """
    df = scored.dropna(subset=["score", "forward_relative_return"]).copy()
    if df.empty:
        return pd.DataFrame(columns=["quantile", "mean_return", "n"])

    df["quantile"] = df.groupby("date")["score"].transform(lambda s: _qcut_safe(s, n_quantiles))
    df = df.dropna(subset=["quantile"])
    if df.empty:
        return pd.DataFrame(columns=["quantile", "mean_return", "n"])
    df["quantile"] = df["quantile"].astype(int) + 1

    return (
        df.groupby("quantile")["forward_relative_return"]
        .agg(mean_return="mean", n="count")
        .reset_index()
        .sort_values("quantile")
        .reset_index(drop=True)
    )


def quantile_spread(quantile_df: pd.DataFrame) -> float | None:
    """Top-quantile-minus-bottom-quantile mean return -- what a model with
    real ranking skill should show as a clear positive spread (the return
    a long-top/short-bottom strategy on the model's own score would earn).
    Near zero or negative means the extremes of the model's own score
    aren't actually separating winners from losers.
    """
    if quantile_df.empty:
        return None
    return float(quantile_df["mean_return"].iloc[-1] - quantile_df["mean_return"].iloc[0])


def compute_all_metrics(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    turnover_series: pd.Series,
    predicted: pd.Series,
    actual: pd.Series,
) -> dict:
    port_equity = (1 + portfolio_returns).cumprod()
    bench_equity = (1 + benchmark_returns).cumprod()
    ic = information_coefficient(predicted, actual)
    return {
        "sharpe": sharpe_ratio(portfolio_returns),
        "cagr": cagr(port_equity),
        "max_drawdown": max_drawdown(port_equity),
        "win_rate": win_rate(portfolio_returns),
        "turnover": float(turnover_series.mean()) if not turnover_series.empty else 0.0,
        "alpha": cagr(port_equity) - cagr(bench_equity),
        "volatility": volatility(portfolio_returns),
        "calibration": ic,
        # Is this IC actually distinguishable from the zero-skill null hypothesis,
        # or is it what len(predicted) points of pure noise would produce some of
        # the time anyway? See src/stats.py's ic_significance docstring.
        "calibration_pvalue": ic_significance(ic, len(predicted)),
    }


def classify_regimes(benchmark_prices: pd.Series, trend_window: int = 60, vol_window: int = 20) -> pd.DataFrame:
    """One row per date in `benchmark_prices`' own index, with a bull/bear
    trend label and a high/low volatility label -- both derived only from
    the benchmark's OWN trailing price history up to and including that
    date, never later dates, so a date's regime label can't be influenced
    by what the market did after it. "High vol" means high relative to
    everything seen up to that date (an expanding median), not relative to
    the whole benchmark series including days not yet reached -- the
    latter would mean a date's label could retroactively change depending
    on how much history the backtest happens to run through, which defeats
    the point of a per-regime breakdown being a stable, reproducible read.
    """
    px = benchmark_prices.sort_index()
    trend = px.pct_change(trend_window)
    vol = px.pct_change().rolling(vol_window).std()
    vol_median = vol.expanding(min_periods=vol_window).median()

    regime_trend = pd.Series("unknown", index=px.index)
    regime_trend[trend >= 0] = "bull"
    regime_trend[trend < 0] = "bear"

    regime_vol = pd.Series("unknown", index=px.index)
    regime_vol[vol >= vol_median] = "high_vol"
    regime_vol[vol < vol_median] = "low_vol"

    return pd.DataFrame({"regime_trend": regime_trend, "regime_vol": regime_vol})


def regime_breakdown_metrics(
    portfolio_returns: pd.Series, benchmark_returns: pd.Series, regime_labels: pd.Series
) -> dict[str, dict]:
    """Per-regime performance for a set of already-simulated daily returns
    (positionally aligned with `regime_labels` -- same length, same order).
    A regime with too few simulated days to mean anything (< 5) is skipped
    rather than reported with a misleadingly precise-looking number.
    """
    breakdown = {}
    for regime in sorted(set(regime_labels) - {"unknown"}):
        mask = (regime_labels == regime).to_numpy()
        port = portfolio_returns[mask]
        bench = benchmark_returns[mask]
        if len(port) < 5:
            continue
        port_equity = (1 + port).cumprod()
        bench_equity = (1 + bench).cumprod()
        breakdown[regime] = {
            "n_days": int(mask.sum()),
            "sharpe": sharpe_ratio(port),
            "cagr": cagr(port_equity),
            "alpha": cagr(port_equity) - cagr(bench_equity),
            "win_rate": win_rate(port),
        }
    return breakdown
