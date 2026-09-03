"""Walk-forward backtester. Built and tested before the models got any more
complex than a plain XGBoost+LightGBM ensemble, per the project's own
"backtester first" rule.

Each fold trains fresh on a rolling window, skips a purge gap of
`models.purge_gap_days` trading days (>= the label horizon, so no test-fold
label ever overlaps data the model trained on), then simulates a top-N
equal-weight long book on the untouched test window, rebalanced every
`backtest.rebalance_days` trading days with transaction costs + slippage
charged on turnover. Reuses `src.models.train._fit_horizon` so backtest
models and production models are trained by identical code.

Selection at each rebalance respects two risk controls (see select_top_n):
a per-sector cap, so a book can't end up all-Technology just because tech
names scored highest that day, and a liquidity floor on trailing dollar
volume, so a name the simulation "buys" is one that could plausibly be
traded at that size without the flat slippage assumption understating the
real cost.
"""
from __future__ import annotations

import json
import logging

import pandas as pd

from src.backtesting.metrics import (
    classify_regimes,
    compute_all_metrics,
    quantile_returns,
    quantile_spread,
    regime_breakdown_metrics,
)
from src.config import get_settings
from src.data.db import get_connection
from src.models.ensemble import ensemble_mean
from src.models.targets import build_targets
from src.models.train import REGRESSOR_NAMES, _fit_horizon, get_feature_cols

logger = logging.getLogger(__name__)


def _build_folds(dates: list, settings: dict) -> list[tuple]:
    bt = settings["backtest"]
    train_window = int(bt["train_window_years"] * 252)
    test_window = int(round(bt["test_window_months"] / 12 * 252))
    step = int(round(bt["step_months"] / 12 * 252))
    purge = settings["models"]["purge_gap_days"]

    n = len(dates)
    folds = []
    i = train_window
    while i + purge + test_window <= n:
        train_start_idx = max(0, i - train_window)
        train_end_idx = i - 1
        test_start_idx = i + purge
        test_end_idx = min(n, test_start_idx + test_window) - 1
        folds.append((dates[train_start_idx], dates[train_end_idx], dates[test_start_idx], dates[test_end_idx]))
        i += step
    return folds


def select_top_n(
    day_scores: pd.Series,
    top_n: int,
    sector_map: dict[str, str] | None = None,
    max_per_sector: int | None = None,
    liquid_tickers: set[str] | None = None,
) -> set[str]:
    """Greedily picks the top-scoring tickers subject to a per-sector cap
    and a liquidity floor, both applied AT SELECTION TIME -- a name that
    fails either constraint is skipped in favor of the next best-scoring
    eligible name, rather than being included anyway or just shrinking the
    book below top_n. Either constraint is disabled by passing None (or
    sector_map=None disables the sector cap regardless of max_per_sector).
    """
    selected: list[str] = []
    sector_counts: dict[str, int] = {}
    for ticker in day_scores.sort_values(ascending=False).index:
        if len(selected) >= top_n:
            break
        if liquid_tickers is not None and ticker not in liquid_tickers:
            continue
        if sector_map is not None and max_per_sector is not None:
            sector = sector_map.get(ticker, "Unknown")
            if sector_counts.get(sector, 0) >= max_per_sector:
                continue
            sector_counts[sector] = sector_counts.get(sector, 0) + 1
        selected.append(ticker)
    return set(selected)


def run_backtest(horizon_days: int) -> dict:
    settings = get_settings()
    purge = settings["models"]["purge_gap_days"]
    if purge < horizon_days:
        # The purge gap only guarantees no train-label overlaps the test
        # window when it's at least as wide as the label horizon itself
        # (see _build_folds and the module docstring). A smaller gap would
        # silently let some training labels see into the test period,
        # inflating this backtest's numbers rather than erroring loudly.
        raise RuntimeError(
            f"models.purge_gap_days ({purge}) is smaller than this backtest's horizon "
            f"({horizon_days}d) -- a training label could see into the test window, "
            "inflating the result. Increase purge_gap_days to at least the horizon."
        )
    con = get_connection()
    features = con.execute("SELECT * FROM features ORDER BY ticker, date").fetchdf()
    # Backtests must use the same adjusted return series as training and
    # features; raw closes introduce artificial split/dividend jumps.
    prices = con.execute(
        "SELECT ticker, date, COALESCE(adj_close, close) AS close, volume FROM prices ORDER BY ticker, date"
    ).fetchdf()
    fundamentals = con.execute("SELECT ticker, as_of, sector FROM fundamentals").fetchdf()
    con.close()

    if features.empty or prices.empty:
        raise RuntimeError("Need prices + features before backtesting")

    feature_cols = get_feature_cols(features)
    targets = build_targets(prices, settings["benchmark"], [horizon_days])
    merged = features.merge(targets, on=["ticker", "date"], how="inner").sort_values("date")

    dates = sorted(merged["date"].unique())
    folds = _build_folds(dates, settings)
    if not folds:
        raise RuntimeError(
            "Not enough history for a single walk-forward fold — need more price "
            "history or a smaller backtest.train_window_years in settings.yaml"
        )

    price_wide = prices.pivot(index="date", columns="ticker", values="close").sort_index()
    volume_wide = prices.pivot(index="date", columns="ticker", values="volume").sort_index()
    daily_returns_wide = price_wide.pct_change()
    # Trailing 20-day average dollar volume, point-in-time (each date only
    # reflects its own trailing window) -- used as a liquidity floor at
    # selection time, not a feature, so this doesn't need to feed the model.
    dollar_volume_wide = (price_wide * volume_wide).rolling(20).mean()
    benchmark = settings["benchmark"]

    # V1 fundamentals are a snapshot, not point-in-time history (see
    # src/features/fundamental.py), so "latest known sector" is the same
    # simplification already used for the sector-relative feature -- a
    # company's sector essentially never changes often enough for this to
    # matter for a risk cap like this one.
    sector_map = (
        fundamentals.sort_values("as_of").groupby("ticker")["sector"].last().to_dict() if not fundamentals.empty else {}
    )

    top_n = settings["backtest"]["top_n"]
    rebalance_days = settings["backtest"]["rebalance_days"]
    cost_frac = (settings["backtest"]["transaction_cost_bps"] + settings["backtest"]["slippage_bps"]) / 10000
    max_per_sector = settings["backtest"].get("max_positions_per_sector")
    min_dollar_volume = settings["backtest"].get("min_avg_dollar_volume")

    all_port_returns, all_bench_returns, all_turnover, all_dates, all_fold = [], [], [], [], []
    all_predicted, all_actual = [], []
    all_scored_frames = []
    folds_used = 0

    for fold_idx, (train_start, train_end, test_start, test_end) in enumerate(folds):
        train_df = merged[(merged["date"] >= train_start) & (merged["date"] <= train_end)]
        test_df = merged[(merged["date"] >= test_start) & (merged["date"] <= test_end)].sort_values("date")
        if train_df.empty or test_df.empty:
            continue
        folds_used += 1

        models = _fit_horizon(train_df, feature_cols, settings)
        X_test = test_df[feature_cols]
        # REGRESSOR_NAMES (imported from train.py, not redefined here) is what
        # keeps this in sync with predict.py's ensemble -- a hardcoded subset
        # here previously left the backtest silently grading a different,
        # weaker model than the one actually making live predictions after
        # CatBoost was added to the other three.
        score = ensemble_mean(*[models[name].predict(X_test) for name in REGRESSOR_NAMES])
        test_df = test_df.assign(score=score)

        all_predicted.extend(score.tolist())
        all_actual.extend(test_df["forward_relative_return"].tolist())
        all_scored_frames.append(test_df[["date", "ticker", "score", "forward_relative_return"]])

        held: set[str] = set()
        for i, d in enumerate(sorted(test_df["date"].unique())):
            if i % rebalance_days == 0:
                day_scores = test_df[test_df["date"] == d].set_index("ticker")["score"]
                liquid_tickers = None
                if min_dollar_volume and d in dollar_volume_wide.index:
                    dv = dollar_volume_wide.loc[d]
                    liquid_tickers = set(dv[dv >= min_dollar_volume].index)
                new_held = select_top_n(
                    day_scores, top_n, sector_map=sector_map, max_per_sector=max_per_sector, liquid_tickers=liquid_tickers
                )
                turnover = len(new_held - held) / max(len(new_held | held), 1) if held else 1.0
                all_turnover.append(turnover)
                held = new_held
                cost_today = turnover * cost_frac
            else:
                cost_today = 0.0

            if d in daily_returns_wide.index and held:
                rets = daily_returns_wide.loc[d, list(held)].dropna()
                port_ret = float(rets.mean()) if not rets.empty else 0.0
            else:
                port_ret = 0.0
            port_ret -= cost_today

            bench_ret = 0.0
            if d in daily_returns_wide.index and benchmark in daily_returns_wide.columns:
                b = daily_returns_wide.loc[d, benchmark]
                bench_ret = float(b) if pd.notna(b) else 0.0

            all_port_returns.append(port_ret)
            all_bench_returns.append(bench_ret)
            all_dates.append(d)
            all_fold.append(fold_idx)

    if not all_port_returns:
        raise RuntimeError("Backtest produced no simulated days — check data coverage")

    port_series = pd.Series(all_port_returns).fillna(0.0)
    bench_series = pd.Series(all_bench_returns).fillna(0.0)
    turnover_series = pd.Series(all_turnover)

    metrics = compute_all_metrics(
        port_series, bench_series, turnover_series, pd.Series(all_predicted), pd.Series(all_actual)
    )

    # Breakdowns so a strong aggregate number can't hide a strategy that
    # only worked in one favorable stretch or one lucky fold.
    # classify_regimes uses only the benchmark's own trailing history for
    # every date, same no-lookahead discipline as every feature in this app.
    regimes_by_date = {}
    if benchmark in price_wide.columns:
        regimes = classify_regimes(price_wide[benchmark])
        regime_trend_dict = regimes["regime_trend"].to_dict()
        regime_vol_dict = regimes["regime_vol"].to_dict()
        regime_trend_labels = pd.Series([regime_trend_dict.get(d, "unknown") for d in all_dates])
        regime_vol_labels = pd.Series([regime_vol_dict.get(d, "unknown") for d in all_dates])
        regimes_by_date = {
            "by_trend": regime_breakdown_metrics(port_series, bench_series, regime_trend_labels),
            "by_volatility": regime_breakdown_metrics(port_series, bench_series, regime_vol_labels),
        }
    fold_labels = pd.Series([f"fold_{i + 1}" for i in all_fold])
    regimes_by_date["by_fold"] = regime_breakdown_metrics(port_series, bench_series, fold_labels)

    # Quantile analysis: does the model's own score actually separate
    # winners from losers, cross-sectionally, on the days it was scoring?
    # A more direct, visual check of ranking skill than the IC alone --
    # see quantile_returns's docstring.
    scored_df = (
        pd.concat(all_scored_frames, ignore_index=True)
        if all_scored_frames
        else pd.DataFrame(columns=["date", "ticker", "score", "forward_relative_return"])
    )
    quantiles = quantile_returns(scored_df)
    regimes_by_date["quantiles"] = quantiles.to_dict("records")
    spread = quantile_spread(quantiles)

    equity_curve = (1 + port_series).cumprod()
    backtest_id = f"bt_{horizon_days}d_{pd.Timestamp.now('UTC'):%Y%m%d%H%M%S}"

    con = get_connection()
    con.execute(
        """
        INSERT OR REPLACE INTO backtests
            (backtest_id, run_at, start_date, end_date, horizon_days, sharpe, cagr, max_drawdown,
             win_rate, turnover, alpha, volatility, n_trades, params_json, equity_curve_json,
             n_folds, regime_breakdown_json, calibration, quantile_spread, calibration_pvalue)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            backtest_id, pd.Timestamp.now("UTC"), dates[0], dates[-1], horizon_days,
            metrics["sharpe"], metrics["cagr"], metrics["max_drawdown"], metrics["win_rate"],
            metrics["turnover"], metrics["alpha"], metrics["volatility"], len(all_turnover),
            json.dumps(settings["backtest"]), json.dumps([round(v, 6) for v in equity_curve.tolist()]),
            folds_used, json.dumps(regimes_by_date), metrics["calibration"], spread,
            metrics["calibration_pvalue"],
        ],
    )
    con.close()
    logger.info("Backtest %s: %s (folds=%d, spread=%s)", backtest_id, metrics, folds_used, spread)
    return {
        "backtest_id": backtest_id,
        "n_folds": folds_used,
        "regime_breakdown": regimes_by_date,
        "quantile_spread": spread,
        **metrics,
    }


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)
    h = int(sys.argv[1]) if len(sys.argv) > 1 else get_settings()["models"]["horizons_days"][0]
    print(run_backtest(h))
