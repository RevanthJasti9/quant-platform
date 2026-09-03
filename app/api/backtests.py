from __future__ import annotations

import json

from fastapi import APIRouter, Request

from app.copy import backtest_caption, metric_label
from app.deps import records, templates
from src.data.db import get_connection

router = APIRouter()

_METRIC_ORDER = ["sharpe", "cagr", "max_drawdown", "alpha", "win_rate", "volatility", "turnover", "calibration", "quantile_spread"]
_REGIME_LABELS = {"bull": "Bull", "bear": "Bear", "high_vol": "High volatility", "low_vol": "Low volatility"}


def _metric_css(key: str, value: float) -> str:
    if key in ("sharpe", "cagr", "alpha", "calibration", "quantile_spread"):
        return "pos" if value >= 0 else "neg"
    if key == "max_drawdown":
        return "neg"
    return ""


def _format_calibration(value: float, pvalue: float | None) -> str:
    """Pairs the IC with whether it's actually distinguishable from zero-skill
    noise at this sample size -- a bare "0.045" invites reading real skill
    into a number that a p-value would show is well within noise range.
    pvalue is None on backtests too small for the significance test to apply
    (or run before this was added) -- shown as the plain IC with no claim
    either way, rather than a misleading "not significant".
    """
    if pvalue is None:
        return f"{value:.3f}"
    verdict = "significant" if pvalue < 0.05 else "not significant"
    return f"{value:.3f} ({verdict}, p={pvalue:.2f})"


def _metric_rows(run: dict) -> list[dict]:
    rows = []
    for key in _METRIC_ORDER:
        value = run.get(key)
        if value is None:  # calibration/quantile_spread are absent on backtests run before this was added
            continue
        label, desc = metric_label(key)
        if key == "calibration":
            formatted = _format_calibration(value, run.get("calibration_pvalue"))
        elif key == "sharpe":
            formatted = f"{value:.2f}"
        else:
            formatted = f"{value * 100:.1f}%"
        rows.append({"label": label, "desc": desc, "value": formatted, "css": _metric_css(key, value)})
    return rows


def _stat_row(label: str, stats: dict) -> dict:
    return {
        "label": label,
        "n_days": stats["n_days"],
        "sharpe": f"{stats['sharpe']:.2f}",
        "alpha": f"{stats['alpha'] * 100:+.1f}%",
        "alpha_css": "pos" if stats["alpha"] >= 0 else "neg",
        "win_rate": f"{stats['win_rate'] * 100:.0f}%",
    }


def _parse_breakdown(regime_breakdown_json: str | None) -> dict:
    if not regime_breakdown_json:
        return {}
    try:
        return json.loads(regime_breakdown_json)
    except (TypeError, ValueError):
        return {}


_NON_REGIME_KEYS = {"by_fold", "quantiles"}


def _regime_rows(breakdown: dict) -> list[dict]:
    """Flattens {"by_trend": {"bull": {...}}, "by_volatility": {...}} into
    a flat list the template can loop over as one section, most days first.
    Excludes by_fold (its own chronological section) and quantiles (a list,
    not a regime->stats dict -- rendered as the bar chart instead).
    """
    rows = [
        _stat_row(_REGIME_LABELS.get(regime, regime), stats)
        for key, group in breakdown.items()
        if key not in _NON_REGIME_KEYS
        for regime, stats in group.items()
    ]
    return sorted(rows, key=lambda r: -r["n_days"])


def _fold_rows(breakdown: dict) -> list[dict]:
    """Kept in chronological order (fold_1 earliest), unlike _regime_rows --
    the point of a per-fold view is spotting whether the edge is fading
    over time, which sorting by size would scramble.
    """
    by_fold = breakdown.get("by_fold", {})
    ordered = sorted(by_fold.items(), key=lambda kv: int(kv[0].split("_")[1]))
    return [_stat_row(fold.replace("_", " ").title(), stats) for fold, stats in ordered]


def _quantile_chart_data(breakdown: dict) -> dict:
    """Chart.js-ready {labels, values} for the quantile bar chart -- a
    model with real ranking skill should show bars climbing left to right.
    """
    quantiles = breakdown.get("quantiles", [])
    return {
        "labels": [f"Q{q['quantile']}" for q in quantiles],
        "values": [round(q["mean_return"] * 100, 3) for q in quantiles],
    }


@router.get("/backtests")
def backtests_page(request: Request):
    con = get_connection()
    runs_df = con.execute("SELECT * FROM backtests ORDER BY run_at DESC").fetchdf()
    con.close()

    latest = None
    metrics = []
    regime_rows = []
    fold_rows = []
    quantile_chart = {"labels": [], "values": []}
    equity_curve = []
    history = []

    if not runs_df.empty:
        runs_df["start_date"] = runs_df["start_date"].dt.date
        runs_df["end_date"] = runs_df["end_date"].dt.date
        runs = records(runs_df.drop(columns=["equity_curve_json", "params_json", "regime_breakdown_json"], errors="ignore"))

        latest = runs[0]
        latest["caption"] = backtest_caption(latest["sharpe"], latest["alpha"])
        metrics = _metric_rows(latest)
        breakdown = _parse_breakdown(runs_df.iloc[0]["regime_breakdown_json"])
        regime_rows = _regime_rows(breakdown)
        fold_rows = _fold_rows(breakdown)
        quantile_chart = _quantile_chart_data(breakdown)
        raw_curve = runs_df.iloc[0]["equity_curve_json"]
        equity_curve = json.loads(raw_curve) if raw_curve else []

        for r in runs[1:]:
            r["caption"] = backtest_caption(r["sharpe"], r["alpha"])
            history.append(r)

    return templates.TemplateResponse(
        request,
        "backtests.html",
        {
            "latest": latest,
            "metrics": metrics,
            "regime_rows": regime_rows,
            "fold_rows": fold_rows,
            "quantile_labels": json.dumps(quantile_chart["labels"]),
            "quantile_values": json.dumps(quantile_chart["values"]),
            "has_quantiles": bool(quantile_chart["labels"]),
            "equity_curve": json.dumps(equity_curve),
            "history": history,
        },
    )
