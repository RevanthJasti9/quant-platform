from __future__ import annotations

import logging

import pandas as pd
import yfinance as yf
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from app.copy import FEATURE_GROUPS, confidence_caption, feature_label, format_feature_value, parse_reasons, sign_class
from app.deps import get_daily_changes, records, templates
from src.data.db import get_connection
from src.stats import wilson_interval

router = APIRouter()
logger = logging.getLogger(__name__)

PRIMARY_HORIZON_DAYS = 5

# Trading-day lookback per range button; None means "all available history".
# Forecast overlay only makes sense once the window is wide enough that a
# 5-20 trading-day projection reads as a forward continuation, not the
# entire visible chart -- so it's off for 1D/1W.
RANGES: dict[str, dict] = {
    "1D": {"days": None, "intraday": True, "caption": "Today, every 5 min"},
    "1W": {"days": 5, "intraday": False, "caption": "Last 5 trading days", "forecast": False},
    "1M": {"days": 21, "intraday": False, "caption": "Last 21 trading days", "forecast": True},
    "3M": {"days": 63, "intraday": False, "caption": "Last 63 trading days", "forecast": True},
    "1Y": {"days": 252, "intraday": False, "caption": "Last 252 trading days", "forecast": True},
    "ALL": {"days": None, "intraday": False, "caption": "Full history", "forecast": True},
}
DEFAULT_RANGE = "3M"


def _build_feature_groups(latest_features: dict) -> list[dict]:
    groups = []
    for group_name, keys in FEATURE_GROUPS:
        rows = []
        for key in keys:
            if key not in latest_features:
                continue
            label, desc = feature_label(key)
            rows.append(
                {
                    "label": label,
                    "desc": desc,
                    "value": format_feature_value(key, latest_features[key]),
                }
            )
        if rows:
            groups.append({"name": group_name, "rows": rows})
    return groups


def _enrich_predictions(rows: list[dict]) -> list[dict]:
    out = []
    for r in rows:
        outcome = None
        if r["actual_relative_return"] is not None:
            predicted_up = r["probability_outperform"] >= 0.5
            actual_up = r["actual_relative_return"] >= 0
            outcome = "correct" if predicted_up == actual_up else "missed"
        out.append(
            {
                **r,
                "outlook_class": sign_class(r["expected_relative_return"]),
                "caption": confidence_caption(r["confidence"]),
                "outcome": outcome,
                "reasons": parse_reasons(r.get("reasons_json")),
                "summary": r.get("reasons_summary"),
            }
        )
    return out


def _intraday_series(ticker: str) -> tuple[list[str], list[float]]:
    """Not stored in DuckDB -- intraday bars are fetched live and only used
    for this one chart render, not part of the modeling pipeline.
    """
    try:
        hist = yf.Ticker(ticker).history(period="1d", interval="5m")
    except Exception:
        logger.warning("Intraday fetch failed for %s", ticker, exc_info=True)
        return [], []
    hist = hist.dropna(subset=["Close"])
    if hist.empty:
        return [], []
    labels = [ts.strftime("%H:%M") for ts in hist.index]
    values = [float(v) for v in hist["Close"]]
    return labels, values


def _forecast_overlay(con, ticker: str, last_price: float, last_date) -> tuple[list, list, list]:
    """Extends the chart with a dashed line projecting price forward using
    the latest expected-relative-return forecasts, one point per horizon.
    Also returns each point's horizon_days so the caller can pick the one
    that matches whatever horizon the page's own headline is describing,
    rather than assuming the furthest-out point (see stock_chart()).
    """
    latest_pred_date = con.execute(
        "SELECT MAX(prediction_date) FROM predictions WHERE ticker = ?", [ticker]
    ).fetchone()[0]
    if latest_pred_date is None:
        return [], [], []

    horizon_rows = con.execute(
        """
        SELECT horizon_days, expected_relative_return FROM predictions
        WHERE ticker = ? AND prediction_date = ? ORDER BY horizon_days
        """,
        [ticker, latest_pred_date],
    ).fetchall()
    if not horizon_rows:
        return [], [], []

    last_date = pd.Timestamp(last_date)
    labels, values, horizons = [], [], []
    for horizon_days, expected_return in horizon_rows:
        forecast_date = last_date + pd.tseries.offsets.BDay(int(horizon_days))
        labels.append(str(forecast_date.date()))
        values.append(last_price * (1 + expected_return))
        horizons.append(horizon_days)
    return labels, values, horizons


@router.get("/stock/{ticker}/chart")
def stock_chart(ticker: str, range: str = DEFAULT_RANGE):
    ticker = ticker.upper()
    range_key = range.upper() if range.upper() in RANGES else DEFAULT_RANGE
    cfg = RANGES[range_key]
    con = get_connection()

    if cfg["intraday"]:
        hist_labels, hist_values = _intraday_series(ticker)
        forecast_labels, forecast_values = [], []
    else:
        prices = con.execute(
            "SELECT date, close FROM prices WHERE ticker = ? ORDER BY date", [ticker]
        ).fetchdf().dropna(subset=["close"])
        if cfg["days"] is not None:
            prices = prices.tail(cfg["days"])
        hist_labels = [str(d) for d in prices["date"]]
        hist_values = [float(v) for v in prices["close"]]

        forecast_labels, forecast_values, forecast_horizons = [], [], []
        if cfg["forecast"] and hist_values:
            forecast_labels, forecast_values, forecast_horizons = _forecast_overlay(
                con, ticker, hist_values[-1], prices["date"].iloc[-1]
            )
    con.close()

    combined_labels = hist_labels + forecast_labels
    price_series = hist_values + [None] * len(forecast_labels)
    if forecast_labels:
        forecast_series = [None] * (len(hist_values) - 1) + [hist_values[-1]] + forecast_values
        # Colored to match the same horizon the page's headline number describes
        # (PRIMARY_HORIZON_DAYS), not just whichever point happens to be furthest
        # out -- otherwise a positive 5D forecast next to a negative 20D one would
        # show a green "up" headline above a red line, which is what this fixes.
        if PRIMARY_HORIZON_DAYS in forecast_horizons:
            primary_value = forecast_values[forecast_horizons.index(PRIMARY_HORIZON_DAYS)]
        else:
            primary_value = forecast_values[-1]
        forecast_up = primary_value >= hist_values[-1]
    else:
        forecast_series = []
        forecast_up = None

    return JSONResponse(
        {
            "labels": combined_labels,
            "price_series": price_series,
            "forecast_series": forecast_series,
            "forecast_up": forecast_up,
            "caption": cfg["caption"],
        }
    )


@router.get("/stock/{ticker}")
def stock_detail(request: Request, ticker: str):
    ticker = ticker.upper()
    con = get_connection()

    change = get_daily_changes(con, [ticker]).get(ticker, {})
    latest_close = change.get("price")
    day_change_pct = change.get("change_pct")
    day_change_dollar = change.get("change_dollar")

    fundamentals_cols = {
        r[0]
        for r in con.execute(
            "SELECT column_name FROM information_schema.columns WHERE table_name = 'fundamentals'"
        ).fetchall()
    }
    company_name, sector = None, None
    if "company_name" in fundamentals_cols:
        company_row = con.execute(
            "SELECT company_name, sector FROM fundamentals WHERE ticker = ? ORDER BY as_of DESC LIMIT 1", [ticker]
        ).fetchone()
        if company_row:
            company_name, sector = company_row

    latest_feat_df = con.execute(
        "SELECT * FROM features WHERE ticker = ? ORDER BY date DESC LIMIT 1", [ticker]
    ).fetchdf()
    feature_groups = []
    if not latest_feat_df.empty:
        latest_features = records(latest_feat_df.drop(columns=["ticker", "date"]))[0]
        feature_groups = _build_feature_groups(latest_features)

    pred_cols = {
        r[0]
        for r in con.execute(
            "SELECT column_name FROM information_schema.columns WHERE table_name = 'predictions'"
        ).fetchall()
    }
    optional_cols = ", reasons_summary" if "reasons_summary" in pred_cols else ""
    predictions_df = con.execute(
        f"""
        SELECT prediction_date, horizon_days, expected_relative_return,
               probability_outperform, confidence, actual_relative_return, reasons_json{optional_cols}
        FROM predictions WHERE ticker = ? ORDER BY prediction_date DESC, horizon_days LIMIT 12
        """,
        [ticker],
    ).fetchdf()
    predictions = []
    primary = None
    current_predictions = []
    if not predictions_df.empty:
        predictions_df["prediction_date"] = predictions_df["prediction_date"].dt.date
        predictions = _enrich_predictions(records(predictions_df))
        primary_matches = [p for p in predictions if p["horizon_days"] == PRIMARY_HORIZON_DAYS]
        primary = primary_matches[0] if primary_matches else predictions[0]

        # Every horizon from the most recent prediction_date, not just the primary
        # one -- a 5D and 20D call can point different directions (see the chart's
        # forecast_up fix), and hiding one behind "Forecast history" let that go
        # unnoticed. predictions is already ordered prediction_date DESC.
        latest_pred_date = predictions[0]["prediction_date"]
        current_predictions = sorted(
            (p for p in predictions if p["prediction_date"] == latest_pred_date),
            key=lambda p: p["horizon_days"],
        )

    track_record_df = con.execute(
        """
        SELECT horizon_days, COUNT(*) AS evaluated,
               SUM(CASE WHEN (probability_outperform >= 0.5) = (actual_relative_return >= 0) THEN 1 ELSE 0 END) AS correct
        FROM predictions WHERE ticker = ? AND actual_relative_return IS NOT NULL
        GROUP BY horizon_days ORDER BY horizon_days
        """,
        [ticker],
    ).fetchdf()
    track_record = []
    for r in records(track_record_df):
        interval = wilson_interval(r["correct"], r["evaluated"])
        track_record.append(
            {
                "horizon_days": r["horizon_days"],
                "correct": r["correct"],
                "evaluated": r["evaluated"],
                "pct": r["correct"] / r["evaluated"] * 100,
                "ci": f"{interval[0] * 100:.0f}-{interval[1] * 100:.0f}%" if interval else None,
            }
        )

    news_digest = con.execute(
        "SELECT summary, headline_count FROM news_digests WHERE ticker = ?", [ticker]
    ).fetchone()
    recent_headlines_df = con.execute(
        """
        WITH ranked AS (
            SELECT headline, source,
                   COALESCE(provider, 'yahoo') AS provider,
                   COALESCE(event_type, 'general_news') AS event_type,
                   url, published_at,
                   ROW_NUMBER() OVER (
                       PARTITION BY COALESCE(duplicate_group, url)
                       ORDER BY reliability_score DESC NULLS LAST, published_at DESC
                   ) AS duplicate_rank
            FROM news_events
            WHERE ticker = ?
        )
        SELECT headline, source, provider, event_type, url, published_at
        FROM ranked
        WHERE duplicate_rank = 1
        ORDER BY published_at DESC
        LIMIT 8
        """,
        [ticker],
    ).fetchdf()
    if not recent_headlines_df.empty:
        recent_headlines_df["published_at"] = pd.to_datetime(recent_headlines_df["published_at"]).dt.strftime("%b %-d")

    con.close()

    return templates.TemplateResponse(
        request,
        "stock.html",
        {
            "ticker": ticker,
            "company_name": company_name,
            "sector": sector,
            "latest_close": latest_close,
            "day_change_pct": day_change_pct,
            "day_change_dollar": day_change_dollar,
            "primary": primary,
            "current_predictions": current_predictions,
            "track_record": track_record,
            "ranges": list(RANGES.keys()),
            "default_range": DEFAULT_RANGE,
            "news_summary": news_digest[0] if news_digest else None,
            "news_headline_count": news_digest[1] if news_digest else 0,
            "recent_headlines": records(recent_headlines_df),
            "feature_groups": feature_groups,
            "predictions": predictions,
        },
    )
