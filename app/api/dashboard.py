from __future__ import annotations

import logging
import socket
from contextlib import contextmanager

import pandas as pd
import yfinance as yf
from fastapi import APIRouter, Query, Request

from app.copy import backtest_caption, confidence_caption, format_bytes, sign_class, table_label, ticker_avatar
from app.deps import (
    get_company_names,
    get_daily_changes,
    get_db_stats,
    get_next_eval_date,
    get_system_stats,
    records,
    templates,
)
from src.config import get_settings
from src.data.db import get_connection
from src.holdings import get_holdings
from src.observability.activity import recent_activity
from src.observability.runtime import active_tasks

router = APIRouter()
logger = logging.getLogger(__name__)
PUBLIC_QUOTE_TIMEOUT_SECONDS = 12


@contextmanager
def _bounded_public_quote_call():
    previous = socket.getdefaulttimeout()
    socket.setdefaulttimeout(PUBLIC_QUOTE_TIMEOUT_SECONDS)
    try:
        yield
    finally:
        socket.setdefaulttimeout(previous)


def _live_public_quotes(tickers: list[str]) -> dict[str, dict[str, float | None]]:
    """Fetch display-only Yahoo Finance quotes; nothing is persisted."""
    quotes: dict[str, dict[str, float | None]] = {}
    for ticker in tickers:
        try:
            with _bounded_public_quote_call():
                fast = yf.Ticker(ticker).fast_info
                price = fast.get("lastPrice")
                previous_close = fast.get("previousClose")
            if price is not None:
                quotes[ticker] = {
                    "price": float(price),
                    "previous_close": float(previous_close) if previous_close is not None else None,
                }
        except Exception:
            logger.warning("Live public quote unavailable for %s", ticker, exc_info=True)
    return quotes


@router.get("/api/activity")
def activity_feed(limit: int = Query(default=100, ge=1, le=300)) -> dict:
    """Recent local application activity for the collapsed dashboard panel."""
    return {"events": recent_activity(limit), "active_tasks": active_tasks()}


def _enrich_outperformers(rows: list[dict]) -> list[dict]:
    out = []
    for r in rows:
        out.append(
            {
                **r,
                "outlook_class": sign_class(r["expected_relative_return"]),
                "caption": confidence_caption(r["confidence"]),
            }
        )
    return out


def _enrich_holdings(con, rows: list[dict], live_quotes: dict[str, dict[str, float | None]] | None = None) -> tuple[list[dict], dict]:
    changes = get_daily_changes(con, [h["ticker"] for h in rows])
    company_names = get_company_names(con, [h["ticker"] for h in rows])

    items = []
    total_value = 0.0
    total_cost = 0.0
    total_day_change = 0.0

    for h in rows:
        ticker, shares, cost_basis = h["ticker"], h["shares"], h["cost_basis"]
        company_name = company_names.get(ticker)
        change = changes.get(ticker, {})
        quote = (live_quotes or {}).get(ticker)
        using_live_price = quote is not None
        current_price = quote["price"] if quote else change.get("price")
        previous_price = quote.get("previous_close") if quote else None
        if using_live_price and previous_price:
            change_dollar = current_price - previous_price
            day_change_pct = change_dollar / previous_price
        else:
            change_dollar = change.get("change_dollar")
            day_change_pct = change.get("change_pct")

        shares_display = f"{shares:.4f}".rstrip("0").rstrip(".")
        item = {
            "ticker": ticker,
            "company_name": company_name,
            "avatar": ticker_avatar(ticker, company_name),
            "shares_display": shares_display,
            "shares": shares,
            "cost_basis": cost_basis,
            "total_paid": shares * cost_basis,
            "current_price": current_price,
            "day_change_pct": day_change_pct,
            "price_source": "Yahoo Finance" if using_live_price else "market close",
        }

        if current_price is not None:
            value = shares * current_price
            cost = shares * cost_basis
            item["value"] = value
            item["gain_dollar"] = value - cost
            item["gain_pct"] = (value - cost) / cost if cost else 0.0
            total_value += value
            total_cost += cost
            if change_dollar is not None:
                total_day_change += shares * change_dollar
        else:
            item["value"] = None

        items.append(item)

    totals = {
        "value": total_value,
        "cost": total_cost,
        "gain_dollar": total_value - total_cost,
        "gain_pct": (total_value - total_cost) / total_cost if total_cost else 0.0,
        "day_change": total_day_change,
        "day_change_pct": (total_day_change / (total_value - total_day_change)) if (total_value - total_day_change) else 0.0,
    }
    return items, totals


@router.get("/dashboard")
def dashboard(request: Request):
    settings = get_settings()
    op_cfg = settings.get("outperformers", {"horizon_days": 5, "top_n": 10})
    # Measured before opening the DB connection (its 0.2s blocking sample
    # would otherwise just extend how long this request holds DuckDB's
    # single-writer lock, for no reason -- see the earlier lock-contention
    # fix in predict.py/news_digest.py for why that matters here).
    system_stats = get_system_stats()
    con = get_connection()

    saved_holdings = get_holdings()
    live_requested = request.query_params.get("refresh_prices") == "1"
    live_quotes = _live_public_quotes([h["ticker"] for h in saved_holdings]) if live_requested else {}
    holdings, holdings_totals = _enrich_holdings(con, saved_holdings, live_quotes)
    broker_snapshot_df = con.execute(
        """
        SELECT total_value, equity_value, cash, crypto_value, synced_at
        FROM broker_portfolio_snapshots
        WHERE provider = 'robinhood'
        """
    ).fetchdf()
    broker_portfolio = records(broker_snapshot_df)[0] if not broker_snapshot_df.empty else None
    if broker_portfolio and broker_portfolio["synced_at"] is not None:
        broker_portfolio["synced_label"] = pd.Timestamp(broker_portfolio["synced_at"]).strftime("%b %d, %I:%M %p")

    outperformers = []
    latest_date_row = con.execute("SELECT MAX(prediction_date) FROM predictions").fetchone()
    latest_date = latest_date_row[0] if latest_date_row else None
    if latest_date:
        df = con.execute(
            """
            SELECT ticker, horizon_days, expected_relative_return, probability_outperform, confidence
            FROM predictions
            WHERE prediction_date = ? AND horizon_days = ?
            ORDER BY probability_outperform DESC
            LIMIT ?
            """,
            [latest_date, op_cfg["horizon_days"], op_cfg["top_n"]],
        ).fetchdf()
        if not df.empty:
            outperformers = _enrich_outperformers(records(df))

    latest_backtest_df = con.execute("SELECT * FROM backtests ORDER BY run_at DESC LIMIT 1").fetchdf()
    backtest = records(latest_backtest_df)[0] if not latest_backtest_df.empty else None
    backtest_note = backtest_caption(backtest["sharpe"], backtest["alpha"]) if backtest else None

    latest_run_df = con.execute("SELECT * FROM ingest_runs ORDER BY started_at DESC LIMIT 1").fetchdf()
    latest_run = records(latest_run_df)[0] if not latest_run_df.empty else None
    source_runs, quality_checks = [], []
    if latest_run:
        latest_run["started_label"] = pd.Timestamp(latest_run["started_at"]).strftime("%b %d, %I:%M %p")
        source_runs = records(
            con.execute(
                "SELECT source, status, row_count, error FROM source_runs WHERE run_id = ? ORDER BY source",
                [latest_run["run_id"]],
            ).fetchdf()
        )
        quality_checks = records(
            con.execute(
                "SELECT check_name, status, detail FROM data_quality_results WHERE run_id = ? ORDER BY check_name",
                [latest_run["run_id"]],
            ).fetchdf()
        )
    health_issue_count = sum(1 for s in source_runs if s["status"] == "failed") + sum(
        1 for c in quality_checks if c["status"] != "pass"
    )

    n_total = con.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
    n_evaluated = con.execute("SELECT COUNT(*) FROM predictions WHERE actual_relative_return IS NOT NULL").fetchone()[0]
    n_correct = con.execute(
        """
        SELECT COUNT(*) FROM predictions
        WHERE actual_relative_return IS NOT NULL
          AND (probability_outperform >= 0.5) = (actual_relative_return >= 0)
        """
    ).fetchone()[0]
    n_prices = con.execute(
        "SELECT COUNT(DISTINCT ticker) FROM prices WHERE ticker != ?", [settings["benchmark"]]
    ).fetchone()[0]

    next_eval_date = get_next_eval_date(con) if n_evaluated == 0 else None
    db_stats = get_db_stats(con)
    db_stats["file_size_label"] = format_bytes(db_stats["file_size_bytes"])
    db_stats["total_rows_label"] = f"{db_stats['total_rows']:,}"
    for t in db_stats["tables"]:
        t["label"] = table_label(t["table_name"])
        t["row_count_label"] = f"{t['row_count']:,} row" + ("" if t["row_count"] == 1 else "s")

    con.close()

    return templates.TemplateResponse(
        request,
        "dashboard.html",
        {
            "holdings": holdings,
            "holdings_totals": holdings_totals,
            "broker_portfolio": broker_portfolio,
            "holdings_error": (
                "Could not get a current public market quote; try again shortly. Your saved holdings were not changed."
                if live_requested and saved_holdings and not live_quotes
                else request.query_params.get("holdings_error")
            ),
            "outperformers": outperformers,
            "outperformers_horizon": op_cfg["horizon_days"],
            "latest_date": latest_date,
            "backtest": backtest,
            "backtest_note": backtest_note,
            "universe_size": len(settings["universe"]),
            "n_prices": n_prices,
            "n_evaluated": n_evaluated,
            "n_correct": n_correct,
            "n_total": n_total,
            "next_eval_date": next_eval_date,
            "system_stats": system_stats,
            "db_stats": db_stats,
            "latest_run": latest_run,
            "source_runs": source_runs,
            "quality_checks": quality_checks,
            "health_issue_count": health_issue_count,
        },
    )
