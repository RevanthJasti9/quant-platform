"""Dedicated accuracy/track-record page: the same "was this prediction
right or wrong" grading already used for the dashboard's Accuracy tile and
each stock page's Forecast History, broken down by horizon and by ticker,
plus a feed of the most recently graded calls. Auto-refreshes (see
accuracy.html) since this is meant to be glanced at, not clicked through.
"""
from __future__ import annotations

from fastapi import APIRouter, Request

from app.copy import confidence_caption, sign_class
from app.deps import get_next_eval_date, records, templates
from src.data.db import get_connection
from src.stats import wilson_interval

router = APIRouter()

_CORRECT_SQL = "(probability_outperform >= 0.5) = (actual_relative_return >= 0)"


def _pct(row: dict) -> float | None:
    return (row["correct"] / row["evaluated"] * 100) if row["evaluated"] else None


def _ci_label(correct: int, evaluated: int) -> str | None:
    """"62%" on its own reads as a precise measurement -- at the small n this
    page usually has early on, the honest range around it can be enormous
    (e.g. 3 of 5 correct is really anywhere from ~30% to ~90%). Returns a
    compact "39-82%" label from the Wilson interval, or None once there's
    nothing to bound.
    """
    interval = wilson_interval(correct, evaluated)
    if interval is None:
        return None
    lo, hi = interval
    return f"{lo * 100:.0f}-{hi * 100:.0f}%"


def build_accuracy_context(con) -> dict:
    """All the reads + grading logic for the accuracy page, as one function
    that takes a connection -- kept separate from the route/template so it's
    directly testable against a temp DuckDB, no HTTP layer involved.
    """
    overall = con.execute(
        f"""
        SELECT COUNT(*) AS evaluated, SUM(CASE WHEN {_CORRECT_SQL} THEN 1 ELSE 0 END) AS correct
        FROM predictions WHERE actual_relative_return IS NOT NULL
        """
    ).fetchone()
    n_evaluated = overall[0] or 0
    n_correct = int(overall[1]) if overall[1] is not None else 0
    overall_pct = (n_correct / n_evaluated * 100) if n_evaluated else None
    overall_ci = _ci_label(n_correct, n_evaluated)
    next_eval_date = get_next_eval_date(con) if n_evaluated == 0 else None

    by_horizon = []
    by_ticker = []
    recent = []

    if n_evaluated:
        by_horizon_df = con.execute(
            f"""
            SELECT horizon_days, COUNT(*) AS evaluated, SUM(CASE WHEN {_CORRECT_SQL} THEN 1 ELSE 0 END) AS correct
            FROM predictions WHERE actual_relative_return IS NOT NULL
            GROUP BY horizon_days ORDER BY horizon_days
            """
        ).fetchdf()
        by_horizon = [{**r, "pct": _pct(r), "ci": _ci_label(r["correct"], r["evaluated"])} for r in records(by_horizon_df)]

        by_ticker_df = con.execute(
            f"""
            SELECT ticker, COUNT(*) AS evaluated, SUM(CASE WHEN {_CORRECT_SQL} THEN 1 ELSE 0 END) AS correct
            FROM predictions WHERE actual_relative_return IS NOT NULL
            GROUP BY ticker ORDER BY evaluated DESC, ticker
            """
        ).fetchdf()
        by_ticker = [{**r, "pct": _pct(r), "ci": _ci_label(r["correct"], r["evaluated"])} for r in records(by_ticker_df)]

        recent_df = con.execute(
            """
            SELECT ticker, prediction_date, horizon_days, probability_outperform,
                   expected_relative_return, actual_relative_return, confidence, evaluated_at
            FROM predictions WHERE actual_relative_return IS NOT NULL
            ORDER BY evaluated_at DESC LIMIT 25
            """
        ).fetchdf()
        for r in records(recent_df):
            predicted_up = r["probability_outperform"] >= 0.5
            actual_up = r["actual_relative_return"] >= 0
            recent.append(
                {
                    **r,
                    "outcome": "correct" if predicted_up == actual_up else "missed",
                    "outlook_class": sign_class(r["expected_relative_return"]),
                    "actual_class": sign_class(r["actual_relative_return"]),
                    "caption": confidence_caption(r["confidence"]),
                }
            )

    return {
        "n_evaluated": n_evaluated,
        "n_correct": n_correct,
        "overall_pct": overall_pct,
        "overall_ci": overall_ci,
        "next_eval_date": next_eval_date,
        "by_horizon": by_horizon,
        "by_ticker": by_ticker,
        "recent": recent,
    }


@router.get("/accuracy")
def accuracy_page(request: Request):
    con = get_connection()
    context = build_accuracy_context(con)
    con.close()
    return templates.TemplateResponse(request, "accuracy.html", context)
