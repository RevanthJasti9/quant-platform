"""Accuracy page data/grading logic (app/api/accuracy.py, app/deps.py's
get_next_eval_date). Tested directly against a temp DuckDB rather than
through the HTTP layer -- build_accuracy_context() takes a connection and
returns the exact template context, so there's no need to spin up a test
client just to verify the SQL aggregation and grading are correct.
"""
from __future__ import annotations

import duckdb
import pandas as pd
import pytest

from app.api.accuracy import _ci_label, build_accuracy_context
from app.deps import get_next_eval_date
from src.data.db import bootstrap_schema, upsert_wide


@pytest.fixture
def db():
    con = duckdb.connect(":memory:")
    bootstrap_schema(con)
    yield con
    con.close()


def _prediction(ticker, prediction_date, horizon_days, probability_outperform, expected_return, actual_return=None):
    return {
        "ticker": ticker,
        "prediction_date": pd.Timestamp(prediction_date).date(),
        "horizon_days": horizon_days,
        "probability_outperform": probability_outperform,
        "expected_relative_return": expected_return,
        "confidence": 80.0,
        "actual_relative_return": actual_return,
        "evaluated_at": pd.Timestamp.now("UTC") if actual_return is not None else None,
    }


def test_empty_predictions_table(db):
    ctx = build_accuracy_context(db)
    assert ctx["n_evaluated"] == 0
    assert ctx["overall_pct"] is None
    assert ctx["by_horizon"] == []
    assert ctx["by_ticker"] == []
    assert ctx["recent"] == []
    assert ctx["next_eval_date"] is None  # nothing pending either


def test_pending_predictions_report_next_eval_date(db):
    rows = pd.DataFrame([_prediction("AAA", "2024-01-10", 5, 0.7, 0.02)])
    upsert_wide(db, "predictions", rows, ("ticker", "prediction_date", "horizon_days"))

    ctx = build_accuracy_context(db)
    assert ctx["n_evaluated"] == 0
    assert ctx["next_eval_date"] is not None
    assert ctx["next_eval_date"] == get_next_eval_date(db)


def test_overall_and_horizon_accuracy_computed_correctly(db):
    rows = pd.DataFrame(
        [
            # 5D: one correct (predicted up, went up), one wrong (predicted up, went down)
            _prediction("AAA", "2024-01-02", 5, 0.7, 0.02, actual_return=0.01),
            _prediction("BBB", "2024-01-02", 5, 0.65, 0.015, actual_return=-0.01),
            # 20D: one correct (predicted down, went down)
            _prediction("AAA", "2024-01-02", 20, 0.3, -0.02, actual_return=-0.03),
        ]
    )
    upsert_wide(db, "predictions", rows, ("ticker", "prediction_date", "horizon_days"))

    ctx = build_accuracy_context(db)
    assert ctx["n_evaluated"] == 3
    assert ctx["n_correct"] == 2
    assert ctx["overall_pct"] == pytest.approx(200 / 3)

    by_horizon = {h["horizon_days"]: h for h in ctx["by_horizon"]}
    assert by_horizon[5]["evaluated"] == 2
    assert by_horizon[5]["correct"] == 1
    assert by_horizon[5]["pct"] == pytest.approx(50.0)
    assert by_horizon[20]["evaluated"] == 1
    assert by_horizon[20]["correct"] == 1
    assert by_horizon[20]["pct"] == pytest.approx(100.0)


def test_by_ticker_breakdown_and_recent_outcomes(db):
    rows = pd.DataFrame(
        [
            _prediction("AAA", "2024-01-02", 5, 0.7, 0.02, actual_return=0.01),  # correct
            _prediction("AAA", "2024-01-09", 5, 0.6, 0.01, actual_return=-0.01),  # missed
        ]
    )
    upsert_wide(db, "predictions", rows, ("ticker", "prediction_date", "horizon_days"))

    ctx = build_accuracy_context(db)
    assert len(ctx["by_ticker"]) == 1
    row = ctx["by_ticker"][0]
    assert (row["ticker"], row["evaluated"], row["correct"], row["pct"]) == ("AAA", 2, 1, pytest.approx(50.0))
    assert row["ci"] is not None  # n=2 is small -- the whole point of the Wilson interval is showing that

    outcomes = {r["prediction_date"]: r["outcome"] for r in ctx["recent"]}
    assert outcomes[pd.Timestamp("2024-01-02")] == "correct"
    assert outcomes[pd.Timestamp("2024-01-09")] == "missed"


def test_recent_list_is_capped_and_most_recent_first(db):
    rows = pd.DataFrame(
        [_prediction("AAA", f"2024-01-{d:02d}", 5, 0.6, 0.01, actual_return=0.01) for d in range(1, 31)]
    )
    upsert_wide(db, "predictions", rows, ("ticker", "prediction_date", "horizon_days"))
    # give each row a distinct evaluated_at so "most recent first" is well-defined
    for i, d in enumerate(range(1, 31)):
        db.execute(
            "UPDATE predictions SET evaluated_at = ? WHERE prediction_date = ?",
            [pd.Timestamp("2024-02-01") + pd.Timedelta(minutes=i), pd.Timestamp(f"2024-01-{d:02d}").date()],
        )

    ctx = build_accuracy_context(db)
    assert len(ctx["recent"]) == 25
    assert ctx["recent"][0]["prediction_date"] == pd.Timestamp("2024-01-30")


def test_ci_label_none_when_nothing_evaluated():
    assert _ci_label(0, 0) is None


def test_ci_label_format():
    label = _ci_label(3, 5)
    assert label is not None
    assert "-" in label and label.endswith("%")


def test_overall_ci_present_once_predictions_are_graded(db):
    rows = pd.DataFrame(
        [
            _prediction("AAA", "2024-01-02", 5, 0.7, 0.02, actual_return=0.01),
            _prediction("BBB", "2024-01-02", 5, 0.65, 0.015, actual_return=-0.01),
        ]
    )
    upsert_wide(db, "predictions", rows, ("ticker", "prediction_date", "horizon_days"))

    ctx = build_accuracy_context(db)
    assert ctx["overall_ci"] is not None
    for h in ctx["by_horizon"]:
        assert h["ci"] is not None


def test_overall_ci_none_when_nothing_evaluated(db):
    ctx = build_accuracy_context(db)
    assert ctx["overall_ci"] is None
