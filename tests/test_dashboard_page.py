"""Full-render smoke test for GET /dashboard. app/api/dashboard.py has no
build_context()-style split (unlike accuracy.py/backtests.py), so a template
bug like a bad Jinja filter can only be caught by actually rendering the
page -- exactly how the db_stats.total_rows "{:,}"|format(...) bug (Jinja's
`format` is printf-style, not Python's str.format) slipped through: every
other check was a unit test of data/logic, none of them rendered the HTML.

Uses TestClient(app) without the `with` block on purpose -- that skips
FastAPI's lifespan (so start_scheduler()/_catch_up_if_missed() never run),
which matters here since it can queue up the real after_close_job.
"""
from __future__ import annotations

import duckdb
from fastapi.testclient import TestClient

from app.main import app
from src.data.db import bootstrap_schema


def test_dashboard_renders_on_a_freshly_bootstrapped_empty_db(tmp_path, monkeypatch):
    db_path = tmp_path / "test.duckdb"
    duckdb.connect(str(db_path)).close()  # bootstrap_schema runs inside get_connection() itself
    monkeypatch.setattr("src.data.db.DB_PATH", db_path)
    monkeypatch.setattr("app.deps.DB_PATH", db_path)

    client = TestClient(app)  # no `with`: skips lifespan/scheduler entirely
    response = client.get("/dashboard")

    assert response.status_code == 200
    assert "AI Quant" in response.text
    assert "Live activity" in response.text
    assert "Refresh prices" in response.text


def test_activity_endpoint_returns_recent_events():
    client = TestClient(app)
    response = client.get("/api/activity")

    assert response.status_code == 200
    assert isinstance(response.json()["events"], list)


def test_dashboard_renders_with_holdings_and_predictions_seeded(tmp_path, monkeypatch):
    db_path = tmp_path / "test.duckdb"
    con = duckdb.connect(str(db_path))
    bootstrap_schema(con)
    con.execute("INSERT INTO holdings (ticker, shares, cost_basis, added_at) VALUES ('AAPL', 2.0, 150.0, now())")
    con.execute(
        "INSERT INTO prices (ticker, date, close) VALUES ('AAPL', CURRENT_DATE, 180.0), ('AAPL', CURRENT_DATE - 1, 178.0)"
    )
    con.close()
    monkeypatch.setattr("src.data.db.DB_PATH", db_path)
    monkeypatch.setattr("app.deps.DB_PATH", db_path)

    client = TestClient(app)
    response = client.get("/dashboard")

    assert response.status_code == 200
    assert "AAPL" in response.text
