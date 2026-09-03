"""GET /stock/{ticker}/chart's forecast_up flag (app/api/stocks.py). Real
bug this covers: expected_relative_return can disagree in sign between the
5D and 20D horizons for the same ticker (a stock can be forecast to beat
the market over the next week while still trailing it over the next
month). forecast_up used to be driven by whichever horizon happened to
sort last (20D), while the page's own headline always describes
PRIMARY_HORIZON_DAYS (5D) -- so the chart line's color could contradict
the headline's own "up"/"down" text. Fixed by keying forecast_up off the
same horizon the headline uses.
"""
from __future__ import annotations

import duckdb
import pandas as pd
from fastapi.testclient import TestClient

from app.main import app
from src.data.db import bootstrap_schema, upsert_wide


def _seed(db_path):
    con = duckdb.connect(str(db_path))
    bootstrap_schema(con)
    prices = pd.DataFrame(
        {"ticker": ["AAA"] * 5, "date": pd.date_range("2024-01-01", periods=5), "close": [100.0, 101, 102, 101, 100.0]}
    )
    upsert_wide(con, "prices", prices, ("ticker", "date"))
    predictions = pd.DataFrame(
        [
            {"ticker": "AAA", "prediction_date": pd.Timestamp("2024-01-05").date(), "horizon_days": 5, "expected_relative_return": 0.02},
            {"ticker": "AAA", "prediction_date": pd.Timestamp("2024-01-05").date(), "horizon_days": 20, "expected_relative_return": -0.05},
        ]
    )
    upsert_wide(con, "predictions", predictions, ("ticker", "prediction_date", "horizon_days"))
    con.close()


def test_forecast_up_matches_the_primary_horizon_even_when_the_longer_horizon_disagrees(tmp_path, monkeypatch):
    db_path = tmp_path / "test.duckdb"
    _seed(db_path)
    monkeypatch.setattr("src.data.db.DB_PATH", db_path)
    monkeypatch.setattr("app.deps.DB_PATH", db_path)

    client = TestClient(app)
    response = client.get("/stock/AAA/chart", params={"range": "3M"})

    assert response.status_code == 200
    body = response.json()
    # 5D is +2% (up) and is what the page's headline describes; 20D is -5% (down).
    # forecast_up must follow the 5D number, not the 20D one it happens to plot last.
    assert body["forecast_up"] is True


def test_forecast_up_falls_back_to_the_only_available_horizon(tmp_path, monkeypatch):
    db_path = tmp_path / "test.duckdb"
    con = duckdb.connect(str(db_path))
    bootstrap_schema(con)
    prices = pd.DataFrame({"ticker": ["BBB"] * 3, "date": pd.date_range("2024-01-01", periods=3), "close": [50.0, 51, 52]})
    upsert_wide(con, "prices", prices, ("ticker", "date"))
    predictions = pd.DataFrame(
        [{"ticker": "BBB", "prediction_date": pd.Timestamp("2024-01-03").date(), "horizon_days": 20, "expected_relative_return": -0.01}]
    )
    upsert_wide(con, "predictions", predictions, ("ticker", "prediction_date", "horizon_days"))
    con.close()
    monkeypatch.setattr("src.data.db.DB_PATH", db_path)
    monkeypatch.setattr("app.deps.DB_PATH", db_path)

    client = TestClient(app)
    response = client.get("/stock/BBB/chart", params={"range": "3M"})

    assert response.status_code == 200
    assert response.json()["forecast_up"] is False
