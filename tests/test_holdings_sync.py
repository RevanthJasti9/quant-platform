from __future__ import annotations

import duckdb

from src import holdings
from src.data.db import bootstrap_schema


def test_complete_broker_snapshot_removes_sold_broker_positions(tmp_path, monkeypatch):
    db_path = tmp_path / "holdings.duckdb"

    def connection():
        con = duckdb.connect(str(db_path))
        bootstrap_schema(con)
        return con

    monkeypatch.setattr(holdings, "get_connection", connection)
    holdings.sync_read_only_holdings(
        [
            {"symbol": "AAA", "quantity": 2, "average_buy_price": 10, "market_price": 11},
            {"symbol": "BBB", "quantity": 1, "average_buy_price": 20, "market_price": 21},
        ],
        replace_positions=True,
    )
    holdings.sync_read_only_holdings(
        [{"symbol": "AAA", "quantity": 2, "average_buy_price": 10, "market_price": 12}],
        replace_positions=True,
    )

    con = connection()
    rows = con.execute("SELECT ticker, broker_market_price, position_source FROM holdings ORDER BY ticker").fetchall()
    con.close()
    assert rows == [("AAA", None, "robinhood")]
