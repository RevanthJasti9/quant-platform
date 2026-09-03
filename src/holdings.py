"""The user's actual portfolio -- stocks they really own, entered by hand
(shares + price paid). This is what "My Return" is computed from. Not
connected to a broker in V1 -- that's Robinhood in V3.

Any ticker held is automatically pulled into every future ingest run too
(see src/data/ingest.py's default ticker list), so a stock you own starts
getting real data and forecasts the moment you add it.
"""
from __future__ import annotations

import logging
from typing import Iterable, Mapping

import pandas as pd

from src.data.db import get_connection

logger = logging.getLogger(__name__)


def get_holdings_tickers() -> list[str]:
    con = get_connection()
    rows = con.execute("SELECT ticker FROM holdings ORDER BY added_at").fetchall()
    con.close()
    return [r[0] for r in rows]


def get_holdings() -> list[dict]:
    con = get_connection()
    df = con.execute(
        """
        SELECT ticker, shares, cost_basis, added_at,
               broker_market_price, broker_previous_close, broker_price_at,
               market_price_source
        FROM holdings
        ORDER BY added_at DESC
        """
    ).fetchdf()
    con.close()
    return df.to_dict("records")


def add_or_update_holding(ticker: str, shares: float, cost_basis: float) -> None:
    """Adding a ticker you already hold combines it with the existing
    position (weighted-average cost basis), the way buying more shares
    actually works -- it doesn't overwrite what you already told it.
    """
    ticker = ticker.upper().strip()
    con = get_connection()
    existing = con.execute("SELECT shares, cost_basis FROM holdings WHERE ticker = ?", [ticker]).fetchone()
    if existing:
        old_shares, old_cost = existing
        total_shares = old_shares + shares
        new_cost_basis = (old_shares * old_cost + shares * cost_basis) / total_shares
        con.execute(
            "UPDATE holdings SET shares = ?, cost_basis = ?, position_source = 'manual' WHERE ticker = ?",
            [total_shares, new_cost_basis, ticker],
        )
    else:
        con.execute(
            "INSERT INTO holdings (ticker, shares, cost_basis, added_at, position_source) VALUES (?, ?, ?, ?, 'manual')",
            [ticker, shares, cost_basis, pd.Timestamp.now("UTC")],
        )
    con.close()
    logger.info("Recorded holding: %s x%s @ %s", ticker, shares, cost_basis)


def update_holding(ticker: str, shares: float, cost_basis: float) -> bool:
    """Replace a holding's saved position details when a manual entry was wrong."""
    ticker = ticker.upper().strip()
    con = get_connection()
    row = con.execute(
        "UPDATE holdings SET shares = ?, cost_basis = ?, position_source = 'manual' WHERE ticker = ? RETURNING ticker",
        [shares, cost_basis, ticker],
    ).fetchone()
    con.close()
    updated = row is not None
    if updated:
        logger.info("Updated holding: %s x%s @ %s", ticker, shares, cost_basis)
    return updated


def sync_read_only_holdings(
    positions: Iterable[Mapping[str, object]],
    portfolio: Mapping[str, object] | None = None,
    provider: str = "robinhood",
    replace_positions: bool = False,
) -> list[str]:
    """Upsert a broker-provided holdings snapshot without any trade capability.

    Each position must contain a symbol, quantity, and average_buy_price.
    A snapshot may also include market_price and previous_close from the
    broker's quote feed, plus an account-level portfolio summary.
    Set replace_positions=True only for a confirmed complete snapshot. In
    that mode, positions no longer returned by the broker are removed; this
    is what keeps a read-only broker mirror accurate after a sale.
    """
    provider = provider.lower().strip()
    valid_positions: list[dict[str, object]] = []
    for position in positions:
        ticker = str(position["symbol"]).upper().strip()
        shares = float(position["quantity"])
        cost_basis = float(position["average_buy_price"])
        if not ticker or shares <= 0 or cost_basis <= 0:
            logger.warning("Skipped invalid broker holding for %s", ticker or "unknown ticker")
            continue
        valid_positions.append(
            {
                "ticker": ticker,
                "shares": shares,
                "cost_basis": cost_basis,
            }
        )

    con = get_connection()
    synced: list[str] = []
    try:
        for position in valid_positions:
            ticker = str(position["ticker"])
            shares = float(position["shares"])
            cost_basis = float(position["cost_basis"])
            con.execute(
                """
                INSERT INTO holdings (
                    ticker, shares, cost_basis, added_at,
                    position_source
                )
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT (ticker) DO UPDATE SET
                    shares = excluded.shares,
                    cost_basis = excluded.cost_basis,
                    added_at = excluded.added_at,
                    broker_market_price = NULL,
                    broker_previous_close = NULL,
                    broker_price_at = NULL,
                    market_price_source = NULL,
                    position_source = excluded.position_source
                """,
                [
                    ticker,
                    shares,
                    cost_basis,
                    pd.Timestamp.now("UTC"),
                    provider,
                ],
            )
            synced.append(ticker)
        if replace_positions:
            if synced:
                placeholders = ", ".join("?" for _ in synced)
                con.execute(
                    f"DELETE FROM holdings WHERE position_source = ? AND ticker NOT IN ({placeholders})",
                    [provider, *synced],
                )
            else:
                con.execute("DELETE FROM holdings WHERE position_source = ?", [provider])
        if portfolio is not None:
            con.execute(
                """
                INSERT INTO broker_portfolio_snapshots (
                    provider, synced_at, total_value, equity_value, cash, crypto_value
                )
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT (provider) DO UPDATE SET
                    synced_at = excluded.synced_at,
                    total_value = excluded.total_value,
                    equity_value = excluded.equity_value,
                    cash = excluded.cash,
                    crypto_value = excluded.crypto_value
                """,
                [
                    provider,
                    pd.Timestamp.now("UTC"),
                    float(portfolio["total_value"]),
                    float(portfolio["equity_value"]),
                    float(portfolio.get("cash", 0)),
                    float(portfolio.get("crypto_value", 0)),
                ],
            )
    finally:
        con.close()
    logger.info("Read-only %s holdings snapshot synced: %s", provider, ", ".join(synced) or "no positions")
    return synced


def remove_holding(ticker: str) -> None:
    ticker = ticker.upper().strip()
    con = get_connection()
    con.execute("DELETE FROM holdings WHERE ticker = ?", [ticker])
    con.close()
    logger.info("Removed holding: %s", ticker)
