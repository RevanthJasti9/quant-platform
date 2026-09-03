"""The prediction journal's other half: fills in actual_relative_return /
error / evaluated_at on `predictions` rows once enough price history exists
to compute the real forward return for that horizon. Every prediction ever
made stays in the table permanently — nothing gets deleted, so this is what
lets you later ask "what patterns show up in the worst 1,000 predictions?"
"""
from __future__ import annotations

import logging

import pandas as pd

from src.config import get_settings
from src.data.db import get_connection
from src.models.targets import build_targets

logger = logging.getLogger(__name__)


def compute_evaluations(pending: pd.DataFrame, prices: pd.DataFrame, settings: dict) -> pd.DataFrame:
    """Pure function, no DB access: given predictions still awaiting an
    outcome and the full price history, returns the subset that can be
    scored right now (enough real price history exists past their horizon)
    with actual_relative_return/error/evaluated_at filled in. A prediction
    whose horizon hasn't elapsed yet -- no future price data exists for
    it -- is correctly excluded, not scored early. This is the actual
    "was this prediction right or wrong" logic, kept separate from the I/O
    below specifically so it can be unit-tested without a database.
    """
    if pending.empty or prices.empty:
        return pending.iloc[0:0]

    targets = build_targets(prices, settings["benchmark"], settings["models"]["horizons_days"])
    targets = targets.rename(columns={"date": "prediction_date", "forward_relative_return": "actual_relative_return"})

    merged = pending.merge(
        targets[["ticker", "prediction_date", "horizon_days", "actual_relative_return"]],
        on=["ticker", "prediction_date", "horizon_days"],
        how="inner",
    )
    if merged.empty:
        return merged

    merged["error"] = merged["actual_relative_return"] - merged["expected_relative_return"]
    merged["evaluated_at"] = pd.Timestamp.now("UTC")
    return merged


def evaluate_predictions() -> int:
    settings = get_settings()
    con = get_connection()
    prices = con.execute("SELECT ticker, date, close FROM prices ORDER BY ticker, date").fetchdf()
    pending = con.execute(
        "SELECT ticker, prediction_date, horizon_days, expected_relative_return "
        "FROM predictions WHERE actual_relative_return IS NULL"
    ).fetchdf()
    con.close()

    merged = compute_evaluations(pending, prices, settings)
    if merged.empty:
        return 0

    con = get_connection()
    con.register("_eval_df", merged)
    con.execute(
        """
        UPDATE predictions p
        SET actual_relative_return = e.actual_relative_return,
            error = e.error,
            evaluated_at = e.evaluated_at
        FROM _eval_df e
        WHERE p.ticker = e.ticker
          AND p.prediction_date = e.prediction_date
          AND p.horizon_days = e.horizon_days
        """
    )
    con.unregister("_eval_df")
    con.close()
    logger.info("Evaluated %d predictions", len(merged))
    return len(merged)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print(evaluate_predictions())
