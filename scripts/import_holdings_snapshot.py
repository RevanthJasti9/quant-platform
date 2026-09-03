"""Import a read-only broker holdings snapshot into the local dashboard.

Read JSON from standard input. Expected shape: {"positions": [...]} or a
plain list, where each position has symbol, quantity, and average_buy_price.
The object form may also contain a portfolio summary and broker market prices.
This script only updates the local DuckDB holdings table; it has no order,
account, or broker-authentication code.
"""
from __future__ import annotations

import json
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.holdings import sync_read_only_holdings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Treat input as a complete broker snapshot and remove broker positions no longer present.",
    )
    args = parser.parse_args()
    payload = json.load(sys.stdin)
    positions = payload.get("positions", []) if isinstance(payload, dict) else payload
    if not isinstance(positions, list):
        raise SystemExit("Snapshot must be a JSON list or an object with a positions list.")
    portfolio = payload.get("portfolio") if isinstance(payload, dict) else None
    synced = sync_read_only_holdings(positions, portfolio=portfolio, replace_positions=args.replace)
    print(f"Synced {len(synced)} holding(s): {', '.join(synced) or 'none'}")


if __name__ == "__main__":
    main()
