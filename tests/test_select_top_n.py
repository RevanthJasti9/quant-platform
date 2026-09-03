"""Backtest stock selection with risk controls (src/backtesting/engine.py:
select_top_n) -- a per-sector cap and a liquidity floor, both applied at
selection time so a name that fails either is skipped in favor of the next
best-scoring eligible name, not just left out of an otherwise-full book.
"""
from __future__ import annotations

import pandas as pd

from src.backtesting.engine import select_top_n


def _scores(pairs):
    tickers, values = zip(*pairs)
    return pd.Series(values, index=list(tickers))


def test_plain_top_n_with_no_constraints():
    scores = _scores([("A", 5), ("B", 4), ("C", 3), ("D", 2), ("E", 1)])
    assert select_top_n(scores, top_n=3) == {"A", "B", "C"}


def test_sector_cap_skips_an_overrepresented_sector_in_favor_of_the_next_best():
    # A, B, C are all Tech and score highest; capping Tech at 1 must skip
    # B and C in favor of D (Health) and E (Energy), not just shrink the book.
    scores = _scores([("A", 5), ("B", 4), ("C", 3), ("D", 2), ("E", 1)])
    sector_map = {"A": "Tech", "B": "Tech", "C": "Tech", "D": "Health", "E": "Energy"}

    selected = select_top_n(scores, top_n=3, sector_map=sector_map, max_per_sector=1)
    assert selected == {"A", "D", "E"}


def test_sector_cap_disabled_when_sector_map_is_none():
    scores = _scores([("A", 5), ("B", 4), ("C", 3)])
    selected = select_top_n(scores, top_n=3, sector_map=None, max_per_sector=1)
    assert selected == {"A", "B", "C"}


def test_liquidity_floor_excludes_illiquid_names_in_favor_of_the_next_best():
    scores = _scores([("A", 5), ("B", 4), ("C", 3)])
    liquid = {"A", "C"}  # B scores 2nd-best but isn't liquid enough

    selected = select_top_n(scores, top_n=2, liquid_tickers=liquid)
    assert selected == {"A", "C"}


def test_liquidity_floor_disabled_when_none():
    scores = _scores([("A", 5), ("B", 4)])
    assert select_top_n(scores, top_n=2, liquid_tickers=None) == {"A", "B"}


def test_sector_cap_and_liquidity_floor_combine():
    scores = _scores([("A", 5), ("B", 4), ("C", 3), ("D", 2)])
    sector_map = {"A": "Tech", "B": "Tech", "C": "Tech", "D": "Health"}
    liquid = {"A", "B", "D"}  # C is both Tech (capped) and illiquid

    selected = select_top_n(scores, top_n=2, sector_map=sector_map, max_per_sector=1, liquid_tickers=liquid)
    assert selected == {"A", "D"}


def test_unknown_sector_tickers_are_grouped_together_for_the_cap():
    scores = _scores([("A", 5), ("B", 4), ("C", 3)])
    sector_map = {"A": "Tech"}  # B, C have no known sector -> both "Unknown"

    selected = select_top_n(scores, top_n=3, sector_map=sector_map, max_per_sector=1)
    assert selected == {"A", "B"}  # C skipped: a second "Unknown" would exceed the cap


def test_fewer_eligible_candidates_than_top_n_returns_what_is_available():
    scores = _scores([("A", 5), ("B", 4), ("C", 3)])
    sector_map = {"A": "Tech", "B": "Tech", "C": "Tech"}
    selected = select_top_n(scores, top_n=5, sector_map=sector_map, max_per_sector=1)
    assert selected == {"A"}
