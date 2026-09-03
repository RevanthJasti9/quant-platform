"""app.copy.ticker_avatar -- the small colored letter badge shown next to
each holding, generated locally from the ticker (no logo API / network
dependency), inspired by how Ghostfolio and similar portfolio trackers mark
each row. Same ticker must always produce the same badge.
"""
from __future__ import annotations

from app.copy import AVATAR_COLOR_COUNT, ticker_avatar


def test_uses_company_name_first_letter_when_available():
    assert ticker_avatar("MSFT", "Microsoft Corporation")["letter"] == "M"
    assert ticker_avatar("GOOGL", "Alphabet Inc.")["letter"] == "A"


def test_falls_back_to_ticker_first_letter_without_a_company_name():
    assert ticker_avatar("MSFT", None)["letter"] == "M"
    assert ticker_avatar("msft", None)["letter"] == "M"  # uppercased


def test_same_ticker_always_gets_the_same_color():
    a = ticker_avatar("AMZN", "Amazon.com, Inc.")
    b = ticker_avatar("AMZN", "Amazon.com, Inc.")
    assert a == b


def test_color_class_is_within_the_defined_palette_range():
    for ticker in ["A", "AAPL", "ZZZZ", "MSFT", "KO", "JNJ", "NVDA"]:
        result = ticker_avatar(ticker)
        index = int(result["color_class"].rsplit("-", 1)[1])
        assert 0 <= index < AVATAR_COLOR_COUNT


def test_different_tickers_are_not_all_the_same_color():
    tickers = ["AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "KO", "JNJ", "V", "MA"]
    colors = {ticker_avatar(t)["color_class"] for t in tickers}
    assert len(colors) > 1
