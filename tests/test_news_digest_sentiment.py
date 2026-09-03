"""Sentiment/summary parsing (src/models/news_digest.py:parse_response).
Small local LLMs aren't perfectly reliable about following a strict output
format, so this covers the well-formatted case and the fallback paths --
malformed sentiment, missing SUMMARY: prefix, and out-of-range values.
"""
from __future__ import annotations

from src.models.news_digest import parse_response


def test_well_formatted_response_parses_both_fields():
    text = "SENTIMENT: 0.7\nSUMMARY: Amazon reported strong earnings and raised guidance."
    sentiment, summary = parse_response(text)
    assert sentiment == 0.7
    assert summary == "Amazon reported strong earnings and raised guidance."


def test_negative_sentiment_parses_correctly():
    sentiment, _ = parse_response("SENTIMENT: -0.8\nSUMMARY: Bad news across the board.")
    assert sentiment == -0.8


def test_out_of_range_sentiment_is_clamped():
    sentiment, _ = parse_response("SENTIMENT: 2.5\nSUMMARY: Wildly overstated.")
    assert sentiment == 1.0

    sentiment, _ = parse_response("SENTIMENT: -4.0\nSUMMARY: Wildly understated.")
    assert sentiment == -1.0


def test_missing_sentiment_line_falls_back_to_no_sentiment_but_keeps_summary():
    text = "SUMMARY: The model skipped the sentiment line but wrote a summary."
    sentiment, summary = parse_response(text)
    assert sentiment is None
    assert summary == "The model skipped the sentiment line but wrote a summary."


def test_response_that_ignores_the_format_entirely_falls_back_to_whole_text_as_summary():
    """If the model doesn't follow SENTIMENT:/SUMMARY: at all, the digest
    feature (which predates sentiment scoring) must not regress -- the raw
    response becomes the summary rather than being silently dropped.
    """
    text = "Amazon had a strong quarter with beat earnings and raised guidance."
    sentiment, summary = parse_response(text)
    assert sentiment is None
    assert summary == text


def test_non_numeric_sentiment_value_is_ignored_not_crashed_on():
    sentiment, summary = parse_response("SENTIMENT: very positive\nSUMMARY: Good news.")
    assert sentiment is None
    assert summary == "Good news."


def test_empty_response_yields_nothing():
    sentiment, summary = parse_response("")
    assert sentiment is None
    assert summary is None
