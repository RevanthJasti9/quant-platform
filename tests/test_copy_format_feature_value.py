"""Display formatting for the new event feature columns (app/copy.py). The
existing formatter for technical/fundamental fields is exercised implicitly
by every stock page render; this covers the branches added for
news/insider/filing features specifically.
"""
from __future__ import annotations

from app.copy import format_feature_value


def test_none_and_nan_render_as_dash():
    assert format_feature_value("insider_buy_count_90d", None) == "—"
    assert format_feature_value("insider_buy_count_90d", float("nan")) == "—"


def test_count_fields_render_as_plain_integers():
    assert format_feature_value("news_event_count_7d", 3.0) == "3"
    assert format_feature_value("insider_sell_count_90d", 24.0) == "24"


def test_days_since_fields_render_as_relative_time():
    assert format_feature_value("insider_days_since_last_txn", 0.0) == "Today"
    assert format_feature_value("news_days_since_last_event", 5.0) == "5d ago"


def test_signed_money_fields_show_sign_and_magnitude():
    assert format_feature_value("insider_net_value_90d", 1_000_000.0) == "+$1.0M"
    assert format_feature_value("insider_net_value_90d", -18_074_160.0) == "-$18.1M"


def test_news_sentiment_shows_qualitative_label_and_raw_score():
    assert format_feature_value("news_sentiment_score", 0.7) == "Positive (+0.70)"
    assert format_feature_value("news_sentiment_score", -0.8) == "Negative (-0.80)"
    assert format_feature_value("news_sentiment_score", 0.1) == "Neutral (+0.10)"
    assert format_feature_value("news_sentiment_score", 0.0) == "Neutral (+0.00)"
