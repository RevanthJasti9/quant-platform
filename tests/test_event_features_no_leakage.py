"""Event features (src/features/events.py) must be point-in-time safe: a
feature computed for date T can't change when a later event is added to the
input, rolling windows must only count events within the window (calendar
days, not row counts), syndicated retellings of the same story must count
once via duplicate_group, and an insider transaction can't be "known" before
its Form 4 was actually filed (transaction_date is when the trade happened;
filing_date is when the market could have found out about it).
"""
from __future__ import annotations

import pandas as pd

from src.features.events import (
    compute_filing_event_features,
    compute_insider_event_features,
    compute_news_event_features,
    compute_news_sentiment_features,
)


def _news_row(ticker, received_at, event_type="general_news", reliability=0.55, group=None):
    return {
        "ticker": ticker,
        "url": f"https://example.com/{group or received_at}",
        "received_at": pd.Timestamp(received_at),
        "event_type": event_type,
        "reliability_score": reliability,
        "duplicate_group": group or f"group-{received_at}",
    }


def _txn(date, code, shares=100.0, price=10.0, ticker="AAA", filing_date=None):
    row = {
        "ticker": ticker,
        "transaction_date": pd.Timestamp(date),
        "transaction_code": code,
        "shares": shares,
        "price": price,
        "value": shares * price,
    }
    if filing_date is not None:
        row["filing_date"] = pd.Timestamp(filing_date)
    return row


def _filing(date, filing_type="8-K", ticker="AAA"):
    return {"ticker": ticker, "filing_type": filing_type, "filing_date": pd.Timestamp(date)}


def test_news_feature_at_date_t_unaffected_by_a_later_event():
    early = pd.DataFrame([_news_row("AAA", "2024-01-05")])
    late = pd.DataFrame([_news_row("AAA", "2024-01-05"), _news_row("AAA", "2024-01-20")])

    row_early = compute_news_event_features(early).set_index("date").loc[pd.Timestamp("2024-01-05")]
    row_late = compute_news_event_features(late).set_index("date").loc[pd.Timestamp("2024-01-05")]

    assert row_early["news_event_count_7d"] == row_late["news_event_count_7d"]
    assert row_early["news_days_since_last_event"] == row_late["news_days_since_last_event"]


def test_news_rolling_count_only_sees_events_within_the_window():
    news = pd.DataFrame([_news_row("AAA", "2024-01-01"), _news_row("AAA", "2024-01-25")])
    row = compute_news_event_features(news).set_index("date").loc[pd.Timestamp("2024-01-25")]
    # the Jan 1 event is 24 days earlier -- inside the 30d window, outside the 7d one
    assert row["news_event_count_7d"] == 1
    assert row["news_event_count_30d"] == 2


def test_duplicate_group_counts_a_syndicated_story_once_at_its_best_reliability():
    news = pd.DataFrame(
        [
            _news_row("AAA", "2024-01-05T09:00:00", reliability=0.55, group="same-story"),
            _news_row("AAA", "2024-01-05T10:00:00", reliability=0.85, group="same-story"),
        ]
    )
    row = compute_news_event_features(news).set_index("date").loc[pd.Timestamp("2024-01-05")]
    assert row["news_event_count_7d"] == 1
    assert row["news_reliability_weighted_7d"] == 0.85


def test_regulatory_and_lawsuit_events_count_as_negative():
    news = pd.DataFrame([_news_row("AAA", "2024-01-05", event_type="regulatory")])
    row = compute_news_event_features(news).set_index("date").loc[pd.Timestamp("2024-01-05")]
    assert row["news_negative_event_count_7d"] == 1


def test_insider_feature_at_date_t_unaffected_by_a_later_transaction():
    early = pd.DataFrame([_txn("2024-02-01", "P")])
    late = pd.DataFrame([_txn("2024-02-01", "P"), _txn("2024-02-20", "S")])

    row_early = compute_insider_event_features(early).set_index("date").loc[pd.Timestamp("2024-02-01")]
    row_late = compute_insider_event_features(late).set_index("date").loc[pd.Timestamp("2024-02-01")]

    assert row_early["insider_net_value_30d"] == row_late["insider_net_value_30d"]
    assert row_early["insider_days_since_last_txn"] == row_late["insider_days_since_last_txn"]


def test_insider_only_counts_open_market_buy_and_sell_codes():
    # 'A' (grant) is compensation, not a voluntary trade -- must not count as a buy.
    txns = pd.DataFrame([_txn("2024-02-01", "P"), _txn("2024-02-01", "A", shares=500.0, price=0.0)])
    row = compute_insider_event_features(txns).set_index("date").loc[pd.Timestamp("2024-02-01")]
    assert row["insider_buy_count_30d"] == 1


def test_insider_net_value_nets_buys_against_sells():
    txns = pd.DataFrame([_txn("2024-02-01", "P", shares=100, price=10), _txn("2024-02-01", "S", shares=40, price=10)])
    row = compute_insider_event_features(txns).set_index("date").loc[pd.Timestamp("2024-02-01")]
    assert row["insider_net_value_30d"] == 1000 - 400


def test_insider_uses_filing_date_not_transaction_date_for_recency():
    """Regression test for a real point-in-time leak: a trade on 1/10 that
    wasn't filed (made public) until 1/20 must not appear in the feature
    until 1/20 -- using transaction_date directly would let the model see it
    up to 10 days before the market actually could have.
    """
    txns = pd.DataFrame(
        [
            _txn("2024-01-01", "S", filing_date="2024-01-02", shares=10, price=10),
            _txn("2024-01-10", "P", filing_date="2024-01-20"),
        ]
    )
    feat = compute_insider_event_features(txns).set_index("date")

    days_since_on_jan10 = (pd.Timestamp("2024-01-10") - pd.Timestamp("2024-01-02")).days
    assert feat.loc[pd.Timestamp("2024-01-10")]["insider_buy_count_30d"] == 0
    assert feat.loc[pd.Timestamp("2024-01-10")]["insider_days_since_last_txn"] == days_since_on_jan10

    assert feat.loc[pd.Timestamp("2024-01-20")]["insider_buy_count_30d"] == 1
    assert feat.loc[pd.Timestamp("2024-01-20")]["insider_days_since_last_txn"] == 0


def test_insider_falls_back_to_transaction_date_when_filing_date_is_missing():
    """Rows ingested before filing_date was tracked have it as NULL --
    those must still work, just less precisely (falls back to transaction_date).
    """
    txns = pd.DataFrame([_txn("2024-02-01", "P"), _txn("2024-02-05", "S", filing_date="2024-02-06")])
    feat = compute_insider_event_features(txns).set_index("date")
    assert feat.loc[pd.Timestamp("2024-02-01")]["insider_buy_count_30d"] == 1


def test_filing_feature_counts_8k_and_tracks_recency():
    filings = pd.DataFrame([_filing("2024-03-01"), _filing("2024-03-20")])
    row = compute_filing_event_features(filings).set_index("date").loc[pd.Timestamp("2024-03-20")]
    assert row["sec_8k_count_30d"] == 2
    assert row["sec_days_since_last_8k"] == 0


def test_filing_feature_ignores_non_8k_filing_types():
    # Form 4s are covered separately (and in more depth) by the insider
    # features; 10-Q/10-K are routine/scheduled, not a surprise signal.
    filings = pd.DataFrame([_filing("2024-03-01", filing_type="10-Q"), _filing("2024-03-20", filing_type="4")])
    assert compute_filing_event_features(filings).empty


def test_filing_feature_at_date_t_unaffected_by_a_later_filing():
    early = pd.DataFrame([_filing("2024-03-01")])
    late = pd.DataFrame([_filing("2024-03-01"), _filing("2024-03-25")])
    row_early = compute_filing_event_features(early).set_index("date").loc[pd.Timestamp("2024-03-01")]
    row_late = compute_filing_event_features(late).set_index("date").loc[pd.Timestamp("2024-03-01")]
    assert row_early["sec_8k_count_30d"] == row_late["sec_8k_count_30d"]


def test_filing_feature_extends_through_date_for_a_sparse_ticker():
    """Regression test: a ticker with a single old 8-K must still produce a
    growing 'days since' and a decaying rolling count on later dates, not
    silently stop existing the day after that filing. Before the
    through_date fix, the output's calendar range stopped at the last
    event's own date, so any date after it -- however recent -- was simply
    absent (NaN after the merge onto the price grid), even though "62 days
    since the last 8-K" is a perfectly valid, computable value.
    """
    filings = pd.DataFrame([_filing("2024-01-01")])
    result = compute_filing_event_features(filings, through_date=pd.Timestamp("2024-03-15"))

    last_row = result.set_index("date").loc[pd.Timestamp("2024-03-15")]
    assert last_row["sec_8k_count_90d"] == 1  # still inside the 90d window
    assert last_row["sec_days_since_last_8k"] == (pd.Timestamp("2024-03-15") - pd.Timestamp("2024-01-01")).days

    # far enough out that the 90d window has fully rolled off
    later = compute_filing_event_features(filings, through_date=pd.Timestamp("2024-06-01"))
    last_row = later.set_index("date").loc[pd.Timestamp("2024-06-01")]
    assert last_row["sec_8k_count_90d"] == 0


def test_through_date_never_shrinks_a_tickers_own_later_events():
    filings = pd.DataFrame([_filing("2024-05-01")])
    result = compute_filing_event_features(filings, through_date=pd.Timestamp("2024-01-01"))
    assert result["date"].max() == pd.Timestamp("2024-05-01")


def _prices(ticker, dates):
    return pd.DataFrame({"ticker": ticker, "date": pd.to_datetime(dates)})


def test_sentiment_persists_forward_until_a_fresher_reading_replaces_it():
    """Unlike the calendar-rolling event features, a sentiment reading is a
    'last known read', not an event count -- it should carry forward
    unchanged on every date after it until a newer reading appears.
    """
    sentiment = pd.DataFrame(
        [
            {"ticker": "AAA", "as_of": "2024-01-05", "sentiment_score": 0.6},
            {"ticker": "AAA", "as_of": "2024-01-15", "sentiment_score": -0.4},
        ]
    )
    prices = _prices("AAA", ["2024-01-06", "2024-01-10", "2024-01-16"])

    feat = compute_news_sentiment_features(sentiment, prices).set_index("date")
    assert feat.loc[pd.Timestamp("2024-01-06")]["news_sentiment_score"] == 0.6
    assert feat.loc[pd.Timestamp("2024-01-10")]["news_sentiment_score"] == 0.6  # still the 1/5 reading
    assert feat.loc[pd.Timestamp("2024-01-16")]["news_sentiment_score"] == -0.4  # the fresher 1/15 reading


def test_sentiment_not_visible_before_it_was_generated():
    sentiment = pd.DataFrame([{"ticker": "AAA", "as_of": "2024-01-15", "sentiment_score": 0.6}])
    prices = _prices("AAA", ["2024-01-05", "2024-01-14"])

    feat = compute_news_sentiment_features(sentiment, prices).set_index("date")
    assert pd.isna(feat.loc[pd.Timestamp("2024-01-05")]["news_sentiment_score"])
    assert pd.isna(feat.loc[pd.Timestamp("2024-01-14")]["news_sentiment_score"])


def test_sentiment_at_date_t_unaffected_by_a_later_reading():
    early = pd.DataFrame([{"ticker": "AAA", "as_of": "2024-01-05", "sentiment_score": 0.6}])
    late = pd.concat(
        [early, pd.DataFrame([{"ticker": "AAA", "as_of": "2024-01-20", "sentiment_score": -0.9}])], ignore_index=True
    )
    prices = _prices("AAA", ["2024-01-06"])

    row_early = compute_news_sentiment_features(early, prices).set_index("date").loc[pd.Timestamp("2024-01-06")]
    row_late = compute_news_sentiment_features(late, prices).set_index("date").loc[pd.Timestamp("2024-01-06")]
    assert row_early["news_sentiment_score"] == row_late["news_sentiment_score"] == 0.6


def test_sentiment_empty_input_returns_empty_frame():
    empty_sentiment = pd.DataFrame(columns=["ticker", "as_of", "sentiment_score"])
    result = compute_news_sentiment_features(empty_sentiment, _prices("AAA", ["2024-01-06"]))
    assert result.empty
    assert list(result.columns) == ["ticker", "date"]


def test_empty_input_returns_empty_frame_with_ticker_date_columns():
    empty_news = pd.DataFrame(
        columns=["ticker", "url", "received_at", "event_type", "reliability_score", "duplicate_group"]
    )
    result = compute_news_event_features(empty_news)
    assert result.empty
    assert list(result.columns) == ["ticker", "date"]

    empty_insider = pd.DataFrame(
        columns=["ticker", "transaction_date", "filing_date", "transaction_code", "shares", "price", "value"]
    )
    result = compute_insider_event_features(empty_insider)
    assert result.empty
    assert list(result.columns) == ["ticker", "date"]

    empty_filings = pd.DataFrame(columns=["ticker", "filing_type", "filing_date"])
    result = compute_filing_event_features(empty_filings)
    assert result.empty
    assert list(result.columns) == ["ticker", "date"]
