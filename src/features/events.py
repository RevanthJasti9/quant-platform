"""Event-derived features from news_events, insider_transactions,
sec_filings, and news_sentiment.

Both are joined onto price dates as true calendar-day rolling windows (not
trading-day row counts, which would silently span extra calendar days across
weekends/holidays since prices only has trading-day rows). A date only ever
reflects events already known by that date:

- News uses `received_at` (when this app actually ingested the article), not
  `published_at` -- a backdated or delayed article can't retroactively
  inform a feature computed for an earlier date.
- Insider transactions use `filing_date` (when the Form 4 actually became
  public), not `transaction_date` (when the trade happened). Insiders get up
  to 2 business days to file, so a feature keyed off transaction_date would
  occasionally use information before it was actually public -- exactly the
  kind of leak that inflates a backtest without holding up live. Falls back
  to transaction_date only for rows ingested before filing_date was tracked.
- SEC filings (8-Ks specifically -- material/unscheduled events, as opposed
  to the routine 10-Q/10-K calendar) use EDGAR's own filing_date directly.

Deterministic/rule-based only, per the product brief -- no NLP scoring, and
`duplicate_group` is used so syndicated retellings of the same story count
once. News coverage only goes back as far as `news.lookback_days` has
accumulated since this app started running (V1 doesn't backfill historical
news), so most of a multi-year backtest range will correctly show NaN here,
same as fundamentals before its first ingest date -- expected, and fine,
since XGBoost/LightGBM both handle missing values natively.
"""
from __future__ import annotations

import pandas as pd

_NEWS_WINDOWS_DAYS = [7, 30]
_INSIDER_WINDOWS_DAYS = [30, 90]
_FILING_WINDOWS_DAYS = [30, 90]
_NEGATIVE_EVENT_TYPES = {"regulatory", "lawsuit"}
_MATERIAL_FILING_TYPES = {"8-K"}


def _calendar_rolling_features(
    daily: pd.DataFrame,
    event_date_col: str,
    metric_cols: list[str],
    windows: list[int],
    recency_label: str | None = None,
    through_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """`daily` has one row per (ticker, event_date) with pre-summed metric
    columns. Reindexes each ticker to a full calendar-day range so
    `.rolling("{w}D")` sums true elapsed days, then optionally adds a
    "days since any metric was nonzero" recency column.

    The range always extends through `through_date` (typically the latest
    price date) even when a ticker's last qualifying event was long before
    it -- without this, a ticker with only one 8-K filing months ago would
    stop producing any output the day after that filing, instead of
    correctly showing an ever-growing "days since" and a rolling count that
    decays back to zero. The range only ever grows to cover through_date,
    never shrinks past a ticker's own last event.
    """
    frames = []
    for ticker, g in daily.groupby("ticker"):
        g = g.set_index(event_date_col).sort_index()[metric_cols]
        range_end = g.index.max()
        if through_date is not None and through_date > range_end:
            range_end = through_date
        full_range = pd.date_range(g.index.min(), range_end, freq="D")
        d = g.reindex(full_range, fill_value=0)

        feat = pd.DataFrame(index=d.index)
        for col in metric_cols:
            for w in windows:
                feat[f"{col}_{w}d"] = d[col].rolling(f"{w}D").sum()

        if recency_label:
            had_event = (d[metric_cols] != 0).any(axis=1)
            last_event_date = d.index.to_series().where(had_event).ffill()
            feat[recency_label] = (d.index.to_series() - last_event_date).dt.days

        feat["ticker"] = ticker
        frames.append(feat.reset_index().rename(columns={"index": "date"}))

    if not frames:
        return pd.DataFrame(columns=["ticker", "date"])
    return pd.concat(frames, ignore_index=True)


def compute_news_event_features(news: pd.DataFrame, through_date: pd.Timestamp | None = None) -> pd.DataFrame:
    """`news` needs columns: ticker, url, received_at, event_type,
    reliability_score, duplicate_group (i.e. news_events as stored).
    `through_date` extends the output through that date even for tickers
    whose most recent story is older -- see _calendar_rolling_features.
    """
    if news.empty:
        return pd.DataFrame(columns=["ticker", "date"])

    n = news.copy()
    n["event_date"] = pd.to_datetime(n["received_at"]).dt.normalize()
    n["reliability_score"] = n["reliability_score"].fillna(0.5)
    n["duplicate_group"] = n["duplicate_group"].fillna(n["url"])
    # Count each real-world story once, keeping its highest-reliability copy.
    n = n.sort_values("reliability_score", ascending=False).drop_duplicates(subset=["ticker", "duplicate_group"])
    n["is_negative"] = n["event_type"].isin(_NEGATIVE_EVENT_TYPES).astype(int)

    daily = (
        n.groupby(["ticker", "event_date"])
        .agg(
            news_event_count=("duplicate_group", "size"),
            news_reliability_weighted=("reliability_score", "sum"),
            news_negative_event_count=("is_negative", "sum"),
        )
        .reset_index()
    )

    return _calendar_rolling_features(
        daily,
        "event_date",
        ["news_event_count", "news_reliability_weighted", "news_negative_event_count"],
        _NEWS_WINDOWS_DAYS,
        recency_label="news_days_since_last_event",
        through_date=through_date,
    )


def compute_insider_event_features(insider: pd.DataFrame, through_date: pd.Timestamp | None = None) -> pd.DataFrame:
    """`insider` needs columns: ticker, transaction_date, transaction_code,
    shares, price, value, and (optional) filing_date, i.e. insider_transactions
    as stored. Only 'P' (open-market purchase) and 'S' (open-market sale)
    codes count as a buy/sell signal -- grants, option exercises, and
    tax-withholding dispositions (A/M/F/G/C/D/I/J) are compensation
    mechanics, not a voluntary trade reflecting the insider's own view.
    `through_date` extends the output through that date even for tickers
    whose most recent transaction is older -- see _calendar_rolling_features.
    """
    if insider.empty:
        return pd.DataFrame(columns=["ticker", "date"])

    ins = insider.copy()
    if "filing_date" in ins.columns:
        known_date = ins["filing_date"].fillna(ins["transaction_date"])
    else:
        known_date = ins["transaction_date"]
    ins["event_date"] = pd.to_datetime(known_date).dt.normalize()
    trade_value = ins["value"].fillna(ins["shares"].abs() * ins["price"]).fillna(0)
    is_buy = ins["transaction_code"] == "P"
    is_sell = ins["transaction_code"] == "S"
    ins["insider_buy_count"] = is_buy.astype(int)
    ins["insider_sell_count"] = is_sell.astype(int)
    ins["insider_net_value"] = trade_value.where(is_buy, 0) - trade_value.where(is_sell, 0)

    daily = (
        ins.groupby(["ticker", "event_date"])
        .agg(
            insider_buy_count=("insider_buy_count", "sum"),
            insider_sell_count=("insider_sell_count", "sum"),
            insider_net_value=("insider_net_value", "sum"),
        )
        .reset_index()
    )

    return _calendar_rolling_features(
        daily,
        "event_date",
        ["insider_buy_count", "insider_sell_count", "insider_net_value"],
        _INSIDER_WINDOWS_DAYS,
        recency_label="insider_days_since_last_txn",
        through_date=through_date,
    )


def compute_filing_event_features(filings: pd.DataFrame, through_date: pd.Timestamp | None = None) -> pd.DataFrame:
    """`filings` needs columns: ticker, filing_type, filing_date (i.e.
    sec_filings as stored). Scoped to 8-Ks only -- unscheduled, material
    events (earnings, executive changes, M&A, etc.), unlike the routine
    10-Q/10-K calendar which mostly just tracks quarter-end. Form 4s are
    covered separately and in more depth by compute_insider_event_features.
    `through_date` extends the output through that date even for tickers
    whose most recent 8-K is older -- see _calendar_rolling_features.
    """
    if filings.empty:
        return pd.DataFrame(columns=["ticker", "date"])

    f = filings[filings["filing_type"].isin(_MATERIAL_FILING_TYPES)].copy()
    if f.empty:
        return pd.DataFrame(columns=["ticker", "date"])

    f["event_date"] = pd.to_datetime(f["filing_date"]).dt.normalize()
    f["sec_8k_count"] = 1
    daily = f.groupby(["ticker", "event_date"]).agg(sec_8k_count=("sec_8k_count", "sum")).reset_index()

    return _calendar_rolling_features(
        daily,
        "event_date",
        ["sec_8k_count"],
        _FILING_WINDOWS_DAYS,
        recency_label="sec_days_since_last_8k",
        through_date=through_date,
    )


def compute_news_sentiment_features(sentiment: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """`sentiment` needs columns: ticker, as_of, sentiment_score (i.e.
    news_sentiment as stored). `prices` needs columns: ticker, date -- used
    only to build the per-ticker date grid this gets joined onto, same role
    it plays in compute_fundamental_features.

    Point-in-time-safe via a strict backward as-of merge (pd.merge_asof,
    same mechanism fundamentals uses): date T only ever sees the most
    recent sentiment score generated on or before T. Deliberately does NOT
    decay on its own the way the calendar-rolling features above do -- a
    sentiment reading is a "last known read", not a count of events in a
    trailing window, so it should persist until a fresher one replaces it.

    Unlike every other feature in this file, sentiment_score is LLM-derived
    (see src/models/news_digest.py), not a deterministic calculation --
    it's a candidate signal, not a validated one. Whether it earns a place
    in the model is a feature-importance question, not an assumption.
    """
    if sentiment.empty or prices.empty:
        return pd.DataFrame(columns=["ticker", "date"])

    px = prices[["ticker", "date"]].copy()
    px["date"] = pd.to_datetime(px["date"])
    sent = sentiment.copy()
    sent["as_of"] = pd.to_datetime(sent["as_of"])

    frames = []
    for ticker, g in px.groupby("ticker"):
        sdata = sent[sent["ticker"] == ticker].sort_values("as_of")
        if sdata.empty:
            continue
        g = g.sort_values("date")
        merged = pd.merge_asof(g, sdata[["as_of", "sentiment_score"]], left_on="date", right_on="as_of", direction="backward")
        frames.append(merged.drop(columns=["as_of"]).rename(columns={"sentiment_score": "news_sentiment_score"}))

    if not frames:
        return pd.DataFrame(columns=["ticker", "date"])
    return pd.concat(frames, ignore_index=True)
