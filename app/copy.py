"""Plain-language translations for the dashboard. All the jargon (RSI,
Sharpe, alpha, ...) lives here in one place so the templates never have to
explain a term inline -- they just call these helpers.
"""
from __future__ import annotations

import json

from src.models.feature_labels import feature_label

FEATURE_GROUPS: list[tuple[str, list[str]]] = [
    ("Price Trend", ["momentum_5d", "momentum_10d", "momentum_20d", "momentum_60d", "rsi", "macd", "macd_signal", "macd_hist"]),
    ("vs. Moving Averages", ["close_to_ma_10", "close_to_ma_20", "close_to_ma_50", "close_to_ma_200"]),
    ("Risk & Activity", ["volatility", "volume_zscore"]),
    ("Value & Financial Health", [
        "pe_ratio", "forward_pe", "price_to_book", "revenue_growth", "earnings_growth",
        "gross_margin", "operating_margin", "profit_margin", "debt_to_equity",
        "return_on_equity", "dividend_yield", "fcf_yield", "log_market_cap",
    ]),
    ("Compared to the Market & Peers", [
        "rel_return_5d_vs_benchmark", "rel_return_5d_vs_sector",
        "rel_return_10d_vs_benchmark", "rel_return_10d_vs_sector",
        "rel_return_20d_vs_benchmark", "rel_return_20d_vs_sector",
    ]),
    ("Recent Events", [
        "news_sentiment_score",
        "news_event_count_7d", "news_negative_event_count_7d", "news_days_since_last_event",
        "insider_buy_count_90d", "insider_sell_count_90d", "insider_net_value_90d", "insider_days_since_last_txn",
        "sec_8k_count_90d", "sec_days_since_last_8k",
    ]),
    ("Company", ["sector", "industry"]),
]

# backtest metric -> (short label, one-line explanation)
METRIC_LABELS: dict[str, tuple[str, str]] = {
    "sharpe": ("Risk-Adjusted Return", "Return earned per unit of risk taken. Above 1 is generally considered good."),
    "cagr": ("Yearly Growth Rate", "How much the strategy grew per year, on average."),
    "max_drawdown": ("Worst Drop", "The biggest peak-to-trough loss during the test period."),
    "win_rate": ("Winning Days", "Share of days the strategy made money."),
    "turnover": ("Trading Frequency", "How much of the portfolio gets swapped out at each rebalance."),
    "alpha": ("Edge vs. the Market", "Extra yearly growth compared to just holding the S&P 500."),
    "volatility": ("Price Swings", "How bumpy the strategy's returns were."),
    "calibration": ("Ranking Skill (IC)", "How well the model's score actually predicts which stocks do better. Near 0 means the ranking isn't much better than chance."),
    "quantile_spread": ("Top vs. Bottom Picks", "Return difference between the model's highest-scored and lowest-scored stocks. A real edge should show a clear positive gap."),
}


_PERCENT_FIELDS = {
    "momentum_5d", "momentum_10d", "momentum_20d", "momentum_60d",
    "close_to_ma_10", "close_to_ma_20", "close_to_ma_50", "close_to_ma_200",
    "volatility", "revenue_growth", "earnings_growth", "gross_margin",
    "operating_margin", "profit_margin", "dividend_yield", "fcf_yield", "return_on_equity",
    "rel_return_5d_vs_benchmark", "rel_return_5d_vs_sector",
    "rel_return_10d_vs_benchmark", "rel_return_10d_vs_sector",
    "rel_return_20d_vs_benchmark", "rel_return_20d_vs_sector",
}
_RATIO_FIELDS = {"pe_ratio", "forward_pe", "price_to_book", "debt_to_equity"}
_COUNT_FIELDS = {
    "news_event_count_7d", "news_event_count_30d", "news_negative_event_count_7d", "news_negative_event_count_30d",
    "insider_buy_count_30d", "insider_buy_count_90d", "insider_sell_count_30d", "insider_sell_count_90d",
    "sec_8k_count_30d", "sec_8k_count_90d",
}
_DAYS_SINCE_FIELDS = {"news_days_since_last_event", "insider_days_since_last_txn", "sec_days_since_last_8k"}
_SIGNED_MONEY_FIELDS = {"insider_net_value_30d", "insider_net_value_90d"}


def _format_money(value: float) -> str:
    for threshold, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M")):
        if abs(value) >= threshold:
            return f"${value / threshold:.1f}{suffix}"
    return f"${value:,.0f}"


def format_bytes(n: float) -> str:
    for threshold, suffix in ((1024**4, "TB"), (1024**3, "GB"), (1024**2, "MB"), (1024, "KB")):
        if n >= threshold:
            return f"{n / threshold:.1f} {suffix}"
    return f"{n:.0f} B"


_TABLE_LABELS = {
    "sec_filings": "SEC filings",
    "news_events": "News events",
    "news_digests": "News digests (LLM)",
    "news_sentiment": "News sentiment (LLM)",
    "insider_transactions": "Insider transactions",
    "data_quality_results": "Data quality checks",
    "broker_portfolio_snapshots": "Broker portfolio snapshots",
    "ingest_runs": "Ingest runs",
    "source_runs": "Source runs",
    "model_versions": "Model versions",
}


def table_label(table_name: str) -> str:
    return _TABLE_LABELS.get(table_name, table_name.replace("_", " ").title())


def _format_signed_money(value: float) -> str:
    sign = "+" if value >= 0 else "-"
    return f"{sign}{_format_money(abs(value))}"


def format_feature_value(key: str, value) -> str:
    if value is None or (isinstance(value, float) and value != value):  # NaN
        return "—"
    if key == "log_market_cap":
        try:
            import math

            return _format_money(math.exp(value))
        except (OverflowError, ValueError):
            return "—"
    if key in ("sector", "industry"):
        return str(value)
    if key == "rsi":
        return f"{value:.0f} / 100"
    if key in ("macd", "macd_signal", "macd_hist"):
        return f"{value:+.2f}"
    if key == "volume_zscore":
        if value >= 1.5:
            return "Much higher than usual"
        if value >= 0.75:
            return "Higher than usual"
        if value <= -1.5:
            return "Much lower than usual"
        if value <= -0.75:
            return "Lower than usual"
        return "About normal"
    if key == "news_sentiment_score":
        label = "Positive" if value > 0.3 else "Negative" if value < -0.3 else "Neutral"
        return f"{label} ({value:+.2f})"
    if key in _PERCENT_FIELDS:
        return f"{value * 100:+.1f}%"
    if key in _RATIO_FIELDS:
        return f"{value:.1f}x"
    if key in _COUNT_FIELDS:
        return f"{value:.0f}"
    if key in _DAYS_SINCE_FIELDS:
        return "Today" if value == 0 else f"{value:.0f}d ago"
    if key in _SIGNED_MONEY_FIELDS:
        return _format_signed_money(value)
    return f"{value:.2f}"


def metric_label(key: str) -> tuple[str, str]:
    return METRIC_LABELS.get(key, (key.replace("_", " ").title(), ""))


def sign_class(value: float) -> str:
    """CSS class ('pos'/'neg') for a signed number -- always matches the
    sign of the number it's colored next to, e.g. don't color a -0.1% red
    number's row green just because the model's separate confidence score
    happens to be high.
    """
    return "pos" if value >= 0 else "neg"


def confidence_caption(confidence: float) -> str:
    """Direction-agnostic on purpose: the classifier's probability-of-
    outperforming and the regressor's expected return can disagree on
    direction (two different model outputs). Pairing a directional
    "62% odds to outperform" caption next to a red expected-return number
    would read as a contradiction; confidence (how much the two models
    agree, regardless of direction) never does.
    """
    return f"{confidence:.0f}% confidence"


AVATAR_COLOR_COUNT = 8  # must match .avatar-0..7 in style.css


def ticker_avatar(ticker: str, company_name: str | None = None) -> dict:
    """A small per-holding identity badge -- same idea as the colored
    letter icons most portfolio trackers (Ghostfolio, Robinhood, etc.) show
    next to each position, but generated locally from the ticker itself
    rather than fetched from a logo API: no network dependency, no rate
    limit, and it stays consistent with this app being local-first. The
    same ticker always maps to the same color and letter.
    """
    letter = (company_name or ticker or "?")[:1].upper()
    color_index = sum(ticker.encode()) % AVATAR_COLOR_COUNT
    return {"letter": letter, "color_class": f"avatar-{color_index}"}


def parse_reasons(reasons_json) -> list[dict]:
    """`reasons_json` is a per-prediction SHAP attribution list (see
    src/models/explain.py): the features that actually drove *this*
    prediction, ranked by how much they pushed it up or down. Translates
    each into a plain-language label + sign for display.
    """
    if not reasons_json:
        return []
    try:
        raw = json.loads(reasons_json)
    except (TypeError, ValueError):
        return []
    out = []
    for r in raw:
        label, _ = feature_label(r["feature"])
        out.append({"label": label, "sign_class": sign_class(r["shap"])})
    return out


def backtest_caption(sharpe: float, alpha: float) -> str:
    direction = "beating" if alpha >= 0 else "trailing"
    return f"Sharpe {sharpe:.2f} · {direction} the S&P 500 by {abs(alpha) * 100:.0f}pt/yr"
