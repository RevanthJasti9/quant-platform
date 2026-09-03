"""Plain-language translation for feature column names. Lives in src/, not
app/, because it's used both by the web layer (app/copy.py re-exports it
for templates) and by the modeling layer (src/models/narrate.py, to build
LLM prompts) -- src/ code must never import from app/.
"""
from __future__ import annotations

# feature_column -> (short label, one-line explanation for people who don't
# know what it means)
FEATURE_LABELS: dict[str, tuple[str, str]] = {
    "momentum_5d": ("5-Day Price Change", "How much the price has moved over the last 5 trading days."),
    "momentum_10d": ("10-Day Price Change", "How much the price has moved over the last 10 trading days."),
    "momentum_20d": ("20-Day Price Change", "How much the price has moved over the last 20 trading days."),
    "momentum_60d": ("60-Day Price Change", "How much the price has moved over the last 60 trading days."),
    "rsi": ("Momentum Strength (RSI)", "0-100 scale. Above 70 usually means “overbought,” below 30 means “oversold.”"),
    "macd": ("Trend Signal", "Positive means the short-term trend is pointing up."),
    "macd_signal": ("Trend Signal (smoothed)", "A smoothed version of the trend signal, used to spot direction changes."),
    "macd_hist": ("Trend Strength", "How strong the current trend is, positive or negative."),
    "close_to_ma_10": ("vs. 10-Day Average Price", "How far today's price is above or below its 10-day average."),
    "close_to_ma_20": ("vs. 20-Day Average Price", "How far today's price is above or below its 20-day average."),
    "close_to_ma_50": ("vs. 50-Day Average Price", "How far today's price is above or below its 50-day average."),
    "close_to_ma_200": ("vs. 200-Day Average Price", "How far today's price is above or below its 200-day average, a common long-term trend marker."),
    "volatility": ("Price Swings", "How much the price bounces around day to day. Higher usually means riskier."),
    "volume_zscore": ("Unusual Trading Activity", "How far today's trading volume is from what's normal for this stock."),
    "pe_ratio": ("Price / Earnings", "How expensive the stock is relative to the company's profit. Lower can mean cheaper."),
    "forward_pe": ("Forward Price / Earnings", "Same idea, but using next year's expected profit."),
    "price_to_book": ("Price / Book Value", "Price compared to the accounting value of the company's assets."),
    "revenue_growth": ("Revenue Growth", "How fast the company's sales are growing."),
    "earnings_growth": ("Earnings Growth", "How fast the company's profit is growing."),
    "gross_margin": ("Gross Margin", "Share of sales left after direct production costs."),
    "operating_margin": ("Operating Margin", "Share of sales left after running the core business."),
    "profit_margin": ("Profit Margin", "Share of sales that ends up as actual profit."),
    "debt_to_equity": ("Debt Level", "How much debt the company carries relative to its own value. Higher means more financial risk."),
    "return_on_equity": ("Return on Equity", "How efficiently the company turns shareholder money into profit."),
    "dividend_yield": ("Dividend Yield", "Annual cash payout to shareholders, as a % of the stock price."),
    "fcf_yield": ("Free Cash Flow Yield", "Actual spare cash the company generates, as a % of its market value."),
    "log_market_cap": ("Company Size", "A size measure — bigger companies tend to move less dramatically."),
    "rel_return_5d_vs_benchmark": ("5-Day Return vs. the Market", "Has it beaten the S&P 500 over the last 5 days?"),
    "rel_return_5d_vs_sector": ("5-Day Return vs. Its Sector", "Has it beaten similar companies over the last 5 days?"),
    "rel_return_10d_vs_benchmark": ("10-Day Return vs. the Market", "Has it beaten the S&P 500 over the last 10 days?"),
    "rel_return_10d_vs_sector": ("10-Day Return vs. Its Sector", "Has it beaten similar companies over the last 10 days?"),
    "rel_return_20d_vs_benchmark": ("20-Day Return vs. the Market", "Has it beaten the S&P 500 over the last 20 days?"),
    "rel_return_20d_vs_sector": ("20-Day Return vs. Its Sector", "Has it beaten similar companies over the last 20 days?"),
    "sector": ("Sector", "The broad industry group the company belongs to."),
    "industry": ("Industry", "The specific line of business the company is in."),
    "news_event_count_7d": ("Recent News Volume (7d)", "How many distinct news stories have come out in the last week."),
    "news_event_count_30d": ("Recent News Volume (30d)", "How many distinct news stories have come out in the last month."),
    "news_negative_event_count_7d": ("Negative News (7d)", "Regulatory or lawsuit stories in the last week."),
    "news_negative_event_count_30d": ("Negative News (30d)", "Regulatory or lawsuit stories in the last month."),
    "news_days_since_last_event": ("Days Since Last News", "How long it's been since any news story came out."),
    "insider_buy_count_30d": ("Insider Buys (30d)", "Open-market purchases by company insiders in the last month."),
    "insider_buy_count_90d": ("Insider Buys (90d)", "Open-market purchases by company insiders in the last quarter."),
    "insider_sell_count_30d": ("Insider Sells (30d)", "Open-market sales by company insiders in the last month."),
    "insider_sell_count_90d": ("Insider Sells (90d)", "Open-market sales by company insiders in the last quarter."),
    "insider_net_value_30d": ("Insider Net $ (30d)", "Insider buying minus selling, in dollars, over the last month."),
    "insider_net_value_90d": ("Insider Net $ (90d)", "Insider buying minus selling, in dollars, over the last quarter."),
    "insider_days_since_last_txn": ("Days Since Insider Trade", "How long it's been since an insider bought or sold shares."),
    "sec_8k_count_30d": ("Material Filings (30d)", "8-K filings (unscheduled material events) in the last month."),
    "sec_8k_count_90d": ("Material Filings (90d)", "8-K filings (unscheduled material events) in the last quarter."),
    "sec_days_since_last_8k": ("Days Since Material Filing", "How long it's been since the company filed an 8-K."),
    "news_sentiment_score": ("News Sentiment (AI read)", "An AI's read of recent headlines' tone, -1 (very negative) to +1 (very positive). Not a verified signal -- see if it actually helped."),
}


def feature_label(key: str) -> tuple[str, str]:
    return FEATURE_LABELS.get(key, (key.replace("_", " ").title(), ""))
