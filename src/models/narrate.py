"""Turn structured SHAP reasons into a deterministic model explanation.

Forecast copy must describe the actual prediction, not an LLM's paraphrase.
Each explanation is therefore built directly from the features and values
that had the largest SHAP effect for that ticker and horizon.
"""
from __future__ import annotations

import json
import math

from src.models.feature_labels import feature_label

_PERCENT_FEATURES = {
    "momentum_5d", "momentum_10d", "momentum_20d", "momentum_60d",
    "close_to_ma_10", "close_to_ma_20", "close_to_ma_50", "close_to_ma_200",
    "volatility", "revenue_growth", "earnings_growth", "gross_margin",
    "operating_margin", "profit_margin", "dividend_yield", "fcf_yield", "return_on_equity",
    "rel_return_5d_vs_benchmark", "rel_return_5d_vs_sector",
    "rel_return_10d_vs_benchmark", "rel_return_10d_vs_sector",
    "rel_return_20d_vs_benchmark", "rel_return_20d_vs_sector",
}
_COUNT_FEATURES = {
    "news_event_count_7d", "news_event_count_30d", "news_negative_event_count_7d", "news_negative_event_count_30d",
    "insider_buy_count_30d", "insider_buy_count_90d", "insider_sell_count_30d", "insider_sell_count_90d",
    "sec_8k_count_30d", "sec_8k_count_90d",
}


def _format_model_value(feature: str, value) -> str:
    """Compact, dependency-free rendering for an input stored with SHAP."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "not available"
    if feature in _PERCENT_FEATURES:
        return f"{float(value) * 100:+.1f}%"
    if feature in _COUNT_FEATURES:
        return f"{float(value):.0f}"
    if feature == "rsi":
        return f"{float(value):.0f}/100"
    if feature == "volume_zscore":
        return f"{float(value):+.1f} standard deviations"
    if feature in {"insider_net_value_30d", "insider_net_value_90d"}:
        return f"${float(value):+,.0f}"
    return f"{float(value):.2f}"


def build_model_summary(ticker: str, horizon_days: int, expected_return: float, reasons_json: str) -> str | None:
    """Describe the top model inputs for one forecast, with their values.

    SHAP sign records whether a feature pushed the ensemble's expected return
    up or down for this exact row. The explanation preserves that distinction
    instead of assuming that a high/low raw value is universally positive.
    """
    try:
        reasons = json.loads(reasons_json)
    except (TypeError, ValueError):
        reasons = []
    if not reasons:
        return None

    direction = "outperform" if expected_return >= 0 else "underperform"
    supporting, offsetting = [], []
    for reason in reasons:
        feature = reason.get("feature")
        if not feature:
            continue
        label, _ = feature_label(feature)
        value = _format_model_value(feature, reason.get("value"))
        text = f"{label} ({value})"
        if float(reason.get("shap", 0)) >= 0:
            supporting.append(text)
        else:
            offsetting.append(text)

    clauses = []
    if supporting:
        clauses.append("supporting signals: " + ", ".join(supporting[:3]))
    if offsetting:
        clauses.append("offsetting signals: " + ", ".join(offsetting[:2]))
    if not clauses:
        return None
    return f"Model forecast: {ticker} to {direction} over {horizon_days} trading days; " + "; ".join(clauses) + "."
