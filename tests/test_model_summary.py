from __future__ import annotations

import json

from src.models.narrate import build_model_summary


def test_model_summary_uses_the_prediction_specific_shap_inputs_and_values():
    summary = build_model_summary(
        "NVDA",
        5,
        0.024,
        json.dumps(
            [
                {"feature": "insider_buy_count_90d", "shap": 0.3, "value": 4},
                {"feature": "close_to_ma_50", "shap": 0.2, "value": 0.08},
                {"feature": "rel_return_5d_vs_benchmark", "shap": -0.1, "value": -0.02},
            ]
        ),
    )

    assert summary == (
        "Model forecast: NVDA to outperform over 5 trading days; supporting signals: "
        "Insider Buys (90d) (4), vs. 50-Day Average Price (+8.0%); "
        "offsetting signals: 5-Day Return vs. the Market (-2.0%)."
    )
