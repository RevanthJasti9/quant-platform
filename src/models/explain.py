"""Per-prediction "why" -- real SHAP feature attribution against the actual
trained regressors, not a heuristic guess. A SHAP value tells you how much
one feature pushed *this specific prediction* away from the average
prediction: positive pushed the expected return up, negative pushed it
down. Averaging the three regressors' SHAP values keeps this consistent
with how the ensemble's own expected-return number is computed (also a
mean of the three regressors).
"""
from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
import shap

from src.config import MODELS_DIR
from src.models.train import REGRESSOR_NAMES

TOP_N_REASONS = 5


def compute_batch_reasons(model_version: str, horizon_days: int, feature_cols: list[str], X: pd.DataFrame) -> list[str]:
    """One SHAP pass per model over the whole batch (all tickers at once)
    instead of reloading models per row -- returns one reasons JSON string
    per row of X, same order.
    """
    version_dir = MODELS_DIR / model_version
    total = np.zeros((len(X), len(feature_cols)))
    for model_name in REGRESSOR_NAMES:
        model = joblib.load(version_dir / f"h{horizon_days}_{model_name}.joblib")
        shap_values = shap.TreeExplainer(model).shap_values(X)
        total += np.asarray(shap_values)
    total /= len(REGRESSOR_NAMES)

    out = []
    for row_index, row in enumerate(total):
        pairs = sorted(zip(feature_cols, row), key=lambda p: abs(p[1]), reverse=True)
        top = [
            {
                "feature": feature,
                "shap": float(shap_value),
                # Store the observed input beside its attribution. This makes
                # the explanation auditable and prevents a prose generator
                # from turning an abstract label into a generic claim.
                "value": _json_value(X.iloc[row_index][feature]),
            }
            for feature, shap_value in pairs[:TOP_N_REASONS]
            if shap_value != 0
        ]
        out.append(json.dumps(top))
    return out


def _json_value(value):
    """Keep reasons_json portable when pandas/numpy values are missing."""
    if pd.isna(value):
        return None
    return float(value) if isinstance(value, np.number) else value
