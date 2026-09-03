"""src.models.train._fit_horizon -- trains the XGBoost + LightGBM + CatBoost
regressor/classifier trio for one horizon. Uses tiny synthetic data and a
handful of trees so this stays fast; the real training run happens through
scripts/run_pipeline.py against real data, this just guards the trio wiring
itself (all six models come back, each one is actually fit and can predict).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.train import CLASSIFIER_NAMES, REGRESSOR_NAMES, _fit_horizon

_TINY_PARAMS = {
    "xgboost": {"n_estimators": 5, "max_depth": 2, "learning_rate": 0.3},
    "lightgbm": {"n_estimators": 5, "max_depth": 2, "learning_rate": 0.3, "min_child_samples": 2, "verbosity": -1},
    "catboost": {"iterations": 5, "depth": 2, "learning_rate": 0.3},
}


@pytest.fixture
def train_df():
    rng = np.random.default_rng(0)
    n = 100
    df = pd.DataFrame(
        {
            "feature_a": rng.normal(size=n),
            "feature_b": rng.normal(size=n),
            "forward_relative_return": rng.normal(scale=0.02, size=n),
        }
    )
    df["outperform"] = (df["forward_relative_return"] > 0).astype(int)
    return df


def test_all_six_models_are_returned_and_fitted(train_df):
    """Locks _fit_horizon's actual output keys to the shared REGRESSOR_NAMES/
    CLASSIFIER_NAMES constants (not a hardcoded literal here) -- those
    constants are what predict.py, engine.py, and explain.py all import to
    stay in sync with whatever _fit_horizon really produces. If someone adds
    a model to one but not the other, this is what catches it.
    """
    feature_cols = ["feature_a", "feature_b"]
    models = _fit_horizon(train_df, feature_cols, {"models": _TINY_PARAMS})

    assert set(models) == set(REGRESSOR_NAMES) | set(CLASSIFIER_NAMES)

    X = train_df[feature_cols]
    for name in REGRESSOR_NAMES:
        preds = models[name].predict(X)
        assert preds.shape == (len(train_df),)

    for name in CLASSIFIER_NAMES:
        proba = models[name].predict_proba(X)
        assert proba.shape == (len(train_df), 2)


def test_catboost_regressor_predictions_are_not_degenerate(train_df):
    """A smoke check that catboost actually learned something rather than
    e.g. silently predicting the same constant for every row (which a
    misconfigured bootstrap/subsample setting could produce).
    """
    feature_cols = ["feature_a", "feature_b"]
    models = _fit_horizon(train_df, feature_cols, {"models": _TINY_PARAMS})
    preds = models["catboost_reg"].predict(train_df[feature_cols])
    assert np.std(preds) > 0
