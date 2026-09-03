"""Trains an XGBoost + LightGBM + CatBoost regressor/classifier trio per
configured horizon and persists them under models/{model_version}/. Model
promotion is manual in V1 (see plan): `predict.py` just picks whichever
model_version has the latest trained_at for a given horizon.

CatBoost was added on top of the original XGBoost+LightGBM pair after
benchmarking research (Qlib's published model comparison) showed tree-based
ensembles substantially outperforming deep learning on this kind of
feature-engineered tabular data, with CatBoost scoring competitively in that
same benchmark and no dependency conflict with this project's pandas
version (unlike alphalens-reloaded, rejected earlier for exactly that).
"""
from __future__ import annotations

import json
import logging
from datetime import date

import joblib
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from xgboost import XGBClassifier, XGBRegressor

from src.config import MODELS_DIR, get_settings
from src.data.db import get_connection
from src.models.targets import build_targets

logger = logging.getLogger(__name__)

# Single source of truth for which models make up the ensemble. predict.py,
# engine.py (the backtester), and explain.py (SHAP reasons) all import these
# rather than each keeping their own list -- the backtester scoring only
# xgb_reg+lgbm_reg after CatBoost was added to the other three is exactly
# the bug this consolidation exists to make structurally impossible: whoever
# adds or removes a model only has to change it here.
REGRESSOR_NAMES = ("xgb_reg", "lgbm_reg", "catboost_reg")
CLASSIFIER_NAMES = ("xgb_clf", "lgbm_clf", "catboost_clf")
RESEARCH_PRICE_BASIS = "adjusted_close_v1"


def get_feature_cols(features_df: pd.DataFrame) -> list[str]:
    return [c for c in features_df.columns if c not in ("ticker", "date")]


def _fit_horizon(train_df: pd.DataFrame, feature_cols: list[str], settings: dict) -> dict:
    X = train_df[feature_cols]
    y_reg = train_df["forward_relative_return"]
    y_clf = train_df["outperform"]

    xgb_p = settings["models"]["xgboost"]
    lgbm_p = settings["models"]["lightgbm"]
    cat_p = settings["models"]["catboost"]

    # CatBoost otherwise creates catboost_info/ in the process working
    # directory. Training should be portable to read-only/service contexts.
    cat_params = {**cat_p, "allow_writing_files": False, "verbose": False}

    return {
        "xgb_reg": XGBRegressor(**xgb_p, verbosity=0).fit(X, y_reg),
        "lgbm_reg": LGBMRegressor(**lgbm_p).fit(X, y_reg),
        "catboost_reg": CatBoostRegressor(**cat_params).fit(X, y_reg),
        "xgb_clf": XGBClassifier(**xgb_p, verbosity=0, eval_metric="logloss").fit(X, y_clf),
        "lgbm_clf": LGBMClassifier(**lgbm_p).fit(X, y_clf),
        "catboost_clf": CatBoostClassifier(**cat_params).fit(X, y_clf),
    }


def train_models(
    model_version: str | None = None,
    train_start: date | None = None,
    train_end: date | None = None,
) -> str:
    settings = get_settings()
    model_version = model_version or f"v{pd.Timestamp.now('UTC'):%Y%m%d%H%M%S}"

    con = get_connection()
    features = con.execute("SELECT * FROM features ORDER BY ticker, date").fetchdf()
    prices = con.execute(
        "SELECT ticker, date, COALESCE(adj_close, close) AS close FROM prices ORDER BY ticker, date"
    ).fetchdf()
    con.close()

    if features.empty:
        raise RuntimeError("No features available — run the feature build first")

    feature_cols = get_feature_cols(features)
    targets = build_targets(prices, settings["benchmark"], settings["models"]["horizons_days"])

    version_dir = MODELS_DIR / model_version
    version_dir.mkdir(parents=True, exist_ok=True)

    con = get_connection()
    for h in settings["models"]["horizons_days"]:
        merged = features.merge(targets[targets["horizon_days"] == h], on=["ticker", "date"], how="inner")
        if train_start is not None:
            merged = merged[merged["date"] >= train_start]
        if train_end is not None:
            merged = merged[merged["date"] <= train_end]
        if merged.empty:
            logger.warning("No training rows for horizon %sd, skipping", h)
            continue

        logger.info("Training %d-day ensemble on %d rows", h, len(merged))
        models = _fit_horizon(merged, feature_cols, settings)
        for name, model in models.items():
            joblib.dump(model, version_dir / f"h{h}_{name}.joblib")

        con.execute(
            """
            INSERT OR REPLACE INTO model_versions
                (model_version, trained_at, algo, horizon_days, params_json, train_start, train_end, feature_cols)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                model_version,
                pd.Timestamp.now("UTC"),
                "xgboost+lightgbm+catboost ensemble",
                h,
                json.dumps(
                    {
                        "xgboost": settings["models"]["xgboost"],
                        "lightgbm": settings["models"]["lightgbm"],
                        "catboost": settings["models"]["catboost"],
                        "research_price_basis": RESEARCH_PRICE_BASIS,
                    }
                ),
                merged["date"].min(),
                merged["date"].max(),
                json.dumps(feature_cols),
            ],
        )
    con.close()
    logger.info("Trained model version %s", model_version)
    return model_version


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print(train_models())
