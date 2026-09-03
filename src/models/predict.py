"""Runs inference with the current model version for each horizon over the
latest feature snapshot and records the ensembled output into `predictions`.
"""
from __future__ import annotations

import json
import logging
from datetime import date

import joblib
import numpy as np
import pandas as pd

from src.config import MODELS_DIR, get_settings
from src.data.db import get_connection, upsert_wide
from src.models.ensemble import ensemble_confidence, ensemble_weighted_mean
from src.models.explain import compute_batch_reasons
from src.models.narrate import build_model_summary
from src.models.train import CLASSIFIER_NAMES, REGRESSOR_NAMES, RESEARCH_PRICE_BASIS

logger = logging.getLogger(__name__)


def _price_ensemble_weights(settings: dict) -> list[float]:
    configured = settings.get("models", {}).get("price_ensemble_weights", {})
    names = ("xgboost", "lightgbm", "catboost")
    return [float(configured.get(name, 1.0)) for name in names]


def _current_model_version(con, horizon: int) -> tuple[str, list[str]] | None:
    rows = con.execute(
        "SELECT model_version, feature_cols, params_json FROM model_versions WHERE horizon_days = ? ORDER BY trained_at DESC",
        [horizon],
    ).fetchall()
    for model_version, feature_cols, params_json in rows:
        try:
            params = json.loads(params_json)
        except (TypeError, json.JSONDecodeError):
            params = {}
        if params.get("research_price_basis") == RESEARCH_PRICE_BASIS:
            return model_version, json.loads(feature_cols)

    if rows:
        logger.error(
            "No compatible model for horizon %sd: existing models predate the adjusted-price feature definition; retrain first",
            horizon,
        )
    return None


def run_predictions(as_of: date | None = None, tickers: list[str] | None = None, run_id: str | None = None) -> int:
    """`tickers`, when given, scopes inference (and the SHAP/LLM reasons
    work that goes with it) to just those -- e.g. the on-demand "add a
    holding" path passes a single ticker so it doesn't end up re-running
    LLM generation for the whole universe just to onboard one new stock.

    `run_id`, when given, links every prediction row to the ingest_runs row
    that produced the data it was built from -- so a prediction can always
    be traced back to what was actually known (and whether that run's
    sources were healthy) at the time it was made.
    """
    settings = get_settings()
    con = get_connection()
    features = con.execute("SELECT * FROM features ORDER BY ticker, date").fetchdf()
    con.close()
    if features.empty:
        logger.warning("No features available — run the feature build first")
        return 0

    if as_of is None:
        as_of = features["date"].max()
    latest = features[features["date"] == as_of].reset_index(drop=True)
    if tickers is not None:
        latest = latest[latest["ticker"].isin(tickers)].reset_index(drop=True)
    if latest.empty:
        logger.warning("No feature rows for as_of=%s", as_of)
        return 0

    con = get_connection()
    model_versions = {h: _current_model_version(con, h) for h in settings["models"]["horizons_days"]}
    con.close()  # closed before the slow part (model inference + LLM calls) so it
    # doesn't hold DuckDB's single writer lock for minutes and block the live
    # dashboard's own queries while this runs (predict.py can take a while
    # once LLM reasons_summary generation is in the loop).

    rows_written = 0
    ensemble_weights = _price_ensemble_weights(settings)
    for h in settings["models"]["horizons_days"]:
        current = model_versions.get(h)
        if current is None:
            logger.warning("No trained model for horizon %sd yet", h)
            continue
        model_version, feature_cols = current
        version_dir = MODELS_DIR / model_version
        X = latest.reindex(columns=feature_cols, fill_value=np.nan)

        regressors = [joblib.load(version_dir / f"h{h}_{name}.joblib") for name in REGRESSOR_NAMES]
        classifiers = [joblib.load(version_dir / f"h{h}_{name}.joblib") for name in CLASSIFIER_NAMES]

        reg_pred = ensemble_weighted_mean([m.predict(X) for m in regressors], ensemble_weights)
        probas = [m.predict_proba(X)[:, 1] for m in classifiers]
        proba = ensemble_weighted_mean(probas, ensemble_weights)
        confidence = ensemble_confidence(*probas)
        reasons = compute_batch_reasons(model_version, h, feature_cols, X)

        horizon_rows = [
            {
                "ticker": ticker,
                "prediction_date": as_of,
                "horizon_days": h,
                "model_version": model_version,
                "expected_relative_return": float(reg_pred[i]),
                "probability_outperform": float(proba[i]),
                "confidence": float(confidence[i]),
                "reasons_json": reasons[i],
                "reasons_summary": build_model_summary(ticker, h, float(reg_pred[i]), reasons[i]),
                "run_id": run_id,
            }
            for i, ticker in enumerate(latest["ticker"])
        ]

        # Persist the actual model output before optional natural-language
        # summaries. A rate limit or local LLM outage must never prevent the
        # dashboard from receiving a valid numeric prediction.
        con = get_connection()
        try:
            rows_written += upsert_wide(
                con, "predictions", pd.DataFrame(horizon_rows), ("ticker", "prediction_date", "horizon_days")
            )
        finally:
            con.close()

    if not rows_written:
        return 0
    logger.info("Recorded %d predictions for %s", rows_written, as_of)
    return rows_written


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print(run_predictions())
