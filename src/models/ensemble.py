"""Combines XGBoost + LightGBM + CatBoost outputs into one score. Kept
intentionally simple (mean averaging) for V1 — a learned/weighted ensemble
is a natural V2 upgrade once there's a backtest baseline to compare it
against.
"""
from __future__ import annotations

import numpy as np


def ensemble_mean(*arrays: np.ndarray) -> np.ndarray:
    return np.mean(np.stack(arrays, axis=0), axis=0)


def ensemble_weighted_mean(arrays: list[np.ndarray], weights: list[float]) -> np.ndarray:
    """Average specialist outputs with validated, normalized configured weights."""
    if not arrays:
        raise ValueError("At least one model output is required")
    if len(arrays) != len(weights):
        raise ValueError("Each model output needs one weight")
    normalized = np.asarray(weights, dtype=float)
    if np.any(normalized < 0) or not np.isfinite(normalized).all() or normalized.sum() <= 0:
        raise ValueError("Ensemble weights must be finite, non-negative, and sum to more than zero")
    return np.average(np.stack(arrays, axis=0), axis=0, weights=normalized)


def ensemble_confidence(*probas: np.ndarray) -> np.ndarray:
    """Higher when the classifiers agree, scaled to 0-100. Spread is the
    widest gap between any two models' probabilities (max - min) -- for
    exactly two models that's identical to the plain absolute difference
    the original two-model version used, so this generalizes to three (or
    more) models without changing any existing behavior.
    """
    stacked = np.stack(probas, axis=0)
    spread = stacked.max(axis=0) - stacked.min(axis=0)
    return (1 - spread) * 100
