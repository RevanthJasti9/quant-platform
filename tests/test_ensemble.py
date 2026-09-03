"""src.models.ensemble -- ensemble_confidence was generalized from a
hardcoded two-model pairwise difference to a max-min spread across however
many models are passed, so CatBoost could be added as a third ensemble
member without changing the two-model math it already used to compute.
"""
from __future__ import annotations

import numpy as np

from src.models.ensemble import ensemble_confidence, ensemble_mean


def test_ensemble_mean_averages_any_number_of_arrays():
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([3.0, 4.0, 5.0])
    c = np.array([2.0, 3.0, 4.0])
    np.testing.assert_allclose(ensemble_mean(a, b), [2.0, 3.0, 4.0])
    np.testing.assert_allclose(ensemble_mean(a, b, c), [2.0, 3.0, 4.0])


def test_confidence_for_two_models_matches_original_pairwise_formula():
    # This is the exact case the original implementation (probas[0]-probas[1]) handled --
    # the generalized max-min version must reproduce it exactly, not just approximately.
    proba_a = np.array([0.9, 0.5, 0.2])
    proba_b = np.array([0.9, 0.5, 0.8])
    expected = (1 - np.abs(proba_a - proba_b)) * 100
    np.testing.assert_allclose(ensemble_confidence(proba_a, proba_b), expected)


def test_confidence_is_100_when_all_models_agree():
    proba = np.array([0.7, 0.3])
    result = ensemble_confidence(proba, proba, proba)
    np.testing.assert_allclose(result, [100.0, 100.0])


def test_confidence_for_three_models_uses_widest_pairwise_gap():
    proba_a = np.array([0.9])
    proba_b = np.array([0.5])
    proba_c = np.array([0.6])  # inside the a/b range -- shouldn't widen the spread
    result = ensemble_confidence(proba_a, proba_b, proba_c)
    np.testing.assert_allclose(result, [(1 - 0.4) * 100])  # max(0.9,0.5,0.6) - min(...) = 0.4


def test_confidence_drops_when_a_third_model_disagrees_more_than_the_other_two():
    two_model = ensemble_confidence(np.array([0.9]), np.array([0.85]))
    three_model = ensemble_confidence(np.array([0.9]), np.array([0.85]), np.array([0.1]))
    assert three_model[0] < two_model[0]
