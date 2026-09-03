from datetime import UTC, datetime

import pytest

from src.cloud.contracts import ResultStatus, SpecialistResult
from src.cloud.model_registry import default_model_specs
from src.cloud.results import PredictionGatekeeper, SpecialistResultStore
from src.cloud.storage import InMemoryStructuredStorage


def _result(model_name: str, prediction: float | None, *, status: ResultStatus = ResultStatus.READY) -> SpecialistResult:
    return SpecialistResult(
        ticker="AMZN",
        as_of=datetime(2026, 9, 3, 21, 0, tzinfo=UTC),
        model_name=model_name,
        model_version="test-v1",
        specialist="price" if model_name != "finbert" else "events",
        horizon_days=5,
        prediction=prediction,
        probability=0.6 if prediction is not None else None,
        confidence=70.0 if prediction is not None else None,
        features_version="features-v1",
        data_timestamp=datetime(2026, 9, 3, 20, 30, tzinfo=UTC),
        status=status,
    )


def test_result_store_round_trips_standardized_specialist_fields():
    storage = InMemoryStructuredStorage()
    store = SpecialistResultStore(storage)
    result = _result("xgboost", 0.02)

    store.write(result)
    stored = store.for_prediction("AMZN", result.as_of, 5)

    assert stored == [result]


def test_gatekeeper_publishes_available_price_models_and_marks_gpu_enhancement_pending():
    gatekeeper = PredictionGatekeeper(default_model_specs())
    results = [_result("xgboost", 0.03), _result("lightgbm", 0.01), _result("finbert", None, status=ResultStatus.DELAYED)]

    published = gatekeeper.publishable(results)

    assert len(published) == 1
    prediction = published[0]
    assert prediction.prediction == pytest.approx((0.03 * 0.34 + 0.01 * 0.33) / 0.67)
    assert prediction.freshness == "partial"
    assert prediction.refresh_required is True
    assert prediction.contributing_models == ("lightgbm", "xgboost")
    assert {"catboost", "chronos-bolt"} <= set(prediction.pending_models)


def test_gatekeeper_does_not_publish_an_event_only_result():
    gatekeeper = PredictionGatekeeper(default_model_specs())

    assert gatekeeper.publishable([_result("finbert", None)]) == []
