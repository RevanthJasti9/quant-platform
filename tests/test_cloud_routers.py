from __future__ import annotations

from datetime import datetime

from src.cloud.contracts import DataRequest, DataResponse, ExecutionRequest, ModelTier, ProviderQuota
from src.cloud.model_registry import default_model_specs
from src.cloud.providers import ModalGPUProvider, StaticDataProvider
from src.cloud.routers import DataRouter, FreeComputeRouter, ModelOrchestrator


def test_free_compute_router_prefers_first_supported_provider_with_free_quota():
    providers = [
        ModalGPUProvider(
            name="modal",
            priority=0,
            quota_state=ProviderQuota("modal", limit=10, used=0),
            minimum_vram_gb=24,
            supported_models={"finbert"},
        ),
        ModalGPUProvider(
            name="backup",
            priority=1,
            quota_state=ProviderQuota("backup", limit=10, used=0),
            minimum_vram_gb=48,
            supported_models={"finbert"},
        ),
    ]
    router = FreeComputeRouter(providers)

    decision = router.route(
        ExecutionRequest(job_type="model_inference", model_name="finbert", requires_gpu=True, minimum_vram_gb=24)
    )

    assert decision.status == "ready"
    assert decision.provider == "modal"


def test_free_compute_router_queues_when_all_supported_providers_are_exhausted():
    providers = [
        ModalGPUProvider(
            name="modal",
            priority=0,
            quota_state=ProviderQuota("modal", limit=10, used=10),
            minimum_vram_gb=24,
            supported_models={"finbert"},
        ),
        ModalGPUProvider(
            name="backup",
            priority=1,
            quota_state=ProviderQuota("backup", limit=5, used=5),
            minimum_vram_gb=48,
            supported_models={"finbert"},
        ),
    ]
    router = FreeComputeRouter(providers)

    decision = router.route(
        ExecutionRequest(job_type="model_inference", model_name="finbert", requires_gpu=True, minimum_vram_gb=24)
    )

    assert decision.status == "queued"
    assert decision.provider is None
    assert "quota exhausted" in decision.reason


def test_data_router_falls_back_to_next_provider_when_primary_is_empty():
    router = DataRouter(
        [
            StaticDataProvider(name="primary", priority=0, responses={"prices": DataResponse("primary", "empty", [])}),
            StaticDataProvider(
                name="secondary",
                priority=1,
                responses={"prices": DataResponse("secondary", "success", [{"ticker": "AAPL"}], coverage=1.0)},
            ),
        ]
    )

    response = router.fetch(DataRequest(dataset="prices"))

    assert response.provider == "secondary"
    assert response.records == [{"ticker": "AAPL"}]


def test_model_orchestrator_plans_partial_publish_when_some_models_are_delayed():
    specs = [spec for spec in default_model_specs() if spec.name in {"xgboost", "chronos-2", "finbert"}]
    orchestrator = ModelOrchestrator(specs)

    requests = orchestrator.build_execution_requests(datetime(2026, 9, 3, 16, 30))
    publish_plan = orchestrator.plan_publication(completed_models=["xgboost", "finbert"], delayed_models=["chronos-2"])

    assert {request.model_name for request in requests} == {"xgboost", "chronos-2", "finbert"}
    assert publish_plan.publish_now is True
    assert publish_plan.refresh_required is True
    assert publish_plan.ready_models == ("finbert", "xgboost")
    assert publish_plan.delayed_models == ("chronos-2",)


def test_specialist_batch_keeps_price_baseline_ahead_of_events_and_experiments():
    specs = default_model_specs()
    exhausted = ModalGPUProvider(
        name="modal",
        priority=0,
        quota_state=ProviderQuota("modal", limit=1, used=1),
        minimum_vram_gb=48,
    )

    plan = ModelOrchestrator(specs).plan_batch(FreeComputeRouter([exhausted]))

    assert [item.model_name for item in plan.routes[:3]] == ["catboost", "lightgbm", "xgboost"]
    assert all(item.tier == ModelTier.PUBLISH_CRITICAL for item in plan.routes[:3])
    assert plan.publication.publish_now is True
    assert plan.publication.refresh_required is True
    assert set(plan.publication.ready_models) == {"xgboost", "lightgbm", "catboost"}
    assert "finbert" in plan.publication.delayed_models
