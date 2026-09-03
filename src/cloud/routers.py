from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from src.cloud.contracts import DataProvider, DataRequest, DataResponse, ExecutionRequest, GPUProvider, JobPriority, ModelSpec


@dataclass(frozen=True)
class RouteDecision:
    status: str
    provider: str | None
    reason: str


class FreeComputeRouter:
    """Chooses only recurring-free GPU providers and hard-stops on exhaustion."""

    def __init__(self, providers: list[GPUProvider]) -> None:
        self.providers = sorted(providers, key=lambda provider: provider.priority)

    def route(self, request: ExecutionRequest) -> RouteDecision:
        supported = [provider for provider in self.providers if provider.supports(request)]
        if not supported:
            return RouteDecision("blocked", None, "No provider supports this request")

        exhausted = []
        for provider in supported:
            quota = provider.quota()
            if quota.exhausted:
                exhausted.append(provider.name)
                continue
            return RouteDecision("ready", provider.name, "Provider selected")

        return RouteDecision(
            "queued",
            None,
            f"Recurring free quota exhausted for: {', '.join(exhausted)}",
        )

    def submit(self, request: ExecutionRequest):
        decision = self.route(request)
        if decision.status != "ready" or decision.provider is None:
            raise RuntimeError(decision.reason)
        provider = next(provider for provider in self.providers if provider.name == decision.provider)
        return provider.submit(request)


class DataRouter:
    def __init__(self, providers: list[DataProvider]) -> None:
        self.providers = sorted(providers, key=lambda provider: provider.priority)

    def fetch(self, request: DataRequest) -> DataResponse:
        last_response: DataResponse | None = None
        for provider in self.providers:
            response = provider.fetch(request)
            last_response = response
            if response.status == "success" and response.records:
                return response
            if not request.allow_fallback:
                return response
        return last_response or DataResponse(provider="none", status="empty")


@dataclass(frozen=True)
class ModelExecutionPlan:
    ready_models: tuple[str, ...]
    delayed_models: tuple[str, ...]
    publish_now: bool
    refresh_required: bool
    priority: JobPriority
    publish_reason: str


class ModelOrchestrator:
    def __init__(self, specs: list[ModelSpec]) -> None:
        self.specs = specs

    def build_execution_requests(self, as_of: datetime | None = None) -> list[ExecutionRequest]:
        requests: list[ExecutionRequest] = []
        for spec in self.specs:
            requests.append(
                ExecutionRequest(
                    job_type="model_inference",
                    model_name=spec.name,
                    requires_gpu=spec.execution.requires_gpu,
                    minimum_vram_gb=spec.execution.minimum_vram_gb,
                    supports_cpu_fallback=spec.execution.supports_cpu_fallback,
                    deadline_at=as_of,
                    payload={"specialist": spec.specialist, "horizons_days": list(spec.horizons_days)},
                    tags=(spec.specialist,),
                )
            )
        return requests

    def plan_publication(
        self,
        completed_models: list[str],
        delayed_models: list[str],
        priority: JobPriority = JobPriority.HIGH,
    ) -> ModelExecutionPlan:
        ready = tuple(sorted(completed_models))
        delayed = tuple(sorted(delayed_models))
        publish_now = bool(ready)
        refresh_required = bool(ready and delayed)
        if ready and delayed:
            reason = "Publishing available specialist results now and queueing a refresh for delayed GPU models"
        elif ready:
            reason = "All required models are available"
        else:
            reason = "No models are ready yet"
        return ModelExecutionPlan(
            ready_models=ready,
            delayed_models=delayed,
            publish_now=publish_now,
            refresh_required=refresh_required,
            priority=priority,
            publish_reason=reason,
        )
