from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime

from src.cloud.contracts import DataRequest, DataResponse, ExecutionRequest, ExecutionResult, GPUProvider, ProviderQuota


@dataclass
class StubGPUProvider(GPUProvider):
    name: str
    priority: int
    quota_state: ProviderQuota
    supports_gpu: bool = True
    minimum_vram_gb: int = 0
    supported_models: set[str] = field(default_factory=set)

    def quota(self) -> ProviderQuota:
        return self.quota_state

    def supports(self, request: ExecutionRequest) -> bool:
        if request.requires_gpu and not self.supports_gpu:
            return False
        if request.minimum_vram_gb > self.minimum_vram_gb:
            return False
        if self.supported_models and request.model_name and request.model_name not in self.supported_models:
            return False
        return True

    def submit(self, request: ExecutionRequest) -> ExecutionResult:
        if self.quota_state.exhausted:
            raise RuntimeError(f"{self.name} quota exhausted")
        self.quota_state = ProviderQuota(
            provider=self.quota_state.provider,
            limit=self.quota_state.limit,
            used=self.quota_state.used + 1,
            reset_at=self.quota_state.reset_at,
            recurring_free_only=self.quota_state.recurring_free_only,
            hard_stop_on_exhaustion=self.quota_state.hard_stop_on_exhaustion,
            unit=self.quota_state.unit,
        )
        return ExecutionResult(
            provider=self.name,
            status="submitted",
            artifact_uri=f"{self.name}://jobs/{request.model_name or request.job_type}",
            metadata={"job_type": request.job_type},
        )


class KaggleGPUProvider(StubGPUProvider):
    pass


class ModalGPUProvider(StubGPUProvider):
    pass


class HuggingFaceZeroGPUProvider(StubGPUProvider):
    pass


@dataclass
class StaticDataProvider:
    name: str
    priority: int
    responses: dict[str, DataResponse]

    def fetch(self, request: DataRequest) -> DataResponse:
        response = self.responses.get(request.dataset)
        if response is None:
            return DataResponse(provider=self.name, status="empty")
        return response


def recurring_free_provider_defaults(now: datetime | None = None) -> list[GPUProvider]:
    now = now or datetime.now(UTC)
    return [
        ModalGPUProvider(
            name="modal",
            priority=0,
            quota_state=ProviderQuota("modal", limit=50, used=0, reset_at=now, unit="gpu_hours"),
            minimum_vram_gb=24,
            supported_models={"finbert", "chronos-bolt", "chronos-2", "timesfm-2.5"},
        ),
        KaggleGPUProvider(
            name="kaggle",
            priority=1,
            quota_state=ProviderQuota("kaggle", limit=30, used=0, reset_at=now, unit="gpu_hours"),
            minimum_vram_gb=16,
            supported_models={"xgboost", "lightgbm", "catboost", "chronos-bolt"},
        ),
        HuggingFaceZeroGPUProvider(
            name="hf-zerogpu",
            priority=2,
            quota_state=ProviderQuota("hf-zerogpu", limit=5, used=0, reset_at=now, unit="gpu_minutes"),
            minimum_vram_gb=48,
            supported_models={"finbert", "chronos-2", "timesfm-2.5"},
        ),
    ]
