from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import IntEnum, StrEnum
from typing import Any, Protocol


class JobPriority(IntEnum):
    CRITICAL = 0
    HIGH = 10
    NORMAL = 20
    LOW = 30
    EXPERIMENT = 40


class JobState(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    WAITING = "waiting"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    BLOCKED = "blocked"
    PUBLISHED_STALE = "published_stale"


class ModelTier(IntEnum):
    """Ordering for specialist work when a fresh prediction deadline exists."""

    PUBLISH_CRITICAL = 0
    EVENT_CRITICAL = 10
    ENHANCEMENT = 20
    EXPERIMENT = 40


class ResultStatus(StrEnum):
    READY = "ready"
    DELAYED = "delayed"
    FAILED = "failed"


@dataclass(frozen=True)
class ProviderQuota:
    provider: str
    limit: int
    used: int
    reset_at: datetime | None = None
    recurring_free_only: bool = True
    hard_stop_on_exhaustion: bool = True
    unit: str = "jobs"

    @property
    def remaining(self) -> int:
        return max(self.limit - self.used, 0)

    @property
    def exhausted(self) -> bool:
        return self.remaining <= 0


@dataclass(frozen=True)
class ExecutionRequest:
    job_type: str
    model_name: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)
    requires_gpu: bool = False
    minimum_vram_gb: int = 0
    supports_cpu_fallback: bool = True
    deadline_at: datetime | None = None
    tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class ExecutionResult:
    provider: str
    status: str
    artifact_uri: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DataRequest:
    dataset: str
    tickers: tuple[str, ...] = ()
    as_of: datetime | None = None
    allow_fallback: bool = True


@dataclass(frozen=True)
class DataResponse:
    provider: str
    status: str
    records: list[dict[str, Any]] = field(default_factory=list)
    coverage: float = 0.0
    freshness_at: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class GPUProvider(Protocol):
    name: str
    priority: int

    def quota(self) -> ProviderQuota: ...

    def supports(self, request: ExecutionRequest) -> bool: ...

    def submit(self, request: ExecutionRequest) -> ExecutionResult: ...


class DataProvider(Protocol):
    name: str
    priority: int

    def fetch(self, request: DataRequest) -> DataResponse: ...


class StructuredStorage(Protocol):
    def write_record(self, table: str, record: dict[str, Any]) -> None: ...

    def read_records(self, table: str) -> list[dict[str, Any]]: ...


class ObjectStorage(Protocol):
    def put_object(self, key: str, payload: bytes, metadata: dict[str, Any] | None = None) -> str: ...

    def get_object(self, key: str) -> bytes: ...


@dataclass(frozen=True)
class ModelCapability:
    name: str
    specialist: str
    requires_gpu: bool = False
    minimum_vram_gb: int = 0
    supports_cpu_fallback: bool = True


@dataclass(frozen=True)
class ModelSpec:
    name: str
    specialist: str
    execution: ModelCapability
    horizons_days: tuple[int, ...]
    publish_partial_results: bool = True
    tier: ModelTier = ModelTier.EXPERIMENT
    ensemble_weight: float = 0.0


@dataclass(frozen=True)
class SpecialistResult:
    """The only data shape a specialist may hand to the ensemble gatekeeper."""

    ticker: str
    as_of: datetime
    model_name: str
    model_version: str
    specialist: str
    horizon_days: int
    prediction: float | None = None
    probability: float | None = None
    confidence: float | None = None
    uncertainty: float | None = None
    features_version: str | None = None
    data_timestamp: datetime | None = None
    status: ResultStatus = ResultStatus.READY
    artifact_uri: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PublishedPrediction:
    ticker: str
    as_of: datetime
    horizon_days: int
    prediction: float
    probability: float | None
    confidence: float | None
    freshness: str
    contributing_models: tuple[str, ...]
    pending_models: tuple[str, ...]
    refresh_required: bool
    data_timestamp: datetime | None


@dataclass
class JobDefinition:
    job_id: str
    job_type: str
    priority: JobPriority
    created_at: datetime
    deadline_at: datetime | None = None
    dependencies: tuple[str, ...] = ()
    max_retries: int = 0
    fallback_job_types: tuple[str, ...] = ()
    payload: dict[str, Any] = field(default_factory=dict)
    allow_partial_publish: bool = False
    refresh_job_types: tuple[str, ...] = ()
    stale_after: timedelta | None = None
    state: JobState = JobState.QUEUED
    attempts: int = 0
    last_error: str | None = None
    result: dict[str, Any] = field(default_factory=dict)
