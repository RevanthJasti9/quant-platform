"""Internal-only read endpoints for the provider-independent control plane."""
from __future__ import annotations

from datetime import UTC, datetime

from fastapi import APIRouter

from src.cloud.control_plane import ControlPlane
from src.cloud.model_registry import default_model_specs
from src.cloud.providers import recurring_free_provider_defaults
from src.cloud.scheduler import default_job_definitions

router = APIRouter(prefix="/api/control-plane", tags=["control-plane"])


def _control_plane() -> ControlPlane:
    control_plane = ControlPlane()
    control_plane.submit_many(default_job_definitions(datetime.now(UTC)))
    return control_plane


@router.get("/status")
def status() -> dict[str, int]:
    """Return the default control-plane queue summary without invoking providers."""
    return _control_plane().summary()


@router.get("/jobs")
def jobs() -> list[dict]:
    control_plane = _control_plane()
    return [
        {
            "job_id": job.job_id,
            "job_type": job.job_type,
            "priority": job.priority.name.lower(),
            "deadline_at": job.deadline_at.isoformat() if job.deadline_at else None,
            "dependencies": list(job.dependencies),
            "allow_partial_publish": job.allow_partial_publish,
        }
        for job in control_plane.ready_jobs()
    ]


@router.get("/gpu-quota")
def gpu_quota() -> list[dict]:
    """Expose configured recurring-free quota limits; this never submits paid work."""
    return [
        {
            "provider": provider.name,
            "remaining": provider.quota().remaining,
            "limit": provider.quota().limit,
            "unit": provider.quota().unit,
            "recurring_free_only": provider.quota().recurring_free_only,
            "hard_stop_on_exhaustion": provider.quota().hard_stop_on_exhaustion,
        }
        for provider in recurring_free_provider_defaults()
    ]


@router.get("/models")
def models() -> list[dict]:
    """Canonical specialist catalog for workers, dashboard, and model arena."""
    return [
        {
            "name": spec.name,
            "specialist": spec.specialist,
            "tier": spec.tier.name.lower(),
            "horizons_days": list(spec.horizons_days),
            "requires_gpu": spec.execution.requires_gpu,
            "minimum_vram_gb": spec.execution.minimum_vram_gb,
            "ensemble_weight": spec.ensemble_weight,
        }
        for spec in default_model_specs()
    ]
