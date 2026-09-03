from __future__ import annotations

from src.cloud.contracts import ModelCapability, ModelSpec


def default_model_specs() -> list[ModelSpec]:
    return [
        ModelSpec(
            name="xgboost",
            specialist="price",
            execution=ModelCapability("xgboost", "price", requires_gpu=False, supports_cpu_fallback=True),
            horizons_days=(5, 20),
        ),
        ModelSpec(
            name="lightgbm",
            specialist="price",
            execution=ModelCapability("lightgbm", "price", requires_gpu=False, supports_cpu_fallback=True),
            horizons_days=(5, 20),
        ),
        ModelSpec(
            name="catboost",
            specialist="price",
            execution=ModelCapability("catboost", "price", requires_gpu=False, supports_cpu_fallback=True),
            horizons_days=(5, 20),
        ),
        ModelSpec(
            name="chronos-bolt",
            specialist="price",
            execution=ModelCapability("chronos-bolt", "price", requires_gpu=True, minimum_vram_gb=16, supports_cpu_fallback=False),
            horizons_days=(5, 20),
        ),
        ModelSpec(
            name="chronos-2",
            specialist="price",
            execution=ModelCapability("chronos-2", "price", requires_gpu=True, minimum_vram_gb=24, supports_cpu_fallback=False),
            horizons_days=(5, 20),
        ),
        ModelSpec(
            name="timesfm-2.5",
            specialist="price",
            execution=ModelCapability("timesfm-2.5", "price", requires_gpu=True, minimum_vram_gb=48, supports_cpu_fallback=False),
            horizons_days=(5, 20),
        ),
        ModelSpec(
            name="finbert",
            specialist="events",
            execution=ModelCapability("finbert", "events", requires_gpu=True, minimum_vram_gb=24, supports_cpu_fallback=False),
            horizons_days=(1, 5),
        ),
    ]
