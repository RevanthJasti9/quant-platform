from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime

from src.cloud.control_plane import ControlPlane
from src.cloud.contracts import ExecutionRequest
from src.cloud.model_registry import default_model_specs
from src.cloud.providers import recurring_free_provider_defaults
from src.cloud.routers import FreeComputeRouter
from src.cloud.scheduler import default_job_definitions


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Cloud-first quant control-plane commands")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("status", help="Show high-level control-plane status")
    sub.add_parser("jobs", help="Show scheduled jobs and priorities")
    sub.add_parser("gpu-quota", help="Show recurring-free GPU quota status")
    sub.add_parser("refresh", help="Queue data refresh jobs")
    sub.add_parser("predict", help="Show model routing plan")
    sub.add_parser("train", help="Show retraining plan")
    sub.add_parser("backtest", help="Show model arena/backtest plan")
    sub.add_parser("deploy", help="Show free-tier deployment checklist")
    return parser


def run_cli(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    control_plane = ControlPlane()
    control_plane.submit_many(default_job_definitions(datetime.now(UTC)))

    if args.command == "status":
        print(json.dumps(control_plane.summary(), indent=2, sort_keys=True))
        return 0
    if args.command == "jobs":
        payload = [
            {
                "job_id": job.job_id,
                "job_type": job.job_type,
                "priority": int(job.priority),
                "deadline_at": job.deadline_at.isoformat() if job.deadline_at else None,
                "dependencies": list(job.dependencies),
            }
            for job in control_plane.ready_jobs()
        ]
        print(json.dumps(payload, indent=2))
        return 0
    if args.command == "gpu-quota":
        payload = [
            {
                "provider": provider.name,
                "remaining": provider.quota().remaining,
                "limit": provider.quota().limit,
                "unit": provider.quota().unit,
                "reset_at": provider.quota().reset_at.isoformat() if provider.quota().reset_at else None,
            }
            for provider in recurring_free_provider_defaults()
        ]
        print(json.dumps(payload, indent=2))
        return 0
    if args.command == "refresh":
        print("Queued data refresh jobs: news_capture, sec_capture, event_capture, eod_prices")
        return 0
    if args.command == "predict":
        router = FreeComputeRouter(recurring_free_provider_defaults())
        decisions = {
            spec.name: router.route(
                request=ExecutionRequest(
                    job_type="model_inference",
                    model_name=spec.name,
                    requires_gpu=spec.execution.requires_gpu,
                    minimum_vram_gb=spec.execution.minimum_vram_gb,
                    supports_cpu_fallback=spec.execution.supports_cpu_fallback,
                )
            ).__dict__
            for spec in default_model_specs()
        }
        print(json.dumps(decisions, indent=2))
        return 0
    if args.command == "train":
        print("Training plan queued: xgboost, lightgbm, catboost locally on free CPU; foundation-model fine-tune only when free GPU quota is available.")
        return 0
    if args.command == "backtest":
        print("Backtest plan queued: walk-forward evaluation, model arena comparison, prediction-journal refresh.")
        return 0
    if args.command == "deploy":
        print(
            "Target free-tier deploy: GitHub Actions -> cloud control plane/API -> object storage + metadata store -> Modal/Kaggle/HF ZeroGPU workers. Paid fallback disabled."
        )
        return 0
    return 1
