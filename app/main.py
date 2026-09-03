from __future__ import annotations

import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from app.api import accuracy, backtests, control_plane, dashboard, holdings, stocks
from app.deps import APP_DIR
from src.scheduler.jobs import start_scheduler
from src.observability.activity import configure_activity_logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
configure_activity_logging()


@asynccontextmanager
async def _lifespan(app: FastAPI):
    scheduler_enabled = os.getenv("ENABLE_PIPELINE_SCHEDULER", "true").lower() in {"1", "true", "yes"}
    app.state.scheduler = start_scheduler() if scheduler_enabled else None
    yield
    if app.state.scheduler is not None:
        app.state.scheduler.shutdown(wait=False)


app = FastAPI(title="AI Quant Platform", lifespan=_lifespan)
app.mount("/static", StaticFiles(directory=str(APP_DIR / "static")), name="static")

app.include_router(dashboard.router)
app.include_router(stocks.router)
app.include_router(backtests.router)
app.include_router(holdings.router)
app.include_router(accuracy.router)
app.include_router(control_plane.router)


@app.get("/")
def root() -> RedirectResponse:
    return RedirectResponse("/dashboard")


@app.get("/health", tags=["operations"])
def health() -> dict[str, str]:
    """Load-balancer health check with no database or provider dependency."""
    return {"status": "ok", "service": "quant-platform"}
