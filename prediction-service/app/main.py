"""API única del motor predictivo: quien consulta no sabe qué modelo responde."""
from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException, Request

from parking_ml import HORIZONS

from .engine import ForecastEngine
from .schemas import (AvailabilityRequest, Branch, FeedbackRequest, ForecastRequest, ForecastResponse,
                      RouterRequest)

ARTIFACTS = Path(os.environ.get("ARTIFACTS_DIR", Path(__file__).parent.parent / "artifacts"))
# Política inicial de la Tabla 2 del paper. El valor calibrado queda en registry.json.
INITIAL_TAU = int(os.environ.get("ROUTER_TAU", "1"))
DRIFT_ERROR = 0.9


@asynccontextmanager
async def lifespan(app: FastAPI):
    torch.set_num_threads(2)
    if not hasattr(app.state, "engine"):
        app.state.engine = ForecastEngine.from_artifacts(ARTIFACTS, INITIAL_TAU)
    yield


app = FastAPI(title="Motor predictivo híbrido", version="1.0", lifespan=lifespan)


def engine_of(request: Request) -> ForecastEngine:
    return request.app.state.engine


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/v1/forecasts", response_model=ForecastResponse)
def forecasts(body: ForecastRequest, request: Request) -> ForecastResponse:
    engine = engine_of(request)
    unknown = {query.zone for query in body.queries} - set(engine.zones)
    if unknown:
        raise HTTPException(422, f"Zonas desconocidas: {sorted(unknown)}")
    return ForecastResponse(results=engine.forecast(body.queries, body.include_attention))


@app.post("/v1/feedback")
def feedback(body: FeedbackRequest, request: Request) -> dict:
    engine = engine_of(request)
    for item in body.items:
        engine.health.record(item.branch, item.horizon, abs(item.predicted - item.observed))
    return {"recorded": len(body.items)}


@app.get("/v1/models")
def models(request: Request) -> dict:
    engine = engine_of(request)
    registry = json.loads((ARTIFACTS / "registry.json").read_text(encoding="utf-8"))
    return {**registry, "tau_active": engine.tau, "health": engine.health.snapshot()}


@app.get("/v1/report")
def report() -> dict:
    return json.loads((ARTIFACTS / "report.json").read_text(encoding="utf-8"))


@app.post("/v1/admin/branches/{branch}")
def set_branch(branch: Branch, body: AvailabilityRequest, request: Request) -> dict:
    engine_of(request).health.set_available(branch, body.available)
    return {"branch": branch, "available": body.available}


@app.post("/v1/admin/router")
def set_router(body: RouterRequest, request: Request) -> dict:
    engine_of(request).tau = body.tau
    return {"tau_active": body.tau}


@app.post("/v1/admin/branches/{branch}/drift")
def inject_drift(branch: Branch, request: Request) -> dict:
    """Simula deriva registrando errores grandes en todos los horizontes de la rama."""
    health = engine_of(request).health
    for horizon in HORIZONS:
        for _ in range(20):
            health.record(branch, horizon, DRIFT_ERROR)
    return {"branch": branch, "drift": True}


@app.post("/v1/admin/reset")
def reset(request: Request) -> dict:
    engine_of(request).health.reset()
    return {"reset": True}
