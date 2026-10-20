"""FastAPI service.

    uvicorn serve.app:app --host 0.0.0.0 --port 8000

POST /classify  {"signal": [[...1000 floats...] x 12]}   (mV, 100 Hz, leads I..V6)
  -> {"status": "ok", "labels": [...], "probabilities": {...}, ...}
  -> {"status": "low_quality", "message": "signal quality too low", ...}
"""
from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from serve.pipeline import ECGPipeline, SignalError


class ECGRequest(BaseModel):
    signal: list[list[float]] = Field(..., description="12 x 1000 array, millivolts, 100 Hz, lead order I, II, III, aVR, aVL, aVF, V1-V6")


def create_app(artifacts_dir: str | None = None, threads: int | None = None) -> FastAPI:
    state: dict = {}

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        state["pipeline"] = ECGPipeline(
            artifacts_dir or os.environ.get("ECG_ARTIFACTS", "artifacts"),
            threads=threads or int(os.environ.get("ECG_THREADS", "4")),
        )
        yield
        state.clear()

    app = FastAPI(title="ECG classifier with signal-quality gate", lifespan=lifespan)

    @app.get("/health")
    def health() -> dict:
        pipe: ECGPipeline = state["pipeline"]
        return {"status": "ok", "classes": pipe.classes, "classifier": pipe.meta["classifier"]["config_name"]}

    @app.post("/classify")
    def classify(req: ECGRequest) -> dict:
        t0 = time.perf_counter()
        try:
            result = state["pipeline"].predict(req.signal)
        except SignalError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        result["latency_ms"] = round((time.perf_counter() - t0) * 1e3, 3)
        return result

    return app


app = create_app()
