"""FastAPI service for network-log anomaly detection.

Exposes /predict, /batch_predict, /healthz, /metrics.
Model and scaler are lazy-loaded so the service can start (and `/healthz` respond)
even if artifacts are missing — predictions will then return 503 until artifacts exist.
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from prometheus_client import Counter, Histogram
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel, Field

from src.config import get_settings
from src.logging_setup import configure_logging, get_request_id, set_request_id

log = logging.getLogger(__name__)

FEATURE_COLUMNS: list[str] = [
    "duration",
    "protocol",
    "src_port",
    "dst_port",
    "packet_count",
    "byte_count",
]

PREDICTIONS_TOTAL = Counter(
    "predictions_total",
    "Total predictions served, labelled by predicted class and endpoint.",
    ["label", "endpoint"],
)
INFERENCE_LATENCY = Histogram(
    "inference_latency_seconds",
    "Time spent in model inference (seconds).",
    ["endpoint"],
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)


class LogEntry(BaseModel):
    duration: float = Field(..., description="Flow duration (seconds).")
    protocol: int = Field(..., description="Protocol code (e.g. 6=TCP, 17=UDP).")
    src_port: int = Field(..., ge=0, le=65535)
    dst_port: int = Field(..., ge=0, le=65535)
    packet_count: int = Field(..., ge=0)
    byte_count: int = Field(..., ge=0)


class PredictionResponse(BaseModel):
    prediction: int
    anomaly_score: float


class BatchPredictRequest(BaseModel):
    items: list[LogEntry] = Field(..., min_length=1, max_length=1000)


class BatchPredictResponse(BaseModel):
    predictions: list[PredictionResponse]


class _ModelBundle:
    def __init__(self) -> None:
        self.model: Any | None = None
        self.scaler: Any | None = None

    def load(self, model_path: str, scaler_path: str) -> None:
        self.model = None
        self.scaler = None
        model_file = Path(model_path)
        scaler_file = Path(scaler_path)
        if not model_file.exists() or not scaler_file.exists():
            log.warning(
                "Model or scaler artifact missing; API will respond 503 on /predict",
                extra={"model_path": str(model_file), "scaler_path": str(scaler_file)},
            )
            return
        self.model = joblib.load(model_file)
        self.scaler = joblib.load(scaler_file)
        log.info("Loaded model and scaler", extra={"model_path": str(model_file)})

    def ready(self) -> bool:
        return self.model is not None and self.scaler is not None


bundle = _ModelBundle()


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ARG001
    settings = get_settings()
    configure_logging(settings.log_level)
    bundle.load(settings.model_path, settings.scaler_path)
    yield


app = FastAPI(title="Anomaly Detection API", version="0.1.0", lifespan=lifespan)


@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    rid = request.headers.get("x-request-id") or set_request_id()
    set_request_id(rid)
    response = await call_next(request)
    response.headers["x-request-id"] = rid
    return response


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    log.info("Validation error", extra={"errors": exc.errors(), "path": request.url.path})
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "detail": "Invalid request payload.",
            "errors": exc.errors(),
            "request_id": get_request_id(),
        },
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    log.exception("Unhandled exception", extra={"path": request.url.path})
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error.", "request_id": get_request_id()},
    )


def _require_ready() -> None:
    if not bundle.ready():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model artifacts not loaded. Train a model and restart the service.",
        )


def _predict_frame(df: pd.DataFrame, endpoint: str) -> list[PredictionResponse]:
    assert bundle.model is not None and bundle.scaler is not None
    start = time.perf_counter()
    scaled = bundle.scaler.transform(df[FEATURE_COLUMNS])
    preds = bundle.model.predict(scaled)
    probs = bundle.model.predict_proba(scaled)[:, 1]
    INFERENCE_LATENCY.labels(endpoint=endpoint).observe(time.perf_counter() - start)
    results: list[PredictionResponse] = []
    for pred, prob in zip(preds, probs, strict=True):
        label = int(pred)
        PREDICTIONS_TOTAL.labels(label=str(label), endpoint=endpoint).inc()
        results.append(
            PredictionResponse(prediction=label, anomaly_score=round(float(prob), 4))
        )
    return results


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "Anomaly Detection API is running."}


@app.get("/healthz")
def healthz() -> dict[str, Any]:
    return {"status": "ok", "model_loaded": bundle.ready()}


@app.post("/predict", response_model=PredictionResponse)
def predict(entry: LogEntry) -> PredictionResponse:
    _require_ready()
    df = pd.DataFrame([entry.model_dump()])
    return _predict_frame(df, endpoint="/predict")[0]


@app.post("/batch_predict", response_model=BatchPredictResponse)
def batch_predict(payload: BatchPredictRequest) -> BatchPredictResponse:
    _require_ready()
    df = pd.DataFrame([item.model_dump() for item in payload.items])
    return BatchPredictResponse(predictions=_predict_frame(df, endpoint="/batch_predict"))


Instrumentator().instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)
