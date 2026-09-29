from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, HTTPException, Request, Response, UploadFile
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from pydantic import BaseModel

from asl.checkpoint import resolve_checkpoint
from asl.infer import Predictor
from asl.monitoring import DriftMonitor, landmark_stats
from asl.video import NoHandsDetected

log = logging.getLogger("asl.serve")
MAX_UPLOAD_MB = int(os.environ.get("ASL_MAX_UPLOAD_MB", "25"))


class LandmarksRequest(BaseModel):
    landmarks: list[list[float]]


class Metrics:
    def __init__(self) -> None:
        self.registry = CollectorRegistry()
        r = self.registry
        self.requests = Counter("asl_requests_total", "requests", ["endpoint", "status"], registry=r)
        self.latency = Histogram(
            "asl_request_latency_seconds", "end to end request latency", ["endpoint"], registry=r
        )
        self.inference = Histogram(
            "asl_inference_seconds", "beam search time", registry=r,
            buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10),
        )
        self.confidence = Histogram(
            "asl_prediction_confidence", "geometric mean token probability", registry=r,
            buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
        )
        self.detection = Histogram(
            "asl_hand_detection_rate", "share of frames with a detected hand", registry=r,
            buckets=(0.1, 0.25, 0.5, 0.75, 0.9, 1.0),
        )
        self.frames = Histogram(
            "asl_input_frames", "frames per request", registry=r,
            buckets=(8, 16, 32, 64, 128, 256, 512),
        )
        self.no_hands = Counter("asl_no_hands_total", "videos with no detectable hands", registry=r)
        self.drift = Gauge("asl_input_drift_psi", "PSI vs training data", ["feature"], registry=r)
        self.info = Gauge("asl_model_info", "loaded model", ["version"], registry=r)


def _model_version(path: Path) -> str:
    env = os.environ.get("ASL_MODEL_VERSION")
    return env or hashlib.sha256(path.read_bytes()).hexdigest()[:8]


def load_default_predictor() -> tuple[Predictor, str]:
    onnx_dir = os.environ.get("ASL_ONNX_DIR")
    if onnx_dir:
        from asl.export import load_onnx_predictor

        variant = os.environ.get("ASL_ONNX_VARIANT", "fp32")
        version = os.environ.get("ASL_MODEL_VERSION") or f"onnx-{variant}"
        return load_onnx_predictor(onnx_dir, variant), version
    model_uri = os.environ.get("ASL_MODEL_URI")
    if model_uri:
        from asl.tracking import download_model

        path = download_model(os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"), model_uri)
    else:
        path = resolve_checkpoint(os.environ.get("ASL_CHECKPOINT"))
    return Predictor.from_checkpoint(path), _model_version(path)


def create_app(
    predictor: Predictor | None = None,
    model_version: str = "unversioned",
    reference: dict | None = None,
) -> FastAPI:
    metrics = Metrics()
    state: dict = {"predictor": predictor, "version": model_version, "monitor": None}
    lock = threading.Lock()

    def _attach_reference(ref: dict | None) -> None:
        if ref is not None:
            state["monitor"] = DriftMonitor(ref["features"] if "features" in ref else ref)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        if state["predictor"] is None:
            state["predictor"], state["version"] = load_default_predictor()
        metrics.info.labels(version=state["version"]).set(1)
        ref_path = os.environ.get("ASL_REFERENCE")
        if reference is None and ref_path and Path(ref_path).exists():
            _attach_reference(json.loads(Path(ref_path).read_text()))
        yield

    app = FastAPI(title="ASL fingerspelling", lifespan=lifespan)
    _attach_reference(reference)

    def _observe_input(landmarks) -> None:
        stats = landmark_stats(landmarks)
        metrics.frames.observe(stats["n_frames"])
        metrics.detection.observe(stats["detection_rate"])
        if state["monitor"] is not None:
            state["monitor"].observe(stats)
            for feature, score in state["monitor"].scores().items():
                metrics.drift.labels(feature=feature).set(score)

    def _finish(endpoint: str, status: int, started: float, request_id: str, **fields) -> None:
        elapsed = time.perf_counter() - started
        metrics.requests.labels(endpoint=endpoint, status=str(status)).inc()
        metrics.latency.labels(endpoint=endpoint).observe(elapsed)
        log.info(json.dumps({
            "request_id": request_id, "endpoint": endpoint, "status": status,
            "latency_ms": round(elapsed * 1000, 1), "model_version": state["version"], **fields,
        }))

    def _predict(landmarks, endpoint: str, started: float, detection_rate=None) -> dict:
        request_id = uuid.uuid4().hex[:12]
        try:
            landmarks = np.asarray(landmarks, dtype=np.float32)
            expected = state["predictor"].model_cfg.feature_size
            if landmarks.ndim != 2 or landmarks.shape[1] != expected or len(landmarks) == 0:
                raise ValueError(f"expected a non-empty (T, {expected}) array, got {landmarks.shape}")
            _observe_input(landmarks)
            t0 = time.perf_counter()
            with lock:
                pred = state["predictor"].predict_landmarks(landmarks)
            metrics.inference.observe(time.perf_counter() - t0)
        except ValueError as err:
            _finish(endpoint, 422, started, request_id, error=str(err))
            raise HTTPException(422, str(err)) from err
        metrics.confidence.observe(pred.confidence)
        _finish(
            endpoint, 200, started, request_id,
            chars=len(pred.text), confidence=round(pred.confidence, 3),
        )
        return {
            "text": pred.text, "confidence": pred.confidence, "n_frames": pred.n_frames,
            "hand_detection_rate": detection_rate, "model_version": state["version"],
            "request_id": request_id,
        }

    @app.get("/health")
    def health() -> dict:
        ready = state["predictor"] is not None
        return {"status": "ok" if ready else "loading", "model_version": state["version"]}

    @app.get("/metrics")
    def prometheus() -> Response:
        return Response(generate_latest(metrics.registry), media_type=CONTENT_TYPE_LATEST)

    @app.post("/predict/landmarks")
    def predict_landmarks(body: LandmarksRequest) -> dict:
        return _predict(body.landmarks, "/predict/landmarks", time.perf_counter())

    @app.post("/predict")
    def predict_video(request: Request, video: UploadFile = File(...)) -> dict:  # noqa: B008
        from asl.video import extract_landmarks

        started = time.perf_counter()
        declared = int(request.headers.get("content-length", 0))
        if declared > MAX_UPLOAD_MB * 1024 * 1024:
            metrics.requests.labels(endpoint="/predict", status="413").inc()
            raise HTTPException(413, f"video larger than {MAX_UPLOAD_MB} MB")
        suffix = Path(video.filename or "clip.mp4").suffix or ".mp4"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(video.file.read())
        try:
            landmarks, rate = extract_landmarks(tmp.name)
        except NoHandsDetected as err:
            metrics.no_hands.inc()
            _finish("/predict", 422, started, uuid.uuid4().hex[:12], error="no hands")
            raise HTTPException(422, str(err)) from err
        finally:
            Path(tmp.name).unlink(missing_ok=True)
        return _predict(landmarks, "/predict", started, detection_rate=rate)

    return app


app = create_app()


def main() -> None:
    import uvicorn

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    uvicorn.run(
        "asl.serve:app", host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", "8000"))
    )


if __name__ == "__main__":
    main()
