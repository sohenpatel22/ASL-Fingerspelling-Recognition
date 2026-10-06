import dataclasses

import numpy as np
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("prometheus_client")
from fastapi.testclient import TestClient  # noqa: E402

import asl.serve as serve  # noqa: E402
from asl.infer import Predictor  # noqa: E402
from asl.monitoring import DriftMonitor, build_reference, landmark_stats, psi  # noqa: E402
from asl.video import NoHandsDetected  # noqa: E402


def _landmarks(t=30, seed=0, right_missing=False):
    rng = np.random.default_rng(seed)
    seq = rng.uniform(0.1, 0.9, (t, 84)).astype(np.float32)
    if right_missing:
        seq[:, 42:] = 0
    return seq


@pytest.fixture()
def client(tiny_cfg, tiny_model, vocab):
    model_cfg = dataclasses.replace(tiny_cfg.model, max_phrase_len=8)  # keeps beam search quick
    predictor = Predictor(tiny_model, vocab, model_cfg, tiny_cfg.decode)
    ref = build_reference([landmark_stats(_landmarks(seed=i)) for i in range(60)])
    with TestClient(serve.create_app(predictor, "test-v1", reference=ref)) as c:
        yield c


def test_health_reports_model_version(client):
    assert client.get("/health").json() == {"status": "ok", "model_version": "test-v1"}


def test_predict_landmarks_ok(client):
    r = client.post("/predict/landmarks", json={"landmarks": _landmarks().tolist()})
    body = r.json()
    assert r.status_code == 200 and isinstance(body["text"], str)
    assert 0 < body["confidence"] <= 1 and body["n_frames"] == 30 and body["model_version"] == "test-v1"


@pytest.mark.parametrize("payload", [[], [[0.0] * 10]])
def test_predict_landmarks_rejects_bad_shape(client, payload):
    assert client.post("/predict/landmarks", json={"landmarks": payload}).status_code == 422


def test_metrics_endpoint_counts_requests(client):
    client.post("/predict/landmarks", json={"landmarks": _landmarks().tolist()})
    client.post("/predict/landmarks", json={"landmarks": []})
    text = client.get("/metrics").text
    assert 'asl_requests_total{endpoint="/predict/landmarks",status="200"} 1.0' in text
    assert 'asl_requests_total{endpoint="/predict/landmarks",status="422"} 1.0' in text
    assert "asl_prediction_confidence_bucket" in text and 'asl_model_info{version="test-v1"} 1.0' in text


def test_video_endpoint_uses_extracted_landmarks(client, monkeypatch):
    monkeypatch.setattr("asl.video.extract_landmarks", lambda path: (_landmarks(40), 0.9))
    r = client.post("/predict", files={"video": ("clip.mp4", b"not really a video", "video/mp4")})
    assert r.status_code == 200 and r.json()["hand_detection_rate"] == 0.9


def test_video_without_hands_is_422_and_counted(client, monkeypatch):
    def no_hands(path):
        raise NoHandsDetected("nothing here")

    monkeypatch.setattr("asl.video.extract_landmarks", no_hands)
    r = client.post("/predict", files={"video": ("clip.mp4", b"x", "video/mp4")})
    assert r.status_code == 422 and "asl_no_hands_total 1.0" in client.get("/metrics").text


def test_oversized_upload_rejected(client, monkeypatch):
    monkeypatch.setattr(serve, "MAX_UPLOAD_MB", 0)
    r = client.post("/predict", files={"video": ("clip.mp4", b"x" * 100, "video/mp4")})
    assert r.status_code == 413


def _gauge(client, feature):
    for line in client.get("/metrics").text.splitlines():
        if line.startswith(f'asl_input_drift_psi{{feature="{feature}"}}'):
            return float(line.split()[-1])
    raise AssertionError(f"no drift gauge for {feature}")


def test_drift_gauge_flags_shifted_inputs(client):
    for i in range(40):
        client.post("/predict/landmarks", json={"landmarks": _landmarks(seed=100 + i).tolist()})
    stable = _gauge(client, "right_rate")
    for i in range(60):  # right hand disappears
        payload = _landmarks(seed=500 + i, right_missing=True).tolist()
        client.post("/predict/landmarks", json={"landmarks": payload})
    assert stable < 0.1 and _gauge(client, "right_rate") > 0.25


def test_landmark_stats_values():
    stats = landmark_stats(_landmarks(20, right_missing=True))
    assert stats["n_frames"] == 20 and stats["left_rate"] == 1.0 and stats["right_rate"] == 0.0
    assert stats["detection_rate"] == 1.0 and stats["motion"] > 0


def test_psi_zero_for_identical_and_large_for_shifted():
    p = np.array([0.25, 0.25, 0.25, 0.25])
    assert psi(p, p) == pytest.approx(0.0, abs=1e-9)
    assert psi(p, np.array([0.7, 0.1, 0.1, 0.1])) > 0.25


def test_drift_monitor_needs_enough_samples():
    ref = build_reference([landmark_stats(_landmarks(seed=i)) for i in range(50)])
    monitor = DriftMonitor(ref)
    for i in range(10):
        monitor.observe(landmark_stats(_landmarks(seed=i)))
    assert monitor.scores() == {}


def test_low_hand_visibility_is_flagged_and_counted(client, monkeypatch):
    for rate in (0.1, 0.9):
        monkeypatch.setattr("asl.video.extract_landmarks", lambda path, r=rate: (_landmarks(40), r))
        body = client.post("/predict", files={"video": ("clip.mp4", b"x", "video/mp4")}).json()
        assert body["low_hand_visibility"] is (rate < 0.3)
    assert "asl_low_visibility_total 1.0" in client.get("/metrics").text
    plain = client.post("/predict/landmarks", json={"landmarks": _landmarks().tolist()}).json()
    assert plain["low_hand_visibility"] is False  # unknown detection rate is never flagged


def test_flagged_predictions_carry_a_reason_and_are_counted(client, monkeypatch):
    monkeypatch.setattr("asl.video.extract_landmarks", lambda path: (_landmarks(40), 0.9))
    files = {"video": ("clip.mp4", b"x", "video/mp4")}
    monkeypatch.setenv("ASL_MIN_CONFIDENCE", "0.0")
    assert client.post("/predict", files=files).json()["flag_reason"] is None
    monkeypatch.setenv("ASL_MIN_CONFIDENCE", "1.1")  # nothing can be that confident
    body = client.post("/predict", files=files).json()
    assert body["flagged"] is True and body["flag_reason"] == "low_confidence"
    assert 'asl_flagged_total{reason="low_confidence"} 1.0' in client.get("/metrics").text


def test_default_predictor_uses_the_deployed_length_penalty(
    tmp_path, tiny_cfg, tiny_model, vocab, monkeypatch
):
    from asl.checkpoint import save_checkpoint
    from asl.infer import DEPLOY_LENGTH_PENALTY

    ckpt = save_checkpoint(tmp_path / "m.pth", tiny_model, tiny_cfg.model, vocab, tiny_cfg.decode)
    monkeypatch.setenv("ASL_CHECKPOINT", str(ckpt))
    monkeypatch.delenv("ASL_LENGTH_PENALTY", raising=False)
    monkeypatch.delenv("ASL_ONNX_DIR", raising=False)
    monkeypatch.delenv("ASL_MODEL_URI", raising=False)
    predictor, _ = serve.load_default_predictor()
    assert tiny_cfg.decode.length_penalty != DEPLOY_LENGTH_PENALTY  # the checkpoint says 0.6
    assert predictor.decode_cfg.length_penalty == DEPLOY_LENGTH_PENALTY
    monkeypatch.setenv("ASL_LENGTH_PENALTY", "0.3")
    assert serve.load_default_predictor()[0].decode_cfg.length_penalty == 0.3
