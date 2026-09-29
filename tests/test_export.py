import dataclasses

import numpy as np
import pytest
import torch

pytest.importorskip("onnxruntime")
pytest.importorskip("onnxscript")

from asl.checkpoint import save_checkpoint  # noqa: E402
from asl.data import build_datasets  # noqa: E402
from asl.export import FILES, ExportDecoder, benchmark, export_onnx, load_onnx_predictor  # noqa: E402
from asl.infer import Predictor  # noqa: E402


@pytest.fixture()
def exported(tmp_path, tiny_cfg, tiny_model, vocab):
    model_cfg = dataclasses.replace(tiny_cfg.model, max_phrase_len=10)
    sizes = export_onnx(tiny_model, vocab, model_cfg, tiny_cfg.decode, tmp_path / "onnx")
    return tmp_path / "onnx", sizes, model_cfg


def test_export_decoder_matches_torch_decoder(tiny_model, tiny_cfg, vocab):
    torch.manual_seed(0)
    x = torch.randn(3, 84, 50)
    tgt = torch.randint(0, vocab.n_classes, (3, 7))
    memory = tiny_model.encoder(x)
    with torch.no_grad():
        wrapped = ExportDecoder(tiny_model.decoder)(tgt, memory)
        torch.testing.assert_close(wrapped, tiny_model.decoder(tgt, memory), rtol=1e-4, atol=1e-4)


def test_onnx_fp32_predictions_match_torch(exported, tiny_cfg, tiny_model, vocab):
    onnx_dir, _, model_cfg = exported
    ref = Predictor(tiny_model, vocab, model_cfg, tiny_cfg.decode)
    onnx = load_onnx_predictor(onnx_dir, "fp32")
    rng = np.random.default_rng(0)
    for frames in (20, 64, 150):  # exercises the dynamic frame axis
        clip = rng.uniform(0.1, 0.9, (frames, 84)).astype(np.float32)
        a, b = ref.predict_landmarks(clip), onnx.predict_landmarks(clip)
        assert a.text == b.text
        assert a.confidence == pytest.approx(b.confidence, rel=1e-3)


def test_int8_model_runs(exported):
    onnx_dir, sizes, _ = exported
    assert set(sizes) == {"fp32", "int8"} and all(v > 0 for s in sizes.values() for v in s.values())
    pred = load_onnx_predictor(onnx_dir, "int8").predict_landmarks(np.random.rand(30, 84).astype(np.float32))
    assert isinstance(pred.text, str) and 0 < pred.confidence <= 1


def test_benchmark_reports_all_variants(tmp_path, tiny_cfg, tiny_model, vocab, exported):
    onnx_dir, _, model_cfg = exported
    ckpt = save_checkpoint(tmp_path / "m.pth", tiny_model, model_cfg, vocab, tiny_cfg.decode)
    report = benchmark(ckpt, onnx_dir, build_datasets(tiny_cfg, vocab)["test"], n=6)
    assert set(report) == {"torch", "onnx_fp32", "onnx_int8"}
    for row in report.values():
        assert row["latency_ms_p50"] > 0 and row["size_mb"] > 0
        assert np.isfinite(row["cer"]) and row["cer"] >= 0
    assert report["torch"]["agreement_with_torch"] == 1.0
    assert all((onnx_dir / f).exists() for pair in FILES.values() for f in pair)


def test_service_loads_onnx_model_from_env(exported, monkeypatch):
    onnx_dir, _, _ = exported
    monkeypatch.setenv("ASL_ONNX_DIR", str(onnx_dir))
    monkeypatch.setenv("ASL_ONNX_VARIANT", "int8")
    from asl.serve import load_default_predictor

    predictor, version = load_default_predictor()
    assert version == "onnx-int8"
    assert isinstance(predictor.predict_landmarks(np.random.rand(25, 84).astype(np.float32)).text, str)
