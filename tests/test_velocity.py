import numpy as np
import pytest
import torch

from asl.checkpoint import load_checkpoint, save_checkpoint
from asl.data import build_datasets
from asl.features import add_velocity, fit_length, resample_or_pad, to_model_input
from asl.infer import Predictor
from asl.model import ASLConformerSeq2Seq
from asl.train import Trainer, seed_everything


def _velocity_cfg(tiny_cfg):
    tiny_cfg.model.velocity = True
    return tiny_cfg


def test_add_velocity_is_the_frame_difference():
    seq = np.arange(12, dtype=np.float32).reshape(4, 3)
    out = add_velocity(seq)
    assert out.shape == (4, 6)
    np.testing.assert_array_equal(out[:, :3], seq)
    np.testing.assert_array_equal(out[0, 3:], 0)
    np.testing.assert_array_equal(out[1:, 3:], np.full((3, 3), 3.0))


@pytest.mark.parametrize("t", [10, 64, 150])
def test_fit_length_without_velocity_matches_the_old_behaviour(t):
    seq = np.random.default_rng(0).random((t, 84)).astype(np.float32)
    np.testing.assert_array_equal(fit_length(seq, 64), resample_or_pad(seq, 64))


def test_fit_length_with_velocity_pads_after_taking_differences():
    seq = np.random.default_rng(0).random((20, 84)).astype(np.float32)
    out = fit_length(seq, 64, velocity=True)
    assert out.shape == (64, 168)
    assert np.all(out[20:] == 0)  # no spike where the padding starts
    assert np.abs(out[1:20, 84:]).sum() > 0


def test_to_model_input_width_follows_the_flag():
    seq = np.random.default_rng(0).random((30, 84)).astype(np.float32)
    assert to_model_input(seq, 64).shape == (84, 64)
    assert to_model_input(seq, 64, velocity=True).shape == (168, 64)


def test_training_and_prediction_with_velocity(tiny_cfg, vocab, tmp_path):
    seed_everything(0)
    cfg = _velocity_cfg(tiny_cfg)
    cfg.data.synthetic_size = 256
    cfg.train.epochs, cfg.train.lr, cfg.train.scheduler = 4, 0.002, "none"
    data = build_datasets(cfg, vocab)
    assert data["train"][0][0].shape == (168, cfg.model.max_seq_len)

    model = ASLConformerSeq2Seq(cfg.model, vocab.vocab_size, vocab.pad_idx)
    history = Trainer(cfg, model, vocab, data["train"], data["val"], torch.device("cpu")).fit()
    assert history[-1]["train_loss"] < history[0]["train_loss"]

    path = save_checkpoint(tmp_path / "vel.pth", model, cfg.model, vocab, cfg.decode)
    loaded, _, ckpt = load_checkpoint(path)
    assert ckpt["_model_config"].velocity and ckpt["_model_config"].input_dim == 168
    predictor = Predictor(loaded, vocab, ckpt["_model_config"], cfg.decode)
    raw = np.random.default_rng(1).uniform(0.1, 0.9, (40, 84)).astype(np.float32)  # still 84 raw columns
    assert isinstance(predictor.predict_landmarks(raw).text, str)


def test_velocity_model_exports_to_onnx(tiny_cfg, vocab, tmp_path):
    pytest.importorskip("onnxscript")
    from asl.export import export_onnx, load_onnx_predictor

    cfg = _velocity_cfg(tiny_cfg)
    model = ASLConformerSeq2Seq(cfg.model, vocab.vocab_size, vocab.pad_idx).eval()
    export_onnx(model, vocab, cfg.model, cfg.decode, tmp_path / "onnx", quantize=False)
    raw = np.random.default_rng(2).uniform(0.1, 0.9, (35, 84)).astype(np.float32)
    ref = Predictor(model, vocab, cfg.model, cfg.decode).predict_landmarks(raw)
    assert load_onnx_predictor(tmp_path / "onnx").predict_landmarks(raw).text == ref.text
