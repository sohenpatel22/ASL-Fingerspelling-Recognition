from types import SimpleNamespace

import numpy as np
import pytest
import torch

from asl.checkpoint import transfer_weights
from asl.config import load_config
from asl.data import build_datasets
from asl.decode import ctc_greedy_decode
from asl.model import ASLConformerSeq2Seq
from asl.train import Trainer, main, seed_everything


def _ctc_cfg(tiny_cfg):
    tiny_cfg.model.ctc = True
    tiny_cfg.train.ctc_weight = 0.3
    return tiny_cfg


def test_ctc_head_is_optional_and_state_dict_compatible(tiny_cfg, vocab):
    plain = ASLConformerSeq2Seq(tiny_cfg.model, vocab.vocab_size, vocab.pad_idx)
    assert plain.ctc_head is None
    assert not [k for k in plain.state_dict() if k.startswith("ctc_head")]

    joint_cfg = _ctc_cfg(tiny_cfg)
    joint = ASLConformerSeq2Seq(joint_cfg.model, vocab.vocab_size, vocab.pad_idx)
    assert {"ctc_head.weight", "ctc_head.bias"} <= set(joint.state_dict())
    stats = transfer_weights(joint, plain.state_dict())  # an older checkpoint still warm-starts it
    assert stats["skipped"] == 0
    torch.testing.assert_close(joint.encoder.proj[0].weight, plain.encoder.proj[0].weight)


def test_forward_joint_shapes(tiny_cfg, vocab):
    cfg = _ctc_cfg(tiny_cfg)
    model = ASLConformerSeq2Seq(cfg.model, vocab.vocab_size, vocab.pad_idx).eval()
    x = torch.randn(2, 84, cfg.model.max_seq_len)
    tgt = torch.randint(0, vocab.n_classes, (2, 6))
    logits, ctc_logits = model.forward_joint(x, tgt)
    assert logits.shape == (2, 6, vocab.vocab_size)
    assert ctc_logits.shape == (2, cfg.model.max_seq_len, vocab.vocab_size)


def test_ctc_greedy_decode_collapses_repeats_and_drops_blanks(vocab):
    blank = vocab.pad_idx
    a, b = vocab.char_to_idx["a"], vocab.char_to_idx["b"]
    frames = [a, a, blank, a, b, b, blank, blank]  # "a", "a" (after a blank), "b"
    logits = torch.full((1, len(frames), vocab.vocab_size), -10.0)
    logits[0, torch.arange(len(frames)), torch.tensor(frames)] = 10.0
    model = SimpleNamespace(encoder=lambda x: x, ctc_head=lambda h: logits)
    assert vocab.decode(ctc_greedy_decode(model, torch.zeros(1, 1, 1), vocab)[0]) == "aab"


def test_ctc_weight_without_ctc_head_is_rejected():
    with pytest.raises(SystemExit, match="model.ctc"):
        main(["--config", "configs/smoke.yaml", "train.ctc_weight=0.3"])


def test_joint_loss_learns_the_toy_task(tiny_cfg, vocab):
    seed_everything(0)
    cfg = _ctc_cfg(tiny_cfg)
    cfg.data.synthetic_size = 512
    cfg.train.epochs, cfg.train.lr, cfg.train.scheduler = 12, 0.002, "none"
    data = build_datasets(cfg, vocab)
    model = ASLConformerSeq2Seq(cfg.model, vocab.vocab_size, vocab.pad_idx)
    history = Trainer(cfg, model, vocab, data["train"], data["val"], torch.device("cpu")).fit()
    assert all(np.isfinite(h["train_loss"]) for h in history)
    assert "val_ctc_cer" in history[-1]
    assert history[-1]["val_greedy_cer"] < 0.4
    assert history[-1]["val_ctc_cer"] < history[0]["val_ctc_cer"]


def test_ctc_config_keys_parse():
    cfg = load_config(None, ["model.ctc=true", "model.velocity=true", "train.ctc_weight=0.25"])
    assert cfg.model.ctc and cfg.model.velocity and cfg.train.ctc_weight == 0.25
    assert cfg.model.input_dim == 168


def test_evaluate_with_the_ctc_decoder(tiny_cfg, vocab):
    from asl.evaluate import evaluate_dataset

    cfg = _ctc_cfg(tiny_cfg)
    model = ASLConformerSeq2Seq(cfg.model, vocab.vocab_size, vocab.pad_idx)
    data = build_datasets(cfg, vocab)["test"]
    res = evaluate_dataset(model, data, vocab, cfg.decode, cfg.model.max_phrase_len, max_samples=6, decoder="ctc")
    assert res["decoder"] == "ctc" and len(res["predictions"]) == 6
