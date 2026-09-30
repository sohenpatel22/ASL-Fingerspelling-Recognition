import json

import numpy as np
import pandas as pd
import pytest
import torch

from asl.checkpoint import (
    convert_checkpoint,
    load_checkpoint,
    load_state_dict_only,
    save_checkpoint,
    transfer_weights,
)
from asl.config import Config, load_config
from asl.data import ASLDataset, build_datasets, participant_split
from asl.evaluate import evaluate_dataset
from asl.infer import Predictor
from asl.model import ASLConformerSeq2Seq
from asl.train import (
    Trainer,
    make_scheduler,
    scheduled_sampling_epsilon,
    seed_everything,
)


def test_config_overrides_and_typo_detection():
    cfg = load_config(None, ["train.epochs=3", "model.dropout=0.1", "train.init_from=null"])
    assert cfg.train.epochs == 3 and cfg.model.dropout == 0.1 and cfg.train.init_from is None
    with pytest.raises(KeyError):
        load_config(None, ["train.epohcs=3"])
    with pytest.raises(KeyError):
        load_config(None, ["trian.epochs=3"])


def test_repo_configs_parse():
    for name in ("default", "smoke", "local_4gb"):
        assert isinstance(load_config(f"configs/{name}.yaml"), Config)


def _meta():
    return pd.DataFrame(
        {"participant_id": np.repeat(np.arange(20), 5), "sequence_id": np.arange(100),
         "phrase": "abc"}
    )


def test_participant_split_is_signer_disjoint_and_deterministic():
    tr, va, te = participant_split(_meta())
    sets = [set(d.participant_id) for d in (tr, va, te)]
    assert not (sets[0] & sets[1]) and not (sets[0] & sets[2]) and not (sets[1] & sets[2])
    assert len(tr) + len(va) + len(te) == 100 and (len(tr), len(va), len(te)) == (70, 15, 15)
    tr2, _, _ = participant_split(_meta())
    assert list(tr.sequence_id) == list(tr2.sequence_id)


def test_participant_split_matches_notebook_procedure():
    meta = _meta()
    np.random.seed(42)
    participants = meta["participant_id"].unique()
    np.random.shuffle(participants)
    expected_train = set(participants[:14])
    tr, _, _ = participant_split(meta)
    assert set(tr.participant_id) == expected_train


def test_asl_dataset_reads_npy(tmp_path, vocab, tiny_cfg):
    np.save(tmp_path / "7.npy", np.random.rand(30, 84).astype(np.float32))
    df = pd.DataFrame({"sequence_id": [7], "phrase": ["hi"], "participant_id": [1]})
    ds = ASLDataset(df, tmp_path, vocab, tiny_cfg.model, augment_data=True)
    x, y = ds[0]
    assert x.shape == (84, 64) and y.shape == (34,) and vocab.decode(y[1:].tolist()) == "hi"


def test_preprocess_parquet_roundtrip(tmp_path):
    pytest.importorskip("pyarrow")
    from asl.data import preprocess_parquets

    cols = ["frame"] + [f"{a}_{h}_hand_{i}" for h in ("left", "right") for a in "xy" for i in range(21)]
    rng = np.random.default_rng(0)
    data = rng.uniform(0.1, 0.9, (6, len(cols)))
    data[:, 0] = [0, 1, 2, 0, 1, 2]
    frame = pd.DataFrame(data, columns=cols, index=pd.Index([10, 10, 10, 11, 11, 11]))
    (tmp_path / "raw").mkdir()
    frame.to_parquet(tmp_path / "raw" / "1.parquet")
    meta = pd.DataFrame({"file_id": [1, 1], "sequence_id": [10, 11]})
    assert preprocess_parquets(meta, tmp_path / "raw", tmp_path / "out") == 2
    seq = np.load(tmp_path / "out" / "10.npy")
    assert seq.shape == (3, 84) and abs(seq[:, 0]).max() < 1e-6  # wrist-centred


def test_scheduler_warmup_then_cosine(tiny_cfg):
    cfg = tiny_cfg.train
    model = torch.nn.Linear(2, 2)
    opt = torch.optim.SGD(model.parameters(), lr=1.0)
    sched = make_scheduler(opt, cfg, steps_per_epoch=10)
    lrs = []
    for _ in range(cfg.epochs * 10):
        lrs.append(opt.param_groups[0]["lr"])
        opt.step()
        sched.step()
    assert lrs[0] < lrs[9] <= 1.0 and lrs[-1] < 0.05 and max(lrs) == pytest.approx(1.0, abs=0.02)


def test_scheduled_sampling_epsilon(tiny_cfg):
    t = tiny_cfg.train
    assert scheduled_sampling_epsilon(t, 3) == 1.0
    t.ss_start_epoch, t.epochs = 2, 10
    assert scheduled_sampling_epsilon(t, 1) == 1.0
    assert scheduled_sampling_epsilon(t, 2) == 1.0
    assert scheduled_sampling_epsilon(t, 10) == pytest.approx(t.ss_eps_end)


def test_training_learns_synthetic_task_and_checkpoint_roundtrips(tiny_cfg, vocab, tmp_path):
    seed_everything(0)
    tiny_cfg.data.synthetic_size = 512
    tiny_cfg.train.epochs, tiny_cfg.train.lr, tiny_cfg.train.scheduler = 10, 0.002, "none"
    tiny_cfg.train.ss_start_epoch = 8
    data = build_datasets(tiny_cfg, vocab)
    model = ASLConformerSeq2Seq(tiny_cfg.model, vocab.vocab_size, vocab.pad_idx)
    logged = []
    trainer = Trainer(
        tiny_cfg, model, vocab, data["train"], data["val"], torch.device("cpu"),
        on_metrics=lambda e, m: logged.append(m),
    )
    history = trainer.fit()
    assert len(logged) == len(history)
    assert history[-1]["train_loss"] < 0.5 * history[0]["train_loss"]
    assert history[-1]["val_greedy_cer"] < history[0]["val_greedy_cer"]
    assert history[-1]["val_greedy_cer"] < 0.3

    out = tmp_path / "run"
    assert (out / "best.pth").exists() and (out / "metrics.jsonl").exists()
    assert json.loads((out / "config.json").read_text())["train"]["epochs"] == 10

    loaded, loaded_vocab, ckpt = load_checkpoint(out / "best.pth")
    assert loaded_vocab.char_to_idx == vocab.char_to_idx
    assert ckpt["_model_config"].d_model == tiny_cfg.model.d_model
    x, _ = data["val"][0]
    model.eval()
    with torch.no_grad():
        best_state = loaded.state_dict()
        assert set(best_state) == set(model.state_dict())
        loaded(x[None], torch.tensor([[vocab.start_idx]]))


def test_resume_continues_from_last_epoch(tiny_cfg, vocab):
    seed_everything(0)
    tiny_cfg.train.epochs = 2
    data = build_datasets(tiny_cfg, vocab)

    def make():
        m = ASLConformerSeq2Seq(tiny_cfg.model, vocab.vocab_size, vocab.pad_idx)
        return Trainer(tiny_cfg, m, vocab, data["train"], data["val"], torch.device("cpu"))

    make().fit()
    tiny_cfg.train.epochs = 3
    t2 = make()
    t2.resume()
    assert t2.start_epoch == 3
    assert len(t2.fit()) == 1


def test_gradient_accumulation_runs(tiny_cfg, vocab):
    tiny_cfg.train.epochs, tiny_cfg.train.grad_accum_steps, tiny_cfg.train.batch_size = 1, 3, 16
    data = build_datasets(tiny_cfg, vocab)
    model = ASLConformerSeq2Seq(tiny_cfg.model, vocab.vocab_size, vocab.pad_idx)
    hist = Trainer(tiny_cfg, model, vocab, data["train"], data["val"], torch.device("cpu")).fit()
    assert np.isfinite(hist[0]["train_loss"])


def test_transfer_weights_handles_vocab_growth(tiny_cfg, vocab):
    new = ASLConformerSeq2Seq(tiny_cfg.model, vocab.vocab_size, vocab.pad_idx)
    old = ASLConformerSeq2Seq(tiny_cfg.model, vocab.vocab_size - 1, vocab.pad_idx - 1)
    stats = transfer_weights(new, old.state_dict())
    assert stats["resized"] == 3 and stats["skipped"] == 0
    torch.testing.assert_close(new.decoder.fc_out.weight[:-1], old.decoder.fc_out.weight)
    torch.testing.assert_close(new.encoder.proj[0].weight, old.encoder.proj[0].weight)


def test_legacy_checkpoint_with_numpy_scalar_converts(tmp_path, tiny_cfg, vocab, tiny_model):
    legacy = tmp_path / "legacy.pth"
    torch.save(
        {"model_state_dict": tiny_model.state_dict(), "test_cer": np.float64(0.43),
         "char_to_idx": vocab.char_to_idx,
         "d_model": 32, "enc_layers": 1, "dec_layers": 1, "n_heads": 4, "ffn_dim": 64,
         "embed_dim": 16, "max_seq_len": 64, "max_phrase_len": 34, "feature_size": 84},
        legacy,
    )
    with pytest.raises(RuntimeError, match="convert"):
        load_checkpoint(legacy)
    model, _, ckpt = load_checkpoint(legacy, allow_unsafe=True)
    assert ckpt["_model_config"].d_model == 32
    safe = convert_checkpoint(legacy, tmp_path / "safe.pth")
    model2, _, _ = load_checkpoint(safe)
    torch.testing.assert_close(model.state_dict()["decoder.fc_out.bias"],
                               model2.state_dict()["decoder.fc_out.bias"])
    assert set(load_state_dict_only(safe)) == set(tiny_model.state_dict())


def test_predictor_end_to_end_on_raw_landmarks(tmp_path, tiny_cfg, vocab, tiny_model):
    path = save_checkpoint(tmp_path / "m.pth", tiny_model, tiny_cfg.model, vocab, tiny_cfg.decode)
    predictor = Predictor.from_checkpoint(path)
    landmarks = np.random.default_rng(0).uniform(0.1, 0.9, (45, 84)).astype(np.float32)
    pred = predictor.predict_landmarks(landmarks)
    assert isinstance(pred.text, str) and 0.0 < pred.confidence <= 1.0 and pred.n_frames == 45
    for bad in (np.zeros((0, 84)), np.zeros((10, 50)), np.zeros(84)):
        with pytest.raises(ValueError):
            predictor.predict_landmarks(bad)


def test_evaluate_dataset_reports_signer_stats(tiny_cfg, vocab, tiny_model):
    ds = build_datasets(tiny_cfg, vocab)["test"]
    res = evaluate_dataset(
        tiny_model, ds, vocab, tiny_cfg.decode, tiny_cfg.model.max_phrase_len, max_samples=10
    )
    assert res["n_samples"] == 10 and 0.0 <= res["cer"] and res["cer_ci95"][0] <= res["cer_ci95"][1]
    assert res["n_signers"] == 5 and len(res["examples"]) == 10


def test_transfer_widens_the_input_conv_for_velocity_and_keeps_old_behaviour(tiny_cfg, vocab):
    plain = ASLConformerSeq2Seq(tiny_cfg.model, vocab.vocab_size, vocab.pad_idx)
    tiny_cfg.model.velocity = True
    wide = ASLConformerSeq2Seq(tiny_cfg.model, vocab.vocab_size, vocab.pad_idx)
    stats = transfer_weights(wide, plain.state_dict())
    assert stats["skipped"] == 0 and stats["resized"] >= 1
    old, new = plain.encoder.proj[0].weight, wide.encoder.proj[0].weight
    torch.testing.assert_close(new[:, : old.shape[1]], old)
    assert torch.count_nonzero(new[:, old.shape[1] :]) == 0  # velocity channels start silent


def test_init_from_accepts_a_hub_path(tmp_path, monkeypatch, tiny_cfg, vocab, tiny_model):
    from asl.checkpoint import fetch_hub_file, save_checkpoint

    saved = save_checkpoint(tmp_path / "m.pth", tiny_model, tiny_cfg.model, vocab)
    calls = {}

    def fake_download(repo_id, filename):
        calls.update(repo_id=repo_id, filename=filename)
        return str(saved)

    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_download)
    assert fetch_hub_file("hf:SohenP/asl-fingerspelling-conformer/asl_v5_best.pth") == saved
    assert calls == {"repo_id": "SohenP/asl-fingerspelling-conformer", "filename": "asl_v5_best.pth"}


def test_corpus_cer_weights_long_phrases_and_evaluation_keeps_every_prediction(tiny_cfg, tiny_model, vocab):
    from asl.config import DecodeConfig
    from asl.data import SyntheticDataset
    from asl.evaluate import evaluate_dataset
    from asl.metrics import corpus_cer, mean_cer

    # one perfect short phrase and one fully wrong long phrase
    assert mean_cer(["ab", "xxxxxxxx"], ["ab", "yyyyyyyy"]) == pytest.approx(0.5)
    assert corpus_cer(["ab", "xxxxxxxx"], ["ab", "yyyyyyyy"]) == pytest.approx(0.8)

    ds = SyntheticDataset(12, vocab, tiny_cfg.model, seed=5)
    res = evaluate_dataset(tiny_model, ds, vocab, DecodeConfig(2, 0.6), 8, max_samples=9)
    assert len(res["predictions"]) == 9
    assert {"signer", "target", "prediction", "cer"} <= set(res["predictions"][0])
    assert res["cer_micro"] >= 0 and len(res["examples"]) == 9


def test_evaluation_records_confidence_and_hand_rate(tmp_path, tiny_cfg, tiny_model, vocab):
    import pandas as pd

    from asl.config import DecodeConfig, ModelConfig
    from asl.data import ASLDataset
    from asl.evaluate import evaluate_dataset

    for seq_id, missing in ((1, 0.0), (2, 0.5)):
        seq = np.random.default_rng(seq_id).uniform(0.1, 0.9, (20, 84)).astype(np.float32)
        seq[int(20 * (1 - missing)) :] = 0.0  # the hand disappears for the last part of the clip
        np.save(tmp_path / f"{seq_id}.npy", seq)
    df = pd.DataFrame({"sequence_id": [1, 2], "phrase": ["ab", "cd"], "participant_id": [7, 7]})
    ds = ASLDataset(df, tmp_path, vocab, ModelConfig(**{**tiny_cfg.model.__dict__}))
    res = evaluate_dataset(tiny_model, ds, vocab, DecodeConfig(2, 0.6), 8)
    rec = res["predictions"]
    assert [round(r["hand_rate"], 2) for r in rec] == [1.0, 0.5]
    assert all(0 < r["confidence"] <= 1 for r in rec)
