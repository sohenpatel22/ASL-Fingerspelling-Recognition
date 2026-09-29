import json

import numpy as np
import pytest
import torch

from asl.checkpoint import save_checkpoint
from asl.decode import beam_search
from asl.infer import Predictor
from asl.lm import EOS, CharNgramLM

WORDS = ["cat", "cats", "bus", "tip", "date", "kenya", "papaya", "norway", "japan", "iceland"]


def test_probabilities_over_the_alphabet_sum_to_one():
    lm = CharNgramLM(order=3, vocab_size=4).fit(["ab", "abc", "ba"])
    for context in ("", "a", "ab", "zzz"):
        total = sum(np.exp(lm.logprob(context, c)) for c in ("a", "b", "c", EOS))
        assert total == pytest.approx(1.0, abs=1e-9)


def test_seen_text_scores_higher_than_scrambled_text():
    lm = CharNgramLM(order=4, vocab_size=60).fit(WORDS * 3)
    assert lm.score("norway") > lm.score("wyaron")
    assert lm.score("papaya") > lm.score("yapapa")
    assert lm.logprob("kenya"[:-1], "a") > lm.logprob("kenya"[:-1], "q")


def test_save_and_load_round_trip(tmp_path):
    lm = CharNgramLM(order=4).fit(WORDS)
    lm.save(tmp_path / "lm.json.gz")
    loaded = CharNgramLM.load(tmp_path / "lm.json.gz")
    for text in ("cat", "japan", "unseen"):
        assert loaded.score(text) == pytest.approx(lm.score(text))


def test_lm_fusion_pulls_beam_search_towards_the_training_text(tiny_model, tiny_cfg, vocab):
    lm = CharNgramLM(order=4, vocab_size=vocab.n_classes + 1).fit(WORDS * 5)
    plain_scores, fused_scores = [], []
    for seed in range(8):
        torch.manual_seed(seed)
        x = torch.randn(84, tiny_cfg.model.max_seq_len)
        plain, _ = beam_search(tiny_model, x, vocab, 4, 10, 0.6)
        fused, _ = beam_search(tiny_model, x, vocab, 4, 10, 0.6, lm=lm, lm_weight=3.0)
        plain_scores.append(lm.score(vocab.decode(plain)))
        fused_scores.append(lm.score(vocab.decode(fused)))
    assert np.mean(fused_scores) > np.mean(plain_scores)


def test_zero_lm_weight_changes_nothing(tiny_model, tiny_cfg, vocab):
    lm = CharNgramLM(order=4).fit(WORDS)
    torch.manual_seed(0)
    x = torch.randn(84, tiny_cfg.model.max_seq_len)
    assert beam_search(tiny_model, x, vocab, 3, 8, 0.6) == beam_search(
        tiny_model, x, vocab, 3, 8, 0.6, lm=lm, lm_weight=0.0
    )


def test_confidence_is_the_model_score_not_the_fused_score(tmp_path, tiny_model, tiny_cfg, vocab):
    lm = CharNgramLM(order=4, vocab_size=60).fit(WORDS)
    lm.save(tmp_path / "lm.json.gz")
    ckpt = save_checkpoint(tmp_path / "m.pth", tiny_model, tiny_cfg.model, vocab, tiny_cfg.decode)
    predictor = Predictor.from_checkpoint(ckpt, lm_path=tmp_path / "lm.json.gz", lm_weight=2.0)
    assert predictor.lm is not None and predictor.decode_cfg.lm_weight == 2.0
    pred = predictor.predict_landmarks(np.random.default_rng(0).uniform(0.1, 0.9, (40, 84)).astype("float32"))
    assert 0 < pred.confidence <= 1


def test_asl_lm_fits_only_on_training_split_phrases_and_the_sweep_runs(tmp_path, tiny_cfg, vocab, tiny_model):
    pytest.importorskip("pyarrow")
    import pandas as pd
    from fake_competition import make_fake_competition

    from asl.data import main as preprocess_main
    from asl.data import participant_split
    from asl.evaluate import main as eval_main
    from asl.lm import main as lm_main

    comp = make_fake_competition(tmp_path / "raw")
    npy_train, npy_supp = tmp_path / "npy_train", tmp_path / "npy_supp"
    preprocess_main(["--comp-dir", str(comp), "--out-train", str(npy_train), "--out-supp", str(npy_supp)])
    data_overrides = [
        f"data.comp_dir={comp.as_posix()}", f"data.npy_train={npy_train.as_posix()}",
        f"data.npy_supp={npy_supp.as_posix()}",
    ]

    lm_path = tmp_path / "lm.json.gz"
    lm_main(["--out", str(lm_path), *data_overrides])
    lm = CharNgramLM.load(lm_path)
    meta = pd.read_csv(comp / "train.csv")
    train, val, test = participant_split(meta)
    supp = pd.read_csv(comp / "supplemental_metadata.csv")
    expected = sum(len(p) + 1 for p in [*train["phrase"], *supp["phrase"]])
    assert sum(lm.counts[""].values()) == expected  # no val or test signer phrases went in
    assert len(val) and len(test)

    ckpt = save_checkpoint(tmp_path / "m.pth", tiny_model, tiny_cfg.model, vocab, tiny_cfg.decode)
    out = tmp_path / "sweep.json"
    eval_main([
        "--checkpoint", str(ckpt), "--split", "val", "--max-samples", "6", "--sweep-lm-weights", "0,1.0",
        "--out", str(out), *data_overrides, f"decode.lm_path={lm_path.as_posix()}",
    ])
    result = json.loads(out.read_text())
    assert set(result["sweep"]) == {"0.0", "1.0"} and result["best_lm_weight"] in (0.0, 1.0)
