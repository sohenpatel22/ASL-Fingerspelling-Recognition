import json

import numpy as np
import pandas as pd
import pytest

from asl.cv import summarize
from asl.data import fold_split, participant_folds

TINY = [
    "model.d_model=32", "model.enc_layers=1", "model.dec_layers=1", "model.n_heads=4",
    "model.ffn_dim=64", "model.embed_dim=16", "model.dropout=0.0",
    "train.batch_size=16", "train.epochs=1", "train.num_workers=0", "train.eval_samples=16",
    "train.device=cpu",
]


def _meta(n_participants=12, per=3):
    return pd.DataFrame({
        "participant_id": np.repeat(np.arange(n_participants), per),
        "sequence_id": np.arange(n_participants * per),
        "phrase": "abc",
    })


def test_folds_partition_the_signers():
    folds = participant_folds(_meta(), k=4)
    assert sum(len(f) for f in folds) == 12
    assert set().union(*folds) == set(range(12))
    assert all(a.isdisjoint(b) for i, a in enumerate(folds) for b in folds[i + 1 :])


def test_every_signer_is_a_test_signer_exactly_once():
    meta, k = _meta(), 4
    tested = []
    for fold in range(k):
        train, val, test = fold_split(meta, k, fold)
        tr_p, va_p, te_p = (set(d.participant_id) for d in (train, val, test))
        assert not (tr_p & va_p) and not (tr_p & te_p) and not (va_p & te_p)
        assert len(train) + len(val) + len(test) == len(meta)
        tested += sorted(te_p)
    assert sorted(tested) == list(range(12))


def test_fold_split_is_deterministic_and_seeded():
    a = fold_split(_meta(), 3, 1)[2].participant_id.tolist()
    assert a == fold_split(_meta(), 3, 1)[2].participant_id.tolist()
    assert a != fold_split(_meta(), 3, 1, seed=7)[2].participant_id.tolist()


def test_summarize():
    s = summarize([0.2, 0.4])
    assert s["mean"] == pytest.approx(0.3) and s["std"] == pytest.approx(0.1) and s["min"] == 0.2


def test_cv_cli_trains_and_scores_each_fold(tmp_path):
    pytest.importorskip("pyarrow")
    from fake_competition import make_fake_competition

    from asl.cv import main as cv_main
    from asl.data import main as preprocess_main

    comp = make_fake_competition(tmp_path / "raw")
    npy_train, npy_supp = tmp_path / "npy_train", tmp_path / "npy_supp"
    preprocess_main(["--comp-dir", str(comp), "--out-train", str(npy_train), "--out-supp", str(npy_supp)])
    out_dir = tmp_path / "cv"
    cv_main([
        "--folds", "3", "--run", "0,2", "--out-dir", str(out_dir), *TINY,
        f"data.comp_dir={comp.as_posix()}", f"data.npy_train={npy_train.as_posix()}",
        f"data.npy_supp={npy_supp.as_posix()}",
    ])
    report = json.loads((out_dir / "cv_results.json").read_text())
    assert set(report["cer_by_fold"]) == {"0", "2"} and report["folds"] == 3
    assert report["min"] <= report["mean"] <= report["max"] and report["std"] >= 0
    assert (out_dir / "fold0" / "best.pth").exists() and (out_dir / "fold2" / "test_summary.json").exists()
