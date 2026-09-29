import json

import pytest

pytest.importorskip("pyarrow")

from fake_competition import make_fake_competition  # noqa: E402

from asl.data import main as preprocess_main  # noqa: E402
from asl.evaluate import main as eval_main  # noqa: E402
from asl.train import main as train_main  # noqa: E402

TINY = [
    "model.d_model=32", "model.enc_layers=1", "model.dec_layers=1", "model.n_heads=4",
    "model.ffn_dim=64", "model.embed_dim=16", "model.dropout=0.0",
    "train.batch_size=16", "train.epochs=2", "train.num_workers=0", "train.eval_samples=16",
    "train.device=cpu",
]


def test_preprocess_train_evaluate_on_competition_format_data(tmp_path):
    comp = make_fake_competition(tmp_path / "raw")
    out_train, out_supp = tmp_path / "npy_train", tmp_path / "npy_supp"
    preprocess_main([
        "--comp-dir", str(comp), "--out-train", str(out_train), "--out-supp", str(out_supp),
    ])
    assert len(list(out_train.glob("*.npy"))) == 60 and len(list(out_supp.glob("*.npy"))) == 18

    overrides = [
        *TINY, f"data.comp_dir={comp.as_posix()}", f"data.npy_train={out_train.as_posix()}",
        f"data.npy_supp={out_supp.as_posix()}", f"train.out_dir={(tmp_path / 'run').as_posix()}",
    ]
    train_main(["--config", "configs/default.yaml", *overrides])
    assert (tmp_path / "run" / "best.pth").exists()

    eval_main([
        "--checkpoint", str(tmp_path / "run" / "best.pth"), "--config", "configs/default.yaml",
        "--split", "test", "--out", str(tmp_path / "summary.json"),
        "--details-out", str(tmp_path / "details.json"), *overrides,
    ])
    summary = json.loads((tmp_path / "summary.json").read_text())
    details = json.loads((tmp_path / "details.json").read_text())
    assert summary["n_samples"] > 0 and "cer" in summary and "cer_by_signer" not in summary
    assert set(details["cer_by_signer"]) and details["examples"]
