import json

import numpy as np
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


def test_strong_augmentation_only_applies_to_training_data(tmp_path):
    from asl.config import load_config
    from asl.data import build_datasets
    from asl.vocab import Vocab

    comp = make_fake_competition(tmp_path / "raw")
    npy_train, npy_supp = tmp_path / "npy_train", tmp_path / "npy_supp"
    preprocess_main(["--comp-dir", str(comp), "--out-train", str(npy_train), "--out-supp", str(npy_supp)])
    overrides = [
        f"data.comp_dir={comp.as_posix()}", f"data.npy_train={npy_train.as_posix()}",
        f"data.npy_supp={npy_supp.as_posix()}",
    ]
    for setting, expected in (("strong", True), ("basic", False)):
        cfg = load_config(None, [*overrides, f"data.augment={setting}"])
        data = build_datasets(cfg, Vocab())
        assert all(part.strong is expected for part in data["train"].datasets)
        assert not data["val"].augment and not data["test"].augment


def test_seeded_random_subset_covers_more_signers_than_the_first_rows(tiny_cfg, tiny_model, vocab):
    import numpy as np

    from asl.config import DecodeConfig
    from asl.data import SyntheticDataset
    from asl.evaluate import evaluate_dataset

    ds = SyntheticDataset(60, vocab, tiny_cfg.model, seed=3)
    ds.df["participant_id"] = np.repeat(np.arange(6), 10)  # rows are grouped by signer, like real data
    common = (tiny_model, ds, vocab, DecodeConfig(2, 0.6), 8)
    first = evaluate_dataset(*common, max_samples=10)
    rand_a = evaluate_dataset(*common, max_samples=10, sample_seed=1)
    rand_b = evaluate_dataset(*common, max_samples=10, sample_seed=1)
    assert first["n_signers"] == 1 and rand_a["n_signers"] > 1
    assert rand_a["cer"] == rand_b["cer"]


def test_sequences_missing_from_disk_are_skipped_not_fatal(tmp_path, capsys):
    import pandas as pd

    from asl.config import ModelConfig
    from asl.data import ASLDataset
    from asl.vocab import Vocab

    rng = np.random.default_rng(0)
    for seq_id in (1, 2, 4):  # sequence 3 is in the metadata but has no file
        np.save(tmp_path / f"{seq_id}.npy", rng.random((20, 84)).astype(np.float32))
    df = pd.DataFrame(
        {"sequence_id": [1, 2, 3, 4], "phrase": list("abcd"), "participant_id": [1, 1, 2, 2]}
    )
    ds = ASLDataset(df, tmp_path, Vocab(), ModelConfig(), augment_data=True)
    assert len(ds) == 3 and list(ds.df.sequence_id) == [1, 2, 4]
    assert "skipping 1 sequences" in capsys.readouterr().out
    for i in range(len(ds)):
        ds[i]  # every remaining row loads
