import json

import pytest

from asl.ablate import RUNS, markdown_table

TINY = [
    "model.d_model=32", "model.enc_layers=1", "model.dec_layers=1", "model.n_heads=4",
    "model.ffn_dim=64", "model.embed_dim=16", "model.dropout=0.0",
    "train.batch_size=16", "train.epochs=1", "train.num_workers=0", "train.eval_samples=16",
    "train.device=cpu", "train.init_from=null",
]


def test_every_variant_is_a_valid_config():
    from asl.config import load_config

    for name, overrides in RUNS.items():
        cfg = load_config("configs/finetune_v5.yaml", overrides)
        assert cfg.train.init_from.startswith("hf:"), name
    everything = load_config("configs/finetune_v5.yaml", RUNS["a5_all"])
    assert everything.model.ctc and everything.model.velocity and everything.data.augment == "strong"
    assert everything.model.max_seq_len == 128 and everything.train.ctc_weight > 0


def test_markdown_table_has_one_row_per_run():
    row = {"best_val_greedy_cer": 0.3, "val_beam_cer": 0.33, "test_cer": 0.4, "test_ci95": [0.38, 0.42],
           "test_signers": 12}
    table = markdown_table({"a0": row, "a1": row})
    assert table.count("\n") == 3 and "| a1 | 0.300 | 0.330 | 0.400 | 0.380 to 0.420 | 12 |" in table


def test_ablation_runner_trains_scores_and_saves_results(tmp_path):
    pytest.importorskip("pyarrow")
    from fake_competition import make_fake_competition

    from asl.ablate import main as ablate_main
    from asl.data import main as preprocess_main

    comp = make_fake_competition(tmp_path / "raw")
    npy_train, npy_supp = tmp_path / "npy_train", tmp_path / "npy_supp"
    preprocess_main(["--comp-dir", str(comp), "--out-train", str(npy_train), "--out-supp", str(npy_supp)])
    out = tmp_path / "runs"
    ablate_main([
        "--only", "a0_baseline,a3_velocity", "--out-dir", str(out),
        "--val-samples", "8", "--test-samples", "8",
        *TINY, f"data.comp_dir={comp.as_posix()}", f"data.npy_train={npy_train.as_posix()}",
        f"data.npy_supp={npy_supp.as_posix()}",
    ])
    results = json.loads((out / "results.json").read_text())
    assert set(results) == {"a0_baseline", "a3_velocity"}
    assert results["a3_velocity"]["overrides"] == ["model.velocity=true"]
    assert all(r["test_cer"] >= 0 and r["epochs_run"] == 1 for r in results.values())
    assert (out / "results.md").read_text().count("a3_velocity") == 1
    assert (out / "a3_velocity" / "best.pth").exists()
