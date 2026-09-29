import pytest

from asl.config import Config, load_config
from asl.model import ASLConformerSeq2Seq
from asl.vocab import Vocab


@pytest.fixture(scope="session")
def vocab() -> Vocab:
    return Vocab()


@pytest.fixture()
def tiny_cfg(tmp_path) -> Config:
    return load_config(
        None,
        [
            "model.d_model=32", "model.enc_layers=1", "model.dec_layers=1", "model.n_heads=4",
            "model.ffn_dim=64", "model.embed_dim=16", "model.dropout=0.0",
            "data.synthetic=true", "data.synthetic_size=128",
            "train.batch_size=32", "train.epochs=4", "train.lr=0.003", "train.warmup_epochs=1",
            "train.label_smoothing=0.0", "train.num_workers=0", "train.eval_samples=32",
            "train.device=cpu", f"train.out_dir={(tmp_path / 'run').as_posix()}",
        ],
    )


@pytest.fixture()
def tiny_model(tiny_cfg, vocab) -> ASLConformerSeq2Seq:
    return ASLConformerSeq2Seq(tiny_cfg.model, vocab.vocab_size, vocab.pad_idx).eval()
