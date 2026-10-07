import numpy as np
import pandas as pd
import pytest
import torch

from asl.config import ModelConfig, load_config
from asl.data import ASLDataset, build_datasets
from asl.features import (
    augment,
    concat_clips,
    finger_dropout,
    resample_to,
    shear,
    temporal_mask,
    wrist_normalize,
)
from asl.model import ASLConformerSeq2Seq
from asl.train import Trainer, mask_decoder_inputs
from asl.vocab import Vocab


def _clip(t=40, seed=0):
    rng = np.random.default_rng(seed)
    return wrist_normalize(rng.uniform(0.1, 0.9, (t, 84)).astype(np.float32))


def test_shear_moves_x_by_y_and_y_by_x():
    seq = _clip()
    out = shear(seq, 0.2, -0.1)
    np.testing.assert_allclose(out[:, :21], seq[:, :21] + 0.2 * seq[:, 21:42], atol=1e-6)
    np.testing.assert_allclose(out[:, 63:84], seq[:, 63:84] - 0.1 * seq[:, 42:63], atol=1e-6)


def test_finger_dropout_blanks_whole_fingers_of_one_hand_only():
    seq = np.ones((30, 84), dtype=np.float32)
    for seed in range(20):
        out = finger_dropout(seq, np.random.default_rng(seed))
        changed = np.where((out != seq).any(axis=0))[0]
        hand = {int(c >= 42) for c in changed}
        assert len(hand) == 1  # one hand
        landmarks = {int(c % 21) for c in changed}
        assert 4 <= len(landmarks) <= 8 and 0 not in landmarks  # one or two fingers, never the wrist
        assert (out[:, changed].sum(axis=1) < len(changed)).any()


def test_temporal_mask_zeroes_contiguous_frames_and_keeps_the_length():
    seq = np.ones((50, 84), dtype=np.float32)
    for seed in range(20):
        out = temporal_mask(seq, np.random.default_rng(seed))
        assert out.shape == seq.shape
        zero = (out == 0).all(axis=1)
        assert 1 <= zero.sum() <= 0.3 * 50 + 1


def test_concat_clips_keeps_everything_if_it_fits_and_squeezes_if_not():
    a, b = _clip(30, 1), _clip(20, 2)
    np.testing.assert_array_equal(concat_clips(a, b, 100), np.concatenate([a, b]))
    squeezed = concat_clips(a, b, 25)
    assert len(squeezed) <= 25 and len(resample_to(a, 10)) == 10


def test_heavy_augmentation_is_valid_and_changes_the_clip():
    seq = _clip(60)
    changed = 0
    for seed in range(40):
        out = augment(seq, np.random.default_rng(seed), heavy=True)
        assert out.shape[1] == 84 and out.dtype == np.float32 and np.isfinite(out).all() and len(out) >= 4
        changed += len(out) != len(seq) or not np.array_equal(out[: len(seq)], seq[: len(out)])
    assert changed > 35


def _dataset(tmp_path, phrases, cutmix):
    tmp_path.mkdir(exist_ok=True)
    rng = np.random.default_rng(0)
    for i, _ in enumerate(phrases):
        np.save(tmp_path / f"{i}.npy", rng.uniform(0.1, 0.9, (30, 84)).astype(np.float32))
    df = pd.DataFrame({"sequence_id": range(len(phrases)), "phrase": phrases, "participant_id": 1})
    return ASLDataset(df, tmp_path, Vocab(), ModelConfig(), augment_data=True, cutmix=cutmix)


def _lengths(ds, vocab, repeats=10):
    return {len(vocab.decode(ds[i][1][1:].tolist())) for i in range(len(ds)) for _ in range(repeats)}


def test_cutmix_glues_two_phrases_when_they_fit_and_leaves_long_ones_alone(tmp_path):
    vocab = Vocab()
    assert _lengths(_dataset(tmp_path / "always", ["ab", "cd", "ef"], cutmix=1.0), vocab) == {4}
    assert _lengths(_dataset(tmp_path / "half", ["ab", "cd", "ef"], cutmix=0.5), vocab, repeats=30) == {2, 4}
    assert _lengths(_dataset(tmp_path / "long", ["x" * 20, "y" * 20], cutmix=1.0), vocab) == {20}  # 40 > 32
    assert _lengths(_dataset(tmp_path / "off", ["ab", "cd"], cutmix=0.0), vocab) == {2}


def test_mask_decoder_inputs_keeps_the_first_token_and_scales_with_probability():
    vocab = Vocab()
    y = torch.randint(0, vocab.n_classes, (200, 12))
    y[:, 0] = vocab.start_idx
    assert torch.equal(mask_decoder_inputs(y, 0.0, vocab.pad_idx), y)
    everything = mask_decoder_inputs(y, 1.0, vocab.pad_idx)
    assert (everything[:, 0] == vocab.start_idx).all() and (everything[:, 1:] == vocab.pad_idx).all()
    half = mask_decoder_inputs(y, 0.5, vocab.pad_idx)
    share = (half[:, 1:] == vocab.pad_idx).float().mean().item()
    assert 0.4 < share < 0.6


def test_training_runs_with_every_new_ingredient_switched_on(tiny_cfg, vocab):
    tiny_cfg.train.decoder_mask, tiny_cfg.train.ema_decay = 0.3, 0.9
    tiny_cfg.train.ctc_weight, tiny_cfg.model.ctc, tiny_cfg.train.epochs = 0.25, True, 1
    tiny_cfg.data.synthetic_size = 64
    data = build_datasets(tiny_cfg, vocab)
    model = ASLConformerSeq2Seq(tiny_cfg.model, vocab.vocab_size, vocab.pad_idx)
    history = Trainer(tiny_cfg, model, vocab, data["train"], data["val"], torch.device("cpu")).fit()
    assert np.isfinite(history[0]["train_loss"])


def test_new_config_keys_parse():
    cfg = load_config(None, ["data.augment=heavy", "data.cutmix=0.5", "train.decoder_mask=0.2"])
    assert cfg.data.augment == "heavy" and cfg.data.cutmix == 0.5 and cfg.train.decoder_mask == 0.2
    with pytest.raises(KeyError):
        load_config(None, ["train.decoder_masks=0.2"])
