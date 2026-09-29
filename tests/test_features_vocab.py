import numpy as np
import pytest

from asl.features import augment, hands_to_row, resample_or_pad, to_model_input, wrist_normalize
from asl.metrics import bootstrap_ci, cer, exact_match, group_cer, mean_cer
from asl.vocab import Vocab


def _reference_wrist_normalize(seq):
    out = seq.copy()
    for offset in [0, 42]:
        lx = out[:, offset : offset + 21]
        ly = out[:, offset + 21 : offset + 42]
        wx, wy = lx[:, 0:1], ly[:, 0:1]
        visible = (lx != 0).any(axis=1, keepdims=True)
        lx = np.where(visible, lx - wx, 0.0)
        ly = np.where(visible, ly - wy, 0.0)
        span = max(float(np.abs(lx).max()), float(np.abs(ly).max()), 1e-6)
        out[:, offset : offset + 21] = lx / span
        out[:, offset + 21 : offset + 42] = ly / span
    return out.astype(np.float32)


def _random_landmarks(t=30, seed=0):
    rng = np.random.default_rng(seed)
    seq = rng.uniform(0.1, 0.9, (t, 84)).astype(np.float32)
    seq[5:10, 42:] = 0.0
    return seq


def test_wrist_normalize_matches_training_implementation():
    seq = _random_landmarks()
    np.testing.assert_allclose(wrist_normalize(seq), _reference_wrist_normalize(seq))


def test_wrist_normalize_properties():
    out = wrist_normalize(_random_landmarks())
    assert out.dtype == np.float32
    assert np.allclose(out[:, 0], 0) and np.allclose(out[:, 21], 0)
    assert np.abs(out).max() <= 1.0 + 1e-6
    assert np.all(out[5:10, 42:] == 0)


def test_wrist_normalize_is_translation_and_input_immutable():
    seq = _random_landmarks()
    before = seq.copy()
    shifted = seq.copy()
    shifted[:, :21] += 0.05
    shifted[:, :21][seq[:, :21] == 0] = 0
    np.testing.assert_array_equal(seq, before)
    np.testing.assert_allclose(
        wrist_normalize(shifted)[:, :42], wrist_normalize(seq)[:, :42], atol=1e-5
    )


@pytest.mark.parametrize("t", [10, 64, 200])
def test_resample_or_pad_length(t):
    out = resample_or_pad(np.ones((t, 84), np.float32), 64)
    assert out.shape == (64, 84)
    if t < 64:
        assert np.all(out[t:] == 0)


def test_to_model_input_handles_nan_and_shape():
    seq = _random_landmarks(40)
    seq[3, 4] = np.nan
    x = to_model_input(seq, 64)
    assert x.shape == (84, 64) and np.isfinite(x).all()


def test_hands_to_row_layout():
    left = np.stack([np.arange(21), np.arange(21) + 100], axis=1).astype(np.float32)
    row = hands_to_row(left, None)
    assert row.shape == (84,)
    np.testing.assert_array_equal(row[:21], left[:, 0])
    np.testing.assert_array_equal(row[21:42], left[:, 1])
    assert np.all(row[42:] == 0)


def test_augment_output_shape_and_dtype():
    rng = np.random.default_rng(0)
    seq = wrist_normalize(_random_landmarks(50))
    for _ in range(50):
        out = augment(seq, rng)
        assert out.shape[1] == 84 and out.dtype == np.float32 and 4 <= len(out) <= 70


def test_default_vocab_matches_competition_map():
    v = Vocab()
    assert v.n_classes == 59 and v.vocab_size == 62
    assert (v.start_idx, v.eos_idx, v.pad_idx) == (59, 60, 61)
    assert v.char_to_idx[" "] == 0 and v.char_to_idx["a"] == 32 and v.char_to_idx["~"] == 58


def test_vocab_roundtrip_and_padding(vocab):
    ids = vocab.encode("hello world", 34)
    assert len(ids) == 34 and ids[0] == vocab.start_idx and ids[12] == vocab.eos_idx
    assert vocab.decode(ids) == "hello world"
    assert vocab.decode(ids[1:]) == "hello world"


def test_vocab_truncates_long_phrases_and_maps_unknown_to_pad(vocab):
    assert len(vocab.encode("a" * 100, 34)) == 34
    assert vocab.encode("A", 6)[1] == vocab.pad_idx


def test_metrics():
    assert cer("cat", "cat") == 0 and cer("cut", "cat") == pytest.approx(1 / 3)
    assert mean_cer(["cat", "xx"], ["cat", ""]) == 0
    assert exact_match(["a", "b"], ["a", "c"]) == 0.5
    groups = group_cer(["cat", "dog"], ["cat", "dot"], [1, 2])
    assert groups == {"1": 0.0, "2": pytest.approx(1 / 3)}
    lo, hi = bootstrap_ci([0.1, 0.2, 0.3, 0.4])
    assert 0.1 <= lo <= 0.25 <= hi <= 0.4


def test_rotate_keeps_radii_and_zeros():
    from asl.features import rotate

    seq = wrist_normalize(_random_landmarks(20))
    out = rotate(seq, np.deg2rad(12))
    for off in (0, 42):
        r_in = np.hypot(seq[:, off : off + 21], seq[:, off + 21 : off + 42])
        r_out = np.hypot(out[:, off : off + 21], out[:, off + 21 : off + 42])
        np.testing.assert_allclose(r_in, r_out, atol=1e-5)
    assert np.all(out[5:10, 42:] == 0)  # missing hand stays missing


def test_stretch_scales_axes_independently():
    from asl.features import stretch

    seq = wrist_normalize(_random_landmarks(10))
    out = stretch(seq, 2.0, 0.5)
    np.testing.assert_allclose(out[:, :21], seq[:, :21] * 2.0)
    np.testing.assert_allclose(out[:, 21:42], seq[:, 21:42] * 0.5)


def test_drop_hand_span_blanks_exactly_one_hand_for_a_short_span():
    from asl.features import drop_hand_span

    seq = np.ones((40, 84), dtype=np.float32)
    for seed in range(20):
        out = drop_hand_span(seq, np.random.default_rng(seed))
        left_blank = (out[:, :42] == 0).all(axis=1)
        right_blank = (out[:, 42:] == 0).all(axis=1)
        assert left_blank.any() != right_blank.any()  # only one hand is touched
        assert 2 <= (left_blank | right_blank).sum() <= 9
        assert out.sum() < seq.sum()


def test_strong_augment_is_valid_and_changes_more_than_basic():
    seq = wrist_normalize(_random_landmarks(50))
    basic_change, strong_change = [], []
    for seed in range(60):
        b = augment(seq, np.random.default_rng(seed), strong=False)
        s = augment(seq, np.random.default_rng(seed), strong=True)
        assert s.shape[1] == 84 and s.dtype == np.float32 and np.isfinite(s).all()
        basic_change.append(np.abs(b[:4] - seq[:4]).mean())
        strong_change.append(np.abs(s[:4] - seq[:4]).mean())
    assert np.mean(strong_change) > np.mean(basic_change)
