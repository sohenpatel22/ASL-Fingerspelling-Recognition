from __future__ import annotations

import numpy as np

N_LANDMARKS = 21
FEATURE_SIZE = 4 * N_LANDMARKS
_HAND_OFFSETS = (0, 2 * N_LANDMARKS)


def wrist_normalize(seq: np.ndarray) -> np.ndarray:
    out = np.array(seq, dtype=np.float32, copy=True)
    n = N_LANDMARKS
    for off in _HAND_OFFSETS:
        lx = out[:, off : off + n]
        ly = out[:, off + n : off + 2 * n]
        wx, wy = lx[:, 0:1], ly[:, 0:1]
        visible = (lx != 0).any(axis=1, keepdims=True)
        lx = np.where(visible, lx - wx, 0.0)
        ly = np.where(visible, ly - wy, 0.0)
        span = max(float(np.abs(lx).max()), float(np.abs(ly).max()), 1e-6)
        out[:, off : off + n] = lx / span
        out[:, off + n : off + 2 * n] = ly / span
    return out.astype(np.float32)


def resample_or_pad(seq: np.ndarray, max_len: int) -> np.ndarray:
    t = len(seq)
    if t > max_len:
        return seq[np.linspace(0, t - 1, max_len, dtype=int)]
    if t < max_len:
        pad = np.zeros((max_len - t, seq.shape[1]), dtype=np.float32)
        return np.vstack([seq, pad])
    return seq


def to_model_input(seq: np.ndarray, max_len: int) -> np.ndarray:
    seq = np.nan_to_num(np.asarray(seq, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    return resample_or_pad(wrist_normalize(seq), max_len).T.astype(np.float32)


def hands_to_row(left_xy: np.ndarray | None, right_xy: np.ndarray | None) -> np.ndarray:
    row = np.zeros(FEATURE_SIZE, dtype=np.float32)
    for off, hand in zip(_HAND_OFFSETS, (left_xy, right_xy), strict=True):
        if hand is not None:
            row[off : off + N_LANDMARKS] = hand[:, 0]
            row[off + N_LANDMARKS : off + 2 * N_LANDMARKS] = hand[:, 1]
    return row


def augment(seq: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    n = len(seq)

    if rng.random() < 0.5 and n > 8:
        keep = np.sort(rng.choice(n, size=int(n * 0.9), replace=False))
        seq = seq[keep]
        n = len(seq)

    if rng.random() < 0.5:
        seq = seq + rng.normal(0.0, 0.01, seq.shape).astype(np.float32)

    if rng.random() < 0.5:
        seq = np.concatenate([seq[:, 42:84], seq[:, 0:42]], axis=1)
        seq[:, 0:21] = -seq[:, 0:21]
        seq[:, 42:63] = -seq[:, 42:63]

    if rng.random() < 0.4:
        t_new = max(4, int(n * rng.uniform(0.8, 1.2)))
        seq = seq[np.linspace(0, n - 1, t_new, dtype=int)]

    return seq.astype(np.float32)
