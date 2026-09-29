from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path

import numpy as np

from asl.features import N_LANDMARKS, wrist_normalize

FEATURES = ("n_frames", "detection_rate", "left_rate", "right_rate", "motion")
N_BINS = 10
MIN_SAMPLES = 30


def landmark_stats(raw: np.ndarray) -> dict[str, float]:
    # computed after wrist normalization so it matches what the reference (stored npy) contains
    seq = wrist_normalize(np.nan_to_num(np.asarray(raw, dtype=np.float32)))
    left = (seq[:, : 2 * N_LANDMARKS] != 0).any(axis=1)
    right = (seq[:, 2 * N_LANDMARKS :] != 0).any(axis=1)
    motion = float(np.abs(np.diff(seq, axis=0)).mean()) if len(seq) > 1 else 0.0
    return {
        "n_frames": float(len(seq)),
        "detection_rate": float((left | right).mean()),
        "left_rate": float(left.mean()),
        "right_rate": float(right.mean()),
        "motion": motion,
    }


def _bin_probs(values, edges: np.ndarray) -> np.ndarray:
    # open-ended bins: everything below edges[0] and above edges[-1] gets its own bin
    idx = np.searchsorted(edges, np.asarray(values), side="right")
    counts = np.bincount(idx, minlength=len(edges) + 1)
    return counts / counts.sum()


def build_reference(stats: list[dict[str, float]]) -> dict[str, dict[str, list[float]]]:
    ref = {}
    for name in FEATURES:
        values = np.array([s[name] for s in stats])
        edges = np.unique(np.quantile(values, np.linspace(0, 1, N_BINS + 1)[1:-1]))
        ref[name] = {"edges": edges.tolist(), "probs": _bin_probs(values, edges).tolist()}
    return ref


def psi(expected: np.ndarray, actual: np.ndarray, eps: float = 1e-4) -> float:
    e, a = np.clip(expected, eps, None), np.clip(actual, eps, None)
    return float(np.sum((a - e) * np.log(a / e)))


class DriftMonitor:
    # rule of thumb for PSI: < 0.1 stable, 0.1-0.25 moderate shift, > 0.25 major shift
    def __init__(self, reference: dict, window: int = 200):
        self.reference = reference
        self.window: deque[dict[str, float]] = deque(maxlen=window)

    def observe(self, stats: dict[str, float]) -> None:
        self.window.append(stats)

    def scores(self) -> dict[str, float]:
        if len(self.window) < MIN_SAMPLES:
            return {}
        out = {}
        for name in FEATURES:
            ref = self.reference[name]
            actual = _bin_probs([s[name] for s in self.window], np.array(ref["edges"]))
            out[name] = psi(np.array(ref["probs"]), actual)
        return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="build the drift reference from preprocessed npy files")
    parser.add_argument("--npy-dir", required=True)
    parser.add_argument("--out", default="monitoring/reference.json")
    parser.add_argument("--n", type=int, default=5000)
    args = parser.parse_args(argv)

    files = sorted(Path(args.npy_dir).glob("*.npy"))
    rng = np.random.default_rng(0)
    if len(files) > args.n:
        files = [files[i] for i in rng.choice(len(files), args.n, replace=False)]
    stats = [landmark_stats(np.load(f)) for f in files]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"n": len(stats), "features": build_reference(stats)}))
    print(f"reference from {len(stats)} sequences -> {args.out}")


if __name__ == "__main__":
    main()
