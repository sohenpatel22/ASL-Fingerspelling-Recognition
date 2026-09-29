import sys
from pathlib import Path

import numpy as np
import pandas as pd

COLS = ["frame"] + [f"{a}_{h}_hand_{i}" for h in ("left", "right") for a in "xy" for i in range(21)]
WORDS = ["cat", "bus", "tip", "date", "kenya", "papaya", "norway", "japan"]


def _write_split(root: Path, name: str, n_participants: int, per_participant: int, start_id: int, rng):
    landmarks_dir = root / f"{name}_landmarks"
    landmarks_dir.mkdir(parents=True, exist_ok=True)
    meta, frames = [], []
    seq_id = start_id
    for p in range(n_participants):
        for _ in range(per_participant):
            phrase = WORDS[int(rng.integers(len(WORDS)))]
            t = int(rng.integers(12, 30))
            data = rng.uniform(0.1, 0.9, (t, len(COLS)))
            data[:, 0] = np.arange(t)
            frames.append(pd.DataFrame(data, columns=COLS, index=pd.Index([seq_id] * t)))
            meta.append({
                "path": f"{name}_landmarks/1.parquet", "file_id": 1, "sequence_id": seq_id,
                "participant_id": 100 + p, "phrase": phrase,
            })
            seq_id += 1
    pd.concat(frames).to_parquet(landmarks_dir / "1.parquet")
    return pd.DataFrame(meta)


def make_fake_competition(root: str | Path, n_participants: int = 10, per_participant: int = 6) -> Path:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    train = _write_split(root, "train", n_participants, per_participant, 0, rng)
    supp = _write_split(root, "supplemental", 3, per_participant, 10_000, rng)
    train.to_csv(root / "train.csv", index=False)
    supp.to_csv(root / "supplemental_metadata.csv", index=False)
    return root


if __name__ == "__main__":
    out = make_fake_competition(sys.argv[1] if len(sys.argv) > 1 else "data/raw/asl-fingerspelling")
    print(f"wrote fake competition data to {out}")
