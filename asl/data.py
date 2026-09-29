from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import ConcatDataset, Dataset

from asl.config import Config, DataConfig, ModelConfig
from asl.features import augment, fit_length, wrist_normalize
from asl.vocab import Vocab

_LH_X = [f"x_left_hand_{i}" for i in range(21)]
_LH_Y = [f"y_left_hand_{i}" for i in range(21)]
_RH_X = [f"x_right_hand_{i}" for i in range(21)]
_RH_Y = [f"y_right_hand_{i}" for i in range(21)]
_LOAD_COLS = ["frame", *_LH_X, *_LH_Y, *_RH_X, *_RH_Y]


def participant_split(
    meta: pd.DataFrame, val_fraction: float = 0.15, test_fraction: float = 0.15, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    participants = meta["participant_id"].unique()
    np.random.RandomState(seed).shuffle(participants)
    n = len(participants)
    n_train = int(n * (1.0 - val_fraction - test_fraction))
    n_val_end = int(n * (1.0 - test_fraction))
    parts = (
        set(participants[:n_train]),
        set(participants[n_train:n_val_end]),
        set(participants[n_val_end:]),
    )
    return tuple(  # type: ignore[return-value]
        meta[meta["participant_id"].isin(p)].reset_index(drop=True) for p in parts
    )


def participant_folds(meta: pd.DataFrame, k: int, seed: int = 42) -> list[set]:
    participants = meta["participant_id"].unique()
    np.random.RandomState(seed).shuffle(participants)
    return [set(chunk) for chunk in np.array_split(participants, k)]


def fold_split(
    meta: pd.DataFrame, k: int, fold: int, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # test = this fold, validation = the next one, train = everything else (all by signer)
    folds = participant_folds(meta, k, seed)
    test_p, val_p = folds[fold % k], folds[(fold + 1) % k]
    pick = lambda mask: meta[mask].reset_index(drop=True)  # noqa: E731
    in_test, in_val = meta["participant_id"].isin(test_p), meta["participant_id"].isin(val_p)
    return pick(~in_test & ~in_val), pick(in_val), pick(in_test)


class ASLDataset(Dataset):

    def __init__(
        self,
        df: pd.DataFrame,
        npy_dir: str | Path,
        vocab: Vocab,
        model_cfg: ModelConfig,
        augment_data: bool = False,
        strong_augment: bool = False,
    ):
        self.npy_dir = Path(npy_dir)
        # a few sequences in the metadata never made it out of the parquet files; drop them up front
        # instead of failing halfway through an epoch
        on_disk = {int(f.stem) for f in self.npy_dir.glob("*.npy")}
        keep = df["sequence_id"].astype(int).isin(on_disk)
        if not keep.all():
            print(f"{self.npy_dir.name}: skipping {int((~keep).sum())} sequences with no .npy file")
        self.df = df[keep].reset_index(drop=True)
        self.vocab = vocab
        self.cfg = model_cfg
        self.augment = augment_data
        self.strong = strong_augment
        self._rng: tuple[int, np.random.Generator] | None = None

    def __len__(self) -> int:
        return len(self.df)

    def _get_rng(self) -> np.random.Generator:
        # separate rng per worker process
        pid = os.getpid()
        if self._rng is None or self._rng[0] != pid:
            self._rng = (pid, np.random.default_rng(torch.initial_seed() % (2**32) + pid))
        return self._rng[1]

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[i]
        seq = np.load(self.npy_dir / f"{row['sequence_id']}.npy").astype(np.float32)
        if self.augment:
            seq = augment(seq, self._get_rng(), self.strong)
        seq = fit_length(seq, self.cfg.max_seq_len, self.cfg.velocity)
        x = torch.from_numpy(np.ascontiguousarray(seq.T)).float()
        y = torch.tensor(
            self.vocab.encode(str(row["phrase"]), self.cfg.max_phrase_len), dtype=torch.long
        )
        return x, y


class SyntheticDataset(Dataset):

    ALPHABET = "abcdefghijklmnopqrstuvwxyz"

    def __init__(self, n: int, vocab: Vocab, model_cfg: ModelConfig, seed: int = 0):
        self.n, self.vocab, self.cfg, self.seed = n, vocab, model_cfg, seed
        proto_rng = np.random.default_rng(1234)
        self.protos = proto_rng.normal(0.0, 0.5, (len(self.ALPHABET), model_cfg.feature_size))
        self.df = pd.DataFrame(
            {"sequence_id": np.arange(n), "participant_id": np.arange(n) % 5, "phrase": ""}
        )
        self.df["phrase"] = [self._phrase(i) for i in range(n)]

    def _phrase(self, i: int) -> str:
        rng = np.random.default_rng(self.seed * 100_003 + i)
        length = int(rng.integers(2, 6))
        return "".join(rng.choice(list(self.ALPHABET), size=length))

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        phrase = self.df.loc[i, "phrase"]
        rng = np.random.default_rng(self.seed * 100_003 + i + 7)
        frames = [
            self.protos[self.ALPHABET.index(c)] + rng.normal(0, 0.05, self.cfg.feature_size)
            for c in phrase
            for _ in range(4)
        ]
        seq = fit_length(np.asarray(frames, dtype=np.float32), self.cfg.max_seq_len, self.cfg.velocity)
        y = torch.tensor(self.vocab.encode(phrase, self.cfg.max_phrase_len), dtype=torch.long)
        return torch.from_numpy(np.ascontiguousarray(seq.T)).float(), y


def build_datasets(cfg: Config, vocab: Vocab) -> dict[str, Dataset]:
    d, m = cfg.data, cfg.model
    if d.synthetic:
        n = d.synthetic_size
        return {
            "train": SyntheticDataset(n, vocab, m, seed=1),
            "val": SyntheticDataset(max(n // 4, 8), vocab, m, seed=2),
            "test": SyntheticDataset(max(n // 4, 8), vocab, m, seed=3),
        }

    comp = Path(d.comp_dir)
    train_meta = pd.read_csv(comp / "train.csv")
    if d.cv_folds > 0:
        tr, va, te = fold_split(train_meta, d.cv_folds, d.cv_fold, d.split_seed)
    else:
        tr, va, te = participant_split(train_meta, d.val_fraction, d.test_fraction, d.split_seed)
    aug = {"augment_data": True, "strong_augment": d.augment == "strong"}
    train_parts: list[Dataset] = [ASLDataset(tr, d.npy_train, vocab, m, **aug)]
    if d.use_supplemental:
        supp = pd.read_csv(comp / "supplemental_metadata.csv")
        train_parts.append(ASLDataset(supp, d.npy_supp, vocab, m, **aug))
    return {
        "train": ConcatDataset(train_parts) if len(train_parts) > 1 else train_parts[0],
        "val": ASLDataset(va, d.npy_train, vocab, m),
        "test": ASLDataset(te, d.npy_train, vocab, m),
    }


def preprocess_parquets(meta: pd.DataFrame, base_dir: str | Path, out_dir: str | Path) -> int:
    from tqdm import tqdm

    base_dir, out_dir = Path(base_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for file_id, group in tqdm(meta.groupby("file_id"), desc=f"-> {out_dir.name}"):
        path = base_dir / f"{file_id}.parquet"
        if not path.exists():
            continue
        frame = pd.read_parquet(path, columns=_LOAD_COLS)
        by_seq = frame.groupby(frame.index)
        for seq_id in group["sequence_id"]:
            if seq_id not in by_seq.groups:
                continue
            rows = by_seq.get_group(seq_id).sort_values("frame")
            seq = np.zeros((len(rows), 84), dtype=np.float32)
            seq[:, 0:21] = rows[_LH_X].to_numpy(np.float32)
            seq[:, 21:42] = rows[_LH_Y].to_numpy(np.float32)
            seq[:, 42:63] = rows[_RH_X].to_numpy(np.float32)
            seq[:, 63:84] = rows[_RH_Y].to_numpy(np.float32)
            seq = np.nan_to_num(seq, nan=0.0, posinf=0.0, neginf=0.0)
            np.save(out_dir / f"{seq_id}.npy", wrist_normalize(seq))
            written += 1
    return written


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Preprocess competition parquet -> npy")
    parser.add_argument("--comp-dir", default=DataConfig.comp_dir)
    parser.add_argument("--out-train", default=DataConfig.npy_train)
    parser.add_argument("--out-supp", default=DataConfig.npy_supp)
    parser.add_argument("--skip-supplemental", action="store_true")
    args = parser.parse_args(argv)

    comp = Path(args.comp_dir)
    n = preprocess_parquets(pd.read_csv(comp / "train.csv"), comp / "train_landmarks", args.out_train)
    print(f"train: wrote {n} sequences to {args.out_train}")
    if not args.skip_supplemental:
        supp = pd.read_csv(comp / "supplemental_metadata.csv")
        n = preprocess_parquets(supp, comp / "supplemental_landmarks", args.out_supp)
        print(f"supplemental: wrote {n} sequences to {args.out_supp}")


if __name__ == "__main__":
    main()
