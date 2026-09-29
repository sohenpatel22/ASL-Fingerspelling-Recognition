from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

import editdistance
import numpy as np


def cer(pred: str, target: str) -> float:
    return editdistance.eval(pred, target) / len(target)


def mean_cer(preds: Sequence[str], targets: Sequence[str]) -> float:
    vals = [cer(p, t) for p, t in zip(preds, targets, strict=True) if len(t) > 0]
    return float(np.mean(vals)) if vals else float("nan")


def exact_match(preds: Sequence[str], targets: Sequence[str]) -> float:
    if not targets:
        return float("nan")
    return float(np.mean([p == t for p, t in zip(preds, targets, strict=True)]))


def group_cer(preds: Sequence[str], targets: Sequence[str], groups: Sequence) -> dict[str, float]:
    buckets: dict[str, list[float]] = defaultdict(list)
    for p, t, g in zip(preds, targets, groups, strict=True):
        if len(t) > 0:
            buckets[str(g)].append(cer(p, t))
    return {g: float(np.mean(v)) for g, v in buckets.items()}


def bootstrap_ci(
    values: Sequence[float], n_boot: int = 1000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = rng.choice(arr, size=(n_boot, arr.size), replace=True).mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))
