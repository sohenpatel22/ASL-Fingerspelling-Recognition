from __future__ import annotations

import os
from collections.abc import Sequence

import numpy as np

# below this share of frames with a detected hand the model tends to make text up
DEFAULT_MIN_HAND_RATE = 0.30


def min_hand_rate() -> float:
    return float(os.environ.get("ASL_MIN_HAND_RATE", DEFAULT_MIN_HAND_RATE))


def low_visibility(rate: float | None, threshold: float | None = None) -> bool:
    return rate is not None and rate < (min_hand_rate() if threshold is None else threshold)


def auroc(scores: Sequence[float], positive: Sequence[bool]) -> float:
    # probability that a random positive gets a higher score than a random negative (ties count half)
    s, y = np.asarray(scores, dtype=float), np.asarray(positive, dtype=bool)
    pos, neg = s[y], s[~y]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    wins = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    return float(wins / (len(pos) * len(neg)))


def coverage_curve(
    records: Sequence[dict], key: str, keep_high: bool = True, points: int = 11
) -> list[dict[str, float]]:
    # keep the clips the score trusts most and report the error on them, for a range of coverages
    rows = [r for r in records if r.get(key) is not None and r.get("cer") is not None]
    rows.sort(key=lambda r: r[key], reverse=keep_high)
    cers = np.array([r["cer"] for r in rows])
    curve = []
    for frac in np.linspace(1.0 / points, 1.0, points):
        k = max(1, int(round(frac * len(rows))))
        curve.append({
            "coverage": float(k / len(rows)),
            "cer": float(cers[:k].mean()),
            "threshold": float(rows[k - 1][key]),
        })
    return curve


def catastrophic(records: Sequence[dict], cutoff: float = 0.9) -> list[bool]:
    return [bool(r["cer"] is not None and r["cer"] >= cutoff) for r in records]


def summarize_gating(records: Sequence[dict], cutoff: float = 0.9) -> dict:
    usable = [r for r in records if r.get("cer") is not None]
    bad = catastrophic(usable, cutoff)
    out = {"n": len(usable), "catastrophic_share": float(np.mean(bad)), "auroc": {}, "curves": {}}
    for key in ("confidence", "hand_rate"):
        rows = [(r[key], b) for r, b in zip(usable, bad, strict=True) if r.get(key) is not None]
        if not rows:
            continue
        # a low score should flag a failure, so score the *negated* value against the failure label
        out["auroc"][key] = auroc([-v for v, _ in rows], [b for _, b in rows])
        out["curves"][key] = coverage_curve(usable, key)
    return out
