import numpy as np
import pytest

from asl.gating import auroc, catastrophic, coverage_curve, summarize_gating


def test_auroc_extremes_ties_and_degenerate_input():
    assert auroc([0.9, 0.8, 0.2, 0.1], [True, True, False, False]) == 1.0
    assert auroc([0.1, 0.2, 0.8, 0.9], [True, True, False, False]) == 0.0
    assert auroc([0.5, 0.5], [True, False]) == 0.5
    assert np.isnan(auroc([0.1, 0.2], [True, True]))


def _records(n=200, seed=0):
    rng = np.random.default_rng(seed)
    hand = rng.uniform(0.1, 1.0, n)
    # clips with few detected hands fail more often, and confidence is only weakly informative
    cer = np.where(hand < 0.35, rng.uniform(0.9, 1.0, n), rng.uniform(0.0, 0.3, n))
    conf = rng.uniform(0, 1, n) * 0.3 + hand * 0.1
    return [
        {"cer": float(c), "hand_rate": float(h), "confidence": float(f)}
        for c, h, f in zip(cer, hand, conf, strict=True)
    ]


def test_coverage_curve_error_grows_as_more_clips_are_accepted():
    curve = coverage_curve(_records(), "hand_rate")
    cers = [p["cer"] for p in curve]
    assert curve[0]["coverage"] < curve[-1]["coverage"] == pytest.approx(1.0)
    assert cers[0] < 0.5 * cers[-1]
    assert curve[0]["threshold"] >= curve[-1]["threshold"]  # keeps the highest scores first


def test_summary_finds_hand_rate_more_informative_than_noisy_confidence():
    summary = summarize_gating(_records())
    assert summary["auroc"]["hand_rate"] > 0.95 > summary["auroc"]["confidence"]
    assert 0.2 < summary["catastrophic_share"] < 0.5 and summary["n"] == 200
    assert len(catastrophic([{"cer": 0.95}, {"cer": 0.2}, {"cer": None}])) == 3
