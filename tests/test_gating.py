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


def test_low_visibility_uses_the_default_and_an_env_override(monkeypatch):
    from asl.gating import DEFAULT_MIN_HAND_RATE, low_visibility

    monkeypatch.delenv("ASL_MIN_HAND_RATE", raising=False)
    assert low_visibility(DEFAULT_MIN_HAND_RATE - 0.01) and not low_visibility(DEFAULT_MIN_HAND_RATE + 0.01)
    assert not low_visibility(None)
    monkeypatch.setenv("ASL_MIN_HAND_RATE", "0.6")
    assert low_visibility(0.5) and not low_visibility(0.7)
    assert low_visibility(0.5, threshold=0.4) is False


def test_app_warns_when_hands_are_rarely_detected(monkeypatch):
    pytest.importorskip("gradio")
    import sys
    from types import SimpleNamespace

    sys.path.insert(0, "app")
    try:
        import app as asl_app
    finally:
        sys.path.remove("app")

    for rate, confidence, expected in (
        (0.1, 0.9, "your hands were only detected"),
        (0.9, 0.5, "not confident"),
        (0.9, 0.9, None),
    ):
        fake = SimpleNamespace(text="hello", confidence=confidence, n_frames=40)
        monkeypatch.setattr(asl_app, "predict", lambda landmarks, f=fake: f)
        monkeypatch.setattr(asl_app, "extract_landmarks", lambda path, r=rate: (np.zeros((40, 84)), r))
        summary, _ = asl_app.translate("clip.mp4", "None", False)
        assert "hello" in summary
        assert (expected in summary) if expected else not summary.startswith("Warning")


def _flag_records(seed, n=400):
    rng = np.random.default_rng(seed)
    hand = rng.uniform(0.05, 1.0, n)
    conf = np.clip(hand * 0.6 + rng.uniform(0.0, 0.5, n), 0, 1)
    failed = (hand < 0.3) | (conf < 0.4)
    cer = np.where(failed, rng.uniform(0.9, 1.0, n), rng.uniform(0.0, 0.2, n))
    return [
        {"cer": float(c), "confidence": float(f), "hand_rate": float(h)}
        for c, f, h in zip(cer, conf, hand, strict=True)
    ]


def test_flag_rule_is_fitted_on_one_set_and_helps_on_another():
    from asl.gating import apply_flag_rule, pick_flag_rule

    rule = pick_flag_rule(_flag_records(0), max_flagged=0.4)
    assert 0 < rule["confidence_min"] <= 0.95 and rule["val_flagged_share"] <= 0.4
    held_out = apply_flag_rule(_flag_records(1), rule)
    assert held_out["accepted_cer"] < 0.5 * held_out["all_cer"]
    assert held_out["caught_catastrophic"] > 0.8
    with pytest.raises(ValueError):
        pick_flag_rule(_flag_records(0), max_flagged=0.0)


def test_flag_reason_prefers_hand_visibility_and_follows_env_overrides(monkeypatch):
    from asl.gating import DEFAULT_MIN_CONFIDENCE, flag_reason

    monkeypatch.delenv("ASL_MIN_HAND_RATE", raising=False)
    monkeypatch.delenv("ASL_MIN_CONFIDENCE", raising=False)
    assert flag_reason(0.99, 0.1) == "low_hand_visibility"
    assert flag_reason(0.1, 0.1) == "low_hand_visibility"  # hands first: low confidence is a consequence
    assert flag_reason(DEFAULT_MIN_CONFIDENCE - 0.01, 0.9) == "low_confidence"
    assert flag_reason(DEFAULT_MIN_CONFIDENCE + 0.01, 0.9) is None
    assert flag_reason(0.1, None) == "low_confidence"  # unknown hand rate (landmark input) is not an error
    monkeypatch.setenv("ASL_MIN_CONFIDENCE", "0.3")
    assert flag_reason(0.5, 0.9) is None
