"""Determinism + logic tests for the deterministic physics rule-list model.

Synthetic arrays only; no network, no GPU, no real video. < 30 s.
Run: OMP_NUM_THREADS=2 python3 -m pytest tests/test_exp_phys_rules.py -q
"""
import os
import sys

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "eval", "exp_phys"))

from phys_rules import (CANDIDATE_RULES, QUANTILES, build_rule_table,  # noqa: E402
                        choose_reject_threshold, choose_reject_threshold_ba,
                        decide, derive_thresholds, score_window)


def _synth_train(n=40, seed=0):
    rng = np.random.default_rng(seed)
    feats = {r["feature"]: rng.normal(0, 1, n) for r in CANDIDATE_RULES}
    feats["full.frozen_frac"] = np.zeros(n)
    return feats


def test_thresholds_are_training_quantiles_quantised_4dp():
    v = np.arange(100.0)
    th = derive_thresholds(v, "low")
    assert len(th) == len(QUANTILES)
    # thresholds are sorted ascending for a low-side rule, and quantised to 4 dp
    assert th == sorted(th)
    for t in th:
        assert abs(t - round(t, 4)) < 1e-12
    assert th[0] < th[-1]


def test_high_side_thresholds_are_mirrored():
    v = np.arange(100.0)
    lo = derive_thresholds(v, "low")
    hi = derive_thresholds(v, "high")
    assert hi == sorted(hi, reverse=True)
    assert hi[0] > hi[-1]


def test_score_is_deterministic_and_monotone():
    feats = _synth_train()
    table = build_rule_table(feats)
    w = {r["feature"]: 0.0 for r in CANDIDATE_RULES}
    a = score_window(w, table)
    b = score_window(dict(w), table)
    assert a == b                       # identical input -> identical score/evidence
    # a clearly bad value cannot lower the score
    w2 = dict(w)
    w2["full.luma_mean"] = -100.0
    assert score_window(w2, table)["score"] >= a["score"]


def test_undecodable_tripwire():
    table = build_rule_table(_synth_train())
    w = {r["feature"]: 0.0 for r in CANDIDATE_RULES}
    w.pop("full.luma_mean")
    sc = score_window(w, table)
    assert "UNDECODABLE" in sc["fired_rules"]
    assert sc["score"] > 100


def test_decide_quality_beats_score_and_review_band():
    table = build_rule_table(_synth_train())
    w = {r["feature"]: 0.0 for r in CANDIDATE_RULES}
    w["full.luma_mean"] = -100.0        # crosses p01/p05/p10/p25 -> hard quality fire
    sc = score_window(w, table)
    d = decide(sc, reject_threshold=999)
    assert d["decision"] == "REJECT" and d["reason"] == "quality_rule"
    assert d["evidence"] and set(d["evidence"]) >= {"rule", "feature", "value", "op", "threshold"}


def test_reject_threshold_derivation():
    scores = np.array([0, 0, 1, 1, 2, 3, 10, 20])
    y = np.array([1, 1, 1, 1, 1, 1, 0, 0])
    t = choose_reject_threshold(scores, y, fr_budget=0.05)
    # <=5% of the 6 good windows may have score >= t -> at most 0 windows
    assert (scores[y == 1] >= t).mean() <= 0.05
    assert choose_reject_threshold_ba(scores, y) >= 1


def test_all_decisions_carry_evidence_shape():
    table = build_rule_table(_synth_train())
    for val in (-50.0, -1.0, 0.0, 5.0):
        w = {r["feature"]: val for r in CANDIDATE_RULES}
        sc = score_window(w, table)
        d = decide(sc, reject_threshold=2)
        if d["decision"] != "ACCEPT":
            assert d["evidence"] is not None
            assert set(d["evidence"]) >= {"rule", "feature", "value", "op", "threshold"}
