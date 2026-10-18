#!/usr/bin/env python3
"""Unit tests for the AUTO stream helpers and audit arithmetic (< 30 s, no network).

Run:  OMP_NUM_THREADS=2 /home/parshu/envs/jadal/bin/python tests/test_dsh_auto.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "tools", "dsh"))

import auto_util as A  # noqa: E402
import common as C  # noqa: E402


def test_rank01_monotone_and_ties():
    r = A.rank01([10.0, 20.0, 30.0])
    assert np.allclose(r, [0.0, 0.5, 1.0]), r
    r = A.rank01([5.0, 5.0, 9.0])
    # the two tied values share the mean rank
    assert r[0] == r[1] and r[2] == 1.0 and 0 < r[0] < 1.0, r
    ra = A.rank_average([1.0, 2.0, 3.0], [3.0, 2.0, 1.0])
    assert np.allclose(ra, [0.5, 0.5, 0.5]), ra


def test_policy_rates_hand_example():
    y = np.array([1, 1, 0, 0])
    pred = np.array([1, 0, 1, 0])
    r = A.policy_rates(y, pred)
    assert r["false_accept_rate"] == 0.5
    assert r["false_reject_rate"] == 0.5
    assert r["bal_acc"] == 0.5
    assert r["accept_share"] == 0.5


def test_group_boot_stat_brackets_the_point():
    rng = np.random.RandomState(0)
    y = np.array([1] * 30 + [0] * 30)
    groups = np.array(["a"] * 20 + ["b"] * 20 + ["c"] * 20)
    s = rng.rand(60) + y * 0.3
    out = A.group_boot_stat(y, groups, lambda d, idx: C.fast_auc(d, s[idx]), n_boot=200)
    assert out["lo"] <= C.fast_auc(y, s) <= out["hi"]
    assert out["n_boot"] > 0


def test_paired_group_boot_diff_identical_predictors_is_null():
    rng = np.random.RandomState(1)
    y = np.array([1] * 40 + [0] * 40)
    groups = np.repeat(list("abcdefgh"), 10)
    s = rng.rand(80) + y * 0.2
    stat = lambda d, idx: C.fast_auc(d, s[idx])  # noqa: E731
    out = A.paired_group_boot_diff(y, groups, stat, stat, n_boot=200)
    assert out["delta"] == 0.0
    assert not out["excludes_zero"]


def test_cost_threshold_matches_the_bayes_rule_on_calibrated_scores():
    """Score IS the true posterior: p ~ U(0,1), y ~ Bernoulli(p).  Then the cost
    minimiser sits near the analytic Bayes threshold c_fa/(c_fa+c_fr)."""
    rng = np.random.RandomState(2)
    n = 20000
    p = rng.rand(n)
    y = (rng.rand(n) < p).astype(int)
    for c_fa, c_fr in ((3.0, 1.0), (1.0, 1.0), (1.0, 3.0)):
        t, _ = A.cost_threshold(y, p, c_fa=c_fa, c_fr=c_fr)
        want = c_fa / (c_fa + c_fr)
        assert abs(t - want) < 0.05, (c_fa, c_fr, t, want)
    # and the asymmetric cost is genuinely lower at the tuned threshold than at 0.5
    t3, _ = A.cost_threshold(y, p, c_fa=3.0, c_fr=1.0)
    pred3 = (p >= t3).astype(int)
    c3 = 3.0 * ((pred3 == 1) & (y == 0)).sum() + 1.0 * ((pred3 == 0) & (y == 1)).sum()
    pred5 = (p >= 0.5).astype(int)
    c5 = 3.0 * ((pred5 == 1) & (y == 0)).sum() + 1.0 * ((pred5 == 0) & (y == 1)).sum()
    assert c3 < c5, (c3, c5)


def test_crc_picks_on_monotone_synthetic_calibration():
    rng = np.random.RandomState(3)
    n = 500
    # confidence uniform in [0,1]; error probability rises with confidence
    conf = rng.rand(n)
    err = rng.rand(n) < (0.05 + 0.5 * conf)
    y = np.where(err, 0, 1)
    p_used = np.where(y == 1, 0.9, 0.1)
    thr = np.full(n, 0.5)
    from auto03 import crc_picks  # noqa: E402  (import after path setup)
    out = crc_picks(conf, p_used, thr, y, 0.20)
    assert out["crc_published"] is not None
    assert out["crc_monotone_envelope"] is not None
    assert out["crc_published"]["risk_bound"] <= 0.20
    # the published pick maximises coverage among the passing candidates
    assert out["crc_published"]["coverage_cal"] >= out["crc_monotone_envelope"]["coverage_cal"]


def test_audit_arithmetic_units():
    # 10 s windows -> 360 per video hour; 10 000 video hours -> 3.6 M windows
    w = 360.0 * 10000.0
    r, a, t = 0.48, 0.02, 30.0
    hours = (r + a) * w * t / 3600.0
    assert abs(hours - 15000.0) < 1e-6, hours
    # per-10 000-window basis is 3.6 M / 10 000 = 360 windows out of every 10 000
    e_out = 0.192
    escaped_per_10k = e_out * (1 - r) * w / (w / 10000.0)
    assert abs(escaped_per_10k - 998.4) < 1.0, escaped_per_10k


def test_wilson_upper_is_above_the_point():
    u = A.wilson_upper(3, 100)
    assert 0.03 < u < 0.12, u
    assert A.wilson_upper(0, 0) != A.wilson_upper(0, 0)  # nan for n=0


def test_every_auto_results_json_has_the_required_schema():
    """No number reaches the report without its provenance block."""
    import json
    for eid in ("AUTO.00", "AUTO.01", "AUTO.03", "AUTO.04", "AUTO.07"):
        fp = os.path.join(REPO, "eval", "results", "auto", f"{eid}.json")
        assert os.path.exists(fp), fp
        d = json.load(open(fp))
        missing = [k for k in C.REQUIRED if k not in d]
        assert not missing, (eid, missing)
        assert d["id"] == eid
        assert isinstance(d["ci95"], list) and len(d["ci95"]) == 2, (eid, d["ci95"])
        assert d["seed"] == 0
        assert len(d["config_sha256"]) == 64
        assert d["git_sha"] and d["device"] and d["cost"] and d["command"]


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in tests:
        fn()
        print("ok", fn.__name__)
    print(f"{len(tests)} tests passed")


if __name__ == "__main__":
    main()
