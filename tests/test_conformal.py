#!/usr/bin/env python3
"""Unit tests for the Z2 conformal / risk-control mathematics (< 30 s, no network).

Checks the pieces the report relies on:
  * conformal_q gives >= 1-alpha marginal coverage on synthetic exchangeable data;
  * the decision map yields review = non-singleton rate and a *proper* coverage
    count (empty sets are NOT covered);
  * the design-effect quantile is conservative (>= split) when n_eff < n;
  * the cluster Hoeffding bound is valid (>= the window bound) and the window
    bound is the anti-conservative one.
"""
import importlib.util
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SPEC = importlib.util.spec_from_file_location(
    "conformal_risk", os.path.join(HERE, "..", "eval", "exp_conformal", "conformal_risk.py"))
cr = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cr)


def _synth(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)                       # 1 good, 0 bad
    # P(good) informative
    p = np.where(y == 1, rng.beta(5, 2, n), rng.beta(2, 5, n))
    return p, y


def test_conformal_coverage_exchangeable():
    p, y = _synth()
    E = cr.true_label_score(p, y)
    n = len(y)
    n_cal = 2000
    ok = 0
    trials = 40
    rng = np.random.default_rng(1)
    for _ in range(trials):
        idx = rng.permutation(n)
        cal, te = idx[:n_cal], idx[n_cal:]
        q = cr.conformal_q(E[cal], 0.10, "split")   # alpha=0.10 -> 90% coverage
        ok += cr.covered_mask(p[te], y[te], q).mean()
    cov = ok / trials
    assert cov >= 0.88, f"coverage too low: {cov:.3f}"   # >= 0.90 up to MC error
    assert cov <= 0.99, f"coverage implausibly high: {cov:.3f}"


def test_decision_review_is_nonsingleton():
    p = np.array([0.95, 0.5, 0.05])
    dec, size = cr.decision(p, 0.4)     # q=0.4 <= 0.5 -> both-set band
    assert size[1] == 2 and dec[1] == "V"
    assert dec[0] == "A" and size[0] == 1
    assert dec[2] == "R" and size[2] == 1


def test_empty_set_not_covered():
    # q=0.7 > 0.5: a window with p=0.5 -> empty set, must NOT be covered
    p = np.array([0.5])
    y = np.array([1])
    q = 0.7
    dec, size = cr.decision(p, q)
    assert size[0] == 0 and dec[0] == "V"
    assert not cr.covered_mask(p, y, q)[0]


def test_design_effect_conservative():
    rng = np.random.default_rng(3)
    E = rng.random(150)
    E_sorted = np.sort(E)
    q_split = cr.conformal_q(E, 0.10, "split")
    q_deff = cr.conformal_q(E, 0.10, "deff", n_eff=14.0)
    assert q_deff >= q_split        # fewer effective samples -> higher threshold


def test_cluster_bound_ge_window_bound():
    rng = np.random.default_rng(4)
    clip = np.repeat([f"c{i}" for i in range(7)], 21)
    loss = (rng.random(len(clip)) < 0.05).astype(float)
    u = cr.all_ucbs(loss, clip, 0.1)
    assert u["cluster_hoeffding"] >= u["window_hoeffding"] - 1e-9
    assert 0.0 <= u["window_bentkus"] <= 1.0


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print("ok", fn.__name__)
    print(f"{len(fns)} tests passed")
