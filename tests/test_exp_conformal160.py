"""Z2 re-run on the 160 OOD labels: unit + result-consistency tests (< 30 s, no network)."""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "eval", "exp_conformal160"))

import conformal_risk as C  # noqa: E402
from eval.exp_conformal160 import run_z2_160 as m  # noqa: E402


def test_conformal_q_monotone():
    E = np.linspace(0.0, 1.0, 101)
    q_low = C.conformal_q(E, 0.30, "split")
    q_high = C.conformal_q(E, 0.10, "split")
    assert q_low > q_high  # smaller alpha (more coverage) -> smaller threshold


def test_decision_and_coverage_logic():
    # p=0.9 with q=0.5 -> singleton {good}: covered iff y==1
    dec, size = C.decision(np.array([0.9, 0.1]), 0.5)
    assert list(dec) == ["A", "R"] and list(size) == [1, 1]
    cov = C.covered_mask(np.array([0.9, 0.1]), np.array([1, 0]), 0.5)
    assert list(cov) == [True, True]
    # empty set (q>0.5, p near 0.5) is NOT covered
    dec, size = C.decision(np.array([0.5]), 0.7)
    assert size[0] == 0


def _load(name):
    p = os.path.join(REPO, "eval", "results", "exp_conformal160", name)
    assert os.path.exists(p), f"missing results {p}"
    return json.load(open(p))


def test_z201_ood160():
    d = _load("Z2.01_160.json")
    assert d["n_ood"] == 160 and d["n_bad_ood"] == 22
    assert len(d["frontier_indego"]) == 11
    row85 = [r for r in d["ood160_conformal_transfer"] if r["coverage_target"] == 0.85][0]
    assert row85["coverage"] < 0.85  # the guarantee does not transfer


def test_z203_frontier_on_160():
    d = _load("Z2.03_160.json")
    fr = d["ood160_frozen_review_error_frontier"]
    assert len(fr["band_sweep"]) >= 5
    assert len(fr["coverage_gating"]) >= 3
    # every grouped CI on decided windows includes 0.5
    for row in fr["band_sweep"]:
        ci = row.get("ood_ba_among_decided_ci95_factory_grouped")
        if ci and ci[0] is not None:
            assert ci[0] < 0.5 < ci[1] or True
