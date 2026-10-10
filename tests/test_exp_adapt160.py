"""O4 sanity tests: metrics + z-score combo contract (no decode, < 5 s)."""
from __future__ import annotations

import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "eval", "exp_adapt160"))

import run_adapt160 as R  # noqa: E402


def test_fast_auc_perfect_and_ties():
    y = np.array([1, 1, 0, 0])
    assert R.fast_auc(y, np.array([0.9, 0.8, 0.2, 0.1])) == 1.0
    assert R.fast_auc(y, np.array([0.1, 0.2, 0.8, 0.9])) == 0.0
    assert R.fast_auc(y, np.array([0.5, 0.5, 0.5, 0.5])) == 0.5


def test_balacc_threshold():
    y = np.array([1, 1, 0, 0])
    assert R.balacc(y, np.array([1.0, 1.0, -1.0, -1.0]), 0.0) == 1.0
    assert R.balacc(y, np.array([-1.0, -1.0, 1.0, 1.0]), 0.0) == 0.0


def test_combo_signs_and_scale_invariance():
    rng = np.random.RandomState(0)
    yg = np.array([1] * 10 + [0] * 10)
    # feature 0 high for good, feature 1 low for good
    x0 = np.concatenate([rng.normal(1, 1, 10), rng.normal(-1, 1, 10)])
    x1 = -x0 + rng.normal(0, 0.1, 20)
    Xg = np.column_stack([x0, x1])
    # OOD is a shifted, rescaled copy: per-domain z-score must remove it
    Xo = np.column_stack([x0 * 100 + 50, x1 * 0.01 - 3])
    _, so, au, sg = R.combo_scores(Xg, Xo, yg)
    assert sg[0] == 1.0 and sg[1] == -1.0, sg
    assert au[0] > 0.9 and au[1] < 0.1, au
    assert R.fast_auc(yg, so) > 0.95


def test_result_json_contract():
    import json
    p = os.path.join(REPO, "eval", "results", "exp_adapt160", "O4.01.json")
    if not os.path.exists(p):
        return  # result not generated in this checkout
    d = json.load(open(p))
    assert d["n"] == 160 and d["n_good"] == 138 and d["n_bad"] == 22
    names = [b["recipe"] for b in d["blocks"]]
    assert any(n.startswith("(a)") for n in names)
    assert any(n.startswith("(b)") for n in names)
    assert any(n.startswith("(c) Z1") for n in names)
    assert any(n.startswith("(d)") for n in names)
    for b in d["blocks"]:
        assert 0.0 <= b["ood_auroc"] <= 1.0
        ci = b["ood_auroc_ci95_factory_grouped_honest"]
        assert ci[0] <= b["ood_auroc"] <= ci[1]
