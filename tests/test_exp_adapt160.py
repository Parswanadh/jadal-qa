"""O4 sanity tests: metrics + z-score combo contract (no decode, < 5 s)."""
from __future__ import annotations

import json
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


def test_group_boot_ci_counts_single_class_skips():
    # single-class groups: a resample that draws only one class is skipped and counted
    y = np.array([1, 1, 0, 0])
    s = np.array([0.9, 0.8, 0.2, 0.1])
    groups = np.array(["a", "a", "b", "b"])
    R.reset_boot_stats()
    R.group_boot_ci(y, s, groups, R.fast_auc, n_boot=200, seed=0)
    st = R.BOOT_STATS["group_boot"]
    assert st["used"] + st["skipped_single_class"] == 200
    assert st["skipped_single_class"] > 0


def test_sign_rule_uses_indego_only():
    # signs come from the IndEgo labels, never from the OOD labels/matrix
    rng = np.random.RandomState(1)
    yg = np.array([1] * 12 + [0] * 12)
    x0 = np.concatenate([rng.normal(1, 1, 12), rng.normal(-1, 1, 12)])
    x1 = -x0 + rng.normal(0, 0.1, 24)
    Xg = np.column_stack([x0, x1])
    Xo_a = np.column_stack([x0 * 5 + 1, x1 * 2])
    Xo_b = np.column_stack([-x0, -x1 + 7])  # different OOD, same expected signs
    _, _, au, sg = R.combo_scores(Xg, Xo_a, yg)
    _, _, au2, sg2 = R.combo_scores(Xg, Xo_b, yg)
    assert (sg == sg2).all()
    assert (sg == np.where(au >= 0.5, 1.0, -1.0)).all()
    assert (au == au2).all()


def test_paired_boot_reports_n_and_skips():
    rng = np.random.RandomState(2)
    y = np.array([1] * 10 + [0] * 10)
    sa = rng.normal(1, 1, 20) + y
    sb = rng.normal(0, 1, 20)
    groups = np.array(["g%d" % (i % 5) for i in range(20)])
    R.reset_boot_stats()
    out = R.paired_group_boot(y, sa, sb, groups, n_boot=300, seed=0)
    assert out["n_boot"] == 300
    assert "ci95" in out and "frac_resamples_gt_0" in out
    assert R.BOOT_STATS["paired_group_boot"]["used"] == 300
    assert R.BOOT_STATS["paired_group_boot"]["skipped_single_class"] == 0


def test_frozen_recipe_sha256_matches_sidecar():
    import hashlib
    obj = json.load(open(R.RECIPE_JSON))
    h = hashlib.sha256(json.dumps(obj, sort_keys=True,
                                  separators=(",", ":")).encode()).hexdigest()
    assert R.recipe_sha256() == h
    if os.path.exists(R.RECIPE_SHA):
        assert open(R.RECIPE_SHA).read().split()[0] == h


def test_result_json_contract():
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
