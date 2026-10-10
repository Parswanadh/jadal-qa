"""Tests for the O2 few-shot / E5.04-final / E5.03-refresh helpers (< 30 s, no network)."""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval.exp_fewshot import o2lib as L  # noqa: E402


def test_labels160_counts():
    lab = L.load_labels160()
    assert lab["n"] == 160
    assert int(lab["y"].sum()) == 138
    assert int((1 - lab["y"]).sum()) == 22
    assert len(set(lab["factory"].tolist())) == 47
    assert set(lab["batch"].tolist()) == {"q1", "q2"}


def test_frozen_scores_cover_labels():
    lab = L.load_labels160()
    sc = L.load_frozen_scores()
    assert len(sc) == 160
    missing = [i for i in lab["ids"] if i not in sc]
    assert missing == []
    for v in sc.values():
        assert 0.0 <= v["usable_prob"] <= 1.0


def test_pixel160_shape_and_balance():
    if not os.path.exists(os.path.join(L.INPUTS, "ood160_pixel.npz")):
        return  # inputs are git-ignored; skip when absent
    z = L.load_ood160_pixel()
    assert z["X10"].shape == (160, 26)
    assert list(z["names"]) == L.FEATURE_NAMES
    assert int(z["y"].sum()) == 138


def test_classify_transfer_rule():
    assert L.classify_transfer(0.50, 0.50) == "chance"
    assert L.classify_transfer(0.52, 0.48) == "chance"      # both within tol
    assert L.classify_transfer(0.63, 0.65) == "stable"
    assert L.classify_transfer(0.30, 0.70) == "flip"
    # informative in one domain only -> NOT chance
    assert L.classify_transfer(0.63, 0.60) == "stable"
    assert L.classify_transfer(0.63, 0.40) == "flip"


def test_group_boot_ci_ordered():
    y = np.asarray([1, 1, 0, 1, 0, 0] * 4)
    s = np.asarray([0.9, 0.8, 0.2, 0.7, 0.3, 0.1] * 4)
    g = np.asarray([f"f{i % 6}" for i in range(len(y))])
    m, lo, hi = L.group_boot_ci(y, s, g, n_boot=60, seed=0)
    assert lo <= hi
    assert m >= 0.4


def test_grouped_cv_no_factory_leak():
    rng = np.random.RandomState(0)
    n = 60
    fac = np.asarray([f"f{i % 12}" for i in range(n)])
    X = rng.randn(n, 4)
    y = (X[:, 0] + rng.randn(n) * 0.1 > 0).astype(int)
    oof, folds = L.grouped_cv_oof(X, y, fac, n_splits=4, seed=0)
    assert np.isfinite(oof).all()
    # every fold's test factories are absent from its train factories
    from sklearn.model_selection import StratifiedGroupKFold
    sg = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=0)
    for tr, te in sg.split(X, y, fac):
        assert not (set(fac[tr]) & set(fac[te]))
    assert any(not f.get("skipped") for f in folds)


def test_subsample_curve_grouped():
    rng = np.random.RandomState(1)
    n = 60
    fac = np.asarray([f"f{i % 12}" for i in range(n)])
    X = rng.randn(n, 5)
    y = (X[:, 0] > 0).astype(int)
    curve = L.grouped_subsample_curve(X, y, fac, [20], repeats=3, seed=0)
    assert curve[0]["k_target"] == 20
    assert curve[0]["auroc_mean"] is None or 0.0 <= curve[0]["auroc_mean"] <= 1.0
