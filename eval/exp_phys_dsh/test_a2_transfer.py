#!/usr/bin/env python3
"""Unit tests for eval/exp_phys_dsh/a2_transfer.py.

Run:  OMP_NUM_THREADS=1 python3 eval/exp_phys_dsh/test_a2_transfer.py
(or `pytest -q eval/exp_phys_dsh/test_a2_transfer.py`)
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

import a2_transfer as A  # noqa: E402

RULES = A.load_rules()
OOD = A.load_npz(os.path.join(A.INPUTS, "ood160_full.npz"))
IND = A.load_npz(os.path.join(A.INPUTS, "indego_full.npz"))
REF = A.indego_ref_stats(RULES, IND["X"], IND["names"])
TMP = "/tmp/a2_tests"
os.makedirs(TMP, exist_ok=True)


def test_quantise_semantics():
    assert A.quantise(0.0723) == 723
    assert A.quantise(0.00004) == 0
    assert A.quantise(-0.00004) == 0
    assert A.quantise(-0.00005) == -1  # half-up away from zero, as in tools/phys/model.py
    assert A.quantise(float("nan")) is None
    assert A.quantise(None) is None
    assert A.quantise("x") is None


def test_frozen_thresholds_are_indego_quantiles_at_levels():
    """The frozen table must be IndEgo quantiles at p01/p05/p10/p25 (mirrored)."""
    worst = 0.0
    for r in RULES["rules"]:
        v = A.masked(A.col(IND["X"], IND["names"], r["feature"]))
        for j, lev in enumerate(A.LEVELS):
            q = lev if r["op"] == "<" else 1.0 - lev
            want = float(np.quantile(v, q))
            worst = max(worst, abs(want - float(r["thresholds"][j])))
    assert worst < 1e-4, f"frozen thresholds are not the claimed quantiles (max dev {worst})"


def test_raw_matches_shipped_model_py():
    """Our integer scorer must agree with tools/phys/model.py on the IndEgo pool."""
    from tools.phys import model as M
    m_rules = M.load_rules(os.path.join(REPO, "tools", "phys", "rules.json"))
    thr = A.fit_thresholds(RULES, IND["X"], IND["names"], "raw")
    sc = A.score_windows(IND["X"], IND["names"], RULES, thr)
    for i in range(0, len(IND["y"]), 7):
        feats = {f: float(IND["X"][i, j]) for j, f in enumerate(IND["names"])}
        ref = M.score_window(feats, m_rules)
        assert int(sc["score"][i]) == int(ref["score"]), (i, sc["score"][i], ref["score"])
        assert bool(sc["strong"][i]) == bool(ref["strong_fired"]), i


def test_decision_semantics():
    """Thresholds are NESTED quantiles: crossing the extreme crosses all softer ones."""
    rules = {"reject_threshold": 14, "rules": [
        {"rule": "Q", "feature": "f", "op": "<", "thresholds": [0.1, 0.2, 0.3, 0.4],
         "quantiles": [0.01, 0.05, 0.1, 0.25], "strength": "quality"},
        {"rule": "W", "feature": "g", "op": ">", "thresholds": [0.9, 0.8, 0.7, 0.6],
         "quantiles": [0.01, 0.05, 0.1, 0.25], "strength": "weak"},
    ]}
    thr = {("Q", "f", j): A.quantise(t) for j, t in enumerate([0.1, 0.2, 0.3, 0.4])}
    thr.update({("W", "g", j): A.quantise(t) for j, t in enumerate([0.9, 0.8, 0.7, 0.6])})
    names = ["f", "g"]
    X = np.array([[0.05, 0.0],    # Q crosses all 4 -> quality hard -> REJECT
                  [0.5, 0.95],    # W crosses all 4 (weak) -> score 4 -> REVIEW @14
                  [0.5, 0.0]])    # nothing -> ACCEPT
    sc = A.score_windows(X, names, rules, thr)
    assert list(sc["score"]) == [4, 4, 0], sc["score"]
    assert list(sc["strong"]) == [True, False, False], sc["strong"]
    assert list(A.decisions(sc["score"], sc["strong"], 14)) == ["REJECT", "REVIEW", "ACCEPT"]
    # a weak rule crossing its extreme must NOT force reject...
    assert list(A.decisions(sc["score"], sc["strong"], 5)) == ["REJECT", "REVIEW", "ACCEPT"]
    # ...but a score >= reject_threshold must reject
    assert list(A.decisions(sc["score"], sc["strong"], 4)) == ["REJECT", "REJECT", "ACCEPT"]


def test_requantile_reproduces_raw_when_target_is_indego():
    """(b) with the IndEgo pool as target must equal the raw frozen table."""
    thr_raw = A.fit_thresholds(RULES, IND["X"], IND["names"], "raw")
    s_raw = A.score_windows(IND["X"], IND["names"], RULES, thr_raw)["score"]
    r = A.evaluate(IND["X"], IND["y"], IND["names"], RULES, "requantile", "self")
    agree = float((r["score"] == s_raw).mean())
    assert agree >= 0.95, f"requantile should be near-identity at home, agreement={agree}"
    assert abs(A.auroc(IND["y"], r["score"]) - A.auroc(IND["y"], s_raw)) < 0.02


def test_rank_and_quantile_variants_agree_on_ood():
    r_b = A.evaluate(OOD["X"], OOD["y"], OOD["names"], RULES, "requantile", "b")
    r_b2 = A.evaluate(OOD["X"], OOD["y"], OOD["names"], RULES, "requantile_rank", "b2")
    a_b = A.auroc(OOD["y"], r_b["score"])
    a_b2 = A.auroc(OOD["y"], r_b2["score"])
    assert abs(a_b - a_b2) < 0.03, (a_b, a_b2)
    # per-feature crossing agreement on the extreme threshold
    for r in RULES["rules"]:
        f = r["feature"]
        v = np.asarray(A.col(OOD["X"], OOD["names"], f), float)
        ok = np.isfinite(v)
        lev = A.level_of(r, 0)
        thr = r_b["thr"][(r["rule"], f, 0)]
        direct = np.array([(A.quantise(x) < thr) if r["op"] == "<" else (A.quantise(x) > thr)
                           for x in v[ok]])
        u = A.transform_values(OOD["X"], OOD["names"], "requantile_rank")[ok,
                                                                          OOD["names"].index(f)]
        rank = np.array([A.quantise(x) < A.quantise(lev) if r["op"] == "<"
                         else A.quantise(x) > A.quantise(lev) for x in u])
        agree = float((direct == rank).mean())
        assert agree >= 0.9, (f, agree)


def test_label_free_no_y_in_fitting():
    """Permuting y must not change any fitted threshold or score."""
    rng = np.random.default_rng(7)
    perm = rng.permutation(len(OOD["y"]))
    base = {}
    for mode in ("raw", "requantile", "requantile_rank", "zscore", "zscore_literal"):
        r = A.evaluate(OOD["X"], OOD["y"], OOD["names"], RULES, mode, mode, ref=REF)
        base[mode] = (r["score"].copy(), dict(r["thr"]))
    # the same calls, but with y replaced by its permutation: nothing may change
    for mode in base:
        r = A.evaluate(OOD["X"], OOD["y"][perm], OOD["names"], RULES, mode, mode, ref=REF)
        assert np.array_equal(r["score"], base[mode][0]), mode
        assert r["thr"] == base[mode][1], mode
    # regime variant too
    desc = A.load_descriptors(OOD["ids"])
    lab, _ = A.regime_split(OOD["ids"], desc, k=2)
    r1 = A.evaluate(OOD["X"], OOD["y"], OOD["names"], RULES, "requantile", "d", groups=lab)
    r2 = A.evaluate(OOD["X"], OOD["y"][perm], OOD["names"], RULES, "requantile", "d", groups=lab)
    assert np.array_equal(r1["score"], r2["score"])


def test_scale_shift_invariance_of_requantile():
    """(b) is invariant to a per-feature affine rescaling of the target.

    Part 1 checks the exact float identity Q_q(a*x+b) = a*Q_q(x)+b.
    Part 2 checks the integer Q=4 decision path under moderate scales, where the
    4-dp threshold quantisation is fine enough not to break ties.
    """
    rng = np.random.default_rng(3)
    a = np.exp(rng.normal(0, 0.5, len(OOD["names"])))
    b = rng.normal(0, 3.0, len(OOD["names"]))
    for j, f in enumerate(OOD["names"]):
        v = A.masked(A.col(OOD["X"], OOD["names"], f))
        for lev in (0.01, 0.25, 0.99):
            want = a[j] * float(np.quantile(v, lev)) + b[j]
            got = float(np.quantile(v * a[j] + b[j], lev))
            assert abs(got - want) <= 1e-6 * max(1.0, abs(want)), (f, lev, got, want)
    X2 = OOD["X"] * a + b
    r1 = A.evaluate(OOD["X"], OOD["y"], OOD["names"], RULES, "requantile", "orig")
    r2 = A.evaluate(X2, OOD["y"], OOD["names"], RULES, "requantile", "scaled")
    agree = float((r1["score"] == r2["score"]).mean())
    assert agree >= 0.95, f"requantile not scale invariant (agreement {agree})"
    r3 = A.evaluate(OOD["X"], OOD["y"], OOD["names"], RULES, "raw", "raw_orig")
    r4 = A.evaluate(X2, OOD["y"], OOD["names"], RULES, "raw", "raw_scaled")
    assert float((r3["score"] == r4["score"]).mean()) < 0.5  # raw is NOT invariant


def test_calib_slice_is_labelfree_and_close():
    """(e) 32-window calibration must land near the full-target (b)."""
    rows = np.sort(np.random.default_rng(1000).choice(len(OOD["y"]), 32, replace=False))
    r_full = A.evaluate(OOD["X"], OOD["y"], OOD["names"], RULES, "requantile", "full")
    r_cal = A.evaluate(OOD["X"], OOD["y"], OOD["names"], RULES, "requantile", "cal", fit_rows=rows)
    d = abs(A.auroc(OOD["y"], r_full["score"]) - A.auroc(OOD["y"], r_cal["score"]))
    assert d < 0.12, d
    # rows outside the calibration slice must be scored (evaluation is on all 160)
    assert r_cal["score"].shape == (160,)


def test_zscore_literal_is_broken():
    """The naive reading of (c) must be detected as broken (fires on everything)."""
    r = A.evaluate(OOD["X"], OOD["y"], OOD["names"], RULES, "zscore_literal", "c1")
    dec = A.decisions(r["score"], r["strong"], 14)
    assert float((dec == "REJECT").mean()) > 0.5


def test_regime_split_is_deterministic_and_covers_all():
    desc = A.load_descriptors(OOD["ids"])
    assert len(desc) == 160, len(desc)
    lab1, info1 = A.regime_split(OOD["ids"], desc, k=2)
    lab2, info2 = A.regime_split(OOD["ids"], desc, k=2)
    assert np.array_equal(lab1, lab2)
    assert info1["sizes"] == info2["sizes"]
    assert (lab1 >= 0).all()
    assert sum(info1["sizes"].values()) == 160


def test_determinism_two_runs_identical():
    r1 = A.evaluate(OOD["X"], OOD["y"], OOD["names"], RULES, "requantile", "x")
    r2 = A.evaluate(OOD["X"], OOD["y"], OOD["names"], RULES, "requantile", "x")
    assert np.array_equal(r1["score"], r2["score"])
    assert all(isinstance(v, (int, type(None))) for v in r1["thr"].values())
    h1 = json.dumps({f"{k}": v for k, v in r1["thr"].items()}, sort_keys=True)
    h2 = json.dumps({f"{k}": v for k, v in r2["thr"].items()}, sort_keys=True)
    assert h1 == h2


def test_reject_threshold_rule():
    assert A.choose_reject_threshold(np.array([0, 1, 2, 3, 4]), np.array([1, 1, 1, 1, 0]), 0.05) == 4
    assert A.choose_reject_threshold(np.array([0, 5, 5]), np.array([1, 1, 0]), 0.0) == 6


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    bad = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except AssertionError as e:
            bad += 1
            print(f"FAIL {fn.__name__}: {e}")
        except Exception as e:  # noqa: BLE001
            bad += 1
            print(f"ERROR {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(fns) - bad}/{len(fns)} passed")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(_run_all())
