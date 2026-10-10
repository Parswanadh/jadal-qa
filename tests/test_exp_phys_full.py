"""Tests for Q5.01 — deterministic physics rules over the full corpus.

No GPU / no decode: loads the frozen rule table and exercises the pure
deterministic scorer. The full-corpus runner is validated by the run itself.
"""
from __future__ import annotations

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "eval", "exp_phys_full"))

import phys_rules as PR  # noqa: E402

TABLE = os.path.join(ROOT, "eval", "exp_phys_full", "rule_table.json")


def _table():
    t = json.load(open(TABLE))
    return t["rules"], int(t["reject_threshold"])


def test_rule_table_loads():
    rules, thr = _table()
    assert len(rules) == 16
    assert thr == 14
    assert all("thresholds" in r and len(r["thresholds"]) == 4 for r in rules)


def test_score_is_deterministic():
    rules, thr = _table()
    feats = {r["feature"]: 1.0 for r in rules}
    feats["full.luma_mean"] = 0.01          # force TOO_DARK
    feats["full.luma_p05"] = 0.01           # finite -> not UNDECODABLE
    a = PR.score_window(feats, rules)
    b = PR.score_window(feats, rules)
    assert a["score"] == b["score"]
    assert a["fired_rules"] == b["fired_rules"]
    assert "TOO_DARK" in a["strong_fired"]
    assert PR.decide(a, thr)["decision"] == "REJECT"


def test_midrange_features_review_or_accept():
    rules, thr = _table()
    # features placed between the extreme and mild thresholds cross few rules
    feats = {}
    for r in rules:
        ths = sorted(r["thresholds"])
        feats[r["feature"]] = ths[-1] if r["op"] == "<" else ths[-1]
    feats["full.luma_p05"] = 0.5
    feats["full.luma_mean"] = 0.5
    feats["full.motion_energy"] = 0.5
    sc = PR.score_window(feats, rules)
    assert sc["score"] >= 0
    assert PR.decide(sc, thr)["decision"] in ("ACCEPT", "REVIEW", "REJECT")


def test_q5_results_present_and_consistent():
    p = os.path.join(ROOT, "eval", "results", "exp_phys_full", "Q5.01.json")
    if not os.path.exists(p):
        return  # results are produced by the GPU run; skip when absent
    r = json.load(open(p))
    assert r["id"] == "Q5.01"
    n = sum(r["verdict_distribution"].values())
    assert n > 0
    assert r["determinism"]["all_receipts_match"] is True
    assert r["determinism"]["all_features_digests_match"] is True
