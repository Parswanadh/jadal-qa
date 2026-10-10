"""Tests for the deterministic v2 physics rule model (tools/phys/model_v2.py).

Fast (< 5 s), no network, no data files: everything is built in-process.
"""
from __future__ import annotations

import json
import os
import sys
import unittest

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     "..", "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from tools.phys import model_v2 as M  # noqa: E402


def toy_table():
    return {
        "name": "toy",
        "t_accept": 1,
        "t_reject": 3,
        "rules": [
            {"rule": "BLUR", "feature": "f.low", "op": "<", "side": "low",
             "strength": "quality", "thresholds_q": [100, 500, 1000, 2500],
             "weight": 1},
            {"rule": "TOO_DARK", "feature": "f.high", "op": ">", "side": "high",
             "strength": "weak", "thresholds_q": [9000, 5000, 1000, 100],
             "weight": 2},
        ],
    }


class TestQuantise(unittest.TestCase):
    def test_round_half_away_from_zero(self):
        self.assertEqual(M.quantise(0.00005), 1)
        self.assertEqual(M.quantise(-0.00005), -1)
        self.assertEqual(M.quantise(0.12345), 1235)
        self.assertEqual(M.quantise(1.0), 10000)

    def test_row_drops_non_finite(self):
        q = M.quantise_row({"a": 1.0, "b": float("nan"), "c": float("inf")})
        self.assertEqual(q, {"a": 10000})


class TestScoring(unittest.TestCase):
    def test_crossing_counts_all_thresholds(self):
        t = toy_table()["rules"][0]
        self.assertEqual(M.crossed(t, 50), 4)
        self.assertEqual(M.crossed(t, 700), 2)
        self.assertEqual(M.crossed(t, 9999), 0)

    def test_weights_multiply_crossings(self):
        tab = toy_table()
        rec = M.score_window({"f.low": 0.02, "f.high": 0.95}, tab)
        # f.low=200 crosses 3 thresholds (w1) -> 3 ; f.high=9500 crosses 4 (w2) -> 8
        self.assertEqual(rec["score"], 11)
        self.assertEqual(rec["verdict"], "REJECT")

    def test_quality_veto_forces_reject(self):
        tab = toy_table()
        rec = M.score_window({"f.low": 0.005}, tab)   # crosses the extreme
        self.assertTrue(rec["vetoed"])
        self.assertEqual(rec["verdict"], "REJECT")

    def test_review_band(self):
        tab = toy_table()
        rec = M.score_window({"f.high": 0.2}, tab)    # crosses 2 thresholds * w2 = 4
        self.assertGreaterEqual(rec["score"], 3)
        self.assertEqual(rec["verdict"], "REJECT")
        tab2 = dict(tab, t_reject=99)
        self.assertEqual(M.score_window({"f.high": 0.2}, tab2)["verdict"], "REVIEW")
        self.assertEqual(M.score_window({}, tab2)["verdict"], "ACCEPT")

    def test_missing_feature_never_fires(self):
        self.assertEqual(M.score_window({"nope": 1}, toy_table())["score"], 0)


class TestDeterminism(unittest.TestCase):
    def test_receipt_is_byte_identical_across_runs_and_key_order(self):
        tab = toy_table()
        a = M.decide({"f.low": 0.02, "f.high": 0.95}, tab, "w1")
        b = M.decide({"f.high": 0.95, "f.low": 0.02}, tab, "w1")
        self.assertEqual(a["receipt_sha256"], b["receipt_sha256"])
        for _ in range(5):
            self.assertEqual(M.decide({"f.low": 0.02, "f.high": 0.95}, tab,
                                      "w1")["receipt_sha256"], a["receipt_sha256"])

    def test_receipt_is_json_serialisable_and_all_int(self):
        rec = M.decide({"f.low": 0.02, "f.high": 0.95}, toy_table(), "w1")
        blob = json.dumps(rec, sort_keys=True)
        self.assertIn("receipt_sha256", json.loads(blob))
        for e in rec["evidence"]:
            self.assertIsInstance(e["value_q"], int)
            self.assertIsInstance(e["threshold_q"], int)

    def test_row_order_does_not_change_scores(self):
        tab = toy_table()
        rows = [{"f.low": 0.02, "f.high": 0.95}, {"f.low": 0.9, "f.high": 0.01}]
        fwd = [r["score"] for r in M.score_records(rows, tab)]
        rev = [r["score"] for r in M.score_records(rows[::-1], tab)][::-1]
        self.assertEqual(fwd, rev)


class TestShippedConfig(unittest.TestCase):
    def test_config_loads_and_is_integer(self):
        p = os.path.join(ROOT, "configs", "phys_rules_v2.json")
        if not os.path.exists(p):
            self.skipTest("config not present")
        tab = M.load_table(p)
        self.assertEqual(tab["quantisation"], 4)
        for r in tab["rules"]:
            for t in r["thresholds_q"]:
                self.assertIsInstance(t, int)
            self.assertIsInstance(r["weight"], int)
            self.assertGreaterEqual(r["weight"], 0)
        self.assertIsInstance(tab["t_reject"], int)


if __name__ == "__main__":
    unittest.main()
