"""Determinism tests for the Q3 deterministic rule scorer (no network, <30 s)."""
from __future__ import annotations

import json
import os
import sys
import unittest
from decimal import Decimal

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "eval", "exp_phys"))

from eval.exp_phys import det_rule_scorer as D  # noqa: E402

HERE = os.path.join(ROOT, "eval", "exp_phys")


class TestCanonicalisation(unittest.TestCase):
    def test_canonical_json_sorted_compact(self):
        a = D.canonical_json({"b": 1, "a": [2, 3]})
        self.assertEqual(a, b'{"a":[2,3],"b":1}')

    def test_canonical_json_order_insensitive(self):
        x = {"z": 1, "a": {"y": 2, "b": 3}}
        y = {"a": {"b": 3, "y": 2}, "z": 1}
        self.assertEqual(D.canonical_json(x), D.canonical_json(y))

    def test_quantise_exact_half_even(self):
        # 0.00005 * 1e4 = 0.5 -> half-even -> 0 ; 0.00015 -> 1.5 -> 2
        self.assertEqual(D.quantise(Decimal("0.00005")), 0)
        self.assertEqual(D.quantise(Decimal("0.00015")), 2)
        self.assertEqual(D.quantise(Decimal("6.4861")), 64861)
        self.assertEqual(D.quantise(Decimal("95.0933")), 950933)
        self.assertEqual(D.quantise(Decimal("339.2751")), 3392751)

    def test_quantise_missing_and_nonfinite(self):
        self.assertEqual(D.quantise(None), D.MISSING)
        self.assertEqual(D.quantise(float("nan")), D.MISSING)
        self.assertEqual(D.quantise(float("inf")), D.MISSING)
        self.assertEqual(D.quantise(True), D.MISSING)


class TestRulePath(unittest.TestCase):
    def setUp(self):
        self.table = D.load_rule_table(os.path.join(HERE, "rule_table.json"))
        self.code = D.code_sha256()

    def test_thresholds_are_integers(self):
        for r in self.table["rules"]:
            self.assertIsInstance(r["thr_int"], int)

    def test_reject_on_low_entropy(self):
        w = {"window_id": "x", "features": {"hist_ent": 5.0,
                                            "blur_norm": 200.0, "ten_norm": 5000.0}}
        r = D.score_window(w, self.table, code_hash=self.code)
        self.assertTrue(r["decision"].startswith("reject"))
        self.assertEqual(r["decision"], "reject:lowent")
        ev = {e["rule"]: e for e in r["evidence"]}
        self.assertEqual(ev["lowent"]["value"], 50000)
        self.assertTrue(all(isinstance(e["value"], int) for e in r["evidence"]))

    def test_pass_when_all_rules_clear(self):
        w = {"window_id": "x", "features": {"hist_ent": 7.5,
                                            "blur_norm": 200.0, "ten_norm": 5000.0}}
        r = D.score_window(w, self.table, code_hash=self.code)
        self.assertEqual(r["decision"], "pass")

    def test_receipt_has_no_forbidden_fields_or_floats(self):
        w = {"window_id": "x", "features": {"hist_ent": 7.5,
                                            "blur_norm": 200.0, "ten_norm": 5000.0}}
        r = D.score_window(w, self.table, code_hash=self.code)
        D.assert_no_forbidden_fields(r)  # raises on float / time / random key

    def test_ten_times_identical(self):
        w = {"window_id": "x", "features": {"hist_ent": 6.0,
                                            "blur_norm": 50.0, "ten_norm": 100.0}}
        hashes = {D.score_window(w, self.table, code_hash=self.code)["receipt_sha256"]
                  for _ in range(10)}
        self.assertEqual(len(hashes), 1)

    def test_feature_order_does_not_change_hash(self):
        f = {"hist_ent": 7.5, "blur_norm": 200.0, "ten_norm": 5000.0}
        w1 = {"window_id": "x", "features": f}
        w2 = {"window_id": "x", "features": dict(reversed(list(f.items())))}
        self.assertEqual(
            D.score_window(w1, self.table, code_hash=self.code)["receipt_sha256"],
            D.score_window(w2, self.table, code_hash=self.code)["receipt_sha256"])


class TestFrozenInput(unittest.TestCase):
    def test_input_set_has_at_least_20_each(self):
        d = json.load(open(os.path.join(HERE, "input_windows.json")))
        self.assertGreaterEqual(d["n_gold"], 20)
        self.assertGreaterEqual(d["n_ood"], 20)
        self.assertEqual(d["n_windows"], d["n_gold"] + d["n_ood"])

    def test_results_proof_passed(self):
        p = os.path.join(ROOT, "eval", "results", "exp_phys", "Q3.01.json")
        if not os.path.exists(p):
            self.skipTest("Q3.01.json not generated yet")
        d = json.load(open(p))
        self.assertTrue(d["all_identical"])
        self.assertEqual(d["n_windows_identical"], d["n_windows"])
        self.assertEqual(d["status"], "PASS")


if __name__ == "__main__":
    unittest.main()
