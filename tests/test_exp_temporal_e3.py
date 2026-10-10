"""Smoke tests for E3 artefacts (stdlib+sklearn-free asserts on committed JSONs)."""
import json
import os
import unittest

D = os.path.join(os.path.dirname(__file__), "..", "eval", "results", "exp_temporal")


def load(n):
    with open(os.path.join(D, n)) as f:
        return json.load(f)


class TestE3(unittest.TestCase):
    def test_e301_present_and_consistent(self):
        r = load("E3.01_policies.json")
        self.assertEqual(r["n_windows"], 150)
        self.assertEqual(r["n_clips"], 7)
        pp = r["per_policy"]
        # cost decreases with sparser grids
        self.assertGreater(pp["every-2"]["mean_frac"], pp["every-4"]["mean_frac"])
        self.assertGreater(pp["every-4"]["mean_frac"], pp["every-8"]["mean_frac"])
        # error grows with sparser grids (label oracle)
        self.assertLess(pp["every-2"]["mean_err_L"], pp["every-4"]["mean_err_L"])
        self.assertLess(pp["every-4"]["mean_err_L"], pp["every-8"]["mean_err_L"])
        # full coverage has zero sampling error
        self.assertEqual(pp["full"]["mean_err_L"], 0.0)
        for v in r["cusum_loco"]["folds"].values():
            self.assertIn("test_err_M", v)

    def test_e302_thin_but_valid(self):
        r = load("E3.02_ood_position.json")
        self.assertEqual(r["n_windows"], 60)
        self.assertEqual(r["n_bad"], 9)
        self.assertGreaterEqual(r["mw_raw_p"], 0.0)
        self.assertLessEqual(r["mw_raw_p"], 1.0)

    def test_e303_curve_saturates(self):
        r = load("E3.03_early_exit.json")
        c = {x["seconds"]: x["auroc"] for x in r["curve"]}
        self.assertGreater(c[4.0], 0.83)
        # 4 s within 0.02 of the best prefix
        self.assertGreaterEqual(c[4.0], max(c.values()) - 0.02)
        # 2 s clearly worse than 4 s
        self.assertLess(c[2.0], c[4.0] - 0.03)

    def test_e304_arithmetic(self):
        r = load("E3.04_compute.json")
        self.assertAlmostEqual(r["corpus_hours"], 4.404, places=2)
        for row in r["per_policy"]:
            self.assertAlmostEqual(row["speedup"], 1.0 / row["window_frac"], places=1)

    def test_e305_decision_exists(self):
        r = load("E3.05_decision.json")
        self.assertIn("every 2", str(r["decisions"]))
        self.assertTrue(any("thin" in k.lower() or "thin_evidence" in k for k in r))


if __name__ == "__main__":
    unittest.main()
