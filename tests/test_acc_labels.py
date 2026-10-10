"""N4 label-strategy + certification-budget tests (< 30 s, CPU only, no network).

Checks the closed-form certification budget, the loader counts, and that the
next-label list excludes already-labelled windows and spans factories.
"""
import json
import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval.exp_acc import acc_lib as L  # noqa: E402


class TestCertBudget(unittest.TestCase):
    def test_formula_matches_closed_form(self):
        # K = ceil( ln(1/delta') / (2 r^2) ), rhat=0
        b = L.cert_budget_table(delta=0.05, grid=46, rhat=0.0)
        for r, expect_single in (("0.05", 600), ("0.10", 150), ("0.20", 38)):
            self.assertEqual(b[r]["single_threshold"]["K_clusters"], expect_single)
        for r, expect_union in (("0.05", 1365), ("0.10", 342), ("0.20", 86)):
            self.assertEqual(b[r]["grid_union"]["K_clusters"], expect_union)
        # sanity: matches the direct formula
        for r in (0.05, 0.10, 0.20):
            k = math.ceil(math.log(1.0 / (0.05 / 46)) / (2 * r ** 2))
            self.assertEqual(k, b[f"{r:.2f}"]["grid_union"]["K_clusters"])

    def test_certifiable_r_inverts(self):
        r = L.certifiable_r(47, delta=0.05, grid=46, rhat=0.0)
        self.assertAlmostEqual(r, math.sqrt(math.log(46 / 0.05) / (2 * 47)), places=9)
        # more clusters -> smaller certifiable r
        self.assertLess(L.certifiable_r(1000, grid=46), L.certifiable_r(47, grid=46))

    def test_union_is_looser_than_single(self):
        b = L.cert_budget_table(delta=0.05, grid=46, rhat=0.0)
        for r in ("0.05", "0.10", "0.20"):
            self.assertGreater(b[r]["grid_union"]["K_clusters"],
                               b[r]["single_threshold"]["K_clusters"])


class TestDataAndTopN(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = L.load_labelled160()

    def test_label_counts(self):
        self.assertEqual(len(self.data["y"]), 160)
        self.assertEqual(int(self.data["y"].sum()), 138)
        self.assertEqual(int((1 - self.data["y"]).sum()), 22)
        self.assertEqual(len(set(self.data["factory"].tolist())), 47)

    def test_features_shape(self):
        self.assertEqual(self.data["X"].shape, (160, 26))

    def test_topn_exists_and_clean(self):
        p = os.path.join(L.OUT, "labels_next.json")
        if not os.path.exists(p):
            self.skipTest("labels_next.json not generated yet")
        d = json.load(open(p))
        labelled = {sid.split("_s_")[-1] for sid in self.data["ids"]}
        self.assertTrue(all(c["window_id"][2:] not in labelled for c in d["windows"]))
        self.assertEqual(d["n"], len(d["windows"]))
        self.assertEqual(d["n_factories_covered"], 47)
        # required schema fields for the hand-off
        for c in d["windows"][:5]:
            for k in ("window_id", "factory", "clip", "novelty_dist"):
                self.assertIn(k, c)


if __name__ == "__main__":
    unittest.main()
