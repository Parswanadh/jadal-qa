"""Tests for the E5 domain-adaptation experiments (no network, <30 s).

Run: PYTHONNOUSERSITE=1 python3 -m unittest tests.test_exp_domain -v
Covers: e5lib pure helpers, and the schema/consistency of the committed
results JSONs (every number must carry n + label_source + protocol + CI).
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.exp_domain import e5lib as L  # noqa: E402

RES = ROOT / "eval" / "results" / "exp_domain"


class TestHelpers(unittest.TestCase):
    def test_sigmoid_bounds(self):
        z = np.array([-50.0, 0.0, 50.0])
        s = L.sigmoid(z)
        self.assertTrue(np.all((s >= 0) & (s <= 1)))
        self.assertAlmostEqual(s[1], 0.5, places=12)

    def test_pool_feats_nan_pad(self):
        P = np.array([[0.2, 0.4, np.nan, np.nan], [0.6, 0.6, 0.6, 0.6]])
        F = L.pool_feats(P)
        self.assertEqual(F.shape, (2, 6))
        self.assertAlmostEqual(F[0, 0], 0.3, places=9)   # mean of valid
        self.assertAlmostEqual(F[0, 1], 0.4, places=9)   # max
        self.assertAlmostEqual(F[1, 4], 1.0, places=9)   # frac>0.5

    def test_boot_ci_degrades_on_single_class(self):
        y = np.zeros(10, int)
        s = np.linspace(0, 1, 10)
        m, lo, hi = L.boot_ci(y, s, lambda a, b: 0.5)
        self.assertTrue(np.isnan(m))

    def test_group_boot_ci_shape(self):
        rng = np.random.RandomState(0)
        y = np.array([1] * 10 + [0] * 10)
        s = y + rng.normal(0, 0.1, 20)
        g = np.array(list("aaaaaaaaaa") + list("bbbbbbbbbb"))
        from sklearn.metrics import roc_auc_score
        m, lo, hi = L.group_boot_ci(y, s, g, n_boot=200, seed=1)
        self.assertLessEqual(lo, hi)
        self.assertTrue(0.0 <= lo <= 1.0 and 0.0 <= hi <= 1.0)

    def test_load_labels_file_schemas(self):
        import tempfile
        cases = [
            ({"latest": {"a": {"verdict": "good"}, "b": {"verdict": "bad"}}}, {"a": 1, "b": 0}),
            ({"items": [{"id": "a", "label": 1}, {"id": "b", "label": 0}]}, {"a": 1, "b": 0}),
        ]
        for doc, want in cases:
            with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
                json.dump(doc, f)
                f.flush()
                path = f.name
            got = L.load_labels_file(path)
            self.assertEqual(got, want)

    def test_feature_names_are_26(self):
        self.assertEqual(len(L.FEATURE_NAMES), 26)
        self.assertEqual(len(set(L.FEATURE_NAMES)), 26)

    def test_classify_transfer_rule_boundary(self):
        # I-1: a feature informative in ONE domain only must NOT be 'chance'.
        self.assertNotEqual(L.classify_transfer(0.62, 0.50), "chance")
        self.assertEqual(L.classify_transfer(0.62, 0.51), "stable")
        self.assertEqual(L.classify_transfer(0.62, 0.49), "flip")
        # both near 0.5 -> chance
        self.assertEqual(L.classify_transfer(0.50, 0.50), "chance")
        self.assertEqual(L.classify_transfer(0.52, 0.48), "chance")
        # tolerance boundary: exactly 0.05 is informative, 0.049 is not
        self.assertNotEqual(L.classify_transfer(0.55, 0.50), "chance")
        self.assertEqual(L.classify_transfer(0.549, 0.50), "chance")
        # identical positive AUROCs in both domains -> stable
        self.assertEqual(L.classify_transfer(0.70, 0.70), "stable")
        self.assertEqual(L.classify_transfer(0.30, 0.30), "stable")

    def test_canonical_config_sha_strips_only_paths(self):
        cfg = {"views": {"eg2": {"embed_model_dir": "/a", "w": 1}},
               "embed_command": "x", "pixel_command": "y", "threshold": 0.5}
        cfg2 = {"views": {"eg2": {"embed_model_dir": "/different", "w": 1}},
                "embed_command": "z", "pixel_command": "w", "threshold": 0.5}
        self.assertEqual(L.canonical_config_sha256(cfg), L.canonical_config_sha256(cfg2))
        cfg3 = {**cfg, "threshold": 0.6}
        self.assertNotEqual(L.canonical_config_sha256(cfg), L.canonical_config_sha256(cfg3))


class TestResultsSchema(unittest.TestCase):
    REQUIRED = ("n", "label_source", "protocol", "device", "git_sha")

    def _load(self, name):
        p = RES / name
        self.assertTrue(p.exists(), f"missing results file {p}")
        return json.load(open(p))

    def test_all_results_exist_with_core_fields(self):
        for name in ("E5.00.json", "E5.01.json", "E5.02.json", "E5.03.json", "E5.04.json"):
            d = self._load(name)
            for k in self.REQUIRED:
                self.assertIn(k, d, f"{name} missing {k}")
            self.assertTrue(str(d["label_source"]).strip())
            self.assertTrue(str(d["command"]).strip(), f"{name} empty command")

    def test_e500_reproduces_frozen_ood_prior(self):
        d = self._load("E5.00.json")
        self.assertEqual(d["n"], 60)
        self.assertEqual(d["n_bad"], 9)
        self.assertEqual(len(d["ci95"]), 2)
        self.assertAlmostEqual(d["ood_auroc"], 0.415, places=2)

    def test_e501_model_vs_model_caveat_and_ci(self):
        d = self._load("E5.01.json")
        self.assertEqual(d["n"], 150)
        self.assertIn("AI", d["label_source"])
        self.assertIn("NOT ground-truth", d["caveat"])
        self.assertIn("pooled_feats_logistic_loco", d["arms"])
        for k, v in d["arms"].items():
            self.assertEqual(len(v["ci95"]), 2, k)
        for v in d["frame_level_cv_ai"].values():
            self.assertEqual(len(v["ci95"]), 2)

    def test_e502_recipes_have_ci_and_min_bad(self):
        d = self._load("E5.02.json")
        self.assertEqual(d["n"], 60)
        self.assertEqual(d["n_bad"], 9)
        for k in ("pixel26_logistic", "eg2_mean_logistic", "two_view_avg"):
            self.assertIn(k, d["recipes"])
            self.assertEqual(len(d["recipes"][k]["ci95"]), 2)
            self.assertIsNotNone(d["recipes"][k]["smallest_n_bad_in_train_fold"])
        self.assertGreater(len(d["learning_curve"]), 0)

    def test_e503_table_complete(self):
        d = self._load("E5.03.json")
        self.assertEqual(len(d["table"]), 26)
        self.assertEqual(len({r["feature"] for r in d["table"]}), 26)
        c = d["counts"]
        self.assertEqual(c["stable"] + c["flip"] + c["chance"], 26)
        for r in d["table"]:
            self.assertEqual(len(r["indego_ci95"]), 2)
            self.assertEqual(len(r["ood_10fps_ci95"]), 2)
            self.assertIn(r["class"], ("stable", "flip", "chance"))

    def test_e503_class_matches_documented_rule(self):
        # I-1: the stored class must equal the one rule, applied to the stored AUROCs.
        d = self._load("E5.03.json")
        for r in d["table"]:
            self.assertEqual(r["class"],
                             L.classify_transfer(r["indego_auroc"], r["ood_10fps_auroc"]),
                             r["feature"])
            # I-2: every row also carries the native-30 class under the same rule.
            self.assertEqual(r["class_native30"],
                             L.classify_transfer(r["indego_auroc"], r["ood_native30_auroc"]),
                             r["feature"])
            self.assertEqual(r["class_agrees_native30_vs_10fps"],
                             r["class"] == r["class_native30"])
            # a feature not chance in either domain is never a one-domain fluke
            if r["class"] != "chance":
                self.assertGreater(max(abs(r["indego_auroc"] - 0.5),
                                       abs(r["ood_10fps_auroc"] - 0.5)), 0.049)
        n30 = d["counts_native30"]
        self.assertEqual(sum(n30.values()), 26)
        self.assertEqual(sorted(d["native30_vs_10fps_class_disagreements"]),
                         sorted(r["feature"] for r in d["table"] if r["class"] != r["class_native30"]))

    def test_e504_inconclusive_when_labels_missing(self):
        d = self._load("E5.04.json")
        self.assertEqual(d["n_scored"], 160)
        if not d["labels2_present"]:
            self.assertTrue(d["labels_missing"])
            self.assertEqual(d["n"], 0)
            self.assertEqual(d["ci95"], "n/a (no labels)")
            self.assertIn("INCONCLUSIVE", d["verdict_note"])
            self.assertTrue(d["descriptive"]["n_scored"] == 160)

    def test_e504_label_source_and_config_assert(self):
        # M-4: the 160 scored windows carry NO labels.
        d = self._load("E5.04.json")
        self.assertIn("NONE", d["label_source"])
        self.assertIn("ood_labels2.json NOT delivered", d["label_source"])
        # M-8: observed config sha stored; canonical content asserted against the frozen one.
        self.assertEqual(len(d["config_sha256_observed"]), 64)
        self.assertEqual(d["config_canonical_sha256_observed"],
                         d["config_canonical_sha256_expected"])
        import hashlib
        cfg = ROOT / "configs" / "ensemble_v1.json"
        with open(cfg, "rb") as f:
            self.assertEqual(hashlib.sha256(f.read()).hexdigest(), d["config_sha256_observed"])


if __name__ == "__main__":
    unittest.main()
