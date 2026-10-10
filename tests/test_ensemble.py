"""Unit tests for the frozen two-view ensemble (synthetic arrays, no GPU/video)."""
import copy
import json
import os
import tempfile
import unittest

import numpy as np

from jadal.ensemble.scorer import (ensemble_prob, features_digest, head_prob, load_config)

CFG = os.path.join(os.path.dirname(__file__), "..", "configs", "ensemble_v1.json")


class TestEnsembleHeads(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg, _ = load_config(CFG)

    def test_config_sha_verifies(self):
        cfg, _ = load_config(CFG)  # raises on sha mismatch
        self.assertEqual(cfg["name"], "ensemble_v1")

    def test_tampered_config_refused(self):
        with open(CFG) as f:
            cfg = json.load(f)
        cfg["threshold"] = 0.99
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "ensemble_v1.json")
            open(p, "w").write(json.dumps(cfg, indent=1))
            open(p + ".sha256", "w").write(open(CFG + ".sha256").read())
            with self.assertRaises(ValueError):
                load_config(p)

    def test_head_prob_sigmoid_at_zero(self):
        h = {"mean": [0.0, 0.0], "scale": [1.0, 1.0],
             "coef": [1.0, -1.0], "intercept": 0.0}
        self.assertAlmostEqual(head_prob(np.zeros(2), h), 0.5, places=12)
        self.assertGreater(head_prob(np.array([2.0, 0.0]), h), 0.5)
        self.assertLess(head_prob(np.array([0.0, 2.0]), h), 0.5)

    def test_head_deterministic(self):
        h = self.cfg["views"]["pixel"]
        rng = np.random.default_rng(0)
        x = rng.normal(size=len(h["mean"]))
        self.assertEqual(head_prob(x, h), head_prob(x, h))

    def test_ensemble_is_equal_average(self):
        rng = np.random.default_rng(1)
        e = rng.normal(size=768)
        feats = {k: float(v) for k, v in
                 zip(self.cfg["views"]["pixel"]["feature_names"], rng.normal(size=26))}
        p, p_e, p_g = ensemble_prob(e, feats, self.cfg)
        self.assertAlmostEqual(p, 0.5 * p_e + 0.5 * p_g, places=12)
        self.assertTrue(0.0 < p < 1.0)

    def test_digest_deterministic_and_sensitive(self):
        feats = {"b": 0.5, "a": 0.25}
        m = np.array([0.1, 0.2, 0.3])
        self.assertEqual(features_digest(feats, m, {}), features_digest(feats, m, {}))
        self.assertNotEqual(features_digest(feats, m, {}),
                            features_digest({"b": 0.5, "a": 0.26}, m, {}))
        self.assertEqual(len(features_digest(feats, m, {})), 64)

    def test_fusion_weights_sum_to_one(self):
        self.assertAlmostEqual(sum(self.cfg["fusion"]["weights"]), 1.0)

    def test_verdict_rule_sane(self):
        self.assertEqual(self.cfg["threshold"], 0.5)
        lo, hi = self.cfg["abstain_band"]
        self.assertTrue(lo < 0.5 < hi or hi == 0.6)


if __name__ == "__main__":
    unittest.main()
