"""Harness tests: scorer metrics + manifest schemas on synthetic arrays.

No GPU, no network, no git-ignored data. < 30 s.
Run: PYTHONNOUSERSITE=1 JADAL_MODELS_DIR=/home/parshu/projects/jadal-qa/models \
     python -m unittest discover -s tests
"""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))
from spectral.score import (apply_head, auroc, clip_bootstrap_ci,  # noqa: E402
                            feat_matrix, fit_full_head, gold_task, make_head)


class TestMetrics(unittest.TestCase):
    def test_auroc_perfect(self):
        self.assertAlmostEqual(auroc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]), 1.0)

    def test_auroc_single_class_none(self):
        self.assertIsNone(auroc([1, 1, 1], [0.5, 0.6, 0.7]))

    def test_bootstrap_ci_ordered(self):
        rng = np.random.default_rng(0)
        y = (rng.random(40) > 0.5).astype(int)
        s = y * 0.6 + rng.random(40) * 0.4
        clips = [f"c{i % 4}" for i in range(40)]
        ci, n = clip_bootstrap_ci(clips, y, s, b=50)
        self.assertEqual(len(ci), 2)
        self.assertLessEqual(ci[0], ci[1])
        self.assertGreater(n, 0)

    def test_head_separable(self):
        X = np.array([[0.0], [0.1], [0.9], [1.0]])
        y = np.array([0, 0, 1, 1])
        head = make_head().fit(X, y)
        self.assertTrue(bool((head.predict_proba(X)[:, 1] > 0.5).tolist() == [False, False, True, True]))


def _fake_gold_doc(n=12):
    rng = np.random.default_rng(1)
    items = {}
    for i in range(n):
        lab = i % 2
        items[f"g{i:03d}"] = {
            "score": None,
            "features": {"blur_var": float(10 + lab * 20 + rng.random()),
                         "motion_mean": float(lab * 5 + rng.random())},
            "latency_s": 0.1}
    return {"arm": "T", "items": items}


def _fake_manifest(n=12):
    return {"items": [{"id": f"g{i:03d}",
                       "clip": "cA" if i < n // 2 else "cB",
                       "label": i % 2} for i in range(n)]}


class TestGoldTask(unittest.TestCase):
    def test_loco_runs(self):
        out = gold_task("T", _fake_gold_doc(), _fake_manifest(),
                        ["blur_var", "motion_mean"])
        self.assertEqual(out["n"], 12)
        self.assertEqual(out["n_clips"], 2)
        self.assertIn("head", out["results"])
        self.assertIsNotNone(out["results"]["head"]["auroc"])

    def test_fit_full_head_threshold(self):
        head, thr, ids = fit_full_head(_fake_gold_doc(), _fake_manifest(),
                                       ["blur_var", "motion_mean"])
        self.assertEqual(len(ids), 12)
        self.assertGreaterEqual(thr, 0.0)
        self.assertLessEqual(thr, 1.0)
        s = apply_head(head, {"blur_var": 35.0, "motion_mean": 6.0},
                       ["blur_var", "motion_mean"])
        self.assertIsNotNone(s)

    def test_feat_matrix_emb(self):
        items = {"a": {"features": {"emb": [0.1, 0.2]}},
                 "b": {"features": {"emb": [0.3, 0.4]}}}
        X, ids = feat_matrix(items, [])
        self.assertEqual(X.shape, (2, 2))
        self.assertEqual(ids, ["a", "b"])


if __name__ == "__main__":
    unittest.main()
