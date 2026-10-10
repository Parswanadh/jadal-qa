"""Focused CPU-only tests for the frames10k arm (synthetic arrays, <30 s).

Covers: sigmoid, per-segment pooling, LOCO OOF shape/contract, clip-bootstrap
CI contract, frame-head z-score application. No GPU, no network, no data files.
"""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval" / "frames10k"))
from analyze_transfer import clip_boot_ci, loco_oof, pool_feats, sigmoid  # noqa: E402


class TestSigmoid(unittest.TestCase):
    def test_extremes(self):
        self.assertAlmostEqual(sigmoid(np.array([1000.0]))[0], 1.0)
        self.assertAlmostEqual(sigmoid(np.array([-1000.0]))[0], 0.0)
        self.assertAlmostEqual(sigmoid(np.array([0.0]))[0], 0.5)


class TestPooling(unittest.TestCase):
    def test_stats(self):
        P = np.array([[0.0, 0.5, 1.0, np.nan]])
        f = pool_feats(P)[0]
        self.assertAlmostEqual(f[0], 0.5)   # mean
        self.assertAlmostEqual(f[1], 1.0)   # max
        self.assertAlmostEqual(f[4], 1 / 3)  # frac>0.5
        self.assertEqual(f.shape[0], 6)

    def test_all_nan(self):
        f = pool_feats(np.full((2, 4), np.nan))
        self.assertTrue(np.all(np.isfinite(f)))
        self.assertEqual(f.shape, (2, 6))


class TestLoco(unittest.TestCase):
    def _toy(self, seed=0):
        rng = np.random.RandomState(seed)
        clips = np.array(["a"] * 20 + ["b"] * 20 + ["c"] * 20)
        y = (rng.rand(60) < 0.5).astype(int)
        X = rng.rand(60, 4) + y[:, None]  # weak signal
        return X, y, clips

    def test_oof_contract(self):
        X, y, clips = self._toy()
        oof = loco_oof(X, y, clips)
        self.assertEqual(oof.shape, (60,))
        self.assertTrue(np.all((oof >= 0) & (oof <= 1)))

    def test_clip_boot_ci(self):
        X, y, clips = self._toy()
        oof = loco_oof(X, y, clips)
        m, lo, hi = clip_boot_ci(y, oof, clips, n_boot=100, seed=0)
        self.assertLessEqual(lo, m)
        self.assertLessEqual(m, hi)
        self.assertTrue(0.0 <= lo <= 1.0 and 0.0 <= hi <= 1.0)


class TestHeadApply(unittest.TestCase):
    def test_zscore_logit(self):
        rng = np.random.RandomState(1)
        X = rng.rand(10, 8)
        mean = X.mean(axis=0)
        scale = X.std(axis=0) + 1e-6
        coef = rng.rand(8)
        z = (X - mean) / scale @ coef
        p = sigmoid(z)
        self.assertTrue(np.all((p > 0) & (p < 1)))


if __name__ == "__main__":
    unittest.main()
