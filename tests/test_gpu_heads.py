"""Fast CPU tests for gpuheads (<30 s, no GPU, synthetic tensors)."""
import unittest

import numpy as np
import torch

from jadal.gpuheads.config import build_grid
from jadal.gpuheads.data import align, pool_window, standardize_fit, standardize_apply
from jadal.gpuheads.heads import BatchedHeads
from jadal.gpuheads.metrics import auroc, bootstrap_ci_auroc
from jadal.gpuheads.train import train_groups


class TestGrid(unittest.TestCase):
    def test_grid_covers_spec(self):
        cfgs = build_grid()
        hs = {c.hidden for c in cfgs}
        self.assertTrue({0, 64, 256, 1024} <= hs)
        self.assertTrue(all(c.dropout == 0.0 for c in cfgs))


class TestBatched(unittest.TestCase):
    def test_forward_shapes(self):
        torch.manual_seed(0)
        for h in (0, 64):
            m = BatchedHeads(10, h, 5, seed=0)
            out = m(torch.randn(7, 10))
            self.assertEqual(tuple(out.shape), (5, 7))

    def test_train_reduces_loss(self):
        torch.manual_seed(0)
        X = torch.randn(64, 8)
        y = (X[:, 0] > 0).float()
        cfgs = build_grid(hiddens=[0], wds=[0.0], seeds=[0], class_weights=[False])
        by_h = {0: cfgs}
        mods, out, _ = train_groups(X, y, by_h, 8, epochs=5, batch=32, device="cpu", seed=0)
        self.assertIn(0, out)
        self.assertEqual(tuple(out[0].shape), (1, 64))

    def test_trains_mid_hidden(self):
        torch.manual_seed(1)
        X = torch.randn(48, 6)
        y = (X[:, 0] + X[:, 1] > 0).float()
        cfgs = build_grid(hiddens=[64], wds=[0.0], seeds=[0, 1], class_weights=[False])
        mods, out, _ = train_groups(X, y, {64: cfgs}, 6, epochs=3, batch=16, device="cpu", seed=1)
        self.assertEqual(tuple(out[64].shape), (2, 48))


class TestDataUtils(unittest.TestCase):
    def test_align_by_id_not_position(self):
        ia, ib = align(np.array(["b", "a"]), np.array(["a", "b"]))
        self.assertEqual(list(ia), [0, 1])
        self.assertEqual(list(ib), [1, 0])

    def test_standardize_train_only(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        mu, sd = standardize_fit(X)
        Z = standardize_apply(X, mu, sd)
        self.assertAlmostEqual(float(Z.mean()), 0.0, places=6)

    def test_pool_window(self):
        v = pool_window(np.array([0.1, 0.9, 0.5, 0.5]))
        self.assertEqual(v.shape, (9,))
        self.assertAlmostEqual(v[0], 0.5)


class TestMetrics(unittest.TestCase):
    def test_auroc_ci(self):
        y = np.array([0, 0, 1, 1])
        s = np.array([0.1, 0.4, 0.6, 0.9])
        self.assertGreater(auroc(y, s), 0.9)
        lo, hi = bootstrap_ci_auroc(y, s, n_boot=50, seed=0)
        self.assertLessEqual(lo, hi)


if __name__ == "__main__":
    unittest.main()
