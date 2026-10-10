"""Tests for the Y2 review-gate analysis (fast, no network, deterministic)."""
from __future__ import annotations

import unittest

import numpy as np

from eval.exp_review.gate import (
    auroc, cls_metrics, evaluate_gate, fit_temperature, gate,
    loco_temperature, quality_fires, wilson_ci,
)


class TestMetrics(unittest.TestCase):
    def test_auroc_perfect_and_reversed(self):
        y = np.array([0, 0, 1, 1])
        self.assertAlmostEqual(auroc([0.1, 0.2, 0.8, 0.9], y), 1.0)
        self.assertAlmostEqual(auroc([0.9, 0.8, 0.2, 0.1], y), 0.0)

    def test_auroc_ties_are_half(self):
        y = np.array([0, 1])
        self.assertAlmostEqual(auroc([0.5, 0.5], y), 0.5)

    def test_wilson_bounds(self):
        lo, hi = wilson_ci(0, 10)
        self.assertEqual(lo, 0.0)
        self.assertGreater(hi, 0.0)


class TestGating(unittest.TestCase):
    def test_coverage_and_review(self):
        p = np.linspace(0.0, 1.0, 101)
        decided, review, _ = gate(p, coverage=0.5, cost_ratio=1)
        self.assertEqual(len(decided), 50)
        self.assertEqual(len(review), 51)

    def test_cost_ratio_moves_boundary(self):
        """5:1 false-accept cost must accept less than 1:1 at full coverage."""
        p = np.linspace(0.0, 1.0, 101)
        _, _, pred1 = gate(p, 1.0, 1)
        _, _, pred5 = gate(p, 1.0, 5)
        self.assertGreater(pred1.sum(), pred5.sum())

    def test_decided_are_most_confident(self):
        p = np.array([0.05, 0.45, 0.5, 0.55, 0.95])
        decided, review, _ = gate(p, coverage=0.4, cost_ratio=1)
        # coverage 0.4 -> 2 decided: the 0.05 and 0.95 (largest margins)
        self.assertEqual(sorted(decided.tolist()), [0, 4])
        self.assertEqual(sorted(review.tolist()), [1, 2, 3])

    def test_evaluate_gate_metrics(self):
        y = np.array([1, 1, 0, 0])
        p = np.array([0.9, 0.8, 0.2, 0.1])
        m, _, _, _ = evaluate_gate(p, y, 1.0, 1)
        self.assertEqual(m["balanced_accuracy"], 1.0)
        self.assertEqual(m["TP"], 2)
        self.assertEqual(m["TN"], 2)


class TestCalibration(unittest.TestCase):
    def test_temperature_is_identityish_on_flat_scores(self):
        p = np.linspace(0.01, 0.99, 50)
        y = (p > 0.5).astype(int)
        T = fit_temperature(p, y)
        self.assertGreater(T, 0.0)

    def test_loco_temperature_shape(self):
        p = np.linspace(0.05, 0.95, 30)
        y = (p > 0.5).astype(int)
        groups = np.array(["a", "b", "c"] * 10)
        out = loco_temperature(p, y, groups)
        self.assertEqual(out.shape, p.shape)
        self.assertTrue(((out > 0) & (out < 1)).all())


class TestQualityRules(unittest.TestCase):
    def test_black_fires_dark(self):
        f = {"luma_mean": 0.0, "dark_frac": 1.0, "bright_frac": 0.0,
             "frozen_frac": 1.0, "zncc_self": 0.0, "motion_energy": 0.0}
        codes = quality_fires(f)
        self.assertIn("TOO_DARK", codes)
        self.assertIn("NO_ACTIVE_WORK", codes)

    def test_clean_does_not_fire(self):
        f = {"luma_mean": 0.28, "dark_frac": 0.18, "bright_frac": 0.0007,
             "frozen_frac": 0.0, "zncc_self": 0.4, "motion_energy": 0.04}
        self.assertEqual(quality_fires(f), [])

    def test_overexposed_fires(self):
        f = {"luma_mean": 0.88, "dark_frac": 0.0, "bright_frac": 0.47,
             "frozen_frac": 0.0, "zncc_self": 0.5, "motion_energy": 0.02}
        self.assertIn("OVEREXPOSED", quality_fires(f))


if __name__ == "__main__":
    unittest.main()
