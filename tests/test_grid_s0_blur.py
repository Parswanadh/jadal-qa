"""Unit tests for the GRID.09 brightness-calibrated sharpness statistic.

Synthetic only (no dataset frames, no video, no network).  The frozen threshold
is checked against the committed calibration artifact: if that file is missing
the test FAILS rather than passing vacuously.

Run: python3 -m unittest tests.test_grid_s0_blur -v
"""
import json
import os
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import poc_run  # noqa: E402
from jadal.mosaic import features as F  # noqa: E402
from jadal.poc import cascade as C  # noqa: E402
from jadal.poc import s0_blur  # noqa: E402

CALIB = os.path.join(ROOT, "eval", "results", "exp_grid", "grid09_calibration.json")


def checker(size=96, seed=0):
    """A sharp synthetic frame: fine checkerboard, luma in [0, 1]."""
    rng = np.random.default_rng(seed)
    y, x = np.indices((size, size))
    c = ((x // 3 + y // 3) % 2).astype(float) * 0.6 + 0.2
    return np.clip(c + rng.normal(0, 0.01, c.shape), 0, 1)


class TestStatistic(unittest.TestCase):
    def test_uniform_is_zero(self):
        for v in (0.0, 0.25, 0.5, 1.0):
            self.assertEqual(s0_blur.lapvar_cal(np.full((45, 80), v)), 0.0)
            self.assertEqual(s0_blur.laplacian_var(np.full((45, 80), v)), 0.0)

    def test_black_is_zero_not_nan(self):
        v = s0_blur.lapvar_cal(np.zeros((45, 80)))
        self.assertTrue(np.isfinite(v))
        self.assertEqual(v, 0.0)

    def test_noise_is_positive(self):
        rng = np.random.default_rng(1)
        self.assertGreater(s0_blur.lapvar_cal(rng.random((45, 80))), 0.1)

    def test_gain_proportional_not_quadratic(self):
        """Raw Laplacian variance scales with gain^2; the calibrated form with gain.
        That is what makes a single threshold usable across exposures.

        The identity holds up to the documented ``+eps`` in the denominator
        (relative deviation ~eps/mean ~ 3e-6 here), so the tolerance is 5 places,
        not 6.
        """
        base = checker()
        g = 0.5
        raw_ratio = s0_blur.laplacian_var(base * g) / s0_blur.laplacian_var(base)
        cal_ratio = s0_blur.lapvar_cal(base * g) / s0_blur.lapvar_cal(base)
        self.assertAlmostEqual(raw_ratio, g ** 2, places=6)
        self.assertAlmostEqual(cal_ratio, g, places=5)
        self.assertNotAlmostEqual(cal_ratio, g ** 2, places=3)

    def test_monotone_in_blur(self):
        """More blur -> strictly smaller lapvar_cal at a fixed gain."""
        import cv2
        base = checker().astype(np.float32)
        vals = [s0_blur.lapvar_cal(base)]
        for sigma in (1.5, 3.0, 8.0):
            k = 2 * int(np.ceil(3 * sigma)) + 1
            vals.append(s0_blur.lapvar_cal(cv2.GaussianBlur(base, (k, k), sigma)))
        self.assertEqual(vals, sorted(vals, reverse=True),
                         f"not monotone in blur: {vals}")
        self.assertLess(vals[-1], 0.2 * vals[0])

    def test_probe_statistic_is_repeat_invariant(self):
        one = checker()
        stack = np.stack([one] * 5)
        self.assertAlmostEqual(s0_blur.probe_lapvar_cal(stack),
                               s0_blur.lapvar_cal(one), places=12)

    def test_features_module_shares_the_definition(self):
        """The mosaic tile statistic must be the SAME function, not a copy."""
        g = checker()
        self.assertEqual(F._lapvar_cal(g), s0_blur.lapvar_cal(g))


class TestProductionProbe(unittest.TestCase):
    def test_s0_from_luma_reports_both_resolutions(self):
        rng = np.random.default_rng(2)
        luma = rng.random((60, 90, 160))
        s0 = poc_run.s0_from_luma(luma)
        for k in ("mean", "p5", "p95", "spatial_var", "temp_e", "lapvar_cal",
                  "lapvar_cal_45"):
            self.assertIn(k, s0)
        # the rule statistic is the 160x90 one, not the 2x2-boxed buffer
        self.assertAlmostEqual(s0["lapvar_cal"], s0_blur.probe_lapvar_cal(luma),
                               places=12)
        self.assertNotAlmostEqual(s0["lapvar_cal"], s0["lapvar_cal_45"], places=6)

    def test_shipped_five_keys_unchanged(self):
        """The five shipped keys keep their exact arithmetic."""
        rng = np.random.default_rng(3)
        luma = rng.random((60, 90, 160))
        s0 = poc_run.s0_from_luma(luma)
        g = luma.reshape(60, 45, 2, 80, 2).mean(axis=(2, 4))
        self.assertAlmostEqual(s0["mean"], float(g.mean()), places=12)
        self.assertAlmostEqual(s0["p5"], float(np.percentile(g, 5)), places=12)
        self.assertAlmostEqual(s0["p95"], float(np.percentile(g, 95)), places=12)
        self.assertAlmostEqual(s0["spatial_var"], float(g[:10].mean(axis=0).var()),
                               places=12)
        self.assertAlmostEqual(s0["temp_e"], float(np.abs(np.diff(g, axis=0)).mean()),
                               places=12)


class TestRule(unittest.TestCase):
    BASE = {"mean": 0.5, "p5": 0.3, "p95": 0.8, "spatial_var": 0.01,
            "temp_e": 0.01, "lapvar_cal": 0.5}

    def test_blur_rule_is_last_and_black_wins(self):
        rules = [r[0] for r in C.S0_RULES]
        self.assertEqual(rules[-1], "BLUR")
        self.assertEqual(rules[:5], ["BLACK", "TOO_DARK", "LENS_BLOCKED",
                                     "OVEREXPOSED", "FROZEN"])
        black = dict(self.BASE, p95=0.0, mean=0.0, lapvar_cal=0.0)
        self.assertEqual(C.s0_decide(black)[0], "BLACK")

    def test_rule_ships_disabled_by_default(self):
        """The shipped default must NOT change any verdict: 0.0 disables the rule.

        Adding BLUR to S0_RULES is the GRID.09 fix, but the stream owner requires
        a verdict-changing rule to ship OFF.  With 0.0 the comparison can never
        fire, so a blurred probe decides exactly as the five shipped rules do.
        """
        self.assertEqual(s0_blur.BLUR_LAPVAR_CAL_MIN, 0.0)
        blurred = dict(self.BASE, lapvar_cal=1e-9)
        self.assertIsNone(C.s0_decide(blurred)[0])
        self.assertEqual(len(C.S0_RULES), 6)
        self.assertEqual(C.S0_RULES[-1][0], "BLUR")

    def test_blur_fires_when_enabled_at_the_calibrated_threshold(self):
        thr = s0_blur.BLUR_LAPVAR_CAL_MIN_CALIBRATED
        self.assertGreater(thr, 0.0, "calibrated threshold is not frozen")
        from jadal.mosaic.s0_parity import decide_with
        rules = list(C.S0_RULES[:5]) + [("BLUR", "lapvar_cal", "<", thr)]
        low = dict(self.BASE, lapvar_cal=thr * 0.5)
        high = dict(self.BASE, lapvar_cal=thr * 2.0)
        self.assertEqual(decide_with(low, rules)[0], "BLUR")
        self.assertIsNone(decide_with(high, rules)[0])
        # and a black clip still reports BLACK, never BLUR
        black = dict(self.BASE, p95=0.0, mean=0.0, lapvar_cal=0.0)
        self.assertEqual(decide_with(black, rules)[0], "BLACK")

    def test_missing_lapvar_cal_still_decides_as_before(self):
        old = {"mean": 0.5, "p5": 0.3, "p95": 0.8, "spatial_var": 0.01,
               "temp_e": 0.01}
        self.assertIsNone(C.s0_decide(old)[0])
        frozen = dict(old, temp_e=1e-6)
        self.assertEqual(C.s0_decide(frozen)[0], "FROZEN")

    def test_blur_threshold_matches_the_committed_calibration(self):
        self.assertTrue(os.path.exists(CALIB),
                        f"calibration artifact missing: {CALIB}")
        cal = json.load(open(CALIB))
        self.assertAlmostEqual(s0_blur.BLUR_LAPVAR_CAL_MIN_CALIBRATED,
                               float(cal["selected_threshold"]), places=15)
        self.assertEqual(s0_blur.BLUR_LAPVAR_CAL_MIN, 0.0,
                         "the shipped default must stay disabled")
        self.assertEqual(cal["phase"], "calibrate")


if __name__ == "__main__":
    unittest.main()
