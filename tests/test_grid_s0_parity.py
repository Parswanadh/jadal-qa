"""Unit tests for the GRID.05 mosaic S0 parity probe.

Synthetic only (numpy + PIL canvas, no dataset frames, no video, no network).
The frozen thresholds are checked against the committed calibration artifact: if
that file is missing the test FAILS rather than passing vacuously.

Run: python3 -m unittest tests.test_grid_s0_parity -v
"""
import json
import os
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from jadal.mosaic import build as B  # noqa: E402
from jadal.mosaic import features as F  # noqa: E402
from jadal.mosaic import s0_parity as P  # noqa: E402
from jadal.poc import cascade as C  # noqa: E402

CALIB = os.path.join(ROOT, "eval", "results", "exp_grid", "grid05_calibration.json")
N = 9


def build(frames, tile_px=224, timestamps=True):
    times = B.tile_times(N)
    idx = B.frame_indices(times, 10.0, 10 ** 6)
    img, side = B.build_mosaic([np.asarray(f, np.uint8) for f in frames], times, idx,
                               B.grid_for_n(N), tile_px, window_id="unit",
                               source="synthetic", fps_used=N / B.WINDOW_S,
                               timestamps=timestamps)
    return np.asarray(img), side


def noisy(seed=0, shape=(480, 480, 3)):
    return np.random.default_rng(seed).integers(0, 255, shape, dtype=np.uint8)


class TestProbeShape(unittest.TestCase):
    def test_probe_carries_the_shipped_five_keys(self):
        c, s = build([noisy(i) for i in range(N)])
        pr = P.mosaic_probe(c, s)
        for k in ("mean", "p5", "p95", "spatial_var", "temp_e", "lapvar_cal"):
            self.assertIn(k, pr)
        # a real factory-like canvas must not fire any shipped rule
        self.assertIsNone(C.s0_decide(pr)[0])
        self.assertIsNone(P.mosaic_s0_decide(pr)[0])

    def test_lapvar_cal_matches_the_feature_vector(self):
        """One statistic, two modules: the probe and sp_lapvar_cal_mean must agree."""
        c, s = build([noisy(i) for i in range(N)])
        fv = F.feature_vector(c, s, groups=("spatial", "intertile"))
        pr = P.mosaic_probe(c, s)
        self.assertAlmostEqual(pr["lapvar_cal"], fv["sp_lapvar_cal_mean"], places=12)
        self.assertAlmostEqual(pr["temp_e"], fv["it_mad_mean"], places=12)

    def test_intertile_frozen_frac_is_recorded(self):
        one = noisy(5)
        c, s = build([one.copy() for _ in range(N)])
        pr = P.mosaic_probe(c, s)
        self.assertEqual(pr["it_frozen_frac"], 1.0)
        self.assertAlmostEqual(pr["temp_e"], 0.0, places=12)


class TestRuleSemantics(unittest.TestCase):
    def test_black_is_black_not_blur(self):
        c, s = build([np.zeros((480, 480, 3), np.uint8) for _ in range(N)])
        pr = P.mosaic_probe(c, s)
        self.assertEqual(P.mosaic_s0_decide(pr)[0], "BLACK")
        self.assertEqual(C.s0_decide(pr)[0], "BLACK")

    def test_uniform_midgrey_is_lens_blocked(self):
        c, s = build([np.full((480, 480, 3), 110, np.uint8) for _ in range(N)])
        pr = P.mosaic_probe(c, s)
        self.assertLess(pr["spatial_var"], 1e-5)
        self.assertEqual(P.mosaic_s0_decide(pr)[0], "LENS_BLOCKED")

    def test_frozen_and_blur_thresholds_are_calibrated_not_disabled(self):
        self.assertGreater(P.MOSAIC_TEMP_E_MAX, 0.0, "FROZEN threshold not frozen")
        self.assertGreater(P.MOSAIC_BLUR_LAPVAR_CAL_MIN, 0.0,
                           "BLUR threshold not frozen")
        self.assertTrue(os.path.exists(CALIB),
                        f"calibration artifact missing: {CALIB}")
        cal = json.load(open(CALIB))
        self.assertAlmostEqual(P.MOSAIC_TEMP_E_MAX,
                               float(cal["selected_temp_e_max"]), places=15)
        self.assertAlmostEqual(P.MOSAIC_BLUR_LAPVAR_CAL_MIN,
                               float(cal["selected_lapvar_cal_min"]), places=15)

    def test_frozen_mosaic_fires_frozen(self):
        one = noisy(7)
        c, s = build([one.copy() for _ in range(N)])
        pr = P.mosaic_probe(c, s)
        self.assertEqual(P.mosaic_s0_decide(pr)[0], "FROZEN")

    def test_rule_order_puts_blur_last(self):
        rules = [r[0] for r in P.mosaic_s0_rules()]
        self.assertEqual(rules, ["BLACK", "TOO_DARK", "LENS_BLOCKED",
                                 "OVEREXPOSED", "FROZEN", "BLUR"])
        self.assertEqual([list(r) for r in P.INHERITED_RULES],
                         [list(r) for r in C.S0_RULES[:4]])


class TestPaddingLeak(unittest.TestCase):
    """Regression test for the bug GRID.05 found and fixed.

    On a non-square source the builder letterboxes with mid-grey.  S1's
    ``letterbox_mask`` drops a row only when >=98 % of it is mid-grey, but the
    burned timestamp box covers the top-right of the padding, so those rows
    survive and their ~50 % luma leaks into the pooled percentiles.  Measured on
    the 640x480 all-black demo-kit clip: p95 0.5020 (TOO_DARK) with the leak,
    0.0000 (BLACK) without it -- the shipped probe reads BLACK.
    """

    def setUp(self):
        self.c, self.s = build([np.zeros((480, 640, 3), np.uint8) for _ in range(N)])

    def test_features_pooling_leaks(self):
        pr = P.mosaic_probe(self.c, self.s, padding_detect="features")
        self.assertGreater(pr["bg_leak_frac"], 0.02)
        self.assertGreater(pr["p95"], 0.03)          # padding is now the p95
        self.assertEqual(P.mosaic_s0_decide(pr)[0], "TOO_DARK")

    def test_strict_pooling_is_clean(self):
        pr = P.mosaic_probe(self.c, self.s, padding_detect="strict")
        self.assertLess(pr["bg_leak_frac"], 0.005)
        self.assertLess(pr["p95"], 0.03)
        self.assertEqual(P.mosaic_s0_decide(pr)[0], "BLACK")

    def test_strict_and_features_agree_on_a_square_black_canvas(self):
        c, s = build([np.zeros((480, 480, 3), np.uint8) for _ in range(N)])
        a = P.mosaic_probe(c, s, padding_detect="strict")
        b = P.mosaic_probe(c, s, padding_detect="features")
        self.assertEqual(a["p95"], b["p95"])
        self.assertLess(a["bg_leak_frac"], 0.005)

    def test_square_sources_are_unaffected(self):
        """On a square source there is no padding, so the two poolings must agree
        EXACTLY on every shared statistic.  Regression guard: an earlier version
        of the strict detector replaced the timestamp mask instead of extending
        it, which put the burned white text into `lapvar_cal` and into every
        pooled percentile while the black-frame test above still passed.

        `bg_leak_frac` is a luma-proximity diagnostic, not a padding detector, so
        it is NOT near zero for arbitrary content (random noise has ~8 % of its
        luma within +-2/255 of mid-grey).  The meaningful assertion is that the
        two poolings see identical pixels here.
        """
        c, s = build([noisy(i) for i in range(N)])
        a = P.mosaic_probe(c, s, padding_detect="strict")
        b = P.mosaic_probe(c, s, padding_detect="features")
        for k in ("mean", "p5", "p95", "spatial_var", "temp_e", "lapvar_cal",
                  "bg_leak_frac", "n_pooled_px"):
            self.assertAlmostEqual(a[k], b[k], places=12,
                                   msg=f"{k} differs: {a[k]} vs {b[k]}")
        # and the strict probe still equals the S1 feature vector's tile statistic
        fv = F.feature_vector(c, s, groups=("spatial",))
        self.assertAlmostEqual(a["lapvar_cal"], fv["sp_lapvar_cal_mean"], places=12)

    def test_unknown_pooling_is_rejected(self):
        with self.assertRaises(ValueError):
            P.pooled_stats(self.c, self.s, padding_detect="nonsense")


class TestDecideWith(unittest.TestCase):
    def test_explicit_table_is_honoured(self):
        pr = {"mean": 0.5, "p5": 0.3, "p95": 0.8, "spatial_var": 0.01,
              "temp_e": 0.01, "lapvar_cal": 0.5}
        self.assertIsNone(P.decide_with(pr, P.INHERITED_RULES)[0])
        tight = list(P.INHERITED_RULES) + [("BLUR", "lapvar_cal", "<", 1.0)]
        self.assertEqual(P.decide_with(pr, tight)[0], "BLUR")

    def test_missing_keys_are_skipped(self):
        self.assertIsNone(P.mosaic_s0_decide({})[0])


if __name__ == "__main__":
    unittest.main()
