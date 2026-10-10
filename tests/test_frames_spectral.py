"""Tests for the frames-spectral study (issue #60). Synthetic arrays only,
no GPU/network/data deps. Must run in < 30 s."""

import unittest

import cv2
import numpy as np


class TestFrameFeatures(unittest.TestCase):
    def test_feature_count_and_finiteness(self):
        from eval.frames_spectral.frame_features import (
            FEATURES, extract_frame_bgr)
        rng = np.random.default_rng(0)
        img = (rng.random((240, 320, 3)) * 255).astype(np.uint8)
        v = extract_frame_bgr(img)
        self.assertEqual(len(v), len(FEATURES))
        self.assertTrue(np.all(np.isfinite(v)), "all features finite")

    def test_determinism(self):
        from eval.frames_spectral.frame_features import extract_frame_bgr
        rng = np.random.default_rng(1)
        img = (rng.random((100, 200, 3)) * 255).astype(np.uint8)
        np.testing.assert_array_equal(extract_frame_bgr(img),
                                      extract_frame_bgr(img.copy()))

    def test_blur_reduces_sharpness(self):
        from eval.frames_spectral.frame_features import FEATURES, extract_frame_bgr
        rng = np.random.default_rng(2)
        img = (rng.random((160, 240, 3)) * 255).astype(np.uint8)
        blur = cv2.GaussianBlur(img, (15, 15), 0)
        a = extract_frame_bgr(img)
        b = extract_frame_bgr(blur)
        i = FEATURES.index("lap_var")
        self.assertGreater(a[i], b[i], "blur must lower lap_var")

    def test_dark_frame_exposure(self):
        from eval.frames_spectral.frame_features import FEATURES, extract_frame_bgr
        dark = np.zeros((120, 200, 3), np.uint8)
        v = extract_frame_bgr(dark)
        i = FEATURES.index("under_frac")
        self.assertGreater(v[i], 0.9, "black frame ~all under-exposed")

    def test_skin_mask_hits_skin_colour(self):
        from eval.frames_spectral.frame_features import FEATURES, extract_frame_bgr
        # BGR approx of skin tone (Cr~150, Cb~100 region)
        skin = np.zeros((120, 200, 3), np.uint8)
        skin[:] = (80, 110, 180)  # B,G,R
        v = extract_frame_bgr(skin)
        i = FEATURES.index("skin_frac")
        self.assertGreater(v[i], 0.5, "skin-coloured frame must have large mask")

    def test_noise_estimators_order(self):
        from eval.frames_spectral.frame_features import FEATURES, extract_frame_bgr
        rng = np.random.default_rng(3)
        base = np.full((120, 200, 3), 128, np.uint8)
        noisy = np.clip(base.astype(float) + rng.normal(0, 25, base.shape),
                        0, 255).astype(np.uint8)
        a = extract_frame_bgr(base)
        b = extract_frame_bgr(noisy)
        i = FEATURES.index("noise_immerkaer")
        j = FEATURES.index("noise_mad_hh1")
        self.assertGreater(b[i], a[i], "noisy frame higher immerkaer sigma")
        self.assertGreater(b[j], a[j], "noisy frame higher MAD-HH1 sigma")


class TestPoolingAndMetrics(unittest.TestCase):
    def test_welch_bands_sum(self):
        from eval.frames_spectral.window_heads import welch_bands
        rng = np.random.default_rng(4)
        s = rng.random(100)
        w = welch_bands(s, 10.0, hann=True)
        self.assertLessEqual(w["lo"] + w["mid"] + w["hi"], 1.0 + 1e-9)
        for v in w.values():
            self.assertTrue(np.isfinite(v) and v >= 0.0)

    def test_pool_series_keys(self):
        from eval.frames_spectral.window_heads import pool_series, CLEAN_DEFAULT
        s = np.linspace(0.1, 0.9, 50)
        p = pool_series(s, 10.0, CLEAN_DEFAULT)
        for k in ("mean", "median", "std", "max", "q10", "q50", "q90",
                  "frac_above", "spec_lo", "spec_mid", "spec_hi", "spec_flick"):
            self.assertIn(k, p)
        self.assertAlmostEqual(p["frac_above"], 0.5, delta=0.15)

    def test_med3_and_detrend(self):
        from eval.frames_spectral.window_heads import detrend_series, med3
        S = np.arange(20, dtype=float).reshape(-1, 1)
        D = detrend_series(S)
        self.assertTrue(np.abs(D).max() < 1e-8, "linear trend removed")
        spike = np.zeros((9, 1))
        spike[4, 0] = 10.0
        self.assertEqual(med3(spike)[4, 0], 0.0, "median-3 kills spike")


if __name__ == "__main__":
    unittest.main()
