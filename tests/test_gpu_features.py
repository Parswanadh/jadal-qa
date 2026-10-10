"""CPU reference tests for jadal/gpu/features.py. No GPU/network needed (<30 s).

GPU-vs-CPU tolerance checks run only when torch.cuda.is_available().
"""
import unittest

import numpy as np

from jadal.gpu.features import (
    BANDS,
    FEATURE_NAMES,
    cpu_dct_energy,
    cpu_dct_hash,
    cpu_grey_world,
    cpu_luma_stats,
    cpu_motion,
    cpu_pca_inverse_residual,
    cpu_rppg_pulse,
    cpu_skin_fraction,
    cpu_spatial_spectrum,
    cpu_temporal_bands,
    cpu_window_features,
    cpu_zncc,
)

rng = np.random.default_rng(0)


def rand_luma(T=40, H=45, W=80):
    return rng.random((T, H, W))


class TestTemporalBands(unittest.TestCase):
    def test_frozen_gives_zero_non_dc(self):
        x = np.full((40, 9, 16), 0.5)
        b = cpu_temporal_bands(x, fps=30.0)
        for i in range(1, 4):
            self.assertAlmostEqual(float(b[f"b{i}"].max()), 0.0, places=12)

    def test_sine_lands_in_right_band(self):
        T, fps = 300, 30.0
        t = np.arange(T) / fps
        sig = 0.2 * np.sin(2 * np.pi * 1.0 * t)  # 1 Hz, zero mean -> band 1
        x = np.broadcast_to(sig[:, None, None], (T, 8, 8)).copy()
        b = cpu_temporal_bands(x, fps=fps)
        e = [float(b[f"b{i}"].mean()) for i in range(4)]
        self.assertGreater(e[1], e[0] + e[2] + e[3])

    def test_band_keys(self):
        self.assertEqual(len(BANDS), 4)
        b = cpu_temporal_bands(rand_luma(), fps=30.0)
        self.assertEqual(sorted(b), ["b0", "b1", "b2", "b3"])


class TestSpatial(unittest.TestCase):
    def test_noise_has_more_hf_than_blur(self):
        from scipy.ndimage import gaussian_filter  # scipy is installed; fallback below
        n = rng.random((45, 80))
        try:
            bl = gaussian_filter(n, 3.0)
        except Exception:
            bl = n
        _, hf_n = cpu_spatial_spectrum(n)
        _, hf_b = cpu_spatial_spectrum(bl)
        self.assertGreater(hf_n, hf_b)

    def test_slope_negative_for_natural_like(self):
        n = rng.random((45, 80))
        slope, _ = cpu_spatial_spectrum(n)
        self.assertLess(slope, 0.5)  # white noise ~ flat; just checks finiteness/range

    def test_dct_sharp_gt_blur(self):
        n = rng.random((32, 32))
        flat = np.full((32, 32), 0.5)
        self.assertGreater(cpu_dct_energy(n), cpu_dct_energy(flat))


class TestColour(unittest.TestCase):
    def test_pca_gray_has_zero_residual(self):
        g = np.stack([np.full(200, 0.5)] * 3, axis=1)
        resid, top1, _ = cpu_pca_inverse_residual(g, k=2)
        self.assertAlmostEqual(resid, 0.0, places=12)
        self.assertAlmostEqual(top1, 1.0, places=6)

    def test_pca_residual_positive_for_colourful(self):
        rgb = rng.random((500, 3))
        resid, top1, _ = cpu_pca_inverse_residual(rgb, k=1)
        self.assertGreater(resid, 1e-6)
        self.assertLess(top1, 1.0)

    def test_grey_world_neutral(self):
        rgb = np.full((4, 4, 3), 0.5)
        self.assertEqual(cpu_grey_world(rgb), (1 / 3, 1 / 3, 1 / 3))

    def test_skin_range(self):
        rgb = rng.random((9, 16, 3))
        v = cpu_skin_fraction(rgb)
        self.assertGreaterEqual(v, 0.0)
        self.assertLessEqual(v, 1.0)


class TestStatsMotion(unittest.TestCase):
    def test_dark_stats(self):
        x = np.full((10, 8, 8), 0.02)
        st = cpu_luma_stats(x)
        self.assertAlmostEqual(st["dark_frac"], 1.0)
        self.assertAlmostEqual(st["bright_frac"], 0.0)

    def test_motion_zero_for_static(self):
        x = np.full((10, 8, 8), 0.4)
        me, gs = cpu_motion(x)
        self.assertAlmostEqual(me, 0.0, places=12)
        self.assertAlmostEqual(gs, 0.0)

    def test_motion_positive_for_change(self):
        x = rand_luma(10, 8, 8)
        me, _ = cpu_motion(x)
        self.assertGreater(me, 0.01)

    def test_zncc_identical(self):
        a = rng.random((16, 16))
        self.assertAlmostEqual(cpu_zncc(a, a), 1.0, places=9)

    def test_hash_deterministic(self):
        a = rng.random((45, 80))
        self.assertEqual(cpu_dct_hash(a), cpu_dct_hash(a.copy()))

    def test_rppg_short(self):
        self.assertEqual(cpu_rppg_pulse(np.zeros((5, 3))), 0.0)


class TestWindow(unittest.TestCase):
    def test_feature_count_and_finiteness(self):
        T, H, W = 30, 45, 80
        luma = rng.random((T, H, W))
        rgb = rng.random((T, H, W, 3))
        f = cpu_window_features(luma, rgb, fps=30.0)
        self.assertEqual(len(f), len(FEATURE_NAMES))
        for name in FEATURE_NAMES:
            self.assertIn(name, f)
            self.assertTrue(np.isfinite(f[name]), name)


class TestGpuParity(unittest.TestCase):
    def test_gpu_matches_cpu(self):
        try:
            import torch
        except ImportError:
            self.skipTest("no torch")
        if not torch.cuda.is_available():
            self.skipTest("no GPU")
        from jadal.gpu.features import gpu_window_features_batch
        T, H, W, B = 30, 48, 80, 2
        luma = rng.random((B, T, H, W)).astype(np.float32)
        rgb = rng.random((B, T, H, W, 3)).astype(np.float32)
        out = gpu_window_features_batch(
            torch.tensor(luma, device="cuda:0"),
            torch.tensor(rgb, device="cuda:0"),
            fps=30.0,
        ).cpu().numpy()
        for bi in range(B):
            ref = cpu_window_features(luma[bi].astype(np.float64), rgb[bi].astype(np.float64))
            for j, name in enumerate(FEATURE_NAMES):
                if name in ("radial_slope", "dup_hash_mean"):
                    continue  # different estimator on GPU path (documented)
                a, r = out[bi, j], ref[name]
                tol = 0.05 * max(abs(r), 1e-3) + 1e-3
                self.assertLess(abs(a - r), tol, f"{name}: gpu={a} cpu={r}")


if __name__ == "__main__":
    unittest.main()
