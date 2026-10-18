"""Unit tests for jadal/mosaic/features.py.

Synthetic only (no dataset frames): tiles generated with numpy, mosaics built with
the real builder so the sidecar/offsets are the production ones.

Run: python3 -m unittest tests.test_mosaic_features -v
"""
import json
import unittest

import numpy as np

from jadal.mosaic import build as B
from jadal.mosaic import features as F


def _tile(seed, size=64, base=None):
    rng = np.random.default_rng(seed)
    if base is None:
        base = rng.integers(0, 255, size=(size, size, 3), dtype=np.uint8)
    return base


def _mosaic(frames, tile_px=224, timestamps=True, grid=None):
    n = len(frames)
    grid = grid or B.grid_for_n(n)
    times = B.tile_times(n)
    idx = B.frame_indices(times, 10.0, 10 ** 6)
    return B.build_mosaic([np.asarray(f, np.uint8) for f in frames], times, idx, grid,
                          tile_px, window_id="unit", source="synthetic", fps_used=10.0,
                          timestamps=timestamps)


class TestNames(unittest.TestCase):
    def test_names_are_deterministic_and_grouped(self):
        a = F.feature_names()
        b = F.feature_names()
        self.assertEqual(a, b)
        self.assertEqual(len(a), len(set(a)), "duplicate feature names")
        self.assertEqual(len(a), len(F.SPATIAL_STATS) * len(F.POOLS)
                         + len(F.INTERTILE_STATS) + len(F.DIFF_STATS))

    def test_subset_groups(self):
        self.assertEqual(F.feature_names(("spatial",), ("mean",)),
                         [f"sp_{s}_mean" for s in F.SPATIAL_STATS])
        self.assertEqual(F.feature_names(("intertile",)), list(F.INTERTILE_STATS))


class TestDeterminism(unittest.TestCase):
    def test_same_canvas_gives_same_features_and_digest(self):
        frames = [_tile(i, 96) for i in range(9)]
        img, side = _mosaic(frames)
        canvas = np.asarray(img)
        a = F.feature_vector(canvas, side, diff_frames=frames)
        b = F.feature_vector(canvas, side, diff_frames=frames)
        self.assertEqual(F.feature_digest(a), F.feature_digest(b))
        self.assertEqual(json.dumps(a, sort_keys=True), json.dumps(b, sort_keys=True))
        self.assertEqual(len(a), len(F.feature_names()))

    def test_digest_changes_with_content(self):
        f1 = [_tile(i, 96) for i in range(4)]
        f2 = list(f1)
        f2[2] = np.zeros((96, 96, 3), np.uint8)
        i1, s1 = _mosaic(f1)
        i2, s2 = _mosaic(f2)
        d1 = F.feature_digest(F.feature_vector(np.asarray(i1), s1, diff_frames=f1))
        d2 = F.feature_digest(F.feature_vector(np.asarray(i2), s2, diff_frames=f2))
        self.assertNotEqual(d1, d2)


class TestTimestampMask(unittest.TestCase):
    """The brief requires proof that the burned timestamp does not leak into features."""

    def test_mask_covers_the_drawn_box(self):
        for tile_px in (112, 160, 224):
            x0, y0, x1, y1 = F._timestamp_box(tile_px)
            m = F.tile_mask(tile_px)
            self.assertFalse(m[y0:y1, x0:x1].any(), f"box not fully masked at {tile_px}")
            self.assertGreater(m.sum(), 0.5 * m.size)

    def test_timestamp_changes_nothing_measurable(self):
        rng = np.random.default_rng(7)
        frames = [rng.integers(0, 255, (96, 96, 3), dtype=np.uint8) for _ in range(4)]
        on, s_on = _mosaic(frames, tile_px=224, timestamps=True)
        off, s_off = _mosaic(frames, tile_px=224, timestamps=False)
        a = F.feature_vector(np.asarray(on), s_on, diff_frames=frames)
        b = F.feature_vector(np.asarray(off), s_off, diff_frames=frames)
        for k in a:
            if np.isnan(a[k]) or np.isnan(b[k]):
                continue
            self.assertAlmostEqual(a[k], b[k], delta=1e-9,
                                   msg=f"timestamp leaked into {k}: {a[k]} vs {b[k]}")


class TestTemporalProxies(unittest.TestCase):
    def test_identical_tiles_look_frozen_and_duplicate(self):
        same = np.full((96, 96, 3), 90, np.uint8)
        frames = [same.copy() for _ in range(4)]
        img, side = _mosaic(frames, tile_px=112)
        v = F.feature_vector(np.asarray(img), side, groups=("intertile",),
                             diff_frames=frames)
        self.assertGreaterEqual(v["it_frozen_frac"], 0.99)
        self.assertLess(v["it_mad_mean"], 1e-6)
        self.assertGreaterEqual(v["it_dup_phash_frac"], 0.99)
        self.assertGreater(v["it_zncc_min"], 0.99)

    def test_moving_tiles_are_not_frozen(self):
        rng = np.random.default_rng(3)
        frames = [rng.integers(0, 255, (96, 96, 3), dtype=np.uint8) for _ in range(4)]
        img, side = _mosaic(frames, tile_px=112)
        v = F.feature_vector(np.asarray(img), side, groups=("intertile",),
                             diff_frames=frames)
        self.assertLess(v["it_frozen_frac"], 0.5)
        self.assertGreater(v["it_mad_mean"], 0.05)

    def test_shift_proxy_finds_a_known_translation(self):
        rng = np.random.default_rng(11)
        base = rng.integers(0, 255, (96, 96), dtype=np.uint8)
        frames = []
        for dx in range(4):
            rgb = np.stack([np.roll(base, dx, axis=1)] * 3, axis=2).astype(np.uint8)
            frames.append(rgb)
        img, side = _mosaic(frames, tile_px=224)
        v = F.feature_vector(np.asarray(img), side, groups=("intertile",))
        self.assertGreater(v["it_shift_mean"], 0.2)


class TestDifferenceMosaic(unittest.TestCase):
    def test_static_sequence_has_zero_difference(self):
        f = np.full((32, 32, 3), 100, np.uint8)
        d = F.diff_mosaic([f.copy() for _ in range(4)], 32)
        self.assertEqual(int(np.abs(d).max()), 0)

    def test_diff_features_are_small_for_static_and_large_for_motion(self):
        static = [np.full((96, 96, 3), 100, np.uint8) for _ in range(4)]
        img, side = _mosaic(static, tile_px=112)
        vs = F.feature_vector(np.asarray(img), side, groups=("diff",),
                              diff_frames=static)["dm_energy_mean"]
        moving = [np.full((96, 96, 3), 0 if i % 2 else 255, np.uint8) for i in range(4)]
        img2, side2 = _mosaic(moving, tile_px=112)
        vm = F.feature_vector(np.asarray(img2), side2, groups=("diff",),
                              diff_frames=moving)["dm_energy_mean"]
        self.assertLess(vs, 1e-6)
        self.assertGreater(vm, 0.3)


class TestBlurStatistic(unittest.TestCase):
    """GRID.09: brightness-calibrated sharpness must separate blur from dimness."""

    def test_blur_tracked_independently_of_brightness(self):
        from PIL import Image, ImageFilter
        rng = np.random.default_rng(5)
        base = (rng.integers(0, 255, (96, 96, 3))).astype(np.uint8)
        img = Image.fromarray(base)
        sharp = np.asarray(img)
        blur = np.asarray(img.filter(ImageFilter.GaussianBlur(3)))
        m = np.ones((96, 96), bool)
        s_sharp = F.spatial_features(sharp, m, F.to_luma(sharp))["lapvar_cal"]
        s_blur = F.spatial_features(blur, m, F.to_luma(blur))["lapvar_cal"]
        self.assertGreater(s_sharp, s_blur * 3,
                           "brightness-calibrated sharpness does not fall under blur")

    def test_letterbox_padding_is_excluded(self):
        content = np.full((60, 120, 3), 200, np.uint8)
        tile = B.letterbox_tile(content, 112)
        arr = np.asarray(tile)
        base = F.tile_mask(112)
        m = F.letterbox_mask(arr, base)
        self.assertLess(m.sum(), base.sum(), "grey padding was not detected")
        self.assertTrue(m[56, 56], "content rows were masked out")


if __name__ == "__main__":
    unittest.main()
