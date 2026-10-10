"""Synthetic-input tests for the E1 fps / frame-selection helpers.

Covers the subsampling index rule {10:1,5:2,3.33:3,2:5,1:10}, the keyframe
selector (fallback must be explicit, off-grid I-frames must change the set),
ffprobe output parsing, and the motion/random selectors. No video, GPU or
network is used.

Run (repo root): PYTHONNOUSERSITE=1 /home/parshu/envs/jadal/bin/python -m unittest discover -s tests
"""

import os
import sys
import unittest

import numpy as np

_EXP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                    "eval", "exp_fps")
sys.path.insert(0, _EXP)

from fps_lib import (FPS_STEPS, NATIVE_FPS, SPATIAL_FEATURES,  # noqa: E402
                      SPAN_DEPENDENT_FEATURES, clip_bootstrap_ci, loco_proba,
                      subsample_idx)
import run_fps  # noqa: E402


class SubsampleRuleTests(unittest.TestCase):
    def test_step_table(self):
        self.assertEqual(FPS_STEPS, {10: 1, 5: 2, 3.33: 3, 2: 5, 1: 10})

    def test_lengths_for_100_native_frames(self):
        expect = {10: 100, 5: 50, 3.33: 34, 2: 20, 1: 10}
        for fps, step in FPS_STEPS.items():
            self.assertEqual(len(subsample_idx(100, step)), expect[fps], fps)

    def test_indices_are_arange(self):
        for step in FPS_STEPS.values():
            np.testing.assert_array_equal(subsample_idx(100, step),
                                          np.arange(0, 100, step))

    def test_effective_fps_matches_key(self):
        for fps, step in FPS_STEPS.items():
            self.assertAlmostEqual(NATIVE_FPS / step, fps, delta=0.01)

    def test_short_window(self):
        np.testing.assert_array_equal(subsample_idx(7, 5), np.array([0, 5]))

    def test_spatial_list_is_26_minus_dt_dependent(self):
        dt_dep = {"temp_b0", "temp_b1", "temp_b2", "temp_b3", "frozen_frac",
                  "flicker_idx", "motion_energy", "rppg_pulse"}
        self.assertEqual(len(SPATIAL_FEATURES), 18)
        self.assertTrue(dt_dep.isdisjoint(SPATIAL_FEATURES))


class KeyframeSelectTests(unittest.TestCase):
    def test_too_few_iframes_is_explicit_fallback(self):
        idx, n, fb = run_fps.keyframe_select([5.0, 7.3], 5.0, 100, budget=20)
        self.assertIsNone(idx)
        self.assertEqual(n, 2)
        self.assertTrue(fb)

    def test_offgrid_iframes_give_non_uniform_set(self):
        start = 5.0
        # 24 I-frames at 0.4 s spacing, offset by 0.05 s (off the 10 fps grid);
        # all land inside the 10 s window at distinct decoded indices
        times = [start + 0.05 + 0.4 * k for k in range(24)]
        idx, n, fb = run_fps.keyframe_select(times, start, 100, budget=20)
        self.assertFalse(fb)
        self.assertEqual(n, 24)
        self.assertEqual(len(idx), 20)
        self.assertEqual(len(set(idx.tolist())), 20)
        pos = {int(round((t - start) * NATIVE_FPS)) for t in times}
        self.assertTrue(set(idx.tolist()) <= pos)
        uniform20 = subsample_idx(100, 5)
        self.assertFalse(np.array_equal(idx, uniform20))

    def test_iframes_outside_window_are_dropped(self):
        idx, n, fb = run_fps.keyframe_select([-1.0, 50.0], 5.0, 100, budget=1)
        self.assertEqual(n, 0)
        self.assertTrue(fb)


class ParseFfprobeTests(unittest.TestCase):
    def test_keeps_iframes_inside_window_only(self):
        out = "-1.0,I\n4.6,I\n5.0,P\n9.9,I\n10.0,I\n12.0,I\n"
        ts = run_fps.parse_ffprobe_iframes(out, 0.0, win_s=10.0)
        self.assertEqual(ts, [4.6, 9.9])

    def test_malformed_timestamp_raises(self):
        with self.assertRaises(ValueError):
            run_fps.parse_ffprobe_iframes("N/A,I\n", 0.0)


class MotionRandomSelectTests(unittest.TestCase):
    def test_motion_select_picks_high_energy_diffs(self):
        T = 100
        luma = np.zeros((T, 2, 2))
        for i in (3, 40, 77):  # jump between frame i and i+1
            luma[i + 1:] += 1.0
        keep = run_fps.motion_select(luma, budget=20)
        self.assertEqual(len(keep), 20)
        self.assertEqual(len(set(keep.tolist())), 20)
        self.assertEqual(keep[0], 0)
        self.assertTrue(np.all(np.diff(keep) > 0))
        self.assertTrue({4, 41, 78} <= set(keep.tolist()))
        self.assertTrue(keep.max() < T)

    def test_random_select_seeded(self):
        a = run_fps.random_select(100, 20, seed=0)
        b = run_fps.random_select(100, 20, seed=0)
        c = run_fps.random_select(100, 20, seed=1)
        np.testing.assert_array_equal(a, b)
        self.assertFalse(np.array_equal(a, c))
        self.assertEqual(len(set(a.tolist())), 20)
        self.assertTrue(np.all(np.diff(a) > 0))
        self.assertTrue(a.min() >= 0 and a.max() < 100)



class LocoAndBootstrapTests(unittest.TestCase):
    """Synthetic inputs only (no files). loco_proba / clip_bootstrap_ci."""

    def test_span_dependent_are_spatial_subset(self):
        self.assertEqual(sorted(SPAN_DEPENDENT_FEATURES),
                         ["dup_hash_mean", "global_shift", "zncc_self"])
        self.assertTrue(set(SPAN_DEPENDENT_FEATURES) <= set(SPATIAL_FEATURES))

    def test_loco_separable_four_clips_auroc_above_0_9(self):
        from sklearn.metrics import roc_auc_score
        rng = np.random.default_rng(0)
        clips = np.repeat(np.array(["c0", "c1", "c2", "c3"]), 30)
        y = np.tile(np.r_[np.ones(15), np.zeros(15)], 4).astype(int)
        X = y[:, None] * 4.0 + rng.normal(0, 0.5, size=(120, 2))
        p = loco_proba(X, y, clips, C=0.3)
        self.assertEqual(p.shape, (120,))
        self.assertTrue(np.all((p >= 0) & (p <= 1)))
        self.assertGreater(roc_auc_score(y, p), 0.9)

    def test_loco_random_labels_ci_contains_half(self):
        rng = np.random.default_rng(1)
        clips = np.repeat(np.array([f"c{i}" for i in range(12)]), 20)
        y = rng.integers(0, 2, size=240)
        X = rng.normal(size=(240, 3))
        p = loco_proba(X, y, clips, C=0.3)
        lo, hi = clip_bootstrap_ci(y, p, clips, B=300, seed=0)
        self.assertLessEqual(lo, 0.5)
        self.assertGreaterEqual(hi, 0.5)
        self.assertLess(lo, hi)

    def test_ci_deterministic_for_fixed_seed(self):
        rng = np.random.default_rng(2)
        clips = np.repeat(np.array([f"c{i}" for i in range(6)]), 15)
        y = rng.integers(0, 2, size=90)
        p = rng.random(90)
        a = clip_bootstrap_ci(y, p, clips, B=200, seed=7)
        b = clip_bootstrap_ci(y, p, clips, B=200, seed=7)
        c = clip_bootstrap_ci(y, p, clips, B=200, seed=8)
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)

    def test_single_class_resamples_are_skipped(self):
        # Two clips: "a" is all-positive, "b" is mixed. A resample that draws
        # "a" twice has a single class, which roc_auc_score rejects with
        # ValueError (P=1/4 per draw, so 200 draws always hit it). It must be
        # dropped, not raise. Perfect scores make every kept draw AUROC 1.
        clips = np.repeat(np.array(["a", "b"]), 10)
        y = np.r_[np.ones(10), np.tile(np.r_[np.ones(5), np.zeros(5)], 1)].astype(int)
        p = y.astype(float)
        lo, hi = clip_bootstrap_ci(y, p, clips, B=200, seed=0)
        self.assertEqual((lo, hi), (1.0, 1.0))


if __name__ == "__main__":
    unittest.main()
