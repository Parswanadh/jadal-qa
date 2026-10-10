"""Unit tests for ``jadal.rhythm.features`` using synthetic arrays only.

Every input here is generated in-process: seeded ``numpy`` RNG arrays plus
``cv2`` filters.  No real video is decoded, no GPU is touched, nothing goes to
the network, and the largest array is 12 x 90 x 160, so the whole module runs
in a fraction of a second on CPU.

Run from the worktree root so that ``jadal`` is importable::

    python -m unittest discover -s tests
    python -m unittest tests.test_rhythm_features -v

Three tests are explicit regression guards for a bug this suite found: the
``cv2.phaseCorrelate`` in-place mutation described in their docstrings.

``@unittest.expectedFailure`` marks behaviour that is currently wrong in
``features.py`` (all of it traces back to one root cause: ``cv2.phaseCorrelate``
mutates its inputs in place, so ``grid_channels`` destroys the frames it is
still working on).  Those tests are the receipt for the bug, not the spec.
"""
from __future__ import annotations

import unittest

import cv2
import numpy as np

from jadal.rhythm import features as F

# ---- shared synthetic fixtures -------------------------------------------- #
T_FRAMES = 12
H, W = F.ANALYSIS_H, F.ANALYSIS_W  # 90 x 160

# channel axis positions, taken from the module's own declared layout
RES_MEAN = F.CHANNEL_NAMES.index("res_mean")
RES_P90 = F.CHANNEL_NAMES.index("res_p90")
SKIN_FRAC = F.CHANNEL_NAMES.index("skin_frac")
TVAR = F.CHANNEL_NAMES.index("tvar")
FLOW_MAG = F.CHANNEL_NAMES.index("flow_mag")
FLOW_COH = F.CHANNEL_NAMES.index("flow_coh")


def _synthetic_stack(seed: int = 0, T: int = T_FRAMES) -> tuple[np.ndarray, np.ndarray]:
    """A (T, 90, 160) float32 gray stack and a matching (T, 90, 160, 3) BGR stack."""
    rng = np.random.default_rng(seed)
    gray = rng.integers(0, 256, (T, H, W)).astype(np.float32)
    color = rng.integers(0, 256, (T, H, W, 3)).astype(np.uint8)
    return gray, color


def _translated_stack(dx: int = 3, dy: int = 2, seed: int = 7,
                      T: int = T_FRAMES) -> tuple[np.ndarray, np.ndarray]:
    """Texture translating by (dx, dy) px per frame.

    ``np.roll`` is a *circular* shift, which is exactly the model phase
    correlation fits, so the true global translation is known to the pixel.
    """
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 256, (H, W)).astype(np.float32)
    gray = np.stack([np.roll(np.roll(base, dx * t, axis=1), dy * t, axis=0)
                     for t in range(T)]).astype(np.float32)
    color = np.repeat(np.clip(gray, 0, 255).astype(np.uint8)[..., None], 3, axis=3)
    return gray, color


def _window_bounds(fps: float, start_s: float, end_s: float) -> tuple[int, int]:
    """The inclusive frame-index bounds ``sample_indices`` promises."""
    return int(np.floor(start_s * fps)), max(int(np.ceil(end_s * fps)) - 1, 0)


class SampleIndicesTests(unittest.TestCase):
    """``sample_indices``: a uniform time grid inside [start_s, end_s)."""

    def test_ten_second_window_at_10fps_gives_100_increasing_indices_in_bounds(self) -> None:
        idx, src_fps = F.sample_indices(10.0, 0.0, 10.0, 10.0)

        self.assertEqual(src_fps, 10.0)
        self.assertEqual(len(idx), 100)  # 10 s * 10 fps
        self.assertTrue(np.issubdtype(idx.dtype, np.integer))
        self.assertTrue(np.all(np.diff(idx) > 0), f"not strictly increasing: {idx}")

        lo, hi = _window_bounds(10.0, 0.0, 10.0)
        self.assertGreaterEqual(int(idx.min()), lo)
        self.assertLessEqual(int(idx.max()), hi)
        self.assertEqual((int(idx.min()), int(idx.max())), (lo, hi))  # covers the window

    def test_ten_second_window_at_30fps_steps_by_about_three(self) -> None:
        idx, _ = F.sample_indices(30.0, 0.0, 10.0, 10.0)

        self.assertTrue(np.all(np.diff(idx) > 0), f"not strictly increasing: {idx}")
        self.assertGreaterEqual(len(idx), 95)  # ~100 samples, one per 0.1 s
        self.assertLessEqual(len(idx), 100)

        steps = np.diff(idx)
        self.assertTrue(np.all(np.abs(steps - 3) <= 1), f"steps not ~3: {steps}")
        self.assertAlmostEqual(float(steps.mean()), 3.0, delta=0.1)

        lo, hi = _window_bounds(30.0, 0.0, 10.0)
        self.assertGreaterEqual(int(idx.min()), lo)
        self.assertLessEqual(int(idx.max()), hi)

    def test_one_second_window_still_gives_at_least_two_indices(self) -> None:
        for fps in (10.0, 30.0):
            with self.subTest(fps=fps):
                idx, _ = F.sample_indices(fps, 0.0, 1.0, 10.0)
                self.assertGreaterEqual(len(idx), 2)
                self.assertTrue(np.all(np.diff(idx) > 0), f"not strictly increasing: {idx}")
                lo, hi = _window_bounds(fps, 0.0, 1.0)
                self.assertGreaterEqual(int(idx.min()), lo)
                self.assertLessEqual(int(idx.max()), hi)

    def test_indices_stay_inside_a_shifted_window(self) -> None:
        idx, _ = F.sample_indices(30.0, 2.5, 12.5, 10.0)
        lo, hi = _window_bounds(30.0, 2.5, 12.5)
        self.assertEqual(len(idx), 100)
        self.assertGreaterEqual(int(idx.min()), lo)
        self.assertLessEqual(int(idx.max()), hi)
        self.assertTrue(np.all(np.diff(idx) > 0))

    def test_is_deterministic(self) -> None:
        first, first_fps = F.sample_indices(30.0, 2.5, 12.5, 10.0)
        second, second_fps = F.sample_indices(30.0, 2.5, 12.5, 10.0)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(first_fps, second_fps)

    def test_non_positive_fps_is_rejected(self) -> None:
        for fps in (0.0, -1.0):
            with self.subTest(fps=fps), self.assertRaises(ValueError):
                F.sample_indices(fps, 0.0, 10.0)


class DctHfTests(unittest.TestCase):
    """``_dct_hf``: high-frequency DCT energy fraction."""

    def test_flat_image_has_near_zero_high_frequency_energy(self) -> None:
        for level in (0.0, 128.0, 255.0):
            with self.subTest(level=level):
                flat = np.full((H, W), level, np.float32)
                self.assertLess(F._dct_hf(flat), 1e-4)

    def test_noise_scores_far_above_blurred_noise(self) -> None:
        rng = np.random.default_rng(0)
        noise = rng.integers(0, 256, (H, W)).astype(np.float32)
        blurred = cv2.GaussianBlur(noise, (0, 0), 3.0)

        hf_noise = F._dct_hf(noise)
        hf_blurred = F._dct_hf(blurred)

        self.assertGreater(hf_noise, 0.1)          # noise really is high frequency
        self.assertLess(hf_blurred, 0.1)           # blurring really removes it
        self.assertGreater(hf_noise, 4.0 * hf_blurred)


class GridChannelsTests(unittest.TestCase):
    """``grid_channels``: shapes, ranges, determinism and the ego-motion ablation."""

    def test_shapes_and_finiteness_for_grid_sizes(self) -> None:
        gray, color = _synthetic_stack(seed=0)
        for G in (1, 4, 6):
            with self.subTest(G=G):
                # fresh copies per call: grid_channels currently mutates its input
                # (see test_grid_channels_does_not_mutate_its_inputs).
                X, shifts = F.grid_channels(gray.copy(), color.copy(), G,
                                            ego_compensate=True, use_flow=True)
                self.assertEqual(X.shape, (T_FRAMES, G, G, F.GRID_CHANNELS))
                self.assertEqual(X.shape[-1], len(F.CHANNEL_NAMES))
                self.assertEqual(X.dtype, np.float32)
                self.assertTrue(np.isfinite(X).all())

                self.assertEqual(shifts.shape, (T_FRAMES,))
                self.assertTrue(np.isfinite(shifts).all())
                self.assertGreaterEqual(float(shifts.min()), 0.0)
                self.assertEqual(float(shifts[0]), 0.0)  # no previous frame to align to

    def test_channel_values_respect_their_documented_ranges(self) -> None:
        gray, color = _synthetic_stack(seed=1)
        for G in (1, 4, 6):
            with self.subTest(G=G):
                X, _ = F.grid_channels(gray.copy(), color.copy(), G,
                                       ego_compensate=True, use_flow=True)

                for name in ("res_mean", "res_p90", "skin_frac"):
                    c = X[..., F.CHANNEL_NAMES.index(name)]
                    self.assertGreaterEqual(float(c.min()), 0.0, name)
                    self.assertLessEqual(float(c.max()), 1.0, name)

                self.assertGreaterEqual(float(X[..., TVAR].min()), 0.0)
                self.assertGreaterEqual(float(X[..., FLOW_MAG].min()), 0.0)
                self.assertLessEqual(float(X[..., FLOW_MAG].max()), 1.0)

                coh = X[..., FLOW_COH]
                self.assertGreaterEqual(float(coh.min()), 0.0)
                self.assertLessEqual(float(coh.max()), 1.0 + 1e-5)
                self.assertGreater(float(coh.max()), 0.1)  # channel is not all zeros

    def test_deterministic_given_pristine_copies(self) -> None:
        """Same *data* in, same bytes out -- as long as the caller passes copies."""
        gray, color = _synthetic_stack(seed=2)
        first, first_shifts = F.grid_channels(gray.copy(), color.copy(), 4,
                                              ego_compensate=True, use_flow=True)
        second, second_shifts = F.grid_channels(gray.copy(), color.copy(), 4,
                                                ego_compensate=True, use_flow=True)
        self.assertEqual(first.tobytes(), second.tobytes())
        np.testing.assert_array_equal(first_shifts, second_shifts)

    def test_grid_channels_deterministic_on_the_same_array(self) -> None:
        """Calling it twice on the same input must give byte-identical output.

        REGRESSION GUARD. ``cv2.phaseCorrelate`` overwrites both of its inputs
        in place on OpenCV 5.0.0, so if ``_phase_shift`` ever stops copying, the
        second call runs on an already-corrupted stack and returns a different
        tensor -- breaking the documented "same input file -> same tensor bytes"
        / receipt-digest contract. This test caught that bug for real.
        """
        gray, color = _synthetic_stack(seed=3)
        first, _ = F.grid_channels(gray, color, 4, ego_compensate=True, use_flow=False)
        second, _ = F.grid_channels(gray, color, 4, ego_compensate=True, use_flow=False)
        self.assertEqual(first.tobytes(), second.tobytes())

    def test_grid_channels_does_not_mutate_its_inputs(self) -> None:
        """A read-only feature extractor must not modify the arrays it is given.

        REGRESSION GUARD. ``cv2.phaseCorrelate`` writes into both source frames,
        so without the copies in ``_phase_shift`` this function silently destroys
        the caller's gray stack -- and its own frames, which it reuses for the
        residual, the Farneback flow and, via ``extract_window``, the global
        channels.
        """
        gray, color = _synthetic_stack(seed=4)
        gray_before = gray.copy()
        color_before = color.copy()

        F.grid_channels(gray, color, 4, ego_compensate=True, use_flow=False)

        self.assertTrue(np.array_equal(gray, gray_before),
                        "grid_channels modified the caller's gray stack in place")
        self.assertTrue(np.array_equal(color, color_before),
                        "grid_channels modified the caller's BGR stack in place")

    def test_res_mean_channel_matches_its_documented_definition(self) -> None:
        """Channel 0 is documented as mean |I_t - warp(I_{t-1})| / 255.

        REGRESSION GUARD. If ``_phase_shift`` ever stops copying, the residual
        is formed from frames it has already overwritten and res_mean comes out
        at roughly half the documented value (measured 0.126 vs 0.287 before the
        fix).
        """
        gray, color = _synthetic_stack(seed=5)
        X, _ = F.grid_channels(gray.copy(), color.copy(), 6,
                               ego_compensate=True, use_flow=False)

        reference = np.zeros((T_FRAMES, 6, 6), np.float32)
        for t in range(1, T_FRAMES):
            # copies: _phase_shift mutates whatever it is handed
            dx, dy = F._phase_shift(gray[t - 1].copy(), gray[t].copy())
            warped = F._warp_translate(gray[t - 1], dx, dy)
            reference[t] = F._grid_mean(np.abs(gray[t] - warped), 6) / 255.0

        self.assertGreater(float(reference[1:].mean()), 0.05)  # not a vacuous check
        np.testing.assert_allclose(X[..., RES_MEAN], reference, atol=1e-6)

    def test_ego_compensate_flag_is_wired_through(self) -> None:
        """With a known global translation the flag must change the tensor."""
        gray, color = _translated_stack(dx=3, dy=2, seed=7)
        G = 4
        # Fresh copies for each call, otherwise the comparison would only be
        # measuring the in-place corruption documented above.
        compensated, _ = F.grid_channels(gray.copy(), color.copy(), G,
                                         ego_compensate=True, use_flow=True)
        uncompensated, _ = F.grid_channels(gray.copy(), color.copy(), G,
                                           ego_compensate=False, use_flow=True)

        self.assertFalse(np.allclose(compensated, uncompensated),
                         "ego_compensate=True and False produced identical tensors")
        # ... and it must be wired the right way round: removing the global
        # translation has to leave less residual motion, not more.
        self.assertLess(float(compensated[1:, ..., RES_MEAN].mean()),
                        float(uncompensated[1:, ..., RES_MEAN].mean()))


class ResampleTTests(unittest.TestCase):
    """``resample_T``: linear interpolation along the time axis."""

    def test_upsample_returns_requested_shape(self) -> None:
        a = np.arange(7 * 4 * 4 * 2, dtype=np.float32).reshape(7, 4, 4, 2)
        out = F.resample_T(a, 13)
        self.assertEqual(out.shape, (13, 4, 4, 2))
        self.assertEqual(out.dtype, np.float32)

    def test_same_length_is_the_identity(self) -> None:
        a = np.arange(7 * 4 * 4 * 2, dtype=np.float32).reshape(7, 4, 4, 2)
        np.testing.assert_array_equal(F.resample_T(a, 7), a)

    def test_linear_ramp_in_time_is_reproduced(self) -> None:
        ramp = np.arange(7, dtype=np.float32).reshape(7, 1, 1, 1)
        for T_out in (4, 13, 100):
            with self.subTest(T_out=T_out):
                out = F.resample_T(ramp, T_out).reshape(-1)
                expected = np.linspace(0.0, 6.0, T_out)
                np.testing.assert_allclose(out, expected, atol=1e-4)
        self.assertAlmostEqual(float(F.resample_T(ramp, 4)[0, 0, 0, 0]), 0.0, delta=1e-4)
        self.assertAlmostEqual(float(F.resample_T(ramp, 4)[-1, 0, 0, 0]), 6.0, delta=1e-4)


class TensorDigestTests(unittest.TestCase):
    """``tensor_digest``: the receipt digest, quantised to a 1e-4 grid."""

    @staticmethod
    def _fixture() -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(0)
        # three-decimal values, i.e. far away from the 5e-5 rounding boundaries,
        # so a +1e-6 perturbation provably cannot cross a quantisation step.
        X = np.round(rng.random((4, 3)), 3).astype(np.float32)
        g = np.round(rng.random((4, 2)), 3).astype(np.float32)
        return X, g

    def test_stable_across_calls_and_hex_sha256(self) -> None:
        X, g = self._fixture()
        first = F.tensor_digest(X, g)
        self.assertEqual(first, F.tensor_digest(X, g))
        self.assertEqual(len(first), 64)
        self.assertTrue(all(c in "0123456789abcdef" for c in first))

    def test_changes_when_an_element_changes(self) -> None:
        X, g = self._fixture()
        base = F.tensor_digest(X, g)

        X_changed = X.copy()
        X_changed[1, 2] += 0.5
        self.assertNotEqual(F.tensor_digest(X_changed, g), base)

        g_changed = g.copy()
        g_changed[0, 0] += 0.5
        self.assertNotEqual(F.tensor_digest(X, g_changed), base)

    def test_ignores_changes_below_the_quantisation_grid(self) -> None:
        X, g = self._fixture()
        base = F.tensor_digest(X, g)
        perturbed = (X + np.float32(1e-6), g + np.float32(1e-6))

        # the perturbation really is below the grid: rounding erases it ...
        np.testing.assert_array_equal(np.round(perturbed[0], 4), np.round(X, 4))
        np.testing.assert_array_equal(np.round(perturbed[1], 4), np.round(g, 4))
        # ... so the receipt must be unchanged.
        self.assertEqual(F.tensor_digest(*perturbed), base)

    def test_changes_above_the_quantisation_grid(self) -> None:
        X, g = self._fixture()
        base = F.tensor_digest(X, g)
        X_bigger = X.copy()
        X_bigger[0, 0] += 1e-3  # one whole grid step
        self.assertNotEqual(F.tensor_digest(X_bigger, g), base)


if __name__ == "__main__":
    unittest.main()
