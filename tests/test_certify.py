"""Tests for jadal/certify.py (L2: calibration, exact bounds, batch rule)."""
from __future__ import annotations

import unittest

import numpy as np

from jadal import certify


class ExactBinomialUpperTest(unittest.TestCase):
    """One-sided Clopper-Pearson upper bound on the error rate."""

    def test_zero_errors_n_100_delta_point_one_is_about_0_0228(self):
        # known value: 1 - 0.1 ** (1/100) = 0.022763
        self.assertAlmostEqual(certify.exact_binomial_upper(0, 100, 0.1), 0.0228, places=4)

    def test_zero_errors_uses_rule_of_three(self):
        # 1 - delta ** (1/n)
        for n, delta in [(1, 0.1), (10, 0.1), (10, 0.05), (250, 0.1)]:
            expected = 1.0 - delta ** (1.0 / n)
            self.assertAlmostEqual(
                certify.exact_binomial_upper(0, n, delta), expected, places=10
            )

    def test_known_values_against_clopper_pearson(self):
        cases = [
            (0, 1, 0.1, 0.9),
            (0, 10, 0.1, 0.205672),
            (1, 10, 0.1, 0.336848),
            (5, 100, 0.1, 0.090771),
            (50, 100, 0.1, 0.568583),
            (3, 20, 0.05, 0.343664),
        ]
        for errors, n, delta, expected in cases:
            with self.subTest(errors=errors, n=n, delta=delta):
                self.assertAlmostEqual(
                    certify.exact_binomial_upper(errors, n, delta), expected, places=5
                )

    def test_all_errors_gives_one(self):
        self.assertEqual(certify.exact_binomial_upper(10, 10, 0.1), 1.0)

    def test_bound_strictly_exceeds_the_empirical_rate(self):
        previous = -1.0
        for errors in range(0, 41):
            bound = certify.exact_binomial_upper(errors, 40, 0.1)
            self.assertGreaterEqual(bound, previous)
            self.assertLessEqual(bound, 1.0)
            if errors < 40:
                self.assertGreater(bound, errors / 40.0)
            else:
                self.assertEqual(bound, 1.0)
            previous = bound

    def test_bound_shrinks_when_delta_grows(self):
        # delta is the error allowed on the bound itself: bigger delta, smaller bound
        looser = certify.exact_binomial_upper(2, 30, 0.2)
        stricter = certify.exact_binomial_upper(2, 30, 0.01)
        self.assertLess(looser, stricter)

    def test_rejects_bad_input(self):
        for bad in [(-1, 10, 0.1), (11, 10, 0.1), (1, 0, 0.1), (1, -3, 0.1)]:
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    certify.exact_binomial_upper(*bad)
        for bad_delta in [0.0, 1.0, -0.1, 1.5]:
            with self.subTest(delta=bad_delta):
                with self.assertRaises(ValueError):
                    certify.exact_binomial_upper(0, 10, bad_delta)


def _softmax(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def _nll(probs: np.ndarray, labels: np.ndarray) -> float:
    picked = np.clip(probs[np.arange(len(labels)), labels], 1e-12, None)
    return float(-np.log(picked).mean())


def _sample_labels(logits: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Draw labels from softmax(logits) with a fixed-seed generator."""
    probs = _softmax(logits)
    u = rng.random(probs.shape[0])
    return (probs.cumsum(axis=1) < u[:, None]).sum(axis=1)


def _calibrated_binary_set() -> tuple[np.ndarray, np.ndarray]:
    """15 bins of 15 rows where the observed accuracy equals the mean confidence."""
    rows: list[list[float]] = []
    labels: list[int] = []
    for j in range(15):
        conf = (j + 1) / 15.0
        correct = j + 1
        for i in range(15):
            y = 1 if i < correct else 0
            rows.append([1.0 - conf, conf])
            labels.append(y)
    return np.array(rows), np.array(labels)


class EceTest(unittest.TestCase):

    def test_perfectly_calibrated_set_has_zero_ece(self):
        probs, y = _calibrated_binary_set()
        self.assertAlmostEqual(certify.ece(probs, y), 0.0, places=12)

    def test_overconfident_set_has_expected_ece(self):
        probs = np.tile([0.05, 0.95], (100, 1))
        y = np.array([1] * 50 + [0] * 50)
        self.assertAlmostEqual(certify.ece(probs, y), 0.45, places=12)

    def test_binary_one_dimensional_confidence_is_supported(self):
        p = np.full(100, 0.9)
        y = np.array([1] * 50 + [0] * 50)
        self.assertAlmostEqual(certify.ece(p, y), 0.4, places=12)

    def test_single_bin_is_the_gap_between_confidence_and_accuracy(self):
        probs = np.tile([0.2, 0.8], (10, 1))
        y = np.ones(10, dtype=int)
        self.assertAlmostEqual(certify.ece(probs, y, bins=1), 0.2, places=12)

    def test_ece_is_bounded_and_shrinks_with_better_calibration(self):
        probs, y = _calibrated_binary_set()
        good = certify.ece(probs, y)
        sharpened = np.clip(probs * 1.5 - 0.25, 0.0, 1.0)
        bad = certify.ece(sharpened, y)
        self.assertGreaterEqual(good, 0.0)
        self.assertLessEqual(good, 1.0)
        self.assertLess(good, bad)

    def test_rejects_mismatched_or_invalid_input(self):
        with self.assertRaises(ValueError):
            certify.ece(np.zeros((3, 2)), np.zeros(4, dtype=int))
        with self.assertRaises(ValueError):
            certify.ece(np.array([[float("nan"), 1.0]]), np.zeros(1, dtype=int))
        with self.assertRaises(ValueError):
            certify.ece(np.array([[0.5, 0.5]]), np.array([5]))
        with self.assertRaises(ValueError):
            certify.ece(np.array([[0.0, 0.0]]), np.zeros(1, dtype=int))
        with self.assertRaises(ValueError):
            certify.ece(np.zeros((0, 2)), np.zeros(0, dtype=int))


class FitTemperatureTest(unittest.TestCase):

    def test_already_calibrated_logits_fit_temperature_near_one(self):
        rng = np.random.default_rng(3)
        base = rng.normal(0.0, 2.0, size=(800, 4))
        y = _sample_labels(base, rng)
        temperature = certify.fit_temperature(base, y)
        self.assertGreater(temperature, 0.6)
        self.assertLess(temperature, 1.6)

    def test_over_confident_logits_fit_temperature_above_one(self):
        rng = np.random.default_rng(11)
        base = rng.normal(0.0, 2.0, size=(900, 4))
        y = _sample_labels(base, rng)
        logits = base * 6.0
        temperature = certify.fit_temperature(logits, y)
        self.assertGreater(temperature, 3.0)
        self.assertLess(temperature, 12.0)
        # the fitted temperature must not make the fit worse than raw logits
        self.assertLessEqual(
            _nll(_softmax(logits / temperature), y), _nll(_softmax(logits), y) + 1e-9
        )

    def test_under_confident_logits_fit_temperature_below_one(self):
        rng = np.random.default_rng(5)
        base = rng.normal(0.0, 2.0, size=(900, 4))
        y = _sample_labels(base, rng)
        temperature = certify.fit_temperature(base / 3.0, y)
        self.assertLess(temperature, 0.6)
        self.assertGreater(temperature, 0.2)

    def test_fitted_temperature_reduces_ece_on_over_confident_logits(self):
        rng = np.random.default_rng(21)
        base = rng.normal(0.0, 1.5, size=(1200, 3))
        y = _sample_labels(base, rng)
        logits = base * 8.0
        before = certify.ece(_softmax(logits), y)
        temperature = certify.fit_temperature(logits, y)
        after = certify.ece(_softmax(logits / temperature), y)
        self.assertLess(after, before)

    def test_one_dimensional_binary_logits_are_supported(self):
        rng = np.random.default_rng(9)
        base = rng.normal(0.0, 2.0, size=(600, 1))
        p_yes = 1.0 / (1.0 + np.exp(-base))
        y = (rng.random(600) < p_yes.ravel()).astype(int)
        temperature = certify.fit_temperature((base * 5.0).ravel(), y)
        self.assertGreater(temperature, 2.0)

    def test_single_row_returns_a_finite_positive_temperature(self):
        temperature = certify.fit_temperature(np.array([[2.0, -1.0]]), np.array([0]))
        self.assertTrue(np.isfinite(temperature))
        self.assertGreater(temperature, 0.0)

    def test_rejects_invalid_input(self):
        with self.assertRaises(ValueError):
            certify.fit_temperature(np.zeros((0, 3)), np.zeros(0, dtype=int))
        with self.assertRaises(ValueError):
            certify.fit_temperature(np.array([[1.0, float("nan")]]), np.array([0]))
        with self.assertRaises(ValueError):
            certify.fit_temperature(np.zeros((3, 2)), np.zeros(4, dtype=int))
        with self.assertRaises(ValueError):
            certify.fit_temperature(np.zeros((3, 2)), np.array([0, 1, 7]))


if __name__ == "__main__":
    unittest.main()
