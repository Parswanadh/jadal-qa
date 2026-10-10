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


class PickThresholdTest(unittest.TestCase):

    @staticmethod
    def _clean_set(n: int = 200) -> tuple[np.ndarray, np.ndarray]:
        """Confidence increasing with correctness; everything at/above 0.5 is right."""
        conf = np.linspace(0.01, 1.0, n)
        correct = conf >= 0.5
        return conf, correct

    @staticmethod
    def _admissible(conf, correct, target, delta, min_accepted):
        """Independent re-derivation of the candidates that meet both constraints."""
        rows = []
        for t in np.unique(conf):
            mask = conf >= t
            accepted = int(mask.sum())
            errors = int((mask & ~correct).sum())
            if accepted < min_accepted:
                continue
            if certify.exact_binomial_upper(errors, accepted, delta) > target:
                continue
            rows.append((float(t), accepted, errors))
        return rows

    def test_returns_the_loosest_threshold_that_still_meets_target_and_min_accepted(self):
        conf, correct = self._clean_set()
        result = certify.pick_threshold(conf, correct, target_err=0.05)
        admissible = self._admissible(conf, correct, 0.05, 0.1, 30)
        self.assertGreater(len(admissible), 1)  # a real scan, not a single choice
        best_threshold, best_accepted, best_errors = max(admissible, key=lambda r: r[1])
        self.assertAlmostEqual(result["threshold"], best_threshold, places=12)
        self.assertEqual(result["accepted"], best_accepted)
        self.assertEqual(result["errors"], best_errors)
        self.assertAlmostEqual(result["coverage"], best_accepted / 200.0, places=12)
        self.assertAlmostEqual(
            result["upper_bound"],
            certify.exact_binomial_upper(best_errors, best_accepted, 0.1),
            places=12,
        )
        self.assertLessEqual(result["upper_bound"], 0.05)
        self.assertGreaterEqual(result["accepted"], 30)

    def test_maximum_coverage_is_preferred_over_the_strictest_admissible_one(self):
        conf, correct = self._clean_set()
        result = certify.pick_threshold(conf, correct, target_err=0.05)
        admissible = self._admissible(conf, correct, 0.05, 0.1, 30)
        strictest = min(admissible, key=lambda r: r[1])
        # the loosest admissible candidate accepts the most segments, so its
        # threshold is the LOWEST of the admissible ones
        self.assertGreater(result["accepted"], strictest[1])
        self.assertLess(result["threshold"], strictest[0])
        # accepting everything is NOT admissible here, so the answer is a real cut
        everything = certify.exact_binomial_upper(int((~correct).sum()), 200, 0.1)
        self.assertGreater(everything, 0.05)
        self.assertLess(result["coverage"], 1.0)

    def test_returns_none_when_the_target_cannot_be_met(self):
        conf = np.linspace(0.01, 1.0, 200)
        correct = conf < 0.5  # errors sit at the top of the confidence range
        result = certify.pick_threshold(conf, correct, target_err=0.05)
        self.assertIsNone(result["threshold"])
        self.assertEqual(result["accepted"], 200)
        self.assertEqual(result["errors"], 200 - int(correct.sum()))
        self.assertEqual(result["coverage"], 1.0)
        self.assertAlmostEqual(
            result["upper_bound"],
            certify.exact_binomial_upper(200 - int(correct.sum()), 200, 0.1),
            places=12,
        )
        self.assertGreater(result["upper_bound"], 0.05)

    def test_min_accepted_limits_coverage(self):
        conf, correct = self._clean_set()
        loose = certify.pick_threshold(conf, correct, target_err=0.5, min_accepted=30)
        strict = certify.pick_threshold(
            conf, correct, target_err=0.5, min_accepted=120
        )
        self.assertGreaterEqual(loose["accepted"], 30)
        self.assertGreaterEqual(strict["accepted"], 120)
        self.assertLessEqual(strict["accepted"], loose["accepted"])
        for result, floor in ((loose, 30), (strict, 120)):
            self.assertLessEqual(result["upper_bound"], 0.5)
            self.assertGreaterEqual(result["accepted"], floor)

    def test_min_accepted_larger_than_the_set_never_passes(self):
        conf, correct = self._clean_set(10)
        result = certify.pick_threshold(conf, correct, target_err=0.05)
        self.assertIsNone(result["threshold"])
        self.assertEqual(result["accepted"], 10)
        self.assertEqual(result["errors"], 5)
        self.assertEqual(result["coverage"], 1.0)
        self.assertAlmostEqual(
            result["upper_bound"], certify.exact_binomial_upper(5, 10, 0.1),
            places=12,
        )
        self.assertGreater(result["upper_bound"], 0.05)

    def test_delta_is_used_for_the_bound(self):
        conf, correct = self._clean_set()
        result = certify.pick_threshold(
            conf, correct, target_err=0.05, delta=0.01, min_accepted=120
        )
        self.assertEqual(
            result["upper_bound"],
            certify.exact_binomial_upper(result["errors"], result["accepted"], 0.01),
        )
        self.assertGreaterEqual(result["accepted"], 120)
        # a stricter delta can only shrink the admissible set
        looser_delta = certify.pick_threshold(
            conf, correct, target_err=0.05, delta=0.1, min_accepted=120
        )
        self.assertLessEqual(result["accepted"], looser_delta["accepted"])
        self.assertLessEqual(
            looser_delta["upper_bound"],
            certify.exact_binomial_upper(
                looser_delta["errors"], looser_delta["accepted"], 0.1
            ),
        )

    def test_empty_input_returns_none(self):
        result = certify.pick_threshold([], [], target_err=0.05)
        self.assertIsNone(result["threshold"])
        self.assertEqual(result["accepted"], 0)
        self.assertEqual(result["errors"], 0)
        self.assertEqual(result["coverage"], 0.0)
        self.assertIsNone(result["upper_bound"])

    def test_rejects_bad_arguments(self):
        conf, correct = self._clean_set(10)
        with self.assertRaises(ValueError):
            certify.pick_threshold([0.1, 0.2], [True], target_err=0.05)
        with self.assertRaises(ValueError):
            certify.pick_threshold(conf, correct, target_err=0.0)
        with self.assertRaises(ValueError):
            certify.pick_threshold(conf, correct, target_err=0.05, delta=0.0)
        with self.assertRaises(ValueError):
            certify.pick_threshold(conf, correct, target_err=0.05, min_accepted=0)


class RiskCoverageTest(unittest.TestCase):

    def test_curve_grows_in_coverage_and_error(self):
        rng = np.random.default_rng(4)
        conf = np.round(rng.random(200), 4)
        correct = rng.random(200) < conf
        curve = certify.risk_coverage(conf, correct)
        self.assertEqual(len(curve), len(np.unique(conf)))
        coverages = [row["coverage"] for row in curve]
        self.assertEqual(coverages, sorted(coverages))
        for row in curve:
            self.assertAlmostEqual(row["error"], round(row["error"], 4), places=12)
            self.assertGreaterEqual(row["error"], 0.0)
            self.assertLessEqual(row["error"], 1.0)
            self.assertGreaterEqual(row["coverage"], 0.0)
            self.assertLessEqual(row["coverage"], 1.0)

    def test_a_confident_and_perfect_set_has_zero_error_above_the_cut(self):
        conf = np.where(np.arange(20) < 10, 0.05, 0.95)
        correct = np.arange(20) >= 10  # the confident half is the correct half
        curve = certify.risk_coverage(conf, correct)
        low, high = curve[0], curve[1]
        self.assertAlmostEqual(low["coverage"], 0.5, places=12)
        self.assertAlmostEqual(low["error"], 0.0, places=12)
        self.assertAlmostEqual(high["coverage"], 1.0, places=12)
        self.assertAlmostEqual(high["error"], 0.5, places=12)

    def test_floats_are_rounded_to_four_decimals(self):
        rng = np.random.default_rng(13)
        conf = rng.random(50)
        correct = rng.random(50) < 0.5
        for row in certify.risk_coverage(conf, correct):
            for key in ("threshold", "coverage", "error"):
                self.assertAlmostEqual(row[key], round(row[key], 4), places=12)

    def test_empty_input_gives_an_empty_curve(self):
        self.assertEqual(certify.risk_coverage([], []), [])

    def test_rejects_mismatched_input(self):
        with self.assertRaises(ValueError):
            certify.risk_coverage([0.1, 0.2], [True, False, True])


class BatchRuleTest(unittest.TestCase):

    @staticmethod
    def _verdict(verdict: str) -> dict:
        return {
            "segment_id": "s1",
            "verdict": verdict,
            "reasons": [],
            "answer_conf": 0.9,
            "threshold": 0.5,
        }

    def test_accepts_when_rejections_are_at_or_below_k(self):
        verdicts = [self._verdict("accept")] * 3 + [self._verdict("reject")]
        self.assertEqual(certify.batch_rule(verdicts, 1), "accept")

    def test_rejects_when_rejections_exceed_k(self):
        verdicts = [self._verdict("accept")] * 3 + [self._verdict("reject")] * 2
        self.assertEqual(certify.batch_rule(verdicts, 1), "reject")

    def test_k_of_zero_rejects_on_the_first_rejection(self):
        verdicts = [self._verdict("reject"), self._verdict("accept")]
        self.assertEqual(certify.batch_rule(verdicts, 0), "reject")

    def test_review_is_not_a_rejection(self):
        verdicts = [self._verdict("review"), self._verdict("review"), self._verdict("accept")]
        self.assertEqual(certify.batch_rule(verdicts, 1), "accept")

    def test_empty_batch_is_accepted(self):
        self.assertEqual(certify.batch_rule([], 1), "accept")

    def test_accepts_a_batch_mapping_and_plain_verdict_strings(self):
        batch = {
            "batch_id": "b1",
            "verdicts": [self._verdict("reject"), self._verdict("reject")],
            "k": 1,
        }
        self.assertEqual(certify.batch_rule(batch, 1), "reject")
        self.assertEqual(certify.batch_rule(["accept", "reject"], 1), "accept")
        self.assertEqual(certify.batch_rule(["reject", "reject"], 1), "reject")

    def test_unknown_verdict_or_bad_k_raises(self):
        with self.assertRaises(ValueError):
            certify.batch_rule([self._verdict("maybe")], 1)
        with self.assertRaises(ValueError):
            certify.batch_rule([self._verdict("accept")], -1)


if __name__ == "__main__":
    unittest.main()
