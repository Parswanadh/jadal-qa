"""Seeded stdlib-random property tests for jadal/certify.py.

Every test builds its inputs from ``random.Random(<fixed>)`` (no hypothesis,
no unseeded randomness); the seed is stated in each test's docstring.
Loop sizes are 200-500 cases; the whole file must run in < 10 s.
"""
from __future__ import annotations

import math
import random
import unittest

import numpy as np

from jadal import certify


def _softmax(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


class ExactBinomialUpperPropertiesTest(unittest.TestCase):
    """Monotonicity / sanity properties of exact_binomial_upper."""

    def test_nondecreasing_in_errors(self):
        """Seed: 3601. 300 random (n, delta, e): bound(e) <= bound(e+1)."""
        rnd = random.Random(3601)
        for _ in range(300):
            n = rnd.randint(1, 60)
            delta = rnd.choice([0.01, 0.05, 0.1, 0.2])
            e = rnd.randint(0, n - 1)
            lo = certify.exact_binomial_upper(e, n, delta)
            hi = certify.exact_binomial_upper(e + 1, n, delta)
            self.assertLessEqual(lo, hi)

    def test_nonincreasing_in_n_for_fixed_errors(self):
        """Seed: 3602. 300 random (e, n, delta): bound(e, n+1) <= bound(e, n)."""
        rnd = random.Random(3602)
        for _ in range(300):
            e = rnd.randint(0, 20)
            n = rnd.randint(max(e, 1), e + 60)
            delta = rnd.choice([0.01, 0.05, 0.1, 0.2])
            small_n = certify.exact_binomial_upper(e, n, delta)
            big_n = certify.exact_binomial_upper(e, n + 1, delta)
            self.assertLessEqual(big_n, small_n)

    def test_always_ge_empirical_rate_and_in_unit_interval(self):
        """Seed: 3603. 300 random (e, n, delta): e/n <= bound <= 1."""
        rnd = random.Random(3603)
        for _ in range(300):
            n = rnd.randint(1, 200)
            e = rnd.randint(0, n)
            delta = rnd.choice([0.01, 0.05, 0.1, 0.2])
            bound = certify.exact_binomial_upper(e, n, delta)
            self.assertGreaterEqual(bound, 0.0)
            self.assertLessEqual(bound, 1.0)
            self.assertGreaterEqual(bound, e / n)

    def test_known_zero_error_value(self):
        """Deterministic anchor: 0 errors, n=100, delta=0.1 -> ~0.0228."""
        bound = certify.exact_binomial_upper(0, 100, 0.1)
        self.assertAlmostEqual(bound, 1.0 - 0.1 ** (1.0 / 100.0), places=10)
        self.assertAlmostEqual(bound, 0.0228, places=4)


def _random_conf_correct(rnd: random.Random) -> tuple[list, list]:
    """Random (conf, correct) data: a few discrete confidence levels with
    repeats (so the strictest candidate can clear min_accepted), correctness
    roughly increasing in confidence with occasional adversarial pockets."""
    nlevels = rnd.randint(3, 6)
    levels = sorted({round(rnd.uniform(0.0, 1.0), 2) for _ in range(nlevels)})
    conf, correct = [], []
    for j, level in enumerate(levels):
        count = rnd.randint(20, 100)
        for _ in range(count):
            if j == len(levels) - 1:
                ok = rnd.random() < 0.995  # clean top: certifiable
            elif rnd.random() < 0.15:
                ok = rnd.random() < 0.3  # adversarial pocket
            else:
                ok = rnd.random() < level
            conf.append(level)
            correct.append(ok)
    combined = list(zip(conf, correct))
    rnd.shuffle(combined)
    conf, correct = zip(*combined)
    return list(conf), list(correct)


class PickThresholdPropertiesTest(unittest.TestCase):
    """Properties of pick_threshold on random (conf, correct) data."""

    def test_recomputed_bound_never_exceeds_target(self):
        """Seed: 3604. 300 datasets: a certified threshold's recomputed
        exact upper bound is <= target and matches the reported bound."""
        rnd = random.Random(3604)
        certified = 0
        for _ in range(300):
            conf, correct = _random_conf_correct(rnd)
            target = rnd.choice([0.05, 0.1, 0.15])
            delta = rnd.choice([0.05, 0.1])
            floor = rnd.randint(5, 30)
            result = certify.pick_threshold(
                conf, correct, target_err=target, delta=delta, min_accepted=floor
            )
            if result["threshold"] is None:
                continue
            certified += 1
            recomputed = certify.exact_binomial_upper(
                result["errors"], result["accepted"], delta
            )
            self.assertAlmostEqual(result["upper_bound"], recomputed, places=12)
            self.assertLessEqual(recomputed, target)
            self.assertGreaterEqual(result["accepted"], floor)
        # sanity: the generator must certify something, or the test is vacuous
        self.assertGreater(certified, 50)

    def test_stops_at_first_failure_not_loosest_passer(self):
        """Deterministic adversarial case: 50 clean @0.99, 5 errors @0.9,
        200 clean @0.5. The scan must stop at 0.9 and return 0.99 even
        though 0.5 alone would also pass the bound (what a
        loosest-admissible scan would wrongly certify)."""
        conf = [0.99] * 50 + [0.9] * 5 + [0.5] * 200
        correct = [True] * 50 + [False] * 5 + [True] * 200
        result = certify.pick_threshold(conf, correct, target_err=0.05)
        self.assertAlmostEqual(result["threshold"], 0.99, places=12)
        self.assertEqual(result["accepted"], 50)
        self.assertEqual(result["errors"], 0)
        # the looser candidate passes on its own: a scan-everything approach
        # would give a different (uncertified) answer
        self.assertLessEqual(certify.exact_binomial_upper(5, 255, 0.1), 0.05)

    def test_respects_min_accepted(self):
        """Seed: 3605. 300 datasets: certified => accepted >= min_accepted;
        coverage == accepted / n; errors <= accepted."""
        rnd = random.Random(3605)
        for _ in range(300):
            conf, correct = _random_conf_correct(rnd)
            floor = rnd.randint(5, 40)
            result = certify.pick_threshold(
                conf, correct, target_err=0.05, min_accepted=floor
            )
            n = len(conf)
            self.assertEqual(result["coverage"], result["accepted"] / n)
            self.assertLessEqual(result["errors"], result["accepted"])
            if result["threshold"] is not None:
                self.assertGreaterEqual(result["accepted"], floor)

    def test_invariant_to_input_order(self):
        """Seed: 3606. 200 datasets: shuffling the input order does not
        change the picked threshold or its statistics."""
        rnd = random.Random(3606)
        for _ in range(200):
            conf, correct = _random_conf_correct(rnd)
            base = certify.pick_threshold(conf, correct, target_err=0.05)
            order = list(range(len(conf)))
            rnd.shuffle(order)
            shuffled = certify.pick_threshold(
                [conf[i] for i in order], [correct[i] for i in order],
                target_err=0.05,
            )
            self.assertEqual(
                base["threshold"] is None, shuffled["threshold"] is None
            )
            if base["threshold"] is not None:
                self.assertAlmostEqual(
                    base["threshold"], shuffled["threshold"], places=12
                )
                self.assertAlmostEqual(
                    base["upper_bound"], shuffled["upper_bound"], places=12
                )
            self.assertEqual(base["accepted"], shuffled["accepted"])
            self.assertEqual(base["errors"], shuffled["errors"])
            self.assertAlmostEqual(base["coverage"], shuffled["coverage"], places=12)


class FitTemperatureEcePropertiesTest(unittest.TestCase):
    """fit_temperature stays positive; ece stays in [0, 1]."""

    def test_temperature_positive_and_ece_bounded(self):
        """Seed: 3607. 200 random logit matrices (40x3, plus binary 1-D):
        0 < T < inf and 0 <= ece <= 1."""
        rnd = random.Random(3607)
        for i in range(200):
            n = 40
            logits = [[rnd.uniform(-3.0, 3.0) for _ in range(3)] for _ in range(n)]
            labels = [rnd.randint(0, 2) for _ in range(n)]
            temperature = certify.fit_temperature(logits, labels)
            self.assertTrue(math.isfinite(temperature))
            self.assertGreater(temperature, 0.0)
            probs = _softmax(np.asarray(logits) / temperature)
            error = certify.ece(probs, labels)
            self.assertGreaterEqual(error, 0.0)
            self.assertLessEqual(error, 1.0)
            if i % 2 == 0:  # binary 1-D spelling must work too
                p = [rnd.random() for _ in range(n)]
                y = [rnd.randint(0, 1) for _ in range(n)]
                one_d = certify.ece(p, y)
                self.assertGreaterEqual(one_d, 0.0)
                self.assertLessEqual(one_d, 1.0)


class BatchRulePropertiesTest(unittest.TestCase):
    """batch_rule is monotone in the rejected count."""

    @staticmethod
    def _verdict(v: str) -> dict:
        return {
            "segment_id": "s",
            "verdict": v,
            "reasons": [],
            "answer_conf": 0.9,
            "threshold": 0.5,
        }

    def test_monotone_in_rejected_count(self):
        """Seed: 3608. 300 random (n, k): as the rejection count grows
        0..n the decision flips accept->reject at most once."""
        rnd = random.Random(3608)
        for _ in range(300):
            n = rnd.randint(0, 8)
            k = rnd.randint(0, n + 2)
            seen_reject = False
            for r in range(n + 1):
                verdicts = [self._verdict("reject")] * r + [
                    self._verdict("accept")
                ] * (n - r)
                decision = certify.batch_rule(verdicts, k)
                if decision == "reject":
                    seen_reject = True
                else:
                    self.assertFalse(
                        seen_reject,
                        f"non-monotone: n={n} k={k} r={r}",
                    )


if __name__ == "__main__":
    unittest.main()
