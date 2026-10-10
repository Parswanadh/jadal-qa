"""Seeded stdlib-random property tests for jadal/receipts.py.

Every test builds its inputs from ``random.Random(<fixed>)`` (no hypothesis,
no unseeded randomness); the seed is stated in each test's docstring.
Loop sizes are 200-500 cases; the whole file must run in < 10 s.
"""
from __future__ import annotations

import copy
import hashlib
import math
import random
import unittest

from jadal import receipts

CREATED_UTC = "2026-10-10T00:00:00+00:00"
EXCLUDED = ("receipt_sha256", "receipt_id", "created_utc")


def _base_inputs(seed_tag: str = "prop") -> tuple[dict, dict, dict, dict]:
    """Deterministic (signals, decisions, verdict) payloads with varied types."""
    digest = hashlib.sha256(seed_tag.encode("utf-8")).digest()
    s = int.from_bytes(digest[:4], "big")
    signals = {
        "decodable": True,
        "blur_var": round(60.0 + (s % 5000) / 100.0, 4),
        "brightness_mean": round(40.0 + (s % 9000) / 100.0, 4),
        "under_exposed_frac": round((s % 7) / 1000.0, 4),
        "over_exposed_frac": round((s % 11) / 1000.0, 4),
        "motion_mean": round((s % 600) / 100.0, 4),
        "hands_visible_frac": round(0.5 + (s % 400) / 1000.0, 4),
        "hand_count_mode": s % 3,
        "faces_found": s % 4,
        "faces_blurred": s % 4,
        "phash": hashlib.sha256(digest).hexdigest()[:16],
        "dup_of": None,
    }
    decisions = {
        "hands_ok": {"p_yes": round((s % 900) / 1000.0 + 0.05, 4)},
        "active_work": {"p_yes": round(((s >> 8) % 900) / 1000.0 + 0.05, 4)},
        "step": {
            "choice": ["trim", "press", "pack"][s % 3],
            "probs": {"trim": 0.4, "press": 0.35, "pack": 0.25},
            "answer_conf": round(((s >> 12) % 900) / 1000.0 + 0.05, 4),
        },
        "model_versions": {"hands": "fake-hands-1.0", "active": "fake-active-1.0"},
    }
    verdict = {
        "segment_id": f"seg-{seed_tag}",
        "verdict": "accept",
        "reasons": ["LOW_CONFIDENCE"] if s % 2 else [],
        "answer_conf": round(((s >> 5) % 900) / 1000.0 + 0.05, 4),
        "threshold": 0.5,
    }
    config = {
        "thresholds": {"target_error": 0.05, "float_decimals": 4},
        "pipeline_version": "9.9.9-test",
        "versions": {"model": "fake-1.0"},
    }
    return signals, decisions, verdict, config


def _segment(tag: str) -> dict:
    return {
        "segment_id": f"seg-{tag}",
        "source_clip": f"clip-{tag}.mp4",
        "worker_id": "w-1",
        "site_id": "site-1",
        "start_s": 0.0,
        "end_s": 20.0,
        "path": f"data_runtime/segments/seg-{tag}.mp4",
        "sha256": hashlib.sha256(f"seg-{tag}".encode("utf-8")).hexdigest(),
    }


def _base_receipt(tag: str = "base") -> dict:
    signals, decisions, verdict, config = _base_inputs(tag)
    if not verdict["reasons"]:
        verdict["reasons"] = ["LOW_CONFIDENCE"]  # keep a list leaf mutable
    return receipts.make_receipt(
        _segment(tag), signals, decisions, verdict, config,
        created_utc=CREATED_UTC,
    )


def _scalar_leaves(obj, prefix: str = "") -> list[str]:
    """Dotted/indexed paths of scalar leaves, in _diff_paths notation."""
    paths: list[str] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if isinstance(value, (dict, list)):
                paths.extend(_scalar_leaves(value, path))
            else:
                paths.append(path)
    elif isinstance(obj, list):
        for i, value in enumerate(obj):
            path = f"{prefix}[{i}]"
            if isinstance(value, (dict, list)):
                paths.extend(_scalar_leaves(value, path))
            else:
                paths.append(path)
    return paths


def _get_path(obj, path: str):
    """Read a dotted/indexed path produced by _scalar_leaves."""
    import re

    current = obj
    for part in path.split("."):
        for token in re.findall(r"[^\[\]]+|\[\d+\]", part):
            if token.startswith("["):
                current = current[int(token[1:-1])]
            else:
                current = current[token] if isinstance(current, dict) else current[int(token)]
    return current


def _set_path(obj, path: str, value) -> None:
    """Set a dotted/indexed path produced by _scalar_leaves."""
    import re

    tokens: list = []
    for part in path.split("."):
        for token in re.findall(r"[^\[\]]+|\[\d+\]", part):
            tokens.append(token)
    current = obj
    for token in tokens[:-1]:
        if token.startswith("["):
            current = current[int(token[1:-1])]
        else:
            current = current[token]
    last = tokens[-1]
    if last.startswith("["):
        current[int(last[1:-1])] = value
    else:
        current[last] = value


def _mutated_value(old, rnd: random.Random):
    """A value guaranteed to differ from ``old`` (JSON-serializable)."""
    if isinstance(old, bool):
        return not old
    if isinstance(old, int):
        return old + rnd.randint(1, 1000) + 1
    if isinstance(old, float):
        return old + 0.5 + rnd.random()
    if isinstance(old, str):
        return old + f"#mut{rnd.randint(0, 10 ** 9)}"
    if old is None:
        return f"mut{rnd.randint(0, 10 ** 9)}"
    return [old, f"mut{rnd.randint(0, 10 ** 9)}"]


def _shuffled(obj, rnd: random.Random):
    """Copy of a JSON-like object with dict key order shuffled recursively."""
    if isinstance(obj, dict):
        items = list(obj.items())
        rnd.shuffle(items)
        return {k: _shuffled(v, rnd) for k, v in items}
    if isinstance(obj, list):
        return [_shuffled(v, rnd) for v in obj]  # list order is meaningful
    return obj


def _random_nested(rnd: random.Random, depth: int = 0):
    """Random JSON-like object: dicts, lists, scalars, unicode."""
    if depth > 3 or rnd.random() < 0.25:
        return rnd.choice(
            [None, True, False, rnd.randint(-99, 99),
             round(rnd.uniform(-99, 99), 4), "कपड़े",
             f"s{rnd.randint(0, 999)}"]
        )
    if rnd.random() < 0.5:
        return {
            f"k{rnd.randint(0, 999)}": _random_nested(rnd, depth + 1)
            for _ in range(rnd.randint(1, 5))
        }
    return [_random_nested(rnd, depth + 1) for _ in range(rnd.randint(0, 4))]


class CanonicalJsonPropertiesTest(unittest.TestCase):
    def test_invariant_under_recursive_key_shuffle(self):
        """Seed: 3611. 300 random nested objects: shuffling dict key
        order recursively does not change canonical_json bytes."""
        rnd = random.Random(3611)
        for _ in range(300):
            obj = _random_nested(rnd)
            self.assertEqual(
                receipts.canonical_json(obj),
                receipts.canonical_json(_shuffled(obj, rnd)),
            )

    def test_stable_across_4dp_float_spellings(self):
        """Seed: 3612. 300 cases: two signals payloads whose floats differ
        only beyond 4 decimals give identical receipt_sha256."""
        rnd = random.Random(3612)
        for i in range(300):
            signals, decisions, verdict, config = _base_inputs(f"spell-{i}")
            key = rnd.choice(["blur_var", "brightness_mean", "motion_mean"])
            # 3-decimal base + tiny epsilon can never cross a 4-dp boundary
            base = round(rnd.uniform(1.0, 120.0), 3)
            eps = rnd.choice([1e-9, 5e-7, -1e-9])
            self.assertEqual(round(base, 4), round(base + eps, 4))
            signals[key] = base
            twin = dict(signals)
            twin[key] = base + eps
            tag = f"spell-{i}"
            a = receipts.make_receipt(
                _segment(tag), signals, decisions, verdict, config,
                created_utc=CREATED_UTC,
            )
            b = receipts.make_receipt(
                _segment(tag), twin, decisions, verdict, config,
                created_utc=CREATED_UTC,
            )
            self.assertEqual(a["receipt_sha256"], b["receipt_sha256"])


class ReceiptHashMutationPropertiesTest(unittest.TestCase):
    def test_every_single_field_mutation_changes_hash(self):
        """Seed: 3613. 500 random single-leaf mutations of the hashed body
        (never the excluded receipt_sha256/receipt_id/created_utc): every
        one changes receipt_sha256."""
        rnd = random.Random(3613)
        base = _base_receipt("mutbase")
        leaves = [
            p for p in _scalar_leaves(base)
            if not any(p == e or p.startswith(e + ".") for e in EXCLUDED)
        ]
        self.assertGreater(len(leaves), 10)
        before = receipts.receipt_hash(base)
        for _ in range(500):
            path = rnd.choice(leaves)
            tampered = copy.deepcopy(base)
            old = _get_path(tampered, path)
            _set_path(tampered, path, _mutated_value(old, rnd))
            self.assertNotEqual(
                receipts.receipt_hash(tampered), before, f"path={path}"
            )

    def test_excluded_fields_do_not_change_hash(self):
        """Seed: 3614. 200 mutations of receipt_sha256/receipt_id/
        created_utc: receipt_hash is unchanged."""
        rnd = random.Random(3614)
        base = _base_receipt("exclbase")
        before = receipts.receipt_hash(base)
        for _ in range(200):
            tampered = copy.deepcopy(base)
            key = rnd.choice(list(EXCLUDED))
            tampered[key] = f"zzz-{rnd.randint(0, 10 ** 9)}"
            self.assertEqual(receipts.receipt_hash(tampered), before)


class NonFiniteAndZeroPropertiesTest(unittest.TestCase):
    def test_nan_inf_rejected(self):
        """Seed: 3615. 200 cases: NaN/+inf/-inf planted at a random float
        leaf of signals/decisions/verdict raises ValueError."""
        rnd = random.Random(3615)
        for i in range(200):
            signals, decisions, verdict, config = _base_inputs(f"nan-{i}")
            section = rnd.choice(["signals", "decisions", "verdict"])
            payload = {"signals": signals, "decisions": decisions,
                       "verdict": verdict}[section]
            leaves = [p for p in _scalar_leaves(payload)
                      if isinstance(_get_path(payload, p), float)]
            path = rnd.choice(leaves)
            _set_path(payload, path, rnd.choice(
                [float("nan"), float("inf"), float("-inf")]))
            with self.assertRaises(ValueError, msg=f"{section}.{path}"):
                receipts.make_receipt(
                    _segment(f"nan-{i}"), signals, decisions, verdict,
                    config, created_utc=CREATED_UTC,
                )

    def test_minus_zero_normalised(self):
        """Seed: 3616. 200 cases: -0.0 (or a tiny negative that rounds to
        it) stored in a float leaf becomes +0.0; no '-0.0' in the bytes."""
        rnd = random.Random(3616)
        for i in range(200):
            signals, decisions, verdict, config = _base_inputs(f"neg-{i}")
            section = rnd.choice(["signals", "decisions", "verdict"])
            payload = {"signals": signals, "decisions": decisions,
                       "verdict": verdict}[section]
            leaves = [p for p in _scalar_leaves(payload)
                      if isinstance(_get_path(payload, p), float)]
            path = rnd.choice(leaves)
            _set_path(payload, path, rnd.choice([-0.0, -1e-9, -0.000001]))
            receipt = receipts.make_receipt(
                _segment(f"neg-{i}"), signals, decisions, verdict,
                config, created_utc=CREATED_UTC,
            )
            stored = _get_path(receipt[section], path)
            self.assertEqual(stored, 0.0)
            self.assertEqual(math.copysign(1.0, stored), 1.0)
            self.assertNotIn(
                b"-0.0", receipts.canonical_json(receipt))


class FakeDecider:
    """Deterministic decider with a one-call batch entry point."""

    def decide(self, segment: dict, signals: dict) -> dict:
        digest = hashlib.sha256(
            ("decide" + segment["segment_id"]).encode("utf-8")).digest()
        seed = int.from_bytes(digest[:4], "big")
        return {
            "hands_ok": {"p_yes": round((seed % 900) / 1000.0 + 0.05, 4)},
            "active_work": {"p_yes": round(((seed >> 8) % 900) / 1000.0 + 0.05, 4)},
            "step": {
                "choice": ["trim", "press", "pack"][seed % 3],
                "probs": {"trim": 0.4, "press": 0.35, "pack": 0.25},
                "answer_conf": round(((seed >> 12) % 900) / 1000.0 + 0.05, 4),
            },
            "model_versions": {"hands": "fake-hands-1.0",
                               "active": "fake-active-1.0"},
        }

    def decide_batch(self, segments: list, signals: list) -> list:
        return [self.decide(s, g) for s, g in zip(segments, signals)]


def _fake_signals(segment: dict) -> dict:
    digest = hashlib.sha256(segment["segment_id"].encode("utf-8")).digest()
    seed = int.from_bytes(digest[:4], "big")
    return {
        "decodable": True,
        "blur_var": round(60.0 + (seed % 5000) / 100.0, 4),
        "brightness_mean": round(40.0 + (seed % 9000) / 100.0, 4),
        "under_exposed_frac": round((seed % 7) / 1000.0, 4),
        "over_exposed_frac": round((seed % 11) / 1000.0, 4),
        "motion_mean": round((seed % 600) / 100.0, 4),
        "hands_visible_frac": round(0.5 + (seed % 400) / 1000.0, 4),
        "hand_count_mode": seed % 3,
        "faces_found": seed % 4,
        "faces_blurred": seed % 4,
        "phash": hashlib.sha256(digest).hexdigest()[:16],
        "dup_of": None,
    }


def _fake_verdict(segment: dict, decisions: dict) -> dict:
    conf = decisions["step"]["answer_conf"]
    return {
        "segment_id": segment["segment_id"],
        "verdict": "accept" if conf >= 0.5 else "review",
        "reasons": [] if conf >= 0.5 else ["LOW_CONFIDENCE"],
        "answer_conf": conf,
        "threshold": 0.5,
    }


class AloneVsBatchPropertiesTest(unittest.TestCase):
    def test_alone_matches_batch_of_five(self):
        """Seed: 3617. 40 random batches of 5 (200 segments): the receipt
        of a segment decided alone equals its receipt from ONE batch call."""
        rnd = random.Random(3617)
        decider = FakeDecider()
        config = {
            "thresholds": {"target_error": 0.05},
            "pipeline_version": "9.9.9-test",
            "versions": {"model": "fake-1.0"},
        }
        for b in range(40):
            tags = [f"ab-{b}-{rnd.randint(0, 10 ** 9)}-{i}" for i in range(5)]
            segments = [_segment(t) for t in tags]
            sigs = [_fake_signals(s) for s in segments]
            batched = decider.decide_batch(segments, sigs)  # ONE call of 5
            for i, segment in enumerate(segments):
                solo_dec = decider.decide(segment, sigs[i])
                solo = receipts.make_receipt(
                    segment, sigs[i], solo_dec,
                    _fake_verdict(segment, solo_dec), config,
                    created_utc="2026-10-10T09:00:00+00:00",
                )
                joint = receipts.make_receipt(
                    segment, sigs[i], batched[i],
                    _fake_verdict(segment, batched[i]), config,
                    created_utc="2026-10-10T11:00:00+00:00",
                )
                self.assertEqual(solo["receipt_sha256"], joint["receipt_sha256"])
                self.assertEqual(solo["receipt_id"], joint["receipt_id"])


class VerifyReceiptPropertiesTest(unittest.TestCase):
    def test_verify_receipt_names_the_mutated_path(self):
        """Seed: 3618. 200 random single-leaf mutations: verify_receipt
        reports match=False and names the mutated path in diff."""
        rnd = random.Random(3618)
        base = _base_receipt("verifybase")
        leaves = [
            p for p in _scalar_leaves(base)
            if not any(p == e or p.startswith(e + ".") for e in EXCLUDED)
        ]
        for _ in range(200):
            path = rnd.choice(leaves)
            tampered = copy.deepcopy(base)
            _set_path(tampered, path,
                      _mutated_value(_get_path(tampered, path), rnd))
            result = receipts.verify_receipt(base, lambda _r, t=tampered: t)
            self.assertFalse(result["match"])
            self.assertIn(path, result["diff"])


if __name__ == "__main__":
    unittest.main()
