"""Tests for jadal/receipts.py (canonical JSON, receipt hashing, re-verify)."""
from __future__ import annotations

import hashlib
import unittest

from jadal import receipts
from jadal.config import Config


def _fake_signals(segment: dict) -> dict:
    """Deterministic fake L0 signals derived from the segment id only."""
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


class FakeDecider:
    """A deterministic decider: same segment in, same decisions out, always.

    Batch position, neighbour segments and wall-clock time cannot reach it, so a
    segment hashed alone must hash identically inside a batch of five.
    """

    model_versions = {"hands": "fake-hands-1.0", "active": "fake-active-1.0"}

    def decide(self, segment: dict, signals: dict) -> dict:
        digest = hashlib.sha256(("decide" + segment["segment_id"]).encode("utf-8")).digest()
        seed = int.from_bytes(digest[:4], "big")
        return {
            "hands_ok": {"p_yes": round((seed % 900) / 1000.0 + 0.05, 4)},
            "active_work": {"p_yes": round(((seed >> 8) % 900) / 1000.0 + 0.05, 4)},
            "step": {
                "choice": ["trim", "press", "pack"][seed % 3] if seed % 5 else None,
                "probs": {"trim": 0.4, "press": 0.35, "pack": 0.25},
                "answer_conf": round(((seed >> 12) % 900) / 1000.0 + 0.05, 4),
            },
            "model_versions": dict(self.model_versions),
        }


def _fake_verdict(segment: dict, decisions: dict, threshold: float) -> dict:
    return {
        "segment_id": segment["segment_id"],
        "verdict": "accept" if decisions["step"]["answer_conf"] >= threshold else "review",
        "reasons": [] if decisions["step"]["answer_conf"] >= threshold else ["LOW_CONFIDENCE"],
        "answer_conf": decisions["step"]["answer_conf"],
        "threshold": threshold,
    }


class CanonicalJsonTest(unittest.TestCase):

    def test_key_order_does_not_change_the_bytes(self):
        a = {"b": 1, "a": {"y": 2, "x": [1, 2]}}
        b = {"a": {"x": [1, 2], "y": 2}, "b": 1}
        self.assertEqual(receipts.canonical_json(a), receipts.canonical_json(b))

    def test_uses_compact_separators_and_sorted_keys(self):
        obj = {"b": 1, "a": 2}
        self.assertEqual(receipts.canonical_json(obj), b'{"a":2,"b":1}')

    def test_non_ascii_is_utf8_not_escaped(self):
        obj = {"text": "कपड़े", "lang": "hi-IN"}
        raw = receipts.canonical_json(obj)
        self.assertIn("कपड़े".encode("utf-8"), raw)
        self.assertNotIn(b"\\u", raw)

    def test_output_is_bytes_and_stable_across_calls(self):
        obj = {"x": [1, 2, 3], "y": {"z": None}}
        first = receipts.canonical_json(obj)
        self.assertIsInstance(first, bytes)
        self.assertEqual(first, receipts.canonical_json(dict(obj)))


def _segment(i: int = 0) -> dict:
    return {
        "segment_id": f"seg-{i:03d}",
        "source_clip": f"clip-{i:03d}.mp4",
        "worker_id": f"w-{i:03d}",
        "site_id": "site-1",
        "start_s": round(i * 20.0, 4),
        "end_s": round(i * 20.0 + 20.0, 4),
        "path": f"data_runtime/segments/seg-{i:03d}.mp4",
        "sha256": hashlib.sha256(f"seg-{i:03d}".encode("utf-8")).hexdigest(),
    }


def _receipt_for(i: int = 0, config=None, created_utc="2026-10-10T00:00:00+00:00") -> dict:
    segment = _segment(i)
    signals = _fake_signals(segment)
    decisions = FakeDecider().decide(segment, signals)
    verdict = _fake_verdict(segment, decisions, 0.5)
    return receipts.make_receipt(
        segment,
        signals,
        decisions,
        verdict,
        config if config is not None else Config(),
        created_utc=created_utc,
    )


class MakeReceiptTest(unittest.TestCase):

    def test_receipt_matches_the_spec_fields(self):
        receipt = _receipt_for()
        self.assertEqual(
            sorted(receipt),
            sorted(
                [
                    "receipt_id",
                    "segment_sha256",
                    "signals",
                    "decisions",
                    "verdict",
                    "config",
                    "receipt_sha256",
                    "created_utc",
                ]
            ),
        )
        self.assertEqual(receipt["segment_sha256"], _segment()["sha256"])
        self.assertEqual(receipt["created_utc"], "2026-10-10T00:00:00+00:00")

    def test_receipt_id_is_the_first_sixteen_hex_chars_of_the_hash(self):
        receipt = _receipt_for()
        self.assertEqual(receipt["receipt_id"], receipt["receipt_sha256"][:16])
        self.assertEqual(len(receipt["receipt_id"]), 16)

    def test_floats_are_rounded_to_four_decimals(self):
        # unrounded raw values go in; the receipt normalizes them on the way in
        segment = _segment(0)
        signals = dict(_fake_signals(segment))
        signals["blur_var"] = 123.456789
        signals["brightness_mean"] = -0.00001
        decisions = FakeDecider().decide(segment, signals)
        decisions["step"]["answer_conf"] = 0.99999
        verdict = _fake_verdict(segment, decisions, 0.123456)
        receipt = receipts.make_receipt(
            segment, signals, decisions, verdict, Config(),
            created_utc="2026-10-10T00:00:00+00:00",
        )
        self.assertEqual(receipt["signals"]["blur_var"], 123.4568)
        self.assertEqual(receipt["signals"]["brightness_mean"], 0.0)
        self.assertEqual(receipt["verdict"]["threshold"], 0.1235)
        self.assertEqual(receipt["verdict"]["answer_conf"], 1.0)
        self.assertEqual(receipt["decisions"]["step"]["answer_conf"], 1.0)
        self.assertNotIn("-0.0", receipts.canonical_json(receipt).decode("utf-8"))

    def test_config_dataclass_becomes_a_config_block(self):
        receipt = _receipt_for(config=Config())
        config_block = receipt["config"]
        self.assertEqual(config_block["pipeline_version"], Config().pipeline_version)
        self.assertIn("thresholds", config_block)
        self.assertIn("versions", config_block)
        self.assertEqual(config_block["thresholds"]["blur_var_min"], 60.0)
        self.assertEqual(config_block["thresholds"]["target_error"], 0.05)
        self.assertEqual(config_block["versions"]["python"], ".".join(map(str, __import__("sys").version_info[:3])))

    def test_config_mapping_is_used_as_is(self):
        block = {
            "thresholds": {"target_error": 0.1},
            "pipeline_version": "9.9.9",
            "versions": {"model": "fake-1.0"},
        }
        receipt = _receipt_for(config=block)
        self.assertEqual(receipt["config"], block)

    def test_different_config_gives_a_different_hash(self):
        a = _receipt_for(config=Config())
        b = _receipt_for(config=Config(target_error=0.2))
        self.assertNotEqual(
            receipts.receipt_hash(a), receipts.receipt_hash(b)
        )

    def test_created_utc_defaults_to_now_and_is_isoformatted(self):
        segment = _segment()
        receipt = receipts.make_receipt(
            segment,
            _fake_signals(segment),
            FakeDecider().decide(segment, _fake_signals(segment)),
            _fake_verdict(segment, FakeDecider().decide(segment, _fake_signals(segment)), 0.5),
            Config(),
        )
        self.assertRegex(receipt["created_utc"], r"^\d{4}-\d{2}-\d{2}T")

    def test_rejects_a_non_mapping_config(self):
        with self.assertRaises(ValueError):
            _receipt_for(config=["not", "a", "config"])


class ReceiptHashTest(unittest.TestCase):

    def test_hash_ignores_key_order_and_excluded_fields(self):
        receipt = _receipt_for()
        reordered = dict(reversed(list(receipt.items())))
        self.assertEqual(
            receipts.receipt_hash(receipt), receipts.receipt_hash(reordered)
        )

    def test_hash_ignores_created_utc_receipt_id_and_hash_itself(self):
        receipt = _receipt_for()
        moved = dict(receipt)
        moved["created_utc"] = "1999-01-01T00:00:00+00:00"
        moved["receipt_id"] = "0" * 16
        moved["receipt_sha256"] = "f" * 64
        self.assertEqual(receipts.receipt_hash(moved), receipt["receipt_sha256"])

    def test_hash_changes_when_a_hashed_field_is_perturbed(self):
        receipt = _receipt_for()
        other = _receipt_for(1)  # a different segment hashes differently
        self.assertNotEqual(receipt["receipt_sha256"], other["receipt_sha256"])
        tampered = dict(receipt)
        tampered["signals"] = dict(receipt["signals"], blur_var=999.0)
        self.assertNotEqual(receipts.receipt_hash(tampered), receipt["receipt_sha256"])

    def test_hash_equals_sha256_of_canonical_payload(self):
        receipt = _receipt_for()
        payload = {
            k: v
            for k, v in receipt.items()
            if k not in {"receipt_sha256", "receipt_id", "created_utc"}
        }
        self.assertEqual(receipt["receipt_sha256"], hashlib.sha256(
            receipts.canonical_json(payload)).hexdigest())


class SegmentIsBatchIndependentTest(unittest.TestCase):
    """SPEC: a segment must hash identically alone or inside a batch of five."""

    @staticmethod
    def _process(segments, decider, threshold, created_utc):
        out = []
        for segment in segments:  # batch size 1: one decider call per segment
            signals = _fake_signals(segment)
            decisions = decider.decide(segment, signals)
            verdict = _fake_verdict(segment, decisions, threshold)
            out.append(
                receipts.make_receipt(
                    segment, signals, decisions, verdict, Config(),
                    created_utc=created_utc,
                )
            )
        return out

    def test_same_hash_alone_and_inside_a_batch_of_five(self):
        decider = FakeDecider()
        batch = [_segment(i) for i in range(5)]
        target = batch[2]
        alone = self._process([target], decider, 0.5, "2026-10-10T10:00:00+00:00")
        in_batch = self._process(batch, decider, 0.5, "2026-10-10T11:00:00+00:00")
        self.assertEqual(len(in_batch), 5)
        self.assertEqual(
            alone[0]["receipt_sha256"], in_batch[2]["receipt_sha256"]
        )
        self.assertEqual(alone[0]["receipt_id"], in_batch[2]["receipt_id"])
        # the neighbours do not matter either: seg-003 hashes the same with
        # neighbours {0, 1, 2, 4} as with neighbours {0, 1, 4}
        reduced = self._process(
            batch[:2] + batch[3:], decider, 0.5, "2026-10-10T12:00:00+00:00"
        )
        self.assertEqual(len(reduced), 4)
        self.assertEqual(reduced[2]["receipt_sha256"], in_batch[3]["receipt_sha256"])
        self.assertNotEqual(alone[0]["created_utc"], in_batch[2]["created_utc"])

    def test_every_segment_in_the_batch_matches_its_alone_hash(self):
        decider = FakeDecider()
        threshold = 0.5
        in_batch = self._process(
            [_segment(i) for i in range(5)], decider, threshold, "2026-10-10T11:00:00+00:00"
        )
        for i, receipt in enumerate(in_batch):
            alone = self._process(
                [_segment(i)], decider, threshold, f"2026-10-10T1{i}:00:00+00:00"
            )
            self.assertEqual(alone[0]["receipt_sha256"], receipt["receipt_sha256"], f"seg {i}")

    def test_segment_sha256_is_the_hash_of_the_original_segment_file(self):
        receipt = _receipt_for()
        self.assertEqual(receipt["segment_sha256"], _segment()["sha256"])
        self.assertNotIn("blurred_sha256", receipt["signals"])


class VerifyReceiptTest(unittest.TestCase):

    def test_matching_receipt_verifies(self):
        receipt = _receipt_for()
        result = receipts.verify_receipt(receipt, lambda r: _receipt_for())
        self.assertTrue(result["match"])
        self.assertEqual(result["expected"], receipt["receipt_sha256"])
        self.assertEqual(result["actual"], receipt["receipt_sha256"])
        self.assertEqual(result["diff"], [])

    def test_recompute_fn_receives_the_stored_receipt(self):
        receipt = _receipt_for()
        seen = []

        def recompute(stored):
            seen.append(stored)
            return _receipt_for()

        receipts.verify_receipt(receipt, recompute)
        self.assertEqual(seen, [receipt])

    def test_perturbed_nested_value_reports_the_field_path(self):
        receipt = _receipt_for()
        good = _receipt_for()

        def recompute(stored):
            tampered = dict(good)
            tampered["verdict"] = dict(good["verdict"], answer_conf=0.1111)
            return tampered

        result = receipts.verify_receipt(receipt, recompute)
        self.assertFalse(result["match"])
        self.assertEqual(result["diff"], ["verdict.answer_conf"])
        self.assertEqual(result["expected"], receipt["receipt_sha256"])
        self.assertNotEqual(result["actual"], receipt["receipt_sha256"])

    def test_perturbed_signals_and_section_report_paths(self):
        receipt = _receipt_for()
        good = _receipt_for()

        def recompute(stored):
            tampered = dict(good)
            tampered["signals"] = dict(good["signals"], blur_var=1.0, motion_mean=2.0)
            del tampered["decisions"]["step"]["probs"]
            tampered["config"] = dict(good["config"], pipeline_version="0.0.0")
            return tampered

        result = receipts.verify_receipt(receipt, recompute)
        self.assertFalse(result["match"])
        self.assertEqual(
            sorted(result["diff"]),
            sorted(
                [
                    "signals.blur_var",
                    "signals.motion_mean",
                    "decisions.step.probs",
                    "config.pipeline_version",
                ]
            ),
        )

    def test_added_key_in_a_nested_dict_is_reported(self):
        receipt = _receipt_for()
        good = _receipt_for()

        def recompute(stored):
            tampered = dict(good)
            tampered["signals"] = dict(good["signals"], extra_signal=1)
            return tampered

        result = receipts.verify_receipt(receipt, recompute)
        self.assertFalse(result["match"])
        self.assertEqual(result["diff"], ["signals.extra_signal"])

    def test_missing_receipt_hash_is_a_mismatch(self):
        receipt = _receipt_for()
        broken = {k: v for k, v in receipt.items() if k != "receipt_sha256"}
        result = receipts.verify_receipt(broken, lambda r: _receipt_for())
        self.assertFalse(result["match"])
        self.assertIn("receipt_sha256", result["diff"])

    def test_created_utc_difference_still_verifies(self):
        receipt = _receipt_for(created_utc="2026-10-10T00:00:00+00:00")
        result = receipts.verify_receipt(
            receipt, lambda r: _receipt_for(created_utc="2027-01-01T09:00:00+00:00")
        )
        self.assertTrue(result["match"])

    def test_a_non_mapping_recompute_result_raises(self):
        receipt = _receipt_for()
        with self.assertRaises(ValueError):
            receipts.verify_receipt(receipt, lambda r: ["not", "a", "receipt"])


if __name__ == "__main__":
    unittest.main()
