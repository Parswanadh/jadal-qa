"""Tests for jadal/funnel.py (GitHub issue #26, Phase 1 interface).

Pure-function contract: no I/O, no randomness, no clock. Every test builds
its own signals/decisions/thresholds fixtures inline.
"""
from __future__ import annotations

import copy
import hashlib
import json
import unittest

from jadal import funnel
from jadal.funnel import queue_rank_key, route


THRESHOLDS = {
    "version": "test",
    "blur_var_min": 60.0,
    "brightness_min": 40.0,
    "under_exposed_frac_max": 0.35,
    "over_exposed_frac_max": 0.35,
    "motion_max": 12.0,
    "motion_min": 0.5,
    "hands_visible_frac_min": 0.5,
    "min_duration_s": 5.0,
    "accept_conf": 0.9,
}

GOOD_SIGNALS = {
    "decodable": True,
    "duration_s": 10.0,
    "blur_var": 120.0,
    "brightness_mean": 110.0,
    "under_exposed_frac": 0.02,
    "over_exposed_frac": 0.02,
    "motion_mean": 4.0,
    "hands_visible_frac": 0.9,
    "dup_of": None,
}

GOOD_DECISIONS = {
    "hands_ok": {"p_yes": 0.95},
    "active_work": {"p_yes": 0.96},
    "step": {"choice": "assemble", "probs": {"assemble": 0.97},
             "answer_conf": 0.97},
}


def sig(**over) -> dict:
    out = copy.deepcopy(GOOD_SIGNALS)
    out.update(over)
    return out


def dec(**over) -> dict:
    out = copy.deepcopy(GOOD_DECISIONS)
    out.update(over)
    return out


class TestAccept(unittest.TestCase):
    def test_clean_clip_accepts(self):
        out = route(sig(), dec(), THRESHOLDS)
        self.assertEqual(out["verdict"], "accept")
        self.assertEqual(out["reasons"], [])
        self.assertEqual(out["answer_conf"], 0.95)  # min of heads
        self.assertEqual(out["queue_priority"], 0.0)

    def test_answer_conf_is_min_of_heads(self):
        out = route(sig(), dec(active_work={"p_yes": 0.91}), THRESHOLDS)
        self.assertEqual(out["verdict"], "accept")
        self.assertEqual(out["answer_conf"], 0.91)


class TestHardReject(unittest.TestCase):
    def check_reject(self, signals, code):
        out = route(signals, dec(), THRESHOLDS)
        self.assertEqual(out["verdict"], "reject", signals)
        self.assertIn(code, out["reasons"])
        codes = [e["code"] for e in out["evidence"]]
        self.assertIn(code, codes)
        entry = next(e for e in out["evidence"] if e["code"] == code)
        for key in ("check", "value", "op", "threshold", "code"):
            self.assertIn(key, entry)

    def test_undecodable(self):
        self.check_reject(sig(decodable=False), "UNDECODABLE")

    def test_too_short(self):
        self.check_reject(sig(duration_s=2.0), "UNDECODABLE")

    def test_blur(self):
        self.check_reject(sig(blur_var=10.0), "BLUR")

    def test_too_dark_brightness(self):
        self.check_reject(sig(brightness_mean=5.0), "TOO_DARK")

    def test_too_dark_under_exposed(self):
        self.check_reject(sig(under_exposed_frac=0.9), "TOO_DARK")

    def test_overexposed(self):
        self.check_reject(sig(over_exposed_frac=0.9), "OVEREXPOSED")

    def test_shaky(self):
        self.check_reject(sig(motion_mean=50.0), "SHAKY")

    def test_static_maps_to_no_active_work(self):
        self.check_reject(sig(motion_mean=0.0), "NO_ACTIVE_WORK")

    def test_hands_not_visible(self):
        self.check_reject(sig(hands_visible_frac=0.1), "HANDS_NOT_VISIBLE")

    def test_duplicate(self):
        self.check_reject(sig(dup_of="seg0001"), "DUPLICATE")


class TestL1CannotReject(unittest.TestCase):
    def test_bad_heads_only_review(self):
        bad = {"hands_ok": {"p_yes": 0.1}, "active_work": {"p_yes": 0.1},
               "step": {"choice": None, "probs": {}, "answer_conf": 0.1}}
        out = route(sig(), bad, THRESHOLDS)
        self.assertEqual(out["verdict"], "review")
        self.assertNotEqual(out["verdict"], "reject")

    def test_wrong_step_only_reviews(self):
        decisions = dec(step={"choice": None, "probs": {}, "answer_conf": 0.99})
        out = route(sig(), decisions, THRESHOLDS)
        self.assertEqual(out["verdict"], "review")
        self.assertIn("WRONG_STEP", out["reasons"])

    def test_low_step_conf_only_reviews(self):
        decisions = dec(step={"choice": "assemble", "probs": {"assemble": 0.4},
                              "answer_conf": 0.4})
        out = route(sig(), decisions, THRESHOLDS)
        self.assertEqual(out["verdict"], "review")
        self.assertIn("LOW_CONFIDENCE", out["reasons"])

    def test_inactive_work_reviews_not_rejects(self):
        out = route(sig(), dec(active_work={"p_yes": 0.2}), THRESHOLDS)
        self.assertEqual(out["verdict"], "review")
        self.assertIn("NO_ACTIVE_WORK", out["reasons"])


class TestOrderingAndDeterminism(unittest.TestCase):
    def test_multiple_failures_all_listed_stable_order(self):
        signals = sig(decodable=False, blur_var=1.0, brightness_mean=1.0,
                      over_exposed_frac=0.9, motion_mean=99.0,
                      hands_visible_frac=0.0, dup_of="seg0001")
        first = route(signals, dec(), THRESHOLDS)
        # input dict order must not matter: reversed insertion, same output
        shuffled = dict(reversed(list(signals.items())))
        second = route(shuffled, dec(), THRESHOLDS)
        self.assertEqual(first["verdict"], "reject")
        self.assertGreaterEqual(len(first["reasons"]), 6)
        order = funnel.REASON_ORDER
        self.assertEqual(first["reasons"],
                         sorted(first["reasons"], key=order.index))
        self.assertEqual(first, second)

    def test_hash_identical_across_calls(self):
        payload = route(sig(), dec(), THRESHOLDS)
        digest = lambda obj: hashlib.sha256(
            json.dumps(obj, sort_keys=True).encode()).hexdigest()
        self.assertEqual(digest(payload),
                         digest(route(sig(), dec(), THRESHOLDS)))

    def test_evidence_floats_rounded_to_4(self):
        signals = sig(blur_var=59.123456789)
        out = route(signals, dec(), THRESHOLDS)
        for entry in out["evidence"]:
            for key in ("value", "threshold"):
                value = entry[key]
                if isinstance(value, float):
                    self.assertEqual(value, round(value, 4))

    def test_queue_priority_monotone_in_uncertainty(self):
        # Both clips demote to review (hands head below the 0.9 gate); the
        # 0.55 clip is nearer the decision boundary => higher priority.
        close = dec(hands_ok={"p_yes": 0.55})
        far = dec(hands_ok={"p_yes": 0.85})
        out_close = route(sig(), close, THRESHOLDS)
        out_far = route(sig(), far, THRESHOLDS)
        self.assertEqual(out_close["verdict"], "review")
        self.assertEqual(out_far["verdict"], "review")
        self.assertGreater(out_close["queue_priority"],
                           out_far["queue_priority"])

    def test_rank_key_tie_break(self):
        items = [{"segment_id": "b", "queue_priority": 0.5},
                 {"segment_id": "a", "queue_priority": 0.5},
                 {"segment_id": "c", "queue_priority": 0.9}]
        ranked = sorted(items, key=lambda it: queue_rank_key(
            it["segment_id"], it["queue_priority"]))
        self.assertEqual([it["segment_id"] for it in ranked], ["c", "a", "b"])


class TestBoundaries(unittest.TestCase):
    def test_exact_threshold_passes(self):
        signals = sig(blur_var=60.0, brightness_mean=40.0,
                      over_exposed_frac=0.35, motion_mean=12.0,
                      hands_visible_frac=0.5, duration_s=5.0)
        out = route(signals, dec(), THRESHOLDS)
        self.assertEqual(out["verdict"], "accept")

    def test_accept_conf_boundary(self):
        decisions = {"hands_ok": {"p_yes": 0.9}, "active_work": {"p_yes": 0.9},
                     "step": {"choice": "s", "probs": {}, "answer_conf": 0.9}}
        out = route(sig(), decisions, THRESHOLDS)
        self.assertEqual(out["verdict"], "accept")
        just_below = {"hands_ok": {"p_yes": 0.8999},
                      "active_work": {"p_yes": 0.9},
                      "step": {"choice": "s", "probs": {},
                               "answer_conf": 0.9}}
        out2 = route(sig(), just_below, THRESHOLDS)
        self.assertEqual(out2["verdict"], "review")


class TestMissingFields(unittest.TestCase):
    def test_missing_l1_reviews_never_crashes_never_accepts(self):
        for decisions in (None, {}, {"hands_ok": {}}, {"step": {}}):
            out = route(sig(), decisions, THRESHOLDS)
            self.assertEqual(out["verdict"], "review", decisions)
            self.assertIn("LOW_CONFIDENCE", out["reasons"])
        self.assertIsNone(route(sig(), None, THRESHOLDS)["answer_conf"])
        self.assertIsNone(route(sig(), {}, THRESHOLDS)["answer_conf"])

    def test_missing_l0_never_rejects(self):
        out = route({}, dec(), THRESHOLDS)
        self.assertIn(out["verdict"], ("review",))
        self.assertIn("LOW_CONFIDENCE", out["reasons"])

    def test_missing_dup_of_is_fine(self):
        signals = sig()
        del signals["dup_of"]
        out = route(signals, dec(), THRESHOLDS)
        self.assertEqual(out["verdict"], "accept")

    def test_none_signals(self):
        out = route(None, None, THRESHOLDS)
        self.assertEqual(out["verdict"], "review")


class TestThresholdHelpers(unittest.TestCase):
    def test_hash_stable_and_pins_body(self):
        loaded = funnel.load_thresholds()
        h1 = funnel.thresholds_hash(loaded)
        h2 = funnel.thresholds_hash(dict(loaded))
        self.assertEqual(h1, h2)
        self.assertEqual(len(h1), 64)
        changed = dict(loaded)
        changed["accept_conf"] = 0.5
        self.assertNotEqual(funnel.thresholds_hash(changed), h1)

    def test_load_thresholds_defaults(self):
        loaded = funnel.load_thresholds()
        self.assertEqual(loaded["blur_var_min"], 60.0)
        self.assertEqual(loaded["calibration"]["status"], "placeholder")


def _mk_record(blur, conf, label, source="synthetic"):
    signals = dict(GOOD_SIGNALS)
    signals["blur_var"] = blur
    decisions = {"hands_ok": {"p_yes": conf},
                 "active_work": {"p_yes": conf},
                 "step": {"choice": "assemble", "probs": {},
                          "answer_conf": conf}}
    return {"signals": signals, "decisions": decisions, "label": label,
            "reason_gold": None, "label_source": source}


class TestCalibrateTwoSided(unittest.TestCase):
    def test_accept_certifies_separable(self):
        records = [_mk_record(150.0, 0.95 + 0.04 * (i % 10) / 9, "good")
                   for i in range(80)]
        records += [_mk_record(150.0, 0.1 + 0.4 * (i % 10) / 9, "bad")
                    for i in range(80)]
        # alpha 0.1: 40 clean calib clips suffice (bound(0,40) ~= 0.056).
        result = funnel.calibrate_two_sided(
            records, {"min_accepted": 10, "seed": 5, "target_alpha": 0.1})
        accept = result["accept"]
        self.assertIsNotNone(accept["threshold"])
        self.assertLessEqual(accept["calib_certified"]["upper_bound"], 0.1)
        self.assertIn("heldout_by_source", accept)
        self.assertEqual(accept["heldout_by_source"]["synthetic"]["fired"],
                         accept["heldout"]["fired"])
        meta = result["meta"]
        self.assertEqual(meta["n_total"], 160)
        self.assertEqual(meta["by_source"], {"synthetic": 160})
        self.assertEqual(meta["n_calib"] + meta["n_heldout"], 160)

    def test_reject_clean_rule_certifies(self):
        records = [_mk_record(5.0 + 15.0 * i / 99, 0.99, "bad")
                   for i in range(100)]
        records += [_mk_record(100.0 + 100.0 * i / 99, 0.99, "good")
                    for i in range(100)]
        result = funnel.calibrate_two_sided(
            records, {"seed": 3, "min_accepted": 1000})
        entry = result["reject_rules"]["blur_var_min"]
        self.assertFalse(entry["disabled"])
        self.assertIsNotNone(entry["threshold"])
        self.assertEqual(entry["calib"]["errors"], 0)
        self.assertLess(entry["threshold"], 150.0)
        # accept side disabled here (min_accepted unreachable) — honest None
        self.assertIsNone(result["accept"]["threshold"])
        self.assertIsNone(result["threshold_updates"]["accept_conf"])

    def test_reject_noisy_rule_disabled(self):
        records = [_mk_record(5.0 + 195.0 * i / 99, 0.99,
                              "bad" if i % 2 else "good")
                   for i in range(100)]
        result = funnel.calibrate_two_sided(records, {"seed": 3})
        entry = result["reject_rules"]["blur_var_min"]
        self.assertTrue(entry["disabled"])
        self.assertIsNone(entry["threshold"])
        self.assertIsNone(result["threshold_updates"]["blur_var_min"])

    def test_mixed_sources_reported_per_source(self):
        records = [_mk_record(150.0, 0.97, "good", "synthetic")
                   for _ in range(50)]
        records += [_mk_record(150.0, 0.97, "good", "gemini")
                    for _ in range(50)]
        records += [_mk_record(150.0, 0.2, "bad", "synthetic")
                    for _ in range(50)]
        records += [_mk_record(150.0, 0.2, "bad", "gemini")
                    for _ in range(50)]
        result = funnel.calibrate_two_sided(
            records, {"min_accepted": 10, "seed": 11})
        self.assertEqual(result["meta"]["by_source"],
                         {"synthetic": 100, "gemini": 100})
        by_source = result["accept"]["heldout_by_source"]
        self.assertEqual(set(by_source), {"synthetic", "gemini"})

    def test_deterministic_and_order_independent(self):
        records = [_mk_record(150.0, 0.97, "good") for _ in range(30)]
        records += [_mk_record(10.0, 0.2, "bad") for _ in range(30)]
        first = funnel.calibrate_two_sided(records, {"seed": 9})
        second = funnel.calibrate_two_sided(list(reversed(records)),
                                            {"seed": 9})
        digest = lambda obj: hashlib.sha256(
            json.dumps(obj, sort_keys=True).encode()).hexdigest()
        strip = lambda res: {k: v for k, v in res.items()}  # date_utc differs
        first["meta"].pop("date_utc")
        second["meta"].pop("date_utc")
        self.assertEqual(digest(strip(first)), digest(strip(second)))

    def test_rejects_unknown_source_and_bad_label(self):
        good = _mk_record(150.0, 0.97, "good")
        bad_source = dict(good, label_source="crowd_unknown")
        with self.assertRaises(ValueError):
            funnel.calibrate_two_sided([good, bad_source])
        bad_label = dict(good, label="maybe")
        with self.assertRaises(ValueError):
            funnel.calibrate_two_sided([good, bad_label])
        with self.assertRaises(ValueError):
            funnel.calibrate_two_sided([])


if __name__ == "__main__":
    unittest.main()
