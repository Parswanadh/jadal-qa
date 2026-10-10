"""Tests for jadal/funnel.py (GitHub issue #26; fix round 1 rulings C1-C4, I1-I2).

Pure-function contract: no I/O, no randomness (fuzz uses random.Random with
fixed seeds), no clock. Every test builds its own fixtures inline, except
TestGoldRecordsBuild which uses tmp dirs (never eval/gold/).
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import random
import tempfile
import unittest
from pathlib import Path

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
    # Round 2: this fixture is an explicitly evidence-rejecting,
    # legacy-hands quality gate (no 'profiles' map => hands enforced
    # everywhere, as before). Default-policy/profile behaviour is pinned
    # in tests/test_funnel_round2.py.
    "hard_reject_policy": "evidence_reject",
}
# THRESHOLDS carries no calibration block => uncalibrated placeholder-like:
# accept requires allow_uncalibrated_accept=True (the demo pattern).

CALIBRATED = {"calibration": {"status": "calibrated"}}

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

REQUIRED_L0 = ["decodable", "duration_s", "blur_var", "brightness_mean",
               "under_exposed_frac", "over_exposed_frac", "motion_mean",
               "hands_visible_frac"]


def sig(**over) -> dict:
    out = copy.deepcopy(GOOD_SIGNALS)
    out.update(over)
    return out


def dec(**over) -> dict:
    out = copy.deepcopy(GOOD_DECISIONS)
    out.update(over)
    return out


def accept_route(signals, decisions, thresholds=THRESHOLDS):
    """route() with the demo's explicit uncalibrated-accept flag."""
    return route(signals, decisions, thresholds,
                 allow_uncalibrated_accept=True)


class TestAccept(unittest.TestCase):
    def test_clean_clip_accepts(self):
        out = accept_route(sig(), dec())
        self.assertEqual(out["verdict"], "accept")
        self.assertEqual(out["reasons"], [])
        self.assertEqual(out["answer_conf"], 0.95)  # min of heads
        self.assertEqual(out["queue_priority"], 0.0)

    def test_answer_conf_is_min_of_heads(self):
        out = accept_route(sig(), dec(active_work={"p_yes": 0.91}))
        self.assertEqual(out["verdict"], "accept")
        self.assertEqual(out["answer_conf"], 0.91)

    def test_uncalibrated_flag_carried_in_evidence(self):
        out = accept_route(sig(), dec())
        flags = [e for e in out["evidence"]
                 if e["check"] == "allow_uncalibrated_accept"]
        self.assertEqual(len(flags), 1)
        self.assertTrue(flags[0]["value"])


class TestCalibrationGate(unittest.TestCase):
    def test_placeholder_status_never_accepts(self):
        out = route(sig(), dec(), THRESHOLDS)  # no calibration block
        self.assertEqual(out["verdict"], "review")
        self.assertEqual(out["reasons"], ["LOW_CONFIDENCE"])
        checks = [e["check"] for e in out["evidence"]]
        self.assertIn("accept_uncalibrated", checks)

    def test_explicit_placeholder_status_never_accepts(self):
        thresholds = dict(THRESHOLDS,
                          calibration={"status": "placeholder"})
        out = route(sig(), dec(), thresholds)
        self.assertEqual(out["verdict"], "review")

    def test_flag_allows_demo_accept(self):
        out = route(sig(), dec(), THRESHOLDS,
                    allow_uncalibrated_accept=True)
        self.assertEqual(out["verdict"], "accept")

    def test_calibrated_status_accepts_without_flag(self):
        thresholds = dict(THRESHOLDS, **CALIBRATED)
        out = route(sig(), dec(), thresholds)
        self.assertEqual(out["verdict"], "accept")
        self.assertNotIn("allow_uncalibrated_accept",
                         [e["check"] for e in out["evidence"]])

    def test_provisional_status_routes_to_review_without_flag(self):
        # Round 2 (audit N1): provisional no longer auto-accepts. A clean,
        # confident clip on provisional thresholds reviews unless the
        # caller passes allow_provisional_accept=True.
        thresholds = dict(THRESHOLDS,
                          calibration={"status": "provisional",
                                       "label_source": "synthetic"})
        out = route(sig(), dec(), thresholds)
        self.assertEqual(out["verdict"], "review")
        self.assertIn("accept_provisional",
                      [e["check"] for e in out["evidence"]])

    def test_provisional_flag_allows_accept_with_status_evidence(self):
        thresholds = dict(THRESHOLDS,
                          calibration={"status": "provisional",
                                       "label_source": "synthetic"})
        out = route(sig(), dec(), thresholds,
                    allow_provisional_accept=True)
        self.assertEqual(out["verdict"], "accept")
        status = [e for e in out["evidence"]
                  if e["check"] == "accept_status"]
        self.assertEqual(len(status), 1)
        self.assertEqual(status[0]["value"], "provisional")
        self.assertEqual(status[0]["label_source"], "synthetic")


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
        payload = accept_route(sig(), dec())
        digest = lambda obj: hashlib.sha256(
            json.dumps(obj, sort_keys=True).encode()).hexdigest()
        self.assertEqual(digest(payload),
                         digest(accept_route(sig(), dec())))

    def test_evidence_floats_rounded_to_4(self):
        signals = sig(blur_var=59.123456789)
        out = route(signals, dec(), THRESHOLDS)
        for entry in out["evidence"]:
            for key in ("value", "threshold"):
                value = entry[key]
                if isinstance(value, float) and value == value:
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
        out = accept_route(signals, dec())
        self.assertEqual(out["verdict"], "accept")

    def test_accept_conf_boundary(self):
        decisions = {"hands_ok": {"p_yes": 0.9}, "active_work": {"p_yes": 0.9},
                     "step": {"choice": "s", "probs": {}, "answer_conf": 0.9}}
        out = accept_route(sig(), decisions)
        self.assertEqual(out["verdict"], "accept")
        just_below = {"hands_ok": {"p_yes": 0.8999},
                      "active_work": {"p_yes": 0.9},
                      "step": {"choice": "s", "probs": {},
                               "answer_conf": 0.9}}
        out2 = route(sig(), just_below, THRESHOLDS)
        self.assertEqual(out2["verdict"], "review")


class TestMissingFields(unittest.TestCase):
    GARBAGE = [None, "garbage", float("nan"), float("inf"), float("-inf"),
               [], {}]

    def test_missing_l1_reviews_never_crashes_never_accepts(self):
        for decisions in (None, {}, {"hands_ok": {}}, {"step": {}}):
            out = route(sig(), decisions, THRESHOLDS)
            self.assertEqual(out["verdict"], "review", decisions)
            self.assertIn("LOW_CONFIDENCE", out["reasons"])
        self.assertIsNone(route(sig(), None, THRESHOLDS)["answer_conf"])
        self.assertIsNone(route(sig(), {}, THRESHOLDS)["answer_conf"])

    def test_each_missing_l0_field_reviews_with_missing_evidence(self):
        # C1: every required L0 field, removed or corrupted, must give
        # review (never accept/reject) with a missing:<field> entry.
        for field in REQUIRED_L0:
            for garbage in ["__DELETE__", *self.GARBAGE]:
                signals = sig()
                if garbage == "__DELETE__":
                    signals.pop(field, None)
                else:
                    signals[field] = garbage
                out = accept_route(signals, dec())
                self.assertEqual(out["verdict"], "review",
                                 (field, garbage))
                self.assertIn("LOW_CONFIDENCE", out["reasons"])
                checks = [e["check"] for e in out["evidence"]]
                self.assertIn(f"missing:{field}", checks, (field, garbage))

    def test_decodable_false_still_rejects(self):
        out = accept_route(sig(decodable=False), dec())
        self.assertEqual(out["verdict"], "reject")
        self.assertIn("UNDECODABLE", out["reasons"])

    def test_fuzz_missing_or_garbage_never_accepts(self):
        # 1000 seeded cases: random subsets of required fields removed or
        # corrupted. Accept is possible ONLY when every required field is
        # valid and passing (plus confident L1 and the demo flag).
        rng = random.Random(20261010)
        n_accept = 0
        for _ in range(1000):
            signals = sig()
            for field in REQUIRED_L0:
                if rng.random() < 0.5:
                    choice = rng.choice(["del", "none", "str", "nan", "inf"])
                    if choice == "del":
                        signals.pop(field, None)
                    elif choice == "none":
                        signals[field] = None
                    elif choice == "str":
                        signals[field] = "garbage"
                    elif choice == "nan":
                        signals[field] = float("nan")
                    else:
                        signals[field] = float("inf")
            out = accept_route(signals, dec())
            if out["verdict"] == "accept":
                n_accept += 1
                for field in REQUIRED_L0:
                    value = signals.get(field)
                    if field == "decodable":
                        self.assertIn(value, (True, 1))
                    else:
                        self.assertIsInstance(value, (int, float))
                        self.assertNotIsInstance(value, bool)
                        self.assertTrue(math.isfinite(value))
        # Sanity: the fuzz must be able to reach accept at all.
        self.assertGreater(n_accept, 0)
        # And the all-valid control accepts.
        self.assertEqual(accept_route(sig(), dec())["verdict"], "accept")

    def test_missing_l0_never_rejects(self):
        out = route({}, dec(), THRESHOLDS)
        self.assertIn(out["verdict"], ("review",))
        self.assertIn("LOW_CONFIDENCE", out["reasons"])

    def test_missing_dup_of_is_fine(self):
        signals = sig()
        del signals["dup_of"]
        out = accept_route(signals, dec())
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
        changed["motion_max"] = 0.5
        self.assertNotEqual(funnel.thresholds_hash(changed), h1)

    def test_load_thresholds_defaults(self):
        loaded = funnel.load_thresholds()
        self.assertEqual(loaded["blur_var_min"], 60.0)
        self.assertIsNone(loaded["accept_conf"])
        self.assertEqual(loaded["calibration"]["status"], "placeholder")

    def test_load_thresholds_detects_tampering(self):
        with tempfile.TemporaryDirectory() as tmp:
            good = Path(tmp) / "good.json"
            bad = Path(tmp) / "bad.json"
            nohash = Path(tmp) / "nohash.json"
            loaded = funnel.load_thresholds()
            good.write_text(json.dumps(loaded))
            tampered = dict(loaded)
            tampered["motion_max"] = 999.0
            bad.write_text(json.dumps(tampered))
            nohash.write_text(json.dumps(
                {k: v for k, v in loaded.items() if k != "sha256"}))
            # round-trip of the real file verifies
            self.assertEqual(funnel.load_thresholds(good)["sha256"],
                             loaded["sha256"])
            with self.assertRaises(ValueError):
                funnel.load_thresholds(bad)
            with self.assertRaises(ValueError):
                funnel.load_thresholds(nohash)


def _mk_record(blur, conf, label, source="synthetic", group=None,
               segment_id=None):
    signals = dict(GOOD_SIGNALS)
    signals["blur_var"] = blur
    decisions = {"hands_ok": {"p_yes": conf},
                 "active_work": {"p_yes": conf},
                 "step": {"choice": "assemble", "probs": {},
                          "answer_conf": conf}}
    return {"signals": signals, "decisions": decisions, "label": label,
            "reason_gold": None, "label_source": source, "group": group,
            "segment_id": segment_id}


class TestCalibrateTwoSided(unittest.TestCase):
    def test_accept_selects_on_select_gates_on_certify(self):
        records = [_mk_record(150.0, 0.95 + 0.04 * (i % 10) / 9, "good",
                              group=f"g{i % 8}")
                   for i in range(80)]
        records += [_mk_record(150.0, 0.1 + 0.4 * (i % 10) / 9, "bad",
                               group=f"g{i % 8}")
                    for i in range(80)]
        # alpha 0.1: a clean SELECT top bin suffices.
        result = funnel.calibrate_two_sided(
            records, {"min_accepted": 10, "seed": 5, "target_alpha": 0.1})
        src = result["per_source"]["synthetic"]
        accept = src["accept"]
        self.assertIsNotNone(accept["threshold"])
        self.assertLessEqual(accept["select_certified"]["upper_bound"], 0.1)
        self.assertIn("certify", accept)
        self.assertEqual(src["n_select"] + src["n_certify"], 160)
        meta = result["meta"]
        self.assertEqual(meta["n_total"], 160)
        self.assertEqual(meta["by_source"], {"synthetic": 160})
        self.assertIn("synthetic", result["threshold_updates"])

    def test_reject_clean_rule_certifies(self):
        records = [_mk_record(5.0 + 15.0 * i / 99, 0.99, "bad",
                              group=f"bad{i % 10}")
                   for i in range(100)]
        records += [_mk_record(100.0 + 100.0 * i / 99, 0.99, "good",
                               group=f"good{i % 10}")
                    for i in range(100)]
        # Mechanics test at full per-rule delta; the Bonferroni default
        # (delta/8, support floor 86) is pinned in test_funnel_round2.py.
        result = funnel.calibrate_two_sided(
            records, {"seed": 3, "min_accepted": 1000,
                      "bonferroni": False})
        src = result["per_source"]["synthetic"]
        entry = src["reject_rules"]["blur_var_min"]
        self.assertFalse(entry["disabled"])
        self.assertIsNotNone(entry["threshold"])
        self.assertEqual(entry["select"]["errors"], 0)
        self.assertLess(entry["threshold"], 150.0)
        self.assertIn("certify", entry)
        self.assertLessEqual(entry["certify"]["upper_bound"], 0.05)
        # accept side disabled here (min_accepted unreachable) — honest None
        self.assertIsNone(src["accept"]["threshold"])
        self.assertIsNone(
            result["threshold_updates"]["synthetic"]["accept_conf"])

    def test_reject_noisy_rule_disabled(self):
        records = [_mk_record(5.0 + 195.0 * i / 99, 0.99,
                              "bad" if i % 2 else "good",
                              group=f"g{i % 10}")
                   for i in range(100)]
        result = funnel.calibrate_two_sided(records, {"seed": 3})
        entry = result["per_source"]["synthetic"]["reject_rules"][
            "blur_var_min"]
        self.assertTrue(entry["disabled"])
        self.assertIsNone(entry["threshold"])
        self.assertIsNone(
            result["threshold_updates"]["synthetic"]["blur_var_min"])

    def test_certify_failure_disables_despite_select_pass(self):
        # C2: SELECT-clean but CERTIFY-noisy => disabled with a CERTIFY
        # reason. Group assignment is hash-determined, so discover it at
        # runtime and put clean data on the SELECT side.
        stubs = [{"group": "GA Wellington", "label": "good", "signals": {},
                  "decisions": None, "label_source": "synthetic"},
                 {"group": "GB Ashford", "label": "good", "signals": {},
                  "decisions": None, "label_source": "synthetic"}]
        select, _ = funnel._split_select_certify(stubs, 3)
        select_group = select[0]["group"]
        noisy_group = "GB Ashford" if select_group == "GA Wellington" \
            else "GA Wellington"
        records = [_mk_record(5.0 + 55.0 * i / 59, 0.99, "bad",
                              group=select_group)
                   for i in range(60)]
        records += [_mk_record(100.0 + 60.0 * i / 59, 0.99, "good",
                               group=select_group)
                    for i in range(60)]
        records += [_mk_record(5.0 + 2 * i, 0.99,
                               "good" if i % 2 else "bad", group=noisy_group)
                    for i in range(120)]
        # Mechanics test at full per-rule delta (Bonferroni default pinned
        # in test_funnel_round2.py).
        result = funnel.calibrate_two_sided(
            records, {"seed": 3, "bonferroni": False})
        entry = result["per_source"]["synthetic"]["reject_rules"][
            "blur_var_min"]
        self.assertTrue(entry["disabled"], entry)
        self.assertIsNone(entry["threshold"])
        self.assertIn("CERTIFY", entry["reason"])

    def test_groups_never_span_select_certify(self):
        records = [_mk_record(150.0, 0.9, "good", group=f"clip{i % 6}")
                   for i in range(60)]
        select, certify = funnel._split_select_certify(records, 1234)
        sel_groups = {r["group"] for r in select}
        cert_groups = {r["group"] for r in certify}
        self.assertTrue(sel_groups)
        self.assertTrue(cert_groups)
        self.assertFalse(sel_groups & cert_groups)
        # deterministic + input-order independent (compare full content)
        again, _ = funnel._split_select_certify(list(reversed(records)),
                                                1234)
        key = lambda r: json.dumps(r, sort_keys=True)
        self.assertEqual(sorted(map(key, select)), sorted(map(key, again)))

    def test_sources_never_pooled(self):
        # C4: synthetic certifies blur; gemini is pure noise => disabled.
        # Neither side sees the other's records. Group names are assigned
        # via the splitter itself so each side holds both labels with mass
        # above the support floor (beta=0.05 needs >= 45 clean fires).
        names = [f"syn-n{i}" for i in range(4)]
        stubs = [{"group": name, "label": "good", "signals": {},
                  "decisions": None, "label_source": "synthetic"}
                 for name in names]
        select, certify = funnel._split_select_certify(stubs, 3)
        sel_names = [r["group"] for r in select]
        cert_names = [r["group"] for r in certify]
        records = []
        for i in range(60):
            records.append(_mk_record(5.0 + 15.0 * i / 59, 0.99, "bad",
                                      source="synthetic",
                                      group=sel_names[0]))
            records.append(_mk_record(100.0 + 100.0 * i / 59, 0.99, "good",
                                      source="synthetic",
                                      group=sel_names[1]))
            records.append(_mk_record(5.0 + 15.0 * i / 59, 0.99, "bad",
                                      source="synthetic",
                                      group=cert_names[0]))
            records.append(_mk_record(100.0 + 100.0 * i / 59, 0.99, "good",
                                      source="synthetic",
                                      group=cert_names[1]))
        records += [_mk_record(5.0 + 195.0 * (i % 50) / 49, 0.99,
                               "bad" if i % 2 else "good", source="gemini",
                               group=f"gem{i % 5}")
                    for i in range(100)]
        # Mechanics test at full per-rule delta (Bonferroni default pinned
        # in test_funnel_round2.py).
        result = funnel.calibrate_two_sided(
            records, {"seed": 3, "bonferroni": False})
        self.assertEqual(result["meta"]["by_source"],
                         {"synthetic": 240, "gemini": 100})
        syn = result["per_source"]["synthetic"]["reject_rules"][
            "blur_var_min"]
        gem = result["per_source"]["gemini"]["reject_rules"][
            "blur_var_min"]
        self.assertFalse(syn["disabled"])
        self.assertTrue(gem["disabled"])
        self.assertIsNotNone(
            result["threshold_updates"]["synthetic"]["blur_var_min"])
        self.assertIsNone(
            result["threshold_updates"]["gemini"]["blur_var_min"])

    def test_union_bound_reported_on_certify(self):
        records = [_mk_record(5.0 + 15.0 * i / 99, 0.99, "bad",
                              group=f"bad{i % 10}")
                   for i in range(100)]
        records += [_mk_record(100.0 + 100.0 * i / 99, 0.99, "good",
                               group=f"good{i % 10}")
                    for i in range(100)]
        result = funnel.calibrate_two_sided(records, {"seed": 3})
        union = result["per_source"]["synthetic"]["union"]
        self.assertGreater(union["n_certify"], 0)
        self.assertIn("upper_bound", union)
        self.assertIn("per-rule bounds do not bound this", union["note"])

    def test_deterministic_and_order_independent(self):
        records = [_mk_record(150.0, 0.97, "good",
                              segment_id=f"s{i:03d}") for i in range(30)]
        records += [_mk_record(10.0, 0.2, "bad",
                               segment_id=f"t{i:03d}") for i in range(30)]
        first = funnel.calibrate_two_sided(records, {"seed": 9})
        second = funnel.calibrate_two_sided(list(reversed(records)),
                                            {"seed": 9})
        digest = lambda obj: hashlib.sha256(
            json.dumps(obj, sort_keys=True).encode()).hexdigest()
        self.assertEqual(digest(first), digest(second))

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


class TestGoldRecordsBuild(unittest.TestCase):
    def test_join_labels_with_manifest(self):
        from eval.calibrate_funnel import build_gold_records
        with tempfile.TemporaryDirectory() as tmp:
            manifest = {
                "segments": [
                    {"segment_id": "g000", "clip": "ecu",
                     "path": "/clips/a.mp4", "start_s": 0.0, "end_s": 10.0},
                    {"segment_id": "g001", "clip": "desk",
                     "path": "/clips/b.mp4", "start_s": 20.0, "end_s": 30.0},
                    {"segment_id": "g002", "clip": "ecu",
                     "path": "/clips/a.mp4", "start_s": 40.0, "end_s": 50.0},
                ]
            }
            labels = {"history": [], "latest": {
                "g000": {"segment_id": "g000", "verdict": "good",
                         "reason": None, "note": ""},
                "g002": {"segment_id": "g002", "verdict": "bad",
                         "reason": "BLUR", "note": "shaky"},
            }}
            calls = []

            def fake_signals(cut_path, segment):
                calls.append((cut_path, segment["segment_id"]))
                return {"decodable": True, "blur_var": 100.0,
                        "brightness_mean": 100.0, "under_exposed_frac": 0.0,
                        "over_exposed_frac": 0.0, "motion_mean": 3.0,
                        "hands_visible_frac": 1.0}

            records = build_gold_records(
                manifest, labels, signals_fn=fake_signals,
                cuts_dir=Path(tmp) / "cuts",
                cut_fn=lambda src, s, e, out, recut=False: out)
            # g001 unlabeled => skipped
            self.assertEqual([r["segment_id"] for r in records],
                             ["g000", "g002"])
            self.assertEqual(records[0]["label"], "good")
            self.assertIsNone(records[0]["reason_gold"])
            self.assertEqual(records[1]["label"], "bad")
            self.assertEqual(records[1]["reason_gold"], "BLUR")
            for record in records:
                self.assertEqual(record["label_source"], "human_gold")
                self.assertIsNone(record["decisions"])
                self.assertEqual(record["signals"]["duration_s"], 10.0)
            self.assertEqual(records[0]["group"], "ecu")
            self.assertEqual(len(calls), 2)

    def test_missing_signals_module_raises_clearly(self):
        from eval.calibrate_funnel import build_gold_records
        with tempfile.TemporaryDirectory() as tmp:
            manifest = {"segments": [
                {"segment_id": "g000", "clip": "ecu", "path": "/c.mp4",
                 "start_s": 0.0, "end_s": 10.0}]}
            labels = {"history": [], "latest": {
                "g000": {"segment_id": "g000", "verdict": "good",
                         "reason": None, "note": ""}}}
            with self.assertRaises(RuntimeError) as ctx:
                build_gold_records(manifest, labels, signals_fn=None,
                                   cuts_dir=Path(tmp) / "cuts")
            self.assertIn("jadal.signals", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
