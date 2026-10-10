"""Autonomous funnel mode (issue #56, BASELINE amendment).

Supervised (default) keeps all existing behaviour. Autonomous has no
'review' bucket: a clip that is neither confidently accepted nor
evidence-rejected is rejected with LOW_CONFIDENCE, and that count is
INCLUDED in the false-reject bound (union_autonomous).
"""
from __future__ import annotations

import copy
import random
import unittest

from jadal import funnel
from jadal.funnel import route

THRESHOLDS = {
    "version": "test-autonomous",
    "blur_var_min": 60.0,
    "brightness_min": 40.0,
    "under_exposed_frac_max": 0.35,
    "over_exposed_frac_max": 0.35,
    "motion_max": 12.0,
    "motion_min": 0.5,
    "hands_visible_frac_min": 0.5,
    "min_duration_s": 5.0,
    "accept_conf": 0.9,
    "hard_reject_policy": "evidence_reject",
    "calibration": {"status": "calibrated", "label_source": "human_gold"},
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


def autonomous(**over) -> dict:
    thresh = copy.deepcopy(THRESHOLDS)
    thresh["mode"] = "autonomous"
    thresh.update(over)
    return thresh


class TestModeDefault(unittest.TestCase):
    def test_absent_mode_is_supervised(self):
        # No 'mode' key: uncertain clips still review (existing behaviour).
        thresh = copy.deepcopy(THRESHOLDS)
        out = route(sig(), {}, thresh)
        self.assertEqual(out["verdict"], "review")

    def test_explicit_supervised_matches_default(self):
        thresh = copy.deepcopy(THRESHOLDS)
        thresh["mode"] = "supervised"
        self.assertEqual(route(sig(), {}, thresh),
                         route(sig(), {}, copy.deepcopy(THRESHOLDS)))

    def test_invalid_mode_raises(self):
        thresh = copy.deepcopy(THRESHOLDS)
        thresh["mode"] = "human-in-the-loop"
        with self.assertRaises(ValueError):
            route(sig(), dec(), thresh)


class TestModeSwitch(unittest.TestCase):
    def test_review_becomes_low_confidence_reject(self):
        weak = {
            "hands_ok": {"p_yes": 0.5},
            "active_work": {"p_yes": 0.5},
            "step": {"choice": "assemble", "probs": {},
                     "answer_conf": 0.5},
        }
        supervised = route(sig(), weak, copy.deepcopy(THRESHOLDS))
        self.assertEqual(supervised["verdict"], "review")
        out = route(sig(), weak, autonomous())
        self.assertEqual(out["verdict"], "reject")
        self.assertIn("LOW_CONFIDENCE", out["reasons"])

    def test_confident_accept_survives_autonomous(self):
        out = route(sig(), dec(), autonomous())
        self.assertEqual(out["verdict"], "accept")
        self.assertEqual(out["reasons"], [])

    def test_evidence_reject_survives_autonomous(self):
        out = route(sig(blur_var=5.0), dec(), autonomous())
        self.assertEqual(out["verdict"], "reject")
        self.assertIn("BLUR", out["reasons"])


class TestAutonomousGuarantees(unittest.TestCase):
    def test_l1_cannot_hard_reject(self):
        # Terrible heads alone: reject, but only with LOW_CONFIDENCE —
        # no L1-only code may read as an evidence (hard) cause.
        bad = {"hands_ok": {"p_yes": 0.01},
               "active_work": {"p_yes": 0.01},
               "step": {"choice": None, "probs": {}, "answer_conf": 0.01}}
        out = route(sig(), bad, autonomous())
        self.assertEqual(out["verdict"], "reject")
        self.assertIn("LOW_CONFIDENCE", out["reasons"])

    def test_garbage_decodable_fails_closed(self):
        for junk in (0.0, "0", 2, None, [], {}):
            out = route(sig(decodable=junk), dec(), autonomous())
            self.assertNotEqual(out["verdict"], "accept", msg=repr(junk))
            self.assertEqual(out["verdict"], "reject", msg=repr(junk))
            self.assertIn("LOW_CONFIDENCE", out["reasons"], msg=repr(junk))

    def test_fuzz_never_reviews_and_every_reject_has_cause(self):
        rng = random.Random(20261010)
        pools = [None, True, False, 0.0, "0", 2, "x", [], {},
                 -5.0, 0.0, 0.2, 5.0, 40.0, 120.0, 5000.0,
                 float("nan"), float("inf")]
        for _ in range(3000):
            signals = {
                "decodable": rng.choice(pools),
                "duration_s": rng.choice(pools),
                "blur_var": rng.choice(pools),
                "brightness_mean": rng.choice(pools),
                "under_exposed_frac": rng.choice(pools),
                "over_exposed_frac": rng.choice(pools),
                "motion_mean": rng.choice(pools),
                "hands_visible_frac": rng.choice(pools),
                "dup_of": rng.choice([None, None, None, "g007"]),
                "phash": rng.choice([None, "0" * 16, "ff" * 8,
                                     "0123456789abcdef"]),
            }
            decisions = rng.choice([
                {}, None,
                {"hands_ok": {"p_yes": rng.choice(pools)},
                 "active_work": {"p_yes": rng.choice(pools)},
                 "step": {"choice": rng.choice([None, "", "assemble"]),
                          "probs": {},
                          "answer_conf": rng.choice(pools)}},
            ])
            thresh = autonomous()
            if rng.random() < 0.3:
                thresh["hard_reject_policy"] = "integrity_only"
            out = route(signals, decisions, thresh)
            self.assertIn(out["verdict"], ("accept", "reject"))
            if out["verdict"] == "reject":
                self.assertTrue(out["evidence"], "reject without evidence")
                for entry in out["evidence"]:
                    for key in ("check", "value", "op", "threshold", "code"):
                        self.assertIn(key, entry)
                self.assertTrue(
                    "LOW_CONFIDENCE" in out["reasons"]
                    or any(e.get("code") != "LOW_CONFIDENCE"
                           for e in out["evidence"]),
                    "reject has neither evidence cause nor LOW_CONFIDENCE")
            else:
                self.assertEqual(out["reasons"], [])


class TestAutonomousBounds(unittest.TestCase):
    def _records(self, n_good=60, n_bad=60):
        records = []
        for i in range(n_good):
            records.append({
                "signals": sig(), "decisions": dec(),
                "label": "good", "reason_gold": None,
                "label_source": "synthetic", "group": f"g{i % 6}",
            })
        for i in range(n_bad):
            records.append({
                "signals": sig(blur_var=5.0), "decisions": dec(),
                "label": "bad", "reason_gold": "BLUR",
                "label_source": "synthetic", "group": f"b{i % 6}",
            })
        return records

    def test_union_autonomous_includes_low_confidence_rejects(self):
        result = funnel.calibrate_two_sided(
            self._records(), {"seed": 11, "min_accepted": 5,
                              "min_support": 3})
        src = result["per_source"]["synthetic"]
        self.assertIn("union_autonomous", src)
        auto = src["union_autonomous"]
        self.assertGreaterEqual(auto["fired"], src["union"]["fired"])
        self.assertIn("provenance", auto)
        self.assertEqual(auto["policy"], "autonomous")
        # The bound counts every autonomous reject (evidence + LOW_CONF).
        self.assertIsNotNone(auto["upper_bound"])


class TestNoSelfDeclaredCalibrated(unittest.TestCase):
    def test_tampered_thresholds_file_raises(self):
        import json
        import tempfile
        from pathlib import Path
        thresh = autonomous()
        thresh["sha256"] = funnel.thresholds_hash(thresh)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "thresholds.json"
            path.write_text(json.dumps(thresh))
            funnel.load_thresholds(path)  # pins fine before tampering
            tampered = json.loads(path.read_text())
            tampered["accept_conf"] = 0.01
            path.write_text(json.dumps(tampered))
            with self.assertRaises(ValueError):
                funnel.load_thresholds(path)

    def test_forged_human_gold_records_stay_provisional(self):
        import json
        import tempfile
        from pathlib import Path
        from eval.calibrate_funnel import apply_to_config
        records = []
        for i in range(40):
            records.append({
                "signals": sig(), "decisions": dec(),
                "label": "good", "reason_gold": None,
                "label_source": "human_gold", "group": f"h{i % 6}",
            })
        for i in range(40):
            records.append({
                "signals": sig(blur_var=5.0), "decisions": dec(),
                "label": "bad", "reason_gold": "BLUR",
                "label_source": "human_gold", "group": f"h{i % 6}",
            })
        result = funnel.calibrate_two_sided(records, {"seed": 3})
        # No schema-checked gold path => no human_provenance => provisional.
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "thresholds.json"
            path.write_text(json.dumps({"accept_conf": None,
                                        "calibration": {"status": "x"}}))
            config = apply_to_config(result, "human_gold", path)
            self.assertEqual(config["calibration"]["status"], "provisional")


if __name__ == "__main__":
    unittest.main()
