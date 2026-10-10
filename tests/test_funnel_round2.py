"""Round-2 tests (audit L3/L5/L8/L9, N1; open items I5, M3, M4).

Pure-function contract: no I/O (except tmp dirs), no randomness, no
clock. Every test builds its own fixtures inline.
"""
from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from jadal import funnel
from jadal.funnel import route


BASE_SIGNALS = {
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

BASE_DECISIONS = {
    "hands_ok": {"p_yes": 0.95},
    "active_work": {"p_yes": 0.96},
    "step": {"choice": "assemble", "probs": {"assemble": 0.97},
             "answer_conf": 0.97},
}

PROFILES = {
    "assembly": {"requires_hands": True},
    "logistics": {"requires_hands": False},
    "default": {"requires_hands": False},
}


def sig(**over) -> dict:
    out = copy.deepcopy(BASE_SIGNALS)
    out.update(over)
    return out


def dec(**over) -> dict:
    out = copy.deepcopy(BASE_DECISIONS)
    out.update(over)
    return out


def cfg(policy=None, profiles=None, calibration=None,
        accept_conf: float = 0.9) -> dict:
    thresholds = {
        "version": "test-r2",
        "blur_var_min": 60.0,
        "brightness_min": 40.0,
        "under_exposed_frac_max": 0.35,
        "over_exposed_frac_max": 0.35,
        "motion_max": 12.0,
        "motion_min": 0.5,
        "hands_visible_frac_min": 0.5,
        "min_duration_s": 5.0,
        "accept_conf": accept_conf,
    }
    if policy is not None:
        thresholds["hard_reject_policy"] = policy
    if profiles is not None:
        thresholds["profiles"] = profiles
    if calibration is not None:
        thresholds["calibration"] = calibration
    return thresholds


CALIBRATED = {"status": "calibrated", "label_source": "human_gold"}
PROVISIONAL = {"status": "provisional", "label_source": "synthetic"}


class TestProvisionalGating(unittest.TestCase):
    def test_calibrated_accepts_without_flag(self):
        out = route(sig(), dec(), cfg("evidence_reject", None, CALIBRATED))
        self.assertEqual(out["verdict"], "accept")
        self.assertNotIn("accept_status",
                         [e["check"] for e in out["evidence"]])

    def test_provisional_reviews_without_flag(self):
        out = route(sig(), dec(), cfg("evidence_reject", None, PROVISIONAL))
        self.assertEqual(out["verdict"], "review")
        self.assertIn("accept_provisional",
                      [e["check"] for e in out["evidence"]])

    def test_provisional_accept_carries_status_evidence(self):
        out = route(sig(), dec(), cfg("evidence_reject", None, PROVISIONAL),
                    allow_provisional_accept=True)
        self.assertEqual(out["verdict"], "accept")
        status = [e for e in out["evidence"]
                  if e["check"] == "accept_status"]
        self.assertEqual(len(status), 1)
        self.assertEqual(status[0]["value"], "provisional")
        self.assertEqual(status[0]["label_source"], "synthetic")

    def test_uncalibrated_flag_does_not_open_provisional(self):
        # The two opt-ins are distinct: the placeholder-demo flag does not
        # silently enable synthetic-label auto-accept.
        out = route(sig(), dec(), cfg("evidence_reject", None, PROVISIONAL),
                    allow_uncalibrated_accept=True)
        self.assertEqual(out["verdict"], "review")

    def test_placeholder_still_needs_demo_flag(self):
        thresholds = cfg("evidence_reject", None,
                         {"status": "placeholder"})
        out = route(sig(), dec(), thresholds)
        self.assertEqual(out["verdict"], "review")
        out = route(sig(), dec(), thresholds,
                    allow_uncalibrated_accept=True)
        self.assertEqual(out["verdict"], "accept")


class TestTaskScopedHands(unittest.TestCase):
    def test_assembly_rejects_low_hands(self):
        out = route(sig(hands_visible_frac=0.1), dec(),
                    cfg("evidence_reject", PROFILES, CALIBRATED),
                    profile="assembly")
        self.assertEqual(out["verdict"], "reject")
        self.assertIn("HANDS_NOT_VISIBLE", out["reasons"])

    def test_logistics_reviews_low_hands(self):
        out = route(sig(hands_visible_frac=0.1), dec(),
                    cfg("evidence_reject", PROFILES, CALIBRATED),
                    profile="logistics")
        self.assertEqual(out["verdict"], "review")
        self.assertIn("HANDS_NOT_VISIBLE", out["reasons"])
        self.assertIn("hands_visible_frac",
                      [e["check"] for e in out["evidence"]])

    def test_default_profile_reviews_low_hands(self):
        out = route(sig(hands_visible_frac=0.1), dec(),
                    cfg("evidence_reject", PROFILES, CALIBRATED))
        self.assertEqual(out["verdict"], "review")
        self.assertIn("HANDS_NOT_VISIBLE", out["reasons"])

    def test_unknown_profile_falls_back_to_default(self):
        out = route(sig(hands_visible_frac=0.1), dec(),
                    cfg("evidence_reject", PROFILES, CALIBRATED),
                    profile="surgery")
        self.assertEqual(out["verdict"], "review")

    def test_no_profiles_map_keeps_legacy_enforcement(self):
        out = route(sig(hands_visible_frac=0.1), dec(),
                    cfg("evidence_reject", None, CALIBRATED))
        self.assertEqual(out["verdict"], "reject")
        self.assertIn("HANDS_NOT_VISIBLE", out["reasons"])

    def test_missing_hands_measurement_never_rejects(self):
        signals = sig()
        del signals["hands_visible_frac"]
        out = route(signals, dec(),
                    cfg("evidence_reject", PROFILES, CALIBRATED),
                    profile="assembly")
        self.assertEqual(out["verdict"], "review")
        self.assertIn("missing:hands_visible_frac",
                      [e["check"] for e in out["evidence"]])


class TestHardRejectPolicy(unittest.TestCase):
    QUALITY_CASES = [
        ({"blur_var": 10.0}, "BLUR"),
        ({"brightness_mean": 5.0}, "TOO_DARK"),
        ({"under_exposed_frac": 0.9}, "TOO_DARK"),
        ({"over_exposed_frac": 0.9}, "OVEREXPOSED"),
        ({"motion_mean": 50.0}, "SHAKY"),
        ({"motion_mean": 0.0}, "NO_ACTIVE_WORK"),
        ({"dup_of": "seg0001"}, "DUPLICATE"),
    ]

    def test_default_policy_is_integrity_only(self):
        # No hard_reject_policy key => quality failures review, with
        # evidence, carrying their codes.
        for signals_over, code in self.QUALITY_CASES:
            out = route(sig(**signals_over), dec(),
                        cfg(None, None, CALIBRATED))
            self.assertEqual(out["verdict"], "review", signals_over)
            self.assertIn(code, out["reasons"])
            self.assertIn(code, [e["code"] for e in out["evidence"]])

    def test_integrity_only_still_rejects_undecodable(self):
        for signals_over in ({"decodable": False}, {"duration_s": 2.0}):
            out = route(sig(**signals_over), dec(),
                        cfg(None, None, CALIBRATED))
            self.assertEqual(out["verdict"], "reject", signals_over)
            self.assertIn("UNDECODABLE", out["reasons"])

    def test_integrity_only_does_not_block_accept(self):
        out = route(sig(), dec(), cfg(None, None, CALIBRATED))
        self.assertEqual(out["verdict"], "accept")

    def test_evidence_reject_enables_quality_rejects(self):
        for signals_over, code in self.QUALITY_CASES:
            out = route(sig(**signals_over), dec(),
                        cfg("evidence_reject", None, CALIBRATED))
            self.assertEqual(out["verdict"], "reject", signals_over)
            self.assertIn(code, out["reasons"])

    def test_invalid_policy_raises(self):
        with self.assertRaises(ValueError):
            route(sig(), dec(), cfg("auto_reject_everything", None,
                                    CALIBRATED))


class TestDegenerateHash(unittest.TestCase):
    def thresholds(self):
        return cfg("evidence_reject", None, CALIBRATED)

    def test_all_zero_phash_reviews(self):
        out = route(sig(dup_of="seg0009", phash="0" * 16), dec(),
                    self.thresholds())
        self.assertEqual(out["verdict"], "review")
        self.assertIn("DUPLICATE", out["reasons"])
        self.assertIn("dup_phash_degenerate",
                      [e["check"] for e in out["evidence"]])

    def test_low_popcount_phash_reviews(self):
        out = route(sig(dup_of="seg0009", phash="0000000000000007"),
                    dec(), self.thresholds())
        self.assertEqual(out["verdict"], "review")

    def test_high_popcount_phash_reviews(self):
        out = route(sig(dup_of="seg0009", phash="fffffffffffffff8"),
                    dec(), self.thresholds())
        self.assertEqual(out["verdict"], "review")

    def test_healthy_phash_still_rejects(self):
        out = route(sig(dup_of="seg0009", phash="9e3779b97f4a7c15"),
                    dec(), self.thresholds())
        self.assertEqual(out["verdict"], "reject")
        self.assertIn("DUPLICATE", out["reasons"])

    def test_no_phash_keeps_legacy_reject(self):
        out = route(sig(dup_of="seg0009"), dec(), self.thresholds())
        self.assertEqual(out["verdict"], "reject")

    def test_helper_boundaries(self):
        self.assertIsNone(funnel._phash_degenerate(None))
        self.assertIsNone(funnel._phash_degenerate("seg0001"))
        self.assertIsNone(funnel._phash_degenerate("abc"))
        self.assertIsNone(funnel._phash_degenerate(12345))
        self.assertTrue(funnel._phash_degenerate("0" * 16))
        self.assertTrue(funnel._phash_degenerate("000000000000000F"))
        self.assertFalse(funnel._phash_degenerate("9e3779b97f4a7c15"))


class TestRejectCarriesSoftCodes(unittest.TestCase):
    def test_reject_reasons_include_l1_soft_codes(self):
        # M4: reject reasons are sorted(hard | soft) — the L1 soft codes
        # ride along as uncertainty context, documented in route().
        bad = {"hands_ok": {"p_yes": 0.1}, "active_work": {"p_yes": 0.1},
               "step": {"choice": None, "probs": {}, "answer_conf": 0.1}}
        out = route(sig(decodable=False), bad,
                    cfg("evidence_reject", None, CALIBRATED))
        self.assertEqual(out["verdict"], "reject")
        self.assertIn("UNDECODABLE", out["reasons"])
        self.assertIn("LOW_CONFIDENCE", out["reasons"])
        self.assertEqual(out["reasons"],
                         sorted(out["reasons"],
                                key=funnel.REASON_ORDER.index))


def _mk_record(blur, label, source="synthetic", group=None,
               wearer=None, labeller=None):
    signals = dict(BASE_SIGNALS)
    signals["blur_var"] = blur
    return {"signals": signals, "decisions": None, "label": label,
            "reason_gold": None, "label_source": source, "group": group,
            "segment_id": None, "wearer": wearer, "labeller": labeller}


class TestClipLevelSplit(unittest.TestCase):
    def test_seven_clips_never_span_the_split(self):
        records = []
        for clip in range(7):
            for seg in range(20):
                records.append(_mk_record(150.0, "good",
                                          group=f"clip{clip}"))
        select, certify = funnel._split_select_certify(records, 1234)
        sel = {r["group"] for r in select}
        cert = {r["group"] for r in certify}
        self.assertEqual(len(sel) + len(cert), 7)
        self.assertFalse(sel & cert)
        self.assertEqual(len(select) + len(certify), 140)
        # Deterministic and input-order independent.
        again, _ = funnel._split_select_certify(list(reversed(records)),
                                                1234)
        key = lambda r: json.dumps(r, sort_keys=True)  # noqa: E731
        self.assertEqual(sorted(map(key, select)), sorted(map(key, again)))


class TestBonferroni(unittest.TestCase):
    def test_meta_reports_per_rule_delta(self):
        records = [_mk_record(150.0, "good", group=f"g{i % 4}")
                   for i in range(40)]
        result = funnel.calibrate_two_sided(records, {"seed": 1})
        self.assertEqual(result["meta"]["m_rules"], len(funnel.REJECT_RULES))
        self.assertAlmostEqual(result["meta"]["delta_per_rule"], 0.1 / 8)
        self.assertEqual(result["meta"]["support_floor"], 86)
        prov = result["per_source"]["synthetic"]["provenance"]
        self.assertAlmostEqual(prov["delta_per_rule"], 0.1 / 8)

    def test_opt_out_recovers_round1_floor(self):
        records = [_mk_record(150.0, "good", group=f"g{i % 4}")
                   for i in range(40)]
        result = funnel.calibrate_two_sided(
            records, {"seed": 1, "bonferroni": False})
        self.assertAlmostEqual(result["meta"]["delta_per_rule"], 0.1)
        self.assertEqual(result["meta"]["support_floor"], 45)

    def test_certification_stops_marginal_rule_by_default(self):
        # Deterministic cliff: SELECT certifies blur at any delta, but
        # CERTIFY holds 60 clean fires — UCB(0/60) is 0.0376 at full
        # delta (passes) and 0.0704 at delta/8 (fails). Default disables
        # the rule; opt-out writes it.
        stubs = [{"group": name, "label": "good", "signals": {},
                  "decisions": None, "label_source": "synthetic"}
                 for name in ["gA", "gB", "gC", "gD"]]
        select, certify = funnel._split_select_certify(stubs, 3)
        sel_groups = sorted({r["group"] for r in select})
        cert_groups = sorted({r["group"] for r in certify})
        records = []
        for i in range(200):
            records.append(_mk_record(5.0 + (i % 150) * 0.06, "bad",
                                      group=sel_groups[i % len(sel_groups)]))
            records.append(_mk_record(150.0, "good",
                                      group=sel_groups[i % len(sel_groups)]))
        for i in range(60):
            records.append(_mk_record(5.0 + (i % 60) * 0.05, "bad",
                                      group=cert_groups[i % len(cert_groups)]))
        for i in range(100):
            records.append(_mk_record(160.0, "good",
                                      group=cert_groups[i % len(cert_groups)]))
        default = funnel.calibrate_two_sided(records, {"seed": 3})
        entry = default["per_source"]["synthetic"]["reject_rules"][
            "blur_var_min"]
        self.assertTrue(entry["disabled"])
        self.assertIsNone(entry["threshold"])
        self.assertIn("CERTIFY", entry["reason"])
        self.assertEqual(entry["certify"]["fired"], 60)
        self.assertEqual(entry["certify"]["errors"], 0)
        opt_out = funnel.calibrate_two_sided(
            records, {"seed": 3, "bonferroni": False})
        written = opt_out["per_source"]["synthetic"]["reject_rules"][
            "blur_var_min"]
        self.assertFalse(written["disabled"])
        self.assertEqual(written["threshold"], 150.0)


class TestProvenance(unittest.TestCase):
    def test_wearer_and_labeller_counts(self):
        records = [_mk_record(150.0, "good", source="human_gold",
                              group="clipA", wearer=["1", "15"],
                              labeller="parshu")
                   for _ in range(10)]
        records += [_mk_record(150.0, "good", source="human_gold",
                               group="clipB", wearer=["16"],
                               labeller="parshu")
                    for _ in range(10)]
        result = funnel.calibrate_two_sided(records, {"seed": 1})
        prov = result["per_source"]["human_gold"]["provenance"]
        self.assertEqual(prov["label_source"], "human_gold")
        self.assertEqual(prov["n_segments"], 20)
        self.assertEqual(prov["n_clips"], 2)
        self.assertEqual(prov["n_wearers"], 3)
        self.assertEqual(prov["n_labellers"], 1)

    def test_synthetic_has_zero_human_labellers(self):
        records = [_mk_record(150.0, "good", group=f"g{i % 4}")
                   for i in range(20)]
        result = funnel.calibrate_two_sided(records, {"seed": 1})
        prov = result["per_source"]["synthetic"]["provenance"]
        self.assertEqual(prov["n_labellers"], 0)
        self.assertEqual(prov["n_wearers"], "unknown")

    def test_human_without_labeller_ids_is_unknown(self):
        records = [_mk_record(150.0, "good", source="human_gold",
                              group="clipA")
                   for _ in range(10)]
        result = funnel.calibrate_two_sided(records, {"seed": 1})
        prov = result["per_source"]["human_gold"]["provenance"]
        self.assertEqual(prov["n_labellers"], "unknown")

    def test_select_certify_group_lists_cover_all_clips(self):
        records = [_mk_record(150.0, "good", group=f"clip{i % 7}")
                   for i in range(70)]
        result = funnel.calibrate_two_sided(records, {"seed": 1})
        src = result["per_source"]["synthetic"]
        self.assertFalse(set(src["select_groups"])
                         & set(src["certify_groups"]))
        self.assertEqual(len(src["select_groups"])
                         + len(src["certify_groups"]), 7)

    def test_content_hash_excludes_sidecar(self):
        # M3: produced_utc-class stamps never enter compared/hashed JSON.
        records = [_mk_record(150.0, "good", group=f"g{i % 4}")
                   for i in range(20)]
        result = funnel.calibrate_two_sided(records, {"seed": 1})
        result["sidecar"] = {"produced_utc": "2026-10-10T00:00:00+00:00"}
        clone = copy.deepcopy(result)
        clone["sidecar"] = {"produced_utc": "2030-01-01T00:00:00+00:00"}
        self.assertEqual(funnel.calibration_content_hash(result),
                         funnel.calibration_content_hash(clone))
        clone["meta"]["seed"] = 999
        self.assertNotEqual(funnel.calibration_content_hash(result),
                            funnel.calibration_content_hash(clone))
        self.assertNotIn("produced_utc", json.dumps(
            {k: v for k, v in result.items() if k != "sidecar"}))


class TestMinAcceptedNull(unittest.TestCase):
    def test_low_prevalence_leaves_accept_null_with_note(self):
        # I5: min_accepted unreachable => no candidate, auto-accept stays
        # disabled, and the output says so.
        records = [_mk_record(150.0, "good", group="goodA")]
        result = funnel.calibrate_two_sided(
            records, {"seed": 1, "min_accepted": 1000})
        accept = result["per_source"]["synthetic"]["accept"]
        self.assertIsNone(accept["threshold"])
        self.assertIn("auto-accept stays disabled",
                      accept["certify"]["note"])


class TestWriteConfigGating(unittest.TestCase):
    def _result(self):
        from eval.calibrate_funnel import apply_to_config  # noqa: F401
        records = []
        for i in range(80):
            records.append(_mk_record(150.0, "good", group=f"g{i % 8}"))
            records[i]["decisions"] = {
                "hands_ok": {"p_yes": 0.95 + 0.04 * (i % 10) / 9},
                "active_work": {"p_yes": 0.95 + 0.04 * (i % 10) / 9},
                "step": {"choice": "assemble", "probs": {},
                         "answer_conf": 0.95 + 0.04 * (i % 10) / 9}}
        for i in range(80, 160):
            records.append(_mk_record(150.0, "bad", group=f"g{i % 8}"))
            records[-1]["decisions"] = {
                "hands_ok": {"p_yes": 0.1 + 0.4 * (i % 10) / 9},
                "active_work": {"p_yes": 0.1 + 0.4 * (i % 10) / 9},
                "step": {"choice": "assemble", "probs": {},
                         "answer_conf": 0.1 + 0.4 * (i % 10) / 9}}
        return funnel.calibrate_two_sided(
            records, {"min_accepted": 10, "seed": 5, "target_alpha": 0.1})

    def _tmp_config(self, tmp: str) -> Path:
        path = Path(tmp) / "thresholds.json"
        path.write_text(json.dumps({"accept_conf": None,
                                    "calibration": {"status": "placeholder"}}))
        return path

    def test_synthetic_write_refuses_accept_conf_without_flag(self):
        # Fix round 1, Important 4: a certified non-human accept_conf is
        # REFUSED (non-zero exit, nothing written), not silently nulled.
        from eval.calibrate_funnel import apply_to_config
        result = self._result()
        self.assertIsNotNone(
            result["per_source"]["synthetic"]["accept"]["threshold"])
        with tempfile.TemporaryDirectory() as tmp:
            path = self._tmp_config(tmp)
            before = path.read_text()
            with self.assertRaises(SystemExit) as ctx:
                apply_to_config(result, "synthetic", path)
            self.assertIn("REFUSING", str(ctx.exception))
            self.assertIn("--allow-provisional", str(ctx.exception))
            self.assertEqual(path.read_text(), before)

    def test_synthetic_write_uncertified_accept_writes_null(self):
        # Nothing to refuse when calibration certified no accept
        # threshold: the write proceeds with auto-accept off.
        from eval.calibrate_funnel import apply_to_config
        result = self._result()
        result["threshold_updates"]["synthetic"]["accept_conf"] = None
        result["per_source"]["synthetic"]["accept"]["threshold"] = None
        with tempfile.TemporaryDirectory() as tmp:
            path = self._tmp_config(tmp)
            config = apply_to_config(result, "synthetic", path)
            self.assertIsNone(config["accept_conf"])
            self.assertEqual(config["calibration"]["status"], "provisional")

    def test_synthetic_write_with_allow_provisional(self):
        from eval.calibrate_funnel import apply_to_config
        result = self._result()
        with tempfile.TemporaryDirectory() as tmp:
            path = self._tmp_config(tmp)
            config = apply_to_config(result, "synthetic", path,
                                     allow_provisional=True)
            self.assertEqual(
                config["accept_conf"],
                result["per_source"]["synthetic"]["accept"]["threshold"])
            self.assertEqual(config["calibration"]["status"], "provisional")

    def test_human_gold_writes_fully(self):
        from eval.calibrate_funnel import apply_to_config
        records = []
        for i in range(80):
            records.append(_mk_record(150.0, "good", source="human_gold",
                                      group=f"h{i % 8}", labeller="parshu"))
            records[i]["decisions"] = {
                "hands_ok": {"p_yes": 0.95}, "active_work": {"p_yes": 0.95},
                "step": {"choice": "assemble", "probs": {},
                         "answer_conf": 0.95}}
        for i in range(80):
            records.append(_mk_record(150.0, "bad", source="human_gold",
                                      group=f"h{i % 8}", labeller="parshu"))
            records[-1]["decisions"] = {
                "hands_ok": {"p_yes": 0.2}, "active_work": {"p_yes": 0.2},
                "step": {"choice": "assemble", "probs": {},
                         "answer_conf": 0.2}}
        result = funnel.calibrate_two_sided(
            records, {"min_accepted": 10, "seed": 5, "target_alpha": 0.1})
        # Simulate the validated gold path: the library result gains the
        # human_provenance the --from-gold driver attaches after
        # schema-checking the real labels file.
        result["meta"]["human_provenance"] = {
            "ok": True, "schema": "gold-labels-v1",
            "file": "eval/gold/labels.json", "sha256": "0" * 64,
            "labellers": "unknown", "n_entries": 160,
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = self._tmp_config(tmp)
            config = apply_to_config(result, "human_gold", path)
            self.assertEqual(
                config["accept_conf"],
                result["per_source"]["human_gold"]["accept"]["threshold"])
            self.assertEqual(config["calibration"]["status"], "calibrated")

    def test_human_gold_without_provenance_stays_provisional(self):
        # Fix round 1, Important 5: human_gold records alone do not earn
        # 'calibrated' — provenance must be established.
        from eval.calibrate_funnel import apply_to_config
        records = []
        for i in range(80):
            records.append(_mk_record(150.0, "good", source="human_gold",
                                      group=f"h{i % 8}", labeller="parshu"))
            records[i]["decisions"] = {
                "hands_ok": {"p_yes": 0.95}, "active_work": {"p_yes": 0.95},
                "step": {"choice": "assemble", "probs": {},
                         "answer_conf": 0.95}}
        for i in range(80):
            records.append(_mk_record(150.0, "bad", source="human_gold",
                                      group=f"h{i % 8}", labeller="parshu"))
            records[-1]["decisions"] = {
                "hands_ok": {"p_yes": 0.2}, "active_work": {"p_yes": 0.2},
                "step": {"choice": "assemble", "probs": {},
                         "answer_conf": 0.2}}
        result = funnel.calibrate_two_sided(
            records, {"min_accepted": 10, "seed": 5, "target_alpha": 0.1})
        with tempfile.TemporaryDirectory() as tmp:
            path = self._tmp_config(tmp)
            config = apply_to_config(result, "human_gold", path)
            self.assertEqual(config["calibration"]["status"], "provisional")
            self.assertIn("human provenance not established",
                          config["calibration"]["note"])


class TestStrictDecodable(unittest.TestCase):
    """Fix round 1, Critical 1: decodable is strictly boolean."""

    def thresholds(self):
        return cfg("evidence_reject", None, CALIBRATED)

    def test_true_passes_false_rejects(self):
        out = route(sig(decodable=True), dec(), self.thresholds())
        self.assertEqual(out["verdict"], "accept")
        out = route(sig(decodable=False), dec(), self.thresholds())
        self.assertEqual(out["verdict"], "reject")
        self.assertEqual(out["reasons"], ["UNDECODABLE"])

    def test_non_bool_never_accepts_never_rejects(self):
        for value in (0.0, "0", "0.0", 2, 0.5, -1e308, 1.5, 1, 0,
                      None, "true", "", [], {}, {"decodable": True}):
            out = route(sig(decodable=value), dec(), self.thresholds())
            self.assertEqual(out["verdict"], "review", value)
            self.assertIn("invalid:decodable",
                          [e["check"] for e in out["evidence"]],
                          value)
            self.assertIn("LOW_CONFIDENCE", out["reasons"])

    def test_missing_decodable_reviews_with_invalid_evidence(self):
        signals = sig()
        del signals["decodable"]
        out = route(signals, dec(), self.thresholds())
        self.assertEqual(out["verdict"], "review")
        entry = next(e for e in out["evidence"]
                     if e["check"] == "invalid:decodable")
        self.assertEqual(entry["value"], "missing")

    def test_evidence_values_are_spec_typed(self):
        # Minor 8: float|int|str only — no raw bools, None or NaN.
        out = route(sig(decodable=False), dec(), self.thresholds())
        for entry in out["evidence"]:
            for key in ("value", "threshold"):
                self.assertIsInstance(entry[key], (float, int, str),
                                      (entry["check"], key, entry[key]))
        out = route(sig(decodable=float("nan")), dec(), self.thresholds())
        entry = next(e for e in out["evidence"]
                     if e["check"] == "invalid:decodable")
        self.assertEqual(entry["value"], "nan")


class TestFuzz5000(unittest.TestCase):
    def test_no_accept_of_garbage_l0(self):
        # Fix round 1: 5000 seeded corruptions incl. type confusion.
        # Accept is possible ONLY when every required field is valid and
        # passing (decodable exactly True) plus confident L1.
        import random
        fields = ["decodable", "duration_s", "blur_var", "brightness_mean",
                  "under_exposed_frac", "over_exposed_frac", "motion_mean",
                  "hands_visible_frac"]
        pool = [None, "", "abc", "nan", "60", "0", "1.5", "false", "true",
                float("nan"), float("inf"), float("-inf"), True, False,
                0, 1, 2, 0.0, 0.5, 1.5, -1, 10**100, -10**100,
                [], {}, [0.9], {"p_yes": 0.9}, {"a": 1},
                "9e999", -1e308, 1e308, "  ", "\x00"]
        rng = random.Random(20261010)
        thresholds = cfg("evidence_reject", None, CALIBRATED)
        n_accept = 0
        for _ in range(5000):
            signals = sig()
            for field in fields:
                if rng.random() < 0.5:
                    choice = rng.choice(["del", "pool"])
                    if choice == "del":
                        signals.pop(field, None)
                    else:
                        signals[field] = rng.choice(pool)
            decisions = dec()
            if rng.random() < 0.3:
                decisions["hands_ok"] = rng.choice(
                    [None, {}, {"p_yes": "x"}, {"p_yes": 2}])
            out = route(signals, decisions, thresholds)
            if out["verdict"] == "accept":
                n_accept += 1
                for field in fields:
                    value = signals.get(field)
                    if field == "decodable":
                        # Fix round 1: only a real bool True passes.
                        self.assertIs(value, True)
                    else:
                        # Numeric validity is exactly _as_float: bools,
                        # strings-that-do-not-parse, NaN/inf and garbage
                        # never parse (passing is enforced by route() —
                        # accept requires every check to clear).
                        self.assertIsNotNone(funnel._as_float(value),
                                             (field, value))
        self.assertGreater(n_accept, 0)
        self.assertEqual(
            route(sig(), dec(), thresholds)["verdict"], "accept")


class TestUnionBothPolicies(unittest.TestCase):
    def test_union_reported_under_both_policies_labelled(self):
        records = []
        for i in range(100):
            records.append(_mk_record(5.0 + 15.0 * i / 99, "bad",
                                      group=f"bad{i % 10}"))
        for i in range(100):
            records.append(_mk_record(100.0 + 100.0 * i / 99, "good",
                                      group=f"good{i % 10}"))
        result = funnel.calibrate_two_sided(
            records, {"seed": 3, "bonferroni": False})
        src = result["per_source"]["synthetic"]
        for key, policy in (("union", "evidence_reject"),
                            ("union_integrity_only", "integrity_only")):
            block = src[key]
            self.assertEqual(block["policy"], policy)
            self.assertEqual(block["profile"], "default")
            for field in ("fired", "errors", "upper_bound", "n_certify",
                          "provenance"):
                self.assertIn(field, block)
            self.assertEqual(block["provenance"]["label_source"], "synthetic")
        # The certified policy actually rejects; the default cannot.
        self.assertGreater(src["union"]["fired"], 0)
        self.assertEqual(src["union_integrity_only"]["fired"], 0)

    def test_provenance_beside_every_bound(self):
        records = [_mk_record(150.0, "good", group=f"g{i % 4}")
                   for i in range(40)]
        result = funnel.calibrate_two_sided(records, {"seed": 1})
        src = result["per_source"]["synthetic"]
        for key, entry in src["reject_rules"].items():
            for field in ("n_segments", "n_clips", "n_wearers",
                          "n_labellers", "label_source", "delta_per_rule"):
                self.assertIn(field, entry["provenance"], (key, field))
        self.assertIn("provenance", src["accept"]["certify"])
        self.assertIn("provenance", src["union"])
        self.assertIn("provenance", src["union_integrity_only"])


class TestGoldSchema(unittest.TestCase):
    def test_real_gold_shape_validates(self):
        from eval.calibrate_funnel import check_gold_labels
        labels = {
            "history": [
                {"ts": "t0", "segment_id": "g000", "verdict": "good",
                 "reason": None, "note": ""},
                {"ts": "t1", "segment_id": "g001", "verdict": "bad",
                 "reason": "BLUR", "note": "x"},
            ],
            "latest": {
                "g000": {"ts": "t0", "segment_id": "g000",
                         "verdict": "good", "reason": None, "note": ""},
                "g001": {"ts": "t1", "segment_id": "g001",
                         "verdict": "bad", "reason": "BLUR", "note": "x"},
            },
        }
        verdict = check_gold_labels(labels)
        self.assertTrue(verdict["ok"], verdict["reasons"])
        self.assertEqual(verdict["n_entries"], 2)
        # The current label tool records no labeller id: said plainly.
        self.assertEqual(verdict["labellers"], "unknown")

    def test_labeller_ids_collected_when_present(self):
        from eval.calibrate_funnel import check_gold_labels
        labels = {
            "history": [
                {"ts": "t0", "segment_id": "g000", "verdict": "good",
                 "reason": None, "note": "", "labeller": "qa-lead"},
            ],
            "latest": {
                "g000": {"ts": "t0", "segment_id": "g000",
                         "verdict": "good", "reason": None, "note": "",
                         "labeller": "qa-lead"},
            },
        }
        verdict = check_gold_labels(labels)
        self.assertTrue(verdict["ok"], verdict["reasons"])
        self.assertEqual(verdict["labellers"], ["qa-lead"])

    def test_non_gold_files_rejected(self):
        from eval.calibrate_funnel import check_gold_labels
        for payload in ([], {}, {"records": []},
                        {"latest": {}, "history": []},
                        {"latest": {"g0": {"segment_id": "g0",
                                           "verdict": "maybe"}},
                         "history": [{"segment_id": "g0"}]},
                        {"latest": {"g0": {"segment_id": "g0",
                                           "verdict": "good"}},
                         "history": []}):
            self.assertFalse(check_gold_labels(payload)["ok"], payload)

    def test_records_self_declared_human_refused_without_flag(self):
        from eval.calibrate_funnel import main
        with tempfile.TemporaryDirectory() as tmp:
            records_path = Path(tmp) / "records.json"
            records_path.write_text(json.dumps([{
                "signals": dict(BASE_SIGNALS), "decisions": None,
                "label": "good", "label_source": "human_gold",
                "group": "clipA"}]))
            out = Path(tmp) / "cal.json"
            with self.assertRaises(SystemExit) as ctx:
                main(["--records", str(records_path),
                      "--out", str(out)])
            self.assertIn("human_gold", str(ctx.exception))
            self.assertIn("--human-gold", str(ctx.exception))
            self.assertFalse(out.is_file())

    def test_records_human_with_flag_records_provenance(self):
        from eval.calibrate_funnel import main
        with tempfile.TemporaryDirectory() as tmp:
            records_path = Path(tmp) / "records.json"
            records_path.write_text(json.dumps([{
                "signals": dict(BASE_SIGNALS), "decisions": None,
                "label": "good", "label_source": "human_gold",
                "group": "clipA", "labeller": "qa-lead"}]))
            out = Path(tmp) / "cal.json"
            self.assertEqual(
                main(["--records", str(records_path), "--human-gold",
                      "--out", str(out)]), 0)
            result = json.loads(out.read_text())
            human_prov = result["meta"]["human_provenance"]
            self.assertTrue(human_prov["ok"])
            self.assertEqual(human_prov["labellers"], ["qa-lead"])
            self.assertEqual(len(human_prov["sha256"]), 64)

    def test_write_config_refusal_from_cli(self):
        # Fix round 1, Important 4 end-to-end: non-zero exit, clear
        # message, config file untouched. (N=1200/seed 7 certifies
        # accept_conf=0.4 on synthetic fixtures — N=400 honestly does
        # not, so there is nothing to refuse there.)
        from eval.calibrate_funnel import main
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / "thresholds.json"
            config.write_text(json.dumps(
                {"accept_conf": None,
                 "calibration": {"status": "placeholder"}}))
            out = Path(tmp) / "cal.json"
            with self.assertRaises(SystemExit) as ctx:
                main(["--synthetic", "1200", "--seed", "7",
                      "--source", "synthetic", "--write-config",
                      "--config", str(config), "--out", str(out)])
            self.assertIn("REFUSING", str(ctx.exception))
            self.assertEqual(json.loads(config.read_text())["accept_conf"],
                             None)
            self.assertEqual(
                json.loads(config.read_text())["calibration"]["status"],
                "placeholder")


class TestGoldPlumbing(unittest.TestCase):
    def test_wearers_from_path(self):
        from eval.calibrate_funnel import wearers_from_path
        self.assertEqual(wearers_from_path(
            "/data/indego/User_15_410_0411_2_480.mp4"), ["15"])
        self.assertEqual(wearers_from_path(
            "/data/User_1_User_15_411_2910_2_1_480.mp4"), ["1", "15"])
        self.assertEqual(wearers_from_path("/data/nouser.mp4"), [])

    def test_from_gold_without_labels_says_so_plainly(self):
        from eval.calibrate_funnel import main
        with tempfile.TemporaryDirectory() as tmp:
            manifest = Path(tmp) / "segments.json"
            manifest.write_text(json.dumps({"segments": []}))
            missing = Path(tmp) / "no-labels.json"
            with self.assertRaises(SystemExit) as ctx:
                main(["--from-gold", "--manifest", str(manifest),
                      "--gold", str(missing)])
            self.assertIn("no human gold labels yet", str(ctx.exception))

    def test_driver_sidecar_holds_produced_utc(self):
        # M3 end-to-end: meta has no clock; sidecar does.
        from eval.calibrate_funnel import main
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "cal.json"
            self.assertEqual(
                main(["--synthetic", "60", "--seed", "7",
                      "--out", str(out)]), 0)
            result = json.loads(out.read_text())
            self.assertIn("produced_utc", result["sidecar"])
            self.assertNotIn("produced_utc", json.dumps(result["meta"]))


if __name__ == "__main__":
    unittest.main()
