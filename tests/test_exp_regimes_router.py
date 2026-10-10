"""Router unit tests: synthetic descriptors only, no video, < 30 s. (E4.04)

Runs under BOTH `python3 -m unittest discover -s tests` (unittest.TestCase
classes) and `pytest`. Every file path is anchored to __file__, so the tests
pass from any working directory.
"""
import glob
import json
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for _p in (os.path.join(ROOT, "jadal", "regimes"), ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from jadal.regimes.router import (  # noqa: E402
    TRAINING_RANGES, choose_profile, receipt_fragment)

BASE = {"width": 480, "height": 480, "fps": 10.0, "shift_mean": 10.0,
        "static_frac": 0.8, "sharp_norm": 8000.0, "luma_mean": 0.30,
        "clip_frac": 0.15, "noise_sigma": 0.005, "_sha256": "abc123"}


class RouterRuleTests(unittest.TestCase):
    def test_clear_stationary(self):
        r = choose_profile(dict(BASE))
        self.assertEqual(r["profile"], "clear_stationary", r)
        self.assertGreaterEqual(r["margin"], 0)
        self.assertFalse(r["unseen_regime"])

    def test_factory_1080p30_by_res_and_fps(self):
        d = dict(BASE, width=1920, height=1080, fps=30.0)
        r = choose_profile(d)
        self.assertEqual(r["profile"], "factory_1080p30", r)

    def test_walking_logistics_by_motion(self):
        d = dict(BASE, shift_mean=35.0, static_frac=0.1)
        r = choose_profile(d)
        self.assertEqual(r["profile"], "walking_logistics", r)

    def test_walking_logistics_by_temp_energy(self):
        d = dict(BASE, temp_energy=0.30)
        r = choose_profile(d)
        self.assertEqual(r["profile"], "walking_logistics", r)

    def test_blurry_lowlight_by_sharpness(self):
        r = choose_profile(dict(BASE, sharp_norm=1200.0))
        self.assertEqual(r["profile"], "blurry_lowlight", r)

    def test_blurry_lowlight_by_luma_and_clip(self):
        self.assertEqual(choose_profile(dict(BASE, luma_mean=0.10))["profile"],
                         "blurry_lowlight")
        self.assertEqual(choose_profile(dict(BASE, clip_frac=0.60))["profile"],
                         "blurry_lowlight")

    def test_priority_hires_over_motion(self):
        d = dict(BASE, width=1920, fps=30.0, shift_mean=20.0, static_frac=0.0)
        self.assertEqual(choose_profile(d)["profile"], "factory_1080p30")


class RouterUnseenRegimeTests(unittest.TestCase):
    def test_rule_descriptor_keys_are_in_training_ranges(self):
        # Finding 1: every descriptor the rules decide on must be range-checked.
        for k in ("clip_frac", "temp_energy", "static_frac"):
            self.assertIn(k, TRAINING_RANGES)

    def test_unseen_regime_falls_back_to_default(self):
        d = dict(BASE, fps=120.0, width=3840)
        r = choose_profile(d)
        self.assertEqual(r["profile"], "default", r)
        self.assertTrue(r["unseen_regime"], r)
        self.assertTrue(any("UNSEEN_REGIME" in x for x in r["reasons"]))

    def test_unseen_regime_out_of_range_clip_frac(self):
        r = choose_profile(dict(BASE, clip_frac=0.90))
        self.assertEqual(r["profile"], "default", r)
        self.assertTrue(r["unseen_regime"], r)
        self.assertTrue(any("clip_frac" in x for x in r["reasons"]))

    def test_unseen_regime_out_of_range_temp_energy(self):
        r = choose_profile(dict(BASE, temp_energy=2.0))
        self.assertEqual(r["profile"], "default", r)
        self.assertTrue(r["unseen_regime"], r)
        self.assertTrue(any("temp_energy" in x for x in r["reasons"]))


class RouterDeterminismAndReceiptTests(unittest.TestCase):
    def test_deterministic_and_receipt_fragment(self):
        d = dict(BASE)
        a, b = choose_profile(d), choose_profile(d)
        self.assertEqual(a, b)
        frag = receipt_fragment(d, a)
        self.assertEqual(frag, {
            "profile": "clear_stationary", "router_margin": a["margin"],
            "router_reasons": a["reasons"], "unseen_regime": False,
            "descriptors_sha": "abc123"}, frag)


class ProfileSchemaTests(unittest.TestCase):
    def test_profiles_validate_against_schema(self):
        with open(os.path.join(ROOT, "profiles", "schema.json")) as fh:
            schema = json.load(fh)
        for p in glob.glob(os.path.join(ROOT, "profiles", "*.json")):
            if os.path.basename(p) == "schema.json":
                continue
            with open(p) as fh:
                prof = json.load(fh)
            for req in schema["required"]:
                self.assertIn(req, prof, (p, req))
            self.assertLess(prof["thresholds"]["reject"],
                            prof["thresholds"]["accept"], p)


if __name__ == "__main__":
    unittest.main()
