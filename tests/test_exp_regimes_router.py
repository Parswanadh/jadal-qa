"""Router unit tests: synthetic descriptors only, no video, < 30 s. (E4.04)"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "jadal", "regimes"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from jadal.regimes.router import TRAINING_RANGES, choose_profile, receipt_fragment

BASE = {"width": 854, "height": 480, "fps": 10.0, "shift_mean": 2.0,
        "static_frac": 0.8, "sharp_norm": 300.0, "luma_mean": 0.45,
        "clip_frac": 0.05, "noise_sigma": 0.01, "_sha256": "abc123"}


def test_clear_stationary():
    r = choose_profile(dict(BASE))
    assert r["profile"] == "clear_stationary", r
    assert r["margin"] >= 0 and not r["unseen_regime"]


def test_factory_1080p30_by_res_and_fps():
    d = dict(BASE, width=1920, height=1080, fps=30.0)
    r = choose_profile(d)
    assert r["profile"] == "factory_1080p30", r


def test_walking_logistics_by_motion():
    d = dict(BASE, shift_mean=12.0, static_frac=0.1)
    r = choose_profile(d)
    assert r["profile"] == "walking_logistics", r


def test_blurry_lowlight_by_sharpness():
    d = dict(BASE, sharp_norm=10.0)
    r = choose_profile(dict(d))["profile"] == "blurry_lowlight"
    assert r


def test_blurry_lowlight_by_luma_and_clip():
    assert choose_profile(dict(BASE, luma_mean=0.10))["profile"] == "blurry_lowlight"
    assert choose_profile(dict(BASE, clip_frac=0.60))["profile"] == "blurry_lowlight"


def test_unseen_regime_falls_back_to_default():
    d = dict(BASE, fps=120.0, width=3840)
    r = choose_profile(d)
    assert r["profile"] == "default" and r["unseen_regime"], r
    assert any("UNSEEN_REGIME" in x for x in r["reasons"])


def test_priority_hires_over_motion():
    d = dict(BASE, width=1920, fps=30.0, shift_mean=20.0, static_frac=0.0)
    assert choose_profile(d)["profile"] == "factory_1080p30"


def test_deterministic_and_receipt_fragment():
    d = dict(BASE)
    a, b = choose_profile(d), choose_profile(d)
    assert a == b
    frag = receipt_fragment(d, a)
    assert frag == {"profile": "clear_stationary", "router_margin": a["margin"],
                    "router_reasons": a["reasons"], "unseen_regime": False,
                    "descriptors_sha": "abc123"}, frag


def test_profiles_validate_against_schema():
    import glob
    schema = json.load(open("profiles/schema.json"))
    for p in glob.glob("profiles/*.json"):
        if os.path.basename(p) == "schema.json":
            continue
        prof = json.load(open(p))
        for req in schema["required"]:
            assert req in prof, (p, req)
        assert prof["thresholds"]["reject"] < prof["thresholds"]["accept"], p
