"""Router unit tests: synthetic descriptors only, no video, < 30 s. (E4.04)"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "jadal", "regimes"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from jadal.regimes.router import TRAINING_RANGES, choose_profile, receipt_fragment

BASE = {"width": 480, "height": 480, "fps": 10.0, "shift_mean": 15.0,
        "temp_energy": 0.12, "static_frac": 0.0, "sharp_norm": 8000.0,
        "luma_mean": 0.30, "clip_frac": 0.17, "noise_sigma": 0.004,
        "_sha256": "abc123"}


def test_clear_stationary():
    r = choose_profile(dict(BASE))
    assert r["profile"] == "clear_stationary", r
    assert r["margin"] >= 0 and not r["unseen_regime"]


def test_factory_1080p30_by_res_and_fps():
    d = dict(BASE, width=1920, height=1080, fps=30.0)
    r = choose_profile(d)
    assert r["profile"] == "factory_1080p30", r


def test_walking_logistics_by_motion():
    d = dict(BASE, shift_mean=35.0, temp_energy=0.30)
    r = choose_profile(d)
    assert r["profile"] == "walking_logistics", r


def test_blurry_lowlight_by_sharpness():
    d = dict(BASE, sharp_norm=1000.0)
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
    d = dict(BASE, width=1920, height=1080, fps=30.0, shift_mean=40.0,
             temp_energy=0.30)
    assert choose_profile(d)["profile"] == "factory_1080p30"


def test_deterministic_and_receipt_fragment():
    d = dict(BASE)
    a, b = choose_profile(d), choose_profile(d)
    assert a == b
    frag = receipt_fragment(d, a)
    assert frag == {"profile": "clear_stationary", "router_margin": a["margin"],
                    "router_reasons": a["reasons"], "unseen_regime": False,
                    "descriptors_sha": "abc123"}, frag


def test_router_covers_e401_pool_without_crash():
    """Route every measured E4.01 descriptor record; report the distribution."""
    import glob as _glob
    from collections import Counter
    files = sorted(_glob.glob("eval/results/exp_regimes/desc_*.jsonl"))
    assert files, "run compute_descriptors.py first"
    cnt, unseen, margins = Counter(), 0, []
    n = 0
    for p in files:
        for line in open(p):
            r = json.loads(line)
            if not r["ok"]:
                continue
            d = choose_profile(r["desc"])
            cnt[d["profile"]] += 1
            unseen += d["unseen_regime"]
            margins.append(d["margin"])
            n += 1
    assert n >= 600, n
    assert unseen == 0, f"{unseen}/{n} unseen (training ranges wrong?)"
    assert set(cnt) >= {"clear_stationary", "factory_1080p30"}, dict(cnt)
    print(f"\nrouter coverage n={n}: {dict(cnt)} unseen={unseen}")


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
