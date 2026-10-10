"""Tests for E8.01 (B2 re-score): contract digests, verdicts, router, decode path.

Fast: the full file runs in ~15 s and needs no network. The two integration
tests are skipped when the video/decode tooling is absent.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

REPO = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "eval", "exp_rescore"))

import contract as C  # noqa: E402

BUNDLE = os.environ.get("JADAL_BUNDLE", "/workspace/jadal_pod_bundle")
MANIFEST = os.path.join(BUNDLE, "queues", "manifest_scale.json")
WINDOW = None
if os.path.exists(MANIFEST):
    _items = sorted(json.load(open(MANIFEST))["items"], key=lambda i: (i["clip"], i["start_s"]))
    WINDOW = _items[0]


def _dummy_window():
    return {"window_id": "w1", "factory": "factory_001", "worker": "worker_006",
            "clip": "factory_001/worker_006/x.mp4", "start_s": 0.0, "end_s": 10.0}


def _dummy_line(**kw):
    feats = {"dark_frac": 0.1, "frozen_frac": 0.2, "luma_mean": 0.4}
    block, dig = C.features_block(feats, [0.1] * 768, 300, 20, 30.0, 0.7, 0.2)
    args = dict(window=_dummy_window(), verdict="review", stage="L1",
                reasons=C.reasons_for(0.45, "review", 0.75, 0.25,
                                      {"reason_evidence": []}, feats),
                score=0.45, features_digest=dig, features=block,
                versions={"recipe": "ensemble_v1"}, latency_ms={"total": 1000.0})
    args.update(kw)
    return C.build_line(**args)


# ------------------------------------------------------------------ canonical/digests
def test_canonical_json_is_stable_and_sorted():
    a = C.canonical_bytes({"b": 1, "a": [1.5, 2]})
    b = C.canonical_bytes({"a": [1.5, 2], "b": 1})
    assert a == b == b'{"a":[1.5,2],"b":1}'


def test_receipt_roundtrip_and_tamper_detection():
    line = _dummy_line()
    assert C.verify_line(line)
    tampered = json.loads(json.dumps(line))
    tampered["score"] = 0.99
    assert not C.verify_line(tampered), "receipt must cover the score"
    tampered2 = json.loads(json.dumps(line))
    tampered2["features"]["luma_mean"] = 0.9
    assert not C.verify_line(tampered2), "features_digest must cover features{}"


def test_contract_fields_exact():
    line = _dummy_line()
    assert tuple(sorted(line)) == tuple(sorted(C.CONTRACT_FIELDS))
    assert line["label_source"] == "none"


def test_features_digest_tracks_content_and_eg2_hash():
    f1, d1 = C.features_block({"a": 1.0}, [1.0, 2.0], 300, 20, 30.0, 0.5, 0.5)
    f2, d2 = C.features_block({"a": 1.0}, [1.0, 2.0], 300, 20, 30.0, 0.5, 0.5)
    f3, d3 = C.features_block({"a": 1.0}, [1.0, 2.5], 300, 20, 30.0, 0.5, 0.5)
    assert d1 == d2 and d1 != d3
    assert f1["eg2_dim"] == 2 and len(f1["eg2_mean_sha256"]) == 64


def test_round_floats_rejects_non_finite():
    assert C.round_floats({"x": 1.00004})["x"] == 1.0
    with pytest.raises(ValueError):
        C.round_floats({"x": float("nan")})


# ------------------------------------------------------------------ verdicts/reasons
def test_verdict_thresholds_follow_profile_operating_point():
    assert C.verdict_for(0.9, 0.75, 0.25) == "accept"
    assert C.verdict_for(0.25, 0.75, 0.25) == "reject"
    assert C.verdict_for(0.5, 0.75, 0.25) == "review"


def test_reasons_carry_op_threshold_evidence():
    r = C.reasons_for(0.45, "review", 0.75, 0.25, {"reason_evidence": []},
                      {"dark_frac": 0.0, "frozen_frac": 0.0})
    assert r[0]["code"] == "LOW_MARGIN_PROFILE_REVIEW"
    ev = r[0]["evidence"]
    assert set(ev) == {"feature", "value", "op", "threshold"} and ev["op"] == "in"
    dark = C.reasons_for(0.1, "reject", 0.75, 0.25, {"reason_evidence": []},
                         {"dark_frac": 0.7, "frozen_frac": 0.0})
    assert [x["code"] for x in dark] == ["SCORE_AT_OR_BELOW_PROFILE_REJECT", "HEURISTIC_DARK"]


def test_router_routes_1080p30_to_factory_profile():
    from jadal.regimes.router import choose_profile
    desc = {"width": 1920, "height": 1080, "fps": 30.0, "sharp_norm": 14000.0,
            "luma_mean": 0.39, "shift_mean": 18.2, "noise_sigma": 0.005,
            "static_frac": 0.5, "clip_frac": 0.14, "temp_energy": 0.17}
    d = choose_profile(desc)
    assert d["profile"] == "factory_1080p30" and d["unseen_regime"] is False
    frag = C.route_from_decision(d, desc)
    codes = [c for c, _ in frag["reason_evidence"]]
    assert "PROFILE_ROUTE_HIRES" in codes
    # out-of-range descriptor -> default + UNSEEN
    d2 = choose_profile({**desc, "luma_mean": 0.9})
    assert d2["profile"] == "default" and d2["unseen_regime"] is True
    assert "UNSEEN_REGIME" in [c for c, _ in C.route_from_decision(d2, desc)["reason_evidence"]]


def test_header_states_no_transfer():
    h = C.header_line(10, "ensemble_v1", {"factory_1080p30": 10}, {"review": 10}, "A40", "abc")
    assert "did not transfer" in h["honesty_note"]
    assert h["label_source"] == "none" and h["type"] == "header"


@pytest.mark.skipif(WINDOW is None or not os.path.exists(WINDOW["path"]),
                    reason="scale corpus / ffmpeg not available")
def test_integration_decode_frames_and_descriptors():
    """One NVDEC pass -> expected frame counts; descriptors match the cv2 protocol."""
    sys.path.insert(0, os.path.join(REPO, "eval", "exp_regimes"))
    from descriptors import describe_window, ffprobe_meta
    from jadal.regimes.router import choose_profile
    from run_e8 import decode_three_branch, describe_from_native_frames, frames_from_raw

    pix, eg, nat = decode_three_branch(WINDOW["path"], 0.0, 10.0)
    f_pix = frames_from_raw(pix, 90, 160)
    assert f_pix.shape == (300, 90, 160, 3)
    assert frames_from_raw(eg, 288, 512).shape == (20, 288, 512, 3)
    nat_frames = list(frames_from_raw(nat, 1080, 1920))
    assert len(nat_frames) == 8 and f_pix[0].std() > 0

    ref = describe_window(WINDOW["path"], 0.0, 10.0)
    mine = describe_from_native_frames(ffprobe_meta(WINDOW["path"]), nat_frames, 0.0, 10.0)
    assert mine["n_frames_used"] == 8
    assert abs(mine["luma_mean"] - ref["luma_mean"]) < 0.03
    assert abs(mine["sharp_norm"] - ref["sharp_norm"]) / ref["sharp_norm"] < 0.25
    assert choose_profile(mine)["profile"] == choose_profile(ref)["profile"] == "factory_1080p30"


def test_ffmpeg_present_and_receipts_helper_cli():
    """ffmpeg exists and jadal.receipts imports (documents the two hard deps)."""
    assert subprocess.run(["ffmpeg", "-version"], capture_output=True).returncode == 0
    import jadal.receipts  # noqa: F401
