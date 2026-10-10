"""LB2 tests: adopted PoC policy constants + integrated verdict artifacts.

Fast (<5 s), CPU-only, no network.  Bundle-dependent assertions skip when the
pod bundle is absent (a clean clone has no caches); the policy-unit tests
always run.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import unittest

try:
    import pytest
except ImportError:  # these modules are pytest-style; `unittest discover` (the merge gate env) has no pytest
    raise unittest.SkipTest("pytest not installed: run these with `python -m pytest tests/test_poc_integration.py`")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from jadal.poc_stream import policy  # noqa: E402

BUNDLE = policy.DEFAULT_BUNDLE or ""  # empty when JADAL_BUNDLE is unset
RESULTS = os.path.join(REPO_ROOT, "eval", "results", "poc_integrated")

needs_bundle = pytest.mark.skipif(
    not BUNDLE or not os.path.exists(os.path.join(BUNDLE, "caches", "ensemble", "ensemble_v1.json")),
    reason="pod bundle caches not present",
)


# ---------------------------------------------------------------- policy units
def test_adopted_constants_are_the_decided_ones():
    assert policy.ADOPTED["sample_every_n_windows"] == 2  # E3.01/E3.05 adopted_policy
    assert policy.ADOPTED["sample_offset"] == 0
    assert policy.ADOPTED["early_exit_s"] == 4.0  # E3.03 saturation
    assert policy.ADOPTED["views"] == ("pixel26", "eg2_2fps")  # E1.01 / E1.02
    assert policy.ADOPTED["fusion"] == "equal_weight_probability_average"
    assert policy.ADOPTED["descriptive_only"] is True


def test_sampling_grid_matches_run_e3_pol_every_k():
    assert policy.sampled_indices(10) == [0, 2, 4, 6, 8]
    assert policy.sampled_indices(5, every=3) == [0, 3]
    assert policy.sampled_indices(7, every=3, offset=1) == [1, 4]
    assert policy.sampled_indices(0) == []
    with pytest.raises(ValueError):
        policy.sampled_indices(5, every=0)
    with pytest.raises(ValueError):
        policy.sampled_indices(5, every=2, offset=2)


def test_prefix_geometry_matches_e3_curve():
    # E3.03: 4 frames -> 2.0 s, 8 -> 4.0 s, 12 -> 6.0 s, 16 -> 8.0 s, 21 -> 10.5 s
    assert policy.eg2_prefix_frames(2.0) == 4
    assert policy.eg2_prefix_frames() == 8
    assert policy.eg2_prefix_frames(6.0) == 12
    assert policy.eg2_prefix_frames(10.5) == 21
    # features.json prefixes are at 2,4,6,8,10 s
    assert policy.pixel_prefix_index() == 1
    assert policy.pixel_prefix_index(2.0) == 0
    assert policy.pixel_prefix_index(10.0) == 4


def test_verdict_mapping_uses_frozen_rule():
    assert policy.window_verdict(0.9) == "accept"
    assert policy.window_verdict(0.3001) == "reject"
    assert policy.window_verdict(0.3999) == "reject"
    assert policy.window_verdict(0.4) == "uncertain"
    assert policy.window_verdict(0.5) == "uncertain"
    assert policy.window_verdict(0.5001) == "accept"
    assert policy.window_verdict(float("nan")) == "uncertain"


def test_clip_verdict_threshold_and_empty():
    assert policy.clip_verdict([True, True, False, False, False]) == "reject"  # 0.4 >= 0.30
    assert policy.clip_verdict([True, False, False, False]) == "accept"  # 0.25 < 0.30
    assert policy.clip_verdict([]) == "uncertain"


def test_fusion_is_equal_weight_and_predict_head_is_monotone():
    assert policy.fuse_probabilities(1.0, 0.0) == pytest.approx(0.5)
    head = {"mean": [0.0], "scale": [1.0], "coef": [1.0], "intercept": 0.0}
    assert policy.predict_head([[0.0]], head)[0] == pytest.approx(0.5)
    assert policy.predict_head([[2.0]], head)[0] > policy.predict_head([[1.0]], head)[0]


def test_cost_ratios_are_the_e304_numbers():
    assert policy.cost_ratio("every_2_speedup") == pytest.approx(1.94)
    assert policy.cost_ratio("every_2_4s_speedup") == pytest.approx(4.86)


def test_policy_fingerprint_is_serialisable_and_labelled_descriptive():
    fp = policy.policy_fingerprint()
    json.dumps(fp)
    assert fp["descriptive_only"] is True
    assert fp["sample_every_n_windows"] == 2
    assert fp["early_exit_s"] == 4.0
    assert "not validated" in fp["status"]


# ------------------------------------------------------------ frozen artifact
@needs_bundle
def test_frozen_ensemble_loads_and_dims_match():
    ens = policy.load_frozen_ensemble(BUNDLE)
    assert ens["name"] == "ensemble_v1"
    assert len(ens["views"]["pixel"]["coef"]) == 26
    assert len(ens["views"]["eg2"]["coef"]) == 768
    assert list(ens["fusion"]["weights"]) == [0.5, 0.5]
    # default resolution must be the byte-exact pre-registered file, not the
    # path-remapped configs/ensemble_v1.json
    resolved = policy.resolve_frozen_prereg(BUNDLE)
    assert resolved.endswith(policy.FROZEN_PREREG_REL)
    assert policy.sha256_file(resolved) == policy.FROZEN_PREREG_SHA256


def test_frozen_ensemble_fails_closed_on_tampered_bytes(tmp_path):
    src = os.path.join(REPO_ROOT, policy.FROZEN_PREREG_REL)
    if not os.path.exists(src):
        pytest.skip("pre-registered frozen file not present")
    data = open(src, "rb").read()
    tampered = data.replace(b'"ensemble_v1"', b'"ensemble_vX"')
    assert tampered != data  # the replacement really happened
    bad = tmp_path / "ensemble_v1.prereg.json"
    bad.write_bytes(tampered)
    with pytest.raises(ValueError, match="sha256 mismatch"):
        policy.load_frozen_ensemble(path=str(bad))


def test_frozen_ensemble_missing_file_raises_and_never_falls_back(tmp_path, monkeypatch):
    with pytest.raises(FileNotFoundError, match="frozen ensemble not found"):
        policy.load_frozen_ensemble(path=str(tmp_path / "nope.json"))
    # Neutralise the real repo copy, otherwise resolve_frozen_prereg legitimately
    # finds configs/frozen/ensemble_v1.prereg.json and the
    # "nothing available -> refuse to fall back" branch is unreachable.
    monkeypatch.setattr(policy, "_REPO_ROOT", str(tmp_path / "no_repo"))
    with pytest.raises(FileNotFoundError, match="refusing to fall back"):
        policy.resolve_frozen_prereg(bundle=str(tmp_path / "empty_bundle"))


@needs_bundle
def test_eg2_prefix_mean_ignores_nan_padding():
    z = np.load(os.path.join(BUNDLE, "caches", "frames10k", "segments.npz"), allow_pickle=True)
    F, counts = z["F"], z["counts"]
    out = policy.eg2_prefix_mean(F, counts)
    assert out.shape == (len(counts), F.shape[2])
    assert np.isfinite(out).all()
    norms = np.linalg.norm(out, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-4)


# --------------------------------------------------------- integrated results
@needs_bundle
def test_window_verdicts_jsonl_schema_and_descriptive_labels():
    path = os.path.join(RESULTS, "window_verdicts.jsonl")
    if not os.path.exists(path):
        pytest.skip("run eval/poc_integrate/run_integration.py first")
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    gold = [r for r in rows if r["label_domain"] == "IndEgo-gold"]
    ood = [r for r in rows if r["label_domain"] == "Egocentric-10K-OOD-human"]
    assert len(gold) == 150 and len(ood) == 60
    for r in rows:
        assert r["verdict"] in {"accept", "reject", "uncertain"}
        assert r["descriptive_only"] is True
        assert r["observed_seconds"] is not None
        assert r["chosen_views"]
        assert 0.0 <= r["score"] <= 1.0
    for r in gold:
        assert r["chosen_views"] == ["pixel26", "eg2_2fps"]
        assert r["observed_seconds"] == 4.0
        assert isinstance(r["sampled"], bool)
        assert "LOCO" in r["score_protocol"]
    for r in ood:
        assert r["claim"].startswith("none")
        assert r["sampled"] is None


@needs_bundle
def test_summary_has_provenance_and_no_ood_claim():
    path = os.path.join(RESULTS, "summary.json")
    if not os.path.exists(path):
        pytest.skip("run eval/poc_integrate/run_integration.py first")
    s = json.load(open(path, encoding="utf-8"))
    assert s["descriptive_only"] is True
    assert s["n"] == 150
    assert "human gold" in s["label_source"]
    assert s["labels_sha256"]
    assert "LOCO" in s["protocol"] and "bootstrap" in s["protocol"]
    assert s["ood_descriptive"]["claim"].startswith("none")
    for name, m in s["metrics"].items():
        assert m["n"] == 150 and len(m["auroc_ci95"]) == 2
    assert s["policy"]["sample_every_n_windows"] == 2
    assert s["policy"]["early_exit_s"] == 4.0
    assert sum(s["verdict_counts"].values()) == 150
