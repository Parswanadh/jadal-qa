"""Deterministic unit tests for the physics model: fixed inputs -> fixed outputs.

Runs with pytest (``python3 -m pytest tools/phys/tests/test_phys_model.py``) or
directly (``python3 tools/phys/tests/test_phys_model.py``).
"""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from tools.phys import model  # noqa: E402

RULES = model.load_rules()


def clean_features() -> dict:
    """A window that crosses no threshold (score 0 -> ACCEPT)."""
    return {
        "full.luma_mean": 0.50, "full.luma_p05": 0.30, "full.motion_energy": 0.001,
        "full.dark_frac": 0.02, "full.bright_frac": 0.001, "full.hf_ratio": 0.10,
        "full.dct_energy": 0.10, "full.temp_b0": 100.0, "full.frozen_frac": 0.0,
        "full.skin_frac": 0.50,
        "phys.flow_act": 0.10, "phys.flow_med": 0.10, "phys.dhash_loop_min": 0.0,
        "phys.mad_mean": 0.0, "phys.frm_bytes_med": 100.0, "phys.p_size_med": 100.0,
        "phys.snr_db": 5.0,
    }


def test_accept_clean_window():
    sc = model.score_window(clean_features(), RULES)
    assert sc["score"] == 0, sc
    assert sc["fired"] == []
    d = model.decide(sc, RULES)
    assert d["decision"] == "ACCEPT"
    assert d["reason"] == "no_rule"


def test_review_band_single_weak_rule():
    f = clean_features()
    f["phys.flow_act"] = 0.70  # crosses only the p75 threshold (0.6394)
    sc = model.score_window(f, RULES)
    assert sc["score"] == 1, sc
    assert sc["strong_fired"] == []
    d = model.decide(sc, RULES)
    assert d["decision"] == "REVIEW"
    assert d["reason"] == "weak_rule"
    # evidence is integer-quantised
    assert d["evidence"]["value"] == 7000
    assert d["evidence"]["threshold"] == 6394
    assert d["evidence"]["op"] == ">"


def test_quality_rule_forces_reject():
    f = clean_features()
    f["full.luma_mean"] = 0.05  # below the p01 threshold 0.0723 -> TOO_DARK
    sc = model.score_window(f, RULES)
    assert "TOO_DARK" in sc["strong_fired"]
    d = model.decide(sc, RULES)
    assert d["decision"] == "REJECT"
    assert d["reason"] == "quality_rule"


def test_soft_quality_crossing_is_review_not_reject():
    # BLUR is a quality rule, but a soft (p25) crossing only adds to the score.
    f = clean_features()
    f["full.hf_ratio"] = 0.0130  # between p10 (0.0097) and p25 (0.0134)
    sc = model.score_window(f, RULES)
    assert sc["strong_fired"] == [], sc
    assert "BLUR" in sc["fired_rules"]
    d = model.decide(sc, RULES)
    assert d["decision"] == "REVIEW"


def test_undecodable_tripwire():
    f = clean_features()
    del f["full.luma_mean"]
    sc = model.score_window(f, RULES)
    assert sc["score"] == RULES["undecodable_score"]
    assert sc["fired_rules"] == ["UNDECODABLE"]
    d = model.decide(sc, RULES)
    assert d["decision"] == "REJECT"
    assert d["reason"] == "quality_rule"


def test_score_reject_threshold_from_integers():
    scored = {"score": RULES["reject_threshold"], "fired": [
        {"rule": "X", "feature": "phys.flow_act", "value": 1, "op": ">",
         "threshold": 1, "strength": "weak", "n_thresholds_crossed": 1}],
        "strong_fired": [], "weak_fired": ["X"], "fired_rules": ["X"]}
    d = model.decide(scored, RULES)
    assert d["decision"] == "REJECT"
    assert d["reason"] == "score>=reject_threshold"


def test_quantise_is_integer_and_deterministic():
    assert model.quantise(0.2678, 4) == 2678
    assert model.quantise(0.0, 4) == 0
    assert model.quantise(float("nan"), 4) is None
    assert model.quantise(float("inf"), 4) is None
    assert isinstance(model.quantise(12.3456, 4), int)


def test_boundary_equality_does_not_cross():
    # value exactly equal to a threshold never crosses (strict inequalities).
    f = clean_features()
    f["full.hf_ratio"] = 0.0134  # == p75 threshold of BLUR (op '<')
    sc = model.score_window(f, RULES)
    assert "BLUR" not in sc["fired_rules"]


def test_receipt_is_canonical_and_has_required_keys():
    rec = model.receipt("g000", clean_features(), RULES)
    for k in ("window_id", "decision", "evidence", "feature_digest",
              "rule_table_sha256", "code_sha256"):
        assert k in rec, k
    txt = model.canonical_json(rec)
    # sorted keys, compact separators, no spaces
    assert ", " not in txt and ": " not in txt
    assert list(json.loads(txt)) == sorted(rec)
    assert rec["rule_table_sha256"] == RULES["_sha256"]
    assert len(rec["code_sha256"]) == 64


def test_receipt_evidence_shape():
    f = clean_features()
    f["phys.flow_act"] = 0.70
    rec = model.receipt("g001", f, RULES)
    assert rec["decision"] == "REVIEW"
    assert len(rec["evidence"]) == 1
    ev = rec["evidence"][0]
    assert set(ev) == {"rule", "feature", "value", "op", "threshold"}
    assert all(ev[k] is None or isinstance(ev[k], int)
               for k in ("value", "threshold"))


def test_fallback_resolves_review():
    f = clean_features()
    f["phys.flow_act"] = 0.70
    rec = model.receipt("g002", f, RULES, fallback_prob=0.9)
    assert rec["decision"] == "ACCEPT"
    assert rec["decision_deterministic"] == "REVIEW"
    assert rec["fallback"]["decision"] == "ACCEPT"
    rec2 = model.receipt("g003", f, RULES, fallback_prob=0.1)
    assert rec2["decision"] == "REJECT"
    # fallback never touches a non-REVIEW window
    rec3 = model.receipt("g004", clean_features(), RULES, fallback_prob=0.1)
    assert rec3["decision"] == "ACCEPT"
    assert rec3.get("fallback") is None


def test_fallback_binds_source_sha256():
    f = clean_features()
    f["phys.flow_act"] = 0.70
    h = "a" * 64
    rec = model.receipt("g010", f, RULES, fallback_prob=0.9, fallback_sha256=h)
    assert rec["fallback"]["source_sha256"] == h
    # when no source file is given the key is still present (null), never absent
    rec2 = model.receipt("g011", f, RULES, fallback_prob=0.9)
    assert "source_sha256" in rec2["fallback"]
    assert rec2["fallback"]["source_sha256"] is None


def test_end_to_end_determinism():
    a = model.canonical_json(model.receipt("g005", clean_features(), RULES))
    b = model.canonical_json(model.receipt("g005", clean_features(), RULES))
    assert a == b
    # feature digest is stable and order-independent
    d1 = model.feature_digest(clean_features(), RULES)
    rev = dict(reversed(list(clean_features().items())))
    assert model.feature_digest(rev, RULES) == d1


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"{len(fns)} tests passed")


if __name__ == "__main__":
    _run_all()
