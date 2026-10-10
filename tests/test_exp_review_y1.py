"""Y1.01 invariants (fast; no network; reads committed results JSON only)."""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "eval/results/exp_review/Y1.01.json")
SUPPORT = os.path.join(ROOT, "eval/results/exp_review/Y1.01_support.json")


def _load(path):
    with open(path) as f:
        return json.load(f)


def test_results_exist_and_baseline():
    d = _load(RESULTS)
    assert d["id"] == "Y1.01"
    assert d["n"]["e8_1546"] == 1546
    assert d["n"]["gold_150"] == 150
    assert d["n"]["ood_60"] == 60 and d["n"]["ood_bad"] == 9
    assert d["reproduction"]["profile_rule_matches_rows"] == 1546
    assert d["reproduction"]["e8_verdict_distribution"] == {"accept": 296, "reject": 83, "review": 1167}


def test_review_monotonic_in_w_and_target_met():
    d = _load(RESULTS)
    rows = d["sweep"]["raw"]["rows"]
    revs = [r["review_share_1546"] for r in rows]
    assert all(b >= a - 1e-9 for a, b in zip(revs, revs[1:])), "review must increase with band width"
    e = d["sweep"]["raw"]["first_w_review_lt_60"]
    assert e["w"] == 0.16
    assert e["review_share_1546"] < 0.60
    assert e["gold_n_decided"] > 0


def test_label_sources_stated_and_ood_not_fit():
    d = _load(RESULTS)
    assert d["label_source"]["e8_1546"] == "none"
    assert "human" in d["label_source"]["gold_150"]
    assert "human" in d["label_source"]["ood_60"]
    assert any("evaluation only" in h for h in d["honesty"])


def test_support_calibration_conclusion():
    s = _load(SUPPORT)
    assert s["id"] == "Y1.01"
    eq = s["calibration_at_equal_error"]["min_review_share_per_curve"]
    # in-sample temperature is ~= raw; LOCO is the artifact and must be lower
    assert abs(eq["temp_insample"] - eq["raw"]) < 0.01
    assert eq["temp_loco"] < eq["raw"] - 0.10
