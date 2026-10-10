"""Z1 re-run on the 160 OOD labels: unit + result-consistency tests (< 30 s, no network)."""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "eval", "exp_domain_adapt160"))

from eval.exp_domain_adapt160 import adaptlib as A  # noqa: E402
from eval.exp_domain_adapt160 import run_z1_160 as m  # noqa: E402


def test_feature_sets():
    assert len(A.FEATURE_NAMES) == 26
    assert len(A.STABLE) == 8
    assert len(A.SHARPNESS) == 3
    assert A.idx_of(A.SHARPNESS) == [6, 7, 8]


def test_clip_boot_ci_shape():
    y = [1, 1, 0, 0, 1, 0]
    s = [0.9, 0.8, 0.2, 0.1, 0.7, 0.3]
    g = ["a", "a", "a", "b", "b", "b"]
    lo, hi = A.clip_boot_ci(y, s, g, n_boot=200, seed=0)
    assert lo <= hi


def _load(name):
    p = os.path.join(REPO, "eval", "results", "exp_domain_adapt160", name)
    assert os.path.exists(p), f"missing results {p}"
    return json.load(open(p))


def test_z101_rows_160():
    d = _load("Z1.01_160.json")
    assert d["n_ood"] == 160 and d["n_bad_ood"] == 22
    methods = [r["method"] for r in d["rows"]]
    assert any("FROZEN" in x for x in methods)
    sharp = [r for r in d["rows"] if "sharpness-3" in r["method"]][0]
    assert sharp["ood160_auroc"] > 0.65
    assert sharp["ood160_ci95_factory_grouped"][0] > 0.5


def test_z103_curve_and_self_training():
    d = _load("Z1.03_160.json")
    assert len(d["fewshot_curve"]) >= 3
    assert all(c["k"] > 0 for c in d["fewshot_curve"])
    assert len(d["self_training"]) >= 1
    assert d["recommendation"]["best_zero_shot"]["ood160_auroc"] > 0.6
