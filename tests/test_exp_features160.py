"""Z3 re-run on the 160 OOD labels: unit + result-consistency tests (< 30 s, no network)."""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "eval", "exp_features160"))

from eval.exp_features160 import run_z3_160 as m  # noqa: E402


def test_classify_transfer_rule():
    assert m.classify_transfer(0.60, 0.70) == "stable"
    assert m.classify_transfer(0.60, 0.40) == "flip"
    assert m.classify_transfer(0.51, 0.52) == "chance"
    # informative in one domain only is NOT chance (E5.03 rule)
    assert m.classify_transfer(0.62, 0.55) == "stable"


def test_fast_auc_extremes():
    assert m.fast_auc([1, 1, 0, 0], [0.9, 0.8, 0.2, 0.1]) == 1.0
    assert m.fast_auc([1, 1, 0, 0], [0.1, 0.2, 0.8, 0.9]) == 0.0


def _load(name):
    p = os.path.join(REPO, "eval", "results", "exp_features160", name)
    assert os.path.exists(p), f"missing results {p}"
    return json.load(open(p))


def test_z301_table_160():
    d = _load("Z3.01_160.json")
    assert d["n"] == 160 and d["n_bad"] == 22 and d["n_factories"] == 47
    assert len(d["table"]) == 20
    assert sum(d["counts_160"].values()) == 20
    for r in d["table"]:
        assert len(r["ood160_ci95_factory_grouped"]) == 2
        assert r["class_160"] in {"stable", "flip", "chance"}


def test_z303_subset_and_gain():
    d = _load("Z3.03_160.json")
    assert len(d["transferable_subset_160"]) == 6
    z6 = d["combined"]["zscore_transferable_only"]
    assert z6["ood160_auroc"] > 0.70
    assert z6["ood160_ci95_factory_grouped"][0] > 0.5
    g = d["paired_factory_grouped_gain_vs_frozen"]["zscore_transferable_only"]
    assert g["ci95"][0] > 0.0
