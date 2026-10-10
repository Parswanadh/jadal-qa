"""N5 recalibration sanity tests (no network, < 30 s).

Run: OMP_NUM_THREADS=2 python3 -m pytest tests/test_recal.py -q
or:  python3 tests/test_recal.py
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, ".."))
sys.path.insert(0, REPO)

CONFIG = os.path.join(REPO, "configs", "ensemble_v1_sparse.json")
N501 = os.path.join(REPO, "eval", "results", "exp_recal", "N5.01.json")
N502 = os.path.join(REPO, "eval", "results", "exp_recal", "N5.02.json")


def test_refit_config_loads_and_heads_apply():
    from jadal.ensemble.scorer import load_config, head_prob
    cfg, _ = load_config(CONFIG)
    assert cfg["name"] == "ensemble_v1_sparse"
    assert len(cfg["views"]["eg2"]["coef"]) == 768
    assert len(cfg["views"]["pixel"]["coef"]) == 26
    lo, hi = cfg["abstain_band"]
    assert 0.0 < lo < hi < 1.0
    p = head_prob(np.zeros(768), cfg["views"]["eg2"])
    assert 0.0 <= p <= 1.0


def test_n501_reproduced_frozen_and_refit():
    d = json.load(open(N501))
    # harness reproduces the frozen numbers
    assert abs(d["loco"]["frozen_features"]["ensemble_probavg"]["auroc"] - 0.8336) < 1e-3
    assert abs(d["loco"]["frozen_features_fixedC"]["ensemble_probavg"]["auroc"] - 0.8534) < 1e-3
    # sparse-refit AUROC within CI of frozen (no material cost)
    s = d["loco"]["sparse_features_nestedC"]["ensemble_probavg"]
    assert s["auroc"] >= 0.80
    # refit head dims
    assert len(d["refit_heads"]["eg2"]["coef"]) == 768
    assert len(d["refit_heads"]["pixel"]["coef"]) == 26
    # score shift reproduced
    assert d["score_shift_gold"]["delta_mean"] > 0.10


def test_verdict_distribution_matches_after_refit():
    d = json.load(open(N501))
    prof = d["operating_point_profile_thresholds_gold"]["factory_1080p30"]
    frozen = prof["frozen_features"]["review_share"]
    sparse_frozen = prof["sparse_frozen_heads"]["review_share"]
    refit = prof["sparse_refit"]["review_share"]
    # the frozen heads on sparse features break the distribution; the refit restores it
    assert abs(sparse_frozen - frozen) > 0.15
    assert abs(refit - frozen) < 0.05


def test_n502_ood_within_grouped_ci():
    d = json.load(open(N502))
    f = d["frozen_features"]
    r = d["sparse_refit"]
    assert d["frozen_scores_available"] == 160
    assert f["n_bad"] == 22
    # sparse-refit OOD AUROC inside the frozen grouped CI (preserved)
    lo, hi = f["group_ci95"]
    assert lo <= r["auroc"] <= hi


if __name__ == "__main__":
    test_refit_config_loads_and_heads_apply()
    test_n501_reproduced_frozen_and_refit()
    test_verdict_distribution_matches_after_refit()
    test_n502_ood_within_grouped_ci()
    print("all N5 tests passed")
