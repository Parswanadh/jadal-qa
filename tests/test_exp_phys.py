"""Tests for the Q2 deterministic rule-list OOD transfer helpers (<30 s, no network)."""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval.exp_phys import run_phys_ood as R  # noqa: E402

RES = R.RES
INPUTS = os.path.join(RES, "inputs")


def test_frozen_rule_table_parses():
    thr, active, v1 = R.load_frozen()
    assert set(v1) == {"lowent", "blur", "ten"}
    assert set(active) == {"blur", "lowent", "madfroz", "pbytes", "ten"}
    assert thr["lowent"]["feature"] == "hist_ent" and thr["lowent"]["op"] == "<"
    assert thr["blur"]["src"] == "th" and thr["blur"]["feature"] == "blur_norm"
    assert thr["ten"]["src"] == "ph" and thr["ten"]["feature"] == "ten_norm"


def test_frozen_thresholds_match_pre_registered_config():
    here = os.path.dirname(os.path.abspath(R.__file__))
    v1 = json.load(open(os.path.join(here, "frozen_perpixel_v1.json")))
    got = {r["name"]: round(r["threshold"], 4) for r in v1["quality_stage"]["rules"]}
    assert got == {"lowent": 6.4861, "blur": 95.0933, "ten": 339.2751}


def test_rule_application_is_deterministic_and_monotone():
    n = 5
    ph = {"hist_ent": np.array([7.0, 6.0, 7.5, 6.2, 8.0]),
          "ten_norm": np.array([400.0, 300.0, 380.0, 500.0, 320.0]),
          "mad_frozen_frac": np.zeros(n), "p_size_med": np.full(n, 3000.0)}
    th = {"blur_norm": np.array([120.0, 80.0, 130.0, 110.0, 90.0])}
    thr, active, v1 = R.load_frozen()
    sb, fired, ev = R.apply_rules([f"w{i}" for i in range(n)], ph, th, thr, v1)
    # lowent fires on 6.0, 6.2 ; blur fires on 80, 90 ; ten fires on 300, 320
    assert fired["lowent"].tolist() == [False, True, False, True, False]
    assert fired["blur"].tolist() == [False, True, False, False, True]
    assert fired["ten"].tolist() == [False, True, False, False, True]
    # OR list: windows 1,3,4 reject (window 2: 7.5>6.49, 130>95, 380>339 -> good)
    assert (sb > 0).tolist() == [False, True, False, True, True]
    assert ev[1] and ev[3] and ev[4] and not ev[0] and not ev[2]


def test_ood_features_present_and_shaped():
    p = os.path.join(INPUTS, "ood_phys.npz")
    if not os.path.exists(p):
        return  # inputs are git-ignored; skip when absent
    z = np.load(p, allow_pickle=True)
    assert z["Xphys"].shape == (160, 16)
    assert z["Xth"].shape == (160, 18)
    assert int(z["y"].sum()) == 138
    assert z["Xphys_indego"].shape == (150, 16)
    assert z["Xth_indego"].shape == (150, 18)
    assert list(z["errors"]).count("") == 160  # no decode errors


def test_result_json_shape_and_verdict():
    p = os.path.join(RES, "Q2.01.json")
    if not os.path.exists(p):
        return  # result is committed, but allow running before generation
    d = json.load(open(p))
    assert d["n"] == 160 and d["n_good"] == 138 and d["n_bad"] == 22
    assert d["n_factories"] == 47
    assert d["conclusion"]["deterministic_rule_list_viable_ood"] is False
    for k in ("perpixel_v1_quality_stage", "active_rules"):
        r = d["rule_sets"][k]
        assert 0.0 <= r["ood_auroc"] <= 1.0
        assert 0.0 <= r["ood_balacc"] <= 1.0
        lo, hi = r["ood_auroc_ci95_factory_grouped"]
        assert lo <= r["ood_auroc"] <= hi
    # below-chance finding: both rule sets < 0.5 AUROC and BA
    assert d["rule_sets"]["perpixel_v1_quality_stage"]["ood_auroc"] < 0.5
    assert d["rule_sets"]["active_rules"]["ood_balacc"] < 0.5
    # baseline reproduces the quoted frozen logistic
    assert abs(d["baseline_frozen_logistic"]["auroc"] - 0.566) < 0.01


def test_group_boot_ci_ordered():
    rng = np.random.RandomState(0)
    y = np.array([1] * 20 + [0] * 5)
    s = np.concatenate([rng.rand(20) * 0.5 + 0.5, rng.rand(5) * 0.5])
    g = np.array([f"f{i % 8}" for i in range(len(y))])
    m, lo, hi = R.group_boot_ci(y, s, g, n_boot=80, seed=0)
    assert lo <= hi
    assert 0.0 <= m <= 1.0
