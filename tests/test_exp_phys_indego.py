"""Q6 determinism + in-domain integration tests for the shipped physics scorer.

Synthetic arrays only for the unit tests; the integration test runs the
leave-one-clip-out evaluation on the cached IndEgo gold matrices when the bundle
is present (skipped otherwise). No network, no GPU, < 30 s.
Run: OMP_NUM_THREADS=2 python3 -m pytest tests/test_exp_phys_indego.py -q
"""
import json
import os
import sys

import numpy as np
import pytest

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "eval", "exp_phys"))

from tools.phys import model as M  # noqa: E402
import phys_rules as PR  # noqa: E402

RULES = os.path.join(ROOT, "tools", "phys", "rules.json")
BUNDLE = os.environ.get("JADAL_DATA", "/workspace/jadal_pod_bundle")
GOLD = os.path.join(BUNDLE, "caches", "perpixel", "gold.npz")
FEATS = os.path.join(BUNDLE, "caches", "gpu", "features.json")
PHYS = os.path.join(BUNDLE, "caches", "perpixel", "physics_feats.npz")


def _synth_features():
    rng = np.random.default_rng(0)
    rules = M.load_rules(RULES)
    feats = {}
    for r in rules["rules"]:
        feats[r["feature"]] = float(rng.uniform(0, 1))
    return feats, rules


def test_frozen_table_loads_with_threshold_14():
    rules = M.load_rules(RULES)
    assert rules["reject_threshold"] == 14
    assert len(rules["rules"]) == 16
    assert rules["quantisation"] == 4
    assert "full.luma_mean" in rules["undecodable_features"]


def test_receipt_is_byte_identical_over_repeats():
    feats, rules = _synth_features()
    recs = {M.canonical_json(M.receipt("w0", feats, rules)) for _ in range(10)}
    assert len(recs) == 1
    digests = {M.receipt("w0", feats, rules)["feature_digest"] for _ in range(10)}
    assert len(digests) == 1


def test_decision_is_pure_function_of_quantised_integers():
    """A sub-quantisation perturbation must not change the receipt."""
    feats, rules = _synth_features()
    base = M.receipt("w0", feats, rules)
    perturbed = {k: v + 1e-6 for k, v in feats.items()}
    assert M.canonical_json(M.receipt("w0", perturbed, rules)) == \
        M.canonical_json(base)


@pytest.mark.skipif(not (os.path.exists(GOLD) and os.path.exists(FEATS)
                         and os.path.exists(PHYS)),
                    reason="IndEgo bundle not present")
def test_loco_auroc_reproduces_q1_on_cached_features():
    from sklearn.metrics import roc_auc_score
    man = {i["id"]: i for i in
           json.load(open(os.path.join(BUNDLE, "labels", "manifest_gold.json")))["items"]}
    gold = np.load(GOLD, allow_pickle=True)
    W = json.load(open(FEATS))["windows"]
    phys = np.load(PHYS, allow_pickle=True)
    ids = [str(x) for x in gold["ids"]]
    y = np.asarray(gold["y"], dtype=int)
    clips = np.asarray([str(c) for c in gold["clips"]])
    full_names = list(W[ids[0]]["full"].keys())
    phys_names = [str(n) for n in phys["names"]]
    pidx = {str(k): i for i, k in enumerate(phys["ids"])}
    feats = []
    for i in ids:
        d = {f"full.{n}": float(W[i]["full"][n]) for n in full_names}
        d.update({f"phys.{n}": float(phys["X"][pidx[i], j])
                  for j, n in enumerate(phys_names)})
        feats.append(d)
    n = len(y)
    scores = np.zeros(n, dtype=int)
    for c in sorted(set(clips.tolist())):
        te = clips == c
        tr = ~te
        tfeats = {k: np.array([feats[i].get(k, np.nan) for i in range(n)])[tr]
                  for k in {r["feature"] for r in PR.CANDIDATE_RULES}}
        table = PR.build_rule_table(tfeats)
        for i in range(n):
            if te[i]:
                scores[i] = PR.score_window(feats[i], table)["score"]
    a = roc_auc_score(1 - y, scores)
    assert abs(a - 0.8224) < 5e-3, a
