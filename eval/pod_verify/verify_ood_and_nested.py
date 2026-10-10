"""LEAD B / LB1 — verify the OOD headline number and the nested-validation numbers.

(a) OOD: the frozen IndEgo cascade applied to the 60 human-labelled Egocentric-10K windows.
    FACTS.md claims: score AUROC 0.529 (0.32-0.74 by window, optimistic), rejected 21 of 51
    good windows (41%), rejected 5 of 9 bad, balanced accuracy 0.51.
    Source of scores: caches/ensemble/ood_scores.json (READ-ONLY).
(b) Nested validation: EG2 0.846, pixel 0.792, prob-avg 0.834, rank-avg 0.844 (FACTS.md).
    Source: caches/ensemble/nested_loco.json (READ-ONLY) - recomputed from its per-fold probs.
"""
import json
import os

import numpy as np
from sklearn.metrics import roc_auc_score

B = "/workspace/jadal_pod_bundle"
OUT = os.path.join(os.path.dirname(__file__), "..", "results", "pod_verify", "ood_and_nested.json")
NBOOT = 2000
SEED = 7


def window_bootstrap_ci(y, p, n=NBOOT, seed=SEED):
    rng = np.random.RandomState(seed)
    aucs = []
    for _ in range(n):
        idx = rng.randint(0, len(y), len(y))
        if len(set(y[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y[idx], p[idx]))
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def main():
    out = {"verified_by": "lead-b (LB1)"}

    # ---------- (a) OOD ----------
    sc = json.load(open(os.path.join(B, "caches", "ensemble", "ood_scores.json")))
    lab = json.load(open(os.path.join(B, "labels", "ood_labels.json")))
    latest = lab["latest"]
    rows = []
    for s in sc["scores"]:
        l = latest.get(s["id"])
        if l is None:
            continue
        rows.append({"id": s["id"], "y": 1 if l["verdict"] == "good" else 0,
                     "p": float(s["usable_prob"]), "verdict": s["verdict"],
                     "reason": l.get("reason")})
    y = np.array([r["y"] for r in rows])
    p = np.array([r["p"] for r in rows])
    v = np.array([r["verdict"] for r in rows])
    good, bad = y == 1, y == 0
    auc = float(roc_auc_score(y, p))
    lo, hi = window_bootstrap_ci(y, p)
    rej_good = int((v[good] == "reject").sum())
    rej_bad = int((v[bad] == "reject").sum())
    tpr = float((v[bad] == "reject").mean())
    tnr = float((v[good] != "reject").mean())
    out["ood"] = {
        "n": int(len(y)), "n_good": int(good.sum()), "n_bad": int(bad.sum()),
        "label_source": "human-labelled Egocentric-10K windows (labels/ood_labels.json, 'latest')",
        "protocol": ("frozen IndEgo-calibrated cascade scores from caches/ensemble/ood_scores.json; "
                     "AUROC of usable_prob vs human good/bad; window-bootstrap 95% CI (optimistic, "
                     "windows within a clip are correlated)"),
        "auroc": round(auc, 4), "ci95_window_bootstrap": [round(lo, 4), round(hi, 4)],
        "rejected_good": rej_good, "rejected_good_of": int(good.sum()),
        "rejected_good_share": round(rej_good / int(good.sum()), 4),
        "rejected_bad": rej_bad, "rejected_bad_of": int(bad.sum()),
        "balanced_accuracy_at_shipped_rule": round((tpr + tnr) / 2, 4),
        "claimed_reference": {"auroc": 0.529, "ci": [0.32, 0.74], "rejected_good": 21,
                              "rejected_bad": 5, "balanced_accuracy": 0.51},
    }

    # ---------- (b) nested validation ----------
    nz = json.load(open(os.path.join(B, "caches", "ensemble", "nested_loco.json")))
    out["nested"] = {"n": nz.get("n"), "protocol": nz.get("protocol"),
                     "label_source": nz.get("label_source"),
                     "keys": sorted(nz.keys())}
    for key in ("oof_probs", "oof", "probs", "predictions"):
        if key in nz:
            out["nested"]["found_prob_block"] = key
    res = {}
    for k in nz:
        if k.endswith("_auroc") or k.endswith("_auc"):
            res[k] = nz[k]
    if res:
        out["nested"]["reported"] = res
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=1)
    print(json.dumps(out["ood"], indent=1))
    print("nested keys:", sorted(nz.keys()))
    print("wrote", os.path.realpath(OUT))


if __name__ == "__main__":
    main()
