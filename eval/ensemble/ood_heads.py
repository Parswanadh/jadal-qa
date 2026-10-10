"""Per-head out-of-domain AUROCs for the frozen ensemble_v1 (EXPLORATORY, n=60).

Reads the cached frozen scores (eval/results/ensemble/ood_scores.json: p_eg2, p_pixel,
usable_prob per window) and the human OOD labels, and writes
eval/results/ensemble/ood_heads.json. Nothing is tuned or refit; the heads are the
frozen ones. Higher score = predicted good (usable).

Labels: env JADAL_OOD_LABELS, default ../ood_labels/labels.json relative to the repo
root (i.e. the sibling jadal-qa-wt/ood_labels/labels.json). Schema:
{history: [...], latest: {window_id: {verdict: good|bad, ...}}} (same loader as
eval/ensemble/eval_ood.py: load_labels_file).

  OMP_NUM_THREADS=2 PYTHONNOUSERSITE=1 /home/parshu/envs/jadal/bin/python \
      eval/ensemble/ood_heads.py [--boot 2000] [--out eval/results/ensemble/ood_heads.json]

Window-bootstrap 95% CIs (seed 0): optimistic, since clips are factory/worker-correlated.
Also reported (in the note field): the AUROC with the sign flipped, for the inverted heads.
"""
import argparse
import json
import os
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _HERE)
from eval_ood import load_labels_file  # noqa: E402

DEFAULT_LABELS = os.path.normpath(os.path.join(_REPO, "..", "ood_labels", "labels.json"))
SCORES = os.path.join(_REPO, "eval", "results", "ensemble", "ood_scores.json")
OUT = os.path.join(_REPO, "eval", "results", "ensemble", "ood_heads.json")
HEADS = ("p_eg2", "p_pixel", "usable_prob")


def head_auroc(y, p, boot, seed=0):
    auc = float(roc_auc_score(y, p))
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(boot):
        b = rng.choice(len(y), len(y), replace=True)
        if len(set(y[b].tolist())) < 2:
            continue
        boots.append(roc_auc_score(y[b], p[b]))
    lo, hi = float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
    return {"auroc": round(auc, 3), "ci95": [round(lo, 3), round(hi, 3)],
            "auroc_if_sign_flipped": round(1.0 - auc, 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default=os.environ.get("JADAL_OOD_LABELS", DEFAULT_LABELS))
    ap.add_argument("--scores", default=SCORES)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    if not os.path.exists(args.labels):
        print(f"labels not found: {args.labels} (set JADAL_OOD_LABELS)")
        sys.exit(1)
    scored = {s["id"]: s for s in json.load(open(args.scores))["scores"] if "error" not in s}
    labels = load_labels_file(args.labels)
    ids = sorted(i for i in labels if i in scored)
    dropped = len(labels) - len(ids)
    y = np.array([labels[i] for i in ids])
    if len(set(y.tolist())) < 2:
        print("REFUSE: need both classes labelled")
        sys.exit(2)
    heads = {h: head_auroc(y, np.array([scored[i][h] for i in ids]), args.boot) for h in HEADS}
    out = {
        "status": "EXPLORATORY (n_bad < 10 pre-registered minimum; not confirmatory)",
        "scores": os.path.relpath(args.scores, _REPO),
        "labels_source": "env JADAL_OOD_LABELS (default ../ood_labels/labels.json)",
        "n": len(ids), "n_good": int(y.sum()), "n_bad": int((y == 0).sum()),
        "labels_dropped_no_score": dropped,
        "bootstrap": {"method": "window-bootstrap", "n": args.boot, "seed": 0},
        "heads": heads,
        "note": ("Frozen heads, no refit. p_pixel is inverted (AUROC < 0.5) on this set. "
                 "Higher score = predicted good."),
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
        f.write("\n")
    for h in HEADS:
        r = heads[h]
        print(f"{h}: AUROC {r['auroc']:.3f} [{r['ci95'][0]:.3f}, {r['ci95'][1]:.3f}] "
              f"(sign-flipped {r['auroc_if_sign_flipped']:.3f})")
    print(f"n={len(ids)} good={int(y.sum())} bad={int((y == 0).sum())} dropped={dropped} -> {args.out}")


if __name__ == "__main__":
    main()
