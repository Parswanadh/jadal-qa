"""Pre-registered OOD evaluation for frozen ensemble_v1 (works on partial labels).

Reads human labels at --labels (same {history, latest} schema as gold labels;
verdict good|bad per window id). Refuses with < 10 labelled windows of each
class. Prints AUROC + window-bootstrap 95% CI (clip=factory/worker here, so
CIs are BY WINDOW and optimistic), balanced accuracy at the frozen 0.5
threshold, and the reject/accept split. Tunes nothing.
"""
import argparse
import json
import os
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_labels(path):
    d = json.load(open(path))
    items = d.get("latest", d if isinstance(d, dict) else d)
    if isinstance(items, dict):
        items = items.get("items", items.get("segments", []))
    out = {}
    for it in items:
        wid = it.get("id", it.get("segment_id"))
        v = str(it.get("verdict", it.get("label", ""))).lower()
        if wid is None or v not in ("good", "bad", "1", "0"):
            continue
        out[wid] = 1 if v in ("good", "1") else 0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--scores", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "results", "ensemble", "ood_scores.json"))
    ap.add_argument("--boot", type=int, default=2000)
    args = ap.parse_args()
    from jadal.ensemble.scorer import load_config
    cfg, used = load_config()
    import hashlib
    frozen_sha = hashlib.sha256(open(used, "rb").read()).hexdigest()
    scored = {s["id"]: s for s in json.load(open(args.scores))["scores"] if "error" not in s}
    labels = load_labels(args.labels)
    ids = [i for i in labels if i in scored]
    y = np.array([labels[i] for i in ids])
    p = np.array([scored[i]["usable_prob"] for i in ids])
    n1, n0 = int(y.sum()), int((y == 0).sum())
    print(f"frozen_sha {frozen_sha}\nlabelled_windows {len(ids)} good={n1} bad={n0}")
    if n1 < 10 or n0 < 10:
        print("REFUSE: need >=10 labelled windows of each class")
        sys.exit(2)
    auc = float(roc_auc_score(y, p))
    rng = np.random.default_rng(0)
    boots = []
    for _ in range(args.boot):
        b = rng.choice(len(ids), len(ids), replace=True)
        if len(set(y[b])) < 2:
            continue
        boots.append(roc_auc_score(y[b], p[b]))
    lo, hi = float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
    pred = (p > cfg["threshold"]).astype(int)
    bal = float(((pred[y == 1] == 1).mean() + (pred[y == 0] == 0).mean()) / 2)
    acc = float((pred == y).mean())
    print(f"OOD AUROC {auc:.3f} [{lo:.3f},{hi:.3f}] (window-bootstrap, optimistic: clip=factory/worker)")
    print(f"balacc@{cfg['threshold']} {bal:.3f} acc {acc:.3f}")
    print(f"accept {int(pred.sum())} / reject-or-review {int((pred == 0).sum())}")
    print("NOTE: no tuning after freezing; frozen threshold applied as-is.")


if __name__ == "__main__":
    main()
