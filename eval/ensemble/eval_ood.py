"""Pre-registered OOD evaluation for frozen ensemble_v1 (works on partial labels).

Label schemas accepted by load_labels_file():
  (a) OOD labelling app: {history, latest: {segment_id: {verdict good|bad, reason, ...}}}
  (b) gold manifest: {items: [{id, label}]}  (c) plain list of {id/segment_id, verdict/label}.
--labels2 merges a second label file (160-window round); labels2 wins on clash.
Confirmatory bar: >=10 labelled windows of each class, else REFUSE (exit 2)
unless --exploratory is passed (numbers then flagged EXPLORATORY, CIs by
window-bootstrap and optimistic since clip=factory/worker). Tunes nothing.
"""
import argparse
import json
import os
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

_HERE = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_SCORES = os.path.normpath(os.path.join(_HERE, "..", "results", "ensemble", "ood_scores.json"))

sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))


def _entry_to_label(wid, entry):
    if isinstance(entry, dict):
        v = str(entry.get("verdict", entry.get("label", ""))).lower()
    else:
        v = str(entry).lower()
    if v in ("good", "1", "1.0", "true"):
        return 1
    if v in ("bad", "0", "0.0", "false"):
        return 0
    return None


def load_labels_file(path):
    """Return {window_id: 0/1}. Handles dict-latest, list, and gold schemas."""
    d = json.load(open(path))
    out = {}
    if isinstance(d, dict) and isinstance(d.get("latest"), dict):
        for wid, entry in d["latest"].items():
            lab = _entry_to_label(wid, entry)
            if lab is not None:
                out[str(wid)] = lab
        return out
    items = None
    if isinstance(d, dict):
        items = d.get("latest", d.get("items", d.get("segments")))
    elif isinstance(d, list):
        items = d
    if isinstance(items, dict):  # {id: entry} without history wrapper
        for wid, entry in items.items():
            lab = _entry_to_label(wid, entry)
            if lab is not None:
                out[str(wid)] = lab
        return out
    for it in (items or []):
        if not isinstance(it, dict):
            continue
        wid = it.get("id", it.get("segment_id"))
        lab = _entry_to_label(wid, it)
        if wid is not None and lab is not None:
            out[str(wid)] = lab
    return out


def evaluate(labels, scored, threshold, boot, tag):
    ids = [i for i in labels if i in scored]
    y = np.array([labels[i] for i in ids])
    p = np.array([scored[i]["usable_prob"] for i in ids])
    n1, n0 = int(y.sum()), int((y == 0).sum())
    print(f"[{tag}] labelled_windows {len(ids)} good={n1} bad={n0}")
    if len(ids) == 0 or len(set(y.tolist())) < 2:
        print(f"[{tag}] REFUSE: need both classes labelled")
        return 2
    auc = float(roc_auc_score(y, p))
    rng = np.random.default_rng(0)
    boots = []
    for _ in range(boot):
        b = rng.choice(len(ids), len(ids), replace=True)
        if len(set(y[b])) < 2:
            continue
        boots.append(roc_auc_score(y[b], p[b]))
    lo, hi = float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
    pred = (p > threshold).astype(int)
    bal = float(((pred[y == 1] == 1).mean() + (pred[y == 0] == 0).mean()) / 2)
    acc = float((pred == y).mean())
    print(f"[{tag}] OOD AUROC {auc:.3f} [{lo:.3f},{hi:.3f}] (window-bootstrap, optimistic: clip=factory/worker)")
    print(f"[{tag}] mean_prob good={p[y == 1].mean():.3f} bad={p[y == 0].mean():.3f}; "
          f"good_below_thr {(p[y == 1] <= threshold).sum()}/{n1}")
    print(f"[{tag}] balacc@{threshold} {bal:.3f} acc {acc:.3f}")
    print(f"[{tag}] accept {int(pred.sum())} / reject-or-review {int((pred == 0).sum())}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--labels2", default=None)
    ap.add_argument("--scores", default=_DEFAULT_SCORES)
    ap.add_argument("--scores2", default=None)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--exploratory", action="store_true",
                    help="proceed even if a class has <10 labels; flags output EXPLORATORY")
    args = ap.parse_args()
    from jadal.ensemble.scorer import load_config
    cfg, used = load_config()
    import hashlib
    frozen_sha = hashlib.sha256(open(used, "rb").read()).hexdigest()
    print(f"frozen_sha {frozen_sha}")
    for path in [args.scores] + ([args.scores2] if args.scores2 else []):
        if not os.path.exists(path):
            print(f"scores cache not found: {path} (run eval/ensemble_score_windows.py first)")
            sys.exit(1)
    if not os.path.exists(args.labels):
        print(f"labels not found: {args.labels} (human labelling in progress?)")
        sys.exit(1)
    scored = {s["id"]: s for s in json.load(open(args.scores))["scores"] if "error" not in s}
    if args.scores2:
        scored.update({s["id"]: s for s in json.load(open(args.scores2))["scores"] if "error" not in s})
    labels = load_labels_file(args.labels)
    tag = "confirmatory"
    if args.labels2:
        if not os.path.exists(args.labels2):
            print(f"labels2 not found: {args.labels2}")
            sys.exit(1)
        labels.update(load_labels_file(args.labels2))
        tag = "merged160"
    n1 = sum(labels.values())
    n0 = len(labels) - n1
    if (n1 < 10 or n0 < 10) and not args.exploratory:
        print(f"[{tag}] REFUSE: need >=10 labelled windows of each class "
              f"(got good={n1} bad={n0}); re-run with --exploratory to record exploratory numbers")
        sys.exit(2)
    if (n1 < 10 or n0 < 10):
        tag += "+EXPLORATORY"
        print(f"[{tag}] WARNING: pre-registered minimum (10/class) not met — numbers are exploratory")
    rc = evaluate(labels, scored, cfg["threshold"], args.boot, tag)
    print("NOTE: no tuning after freezing; frozen threshold applied as-is.")
    sys.exit(rc)


if __name__ == "__main__":
    main()
