"""Freeze a deployable artifact: config JSON + small weights .npz + sha256.

The frozen scorer is a logistic head on the SAME feature vector that
`jadal/rhythm/scorer.py::_model_features` computes on the CPU path, so the CPU
reference and the shipped artifact agree by construction. The torch ensemble is
reported separately in docs/reports/rhythm.md and is NOT what this file freezes
(freezing a 7-fold ensemble's members would need the torch graph at inference).

Usage
  python eval/rhythm/freeze.py --out-dir configs --tag v1
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402

from jadal.rhythm import features as F  # noqa: E402

GOLD = Path("/home/parshu/projects/jadal-qa-wt/spectral_data/manifest_gold.json")
CACHE = Path("/home/parshu/projects/jadal-qa-wt/custom_algo_data/cache/main")
G = 6


def build_matrix():
    items = json.load(open(GOLD))["items"]
    rows, y, clip = [], [], []
    for it in items:
        c = F.load_cache(CACHE, it["id"], G)
        if c is None:
            continue
        X, g = c["X"], c["g"]
        T, GG, _, C = X.shape
        feats = []
        for ch in range(C):
            chm = X[:, :, :, ch]
            cell = chm.mean(axis=0).reshape(-1)
            feats += [float(cell.mean()), float(cell.std()), float(cell.max()),
                      float(cell.max() / (cell.sum() + 1e-6))]
        for ch in range(g.shape[1]):
            col = g[:, ch]
            feats += [float(col.mean()), float(col.std()),
                      float(np.percentile(col, 10)), float(np.percentile(col, 90))]
        rows.append(feats)
        y.append(int(it["label"]))
        clip.append(it["clip"])
    return np.nan_to_num(np.asarray(rows, np.float32)), np.array(y), np.array(clip)


def loco_probs(X, y, clip, C=0.3):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    p = np.zeros(len(y))
    for c in np.unique(clip):
        tr, te = clip != c, clip == c
        sc = StandardScaler().fit(X[tr])
        m = LogisticRegression(max_iter=5000, C=C, class_weight="balanced").fit(sc.transform(X[tr]), y[tr])
        p[te] = m.predict_proba(sc.transform(X[te]))[:, 1]
    return p


def auroc(y, p):
    y, p = np.asarray(y), np.asarray(p)
    order = np.argsort(p, kind="mergesort")
    sp = p[order]
    ranks = np.empty(len(p), float)
    i = 0
    while i < len(sp):
        j = i
        while j + 1 < len(sp) and sp[j + 1] == sp[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    npos, nneg = int((y == 1).sum()), int((y == 0).sum())
    return float((ranks[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="configs")
    ap.add_argument("--tag", default="v1")
    args = ap.parse_args()

    X, y, clip = build_matrix()
    # final head is fit on ALL windows; the honest score is the LOCO estimate
    loo = auroc(y, loco_probs(X, y, clip))
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(X)
    m = LogisticRegression(max_iter=5000, C=0.3, class_weight="balanced").fit(sc.transform(X), y)
    w = m.coef_.ravel().astype(np.float32)
    b = float(m.intercept_[0])

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    wpath = out / f"rhythm_{args.tag}.npz"
    np.savez_compressed(wpath, mu=sc.mean_.astype(np.float32), sd=sc.scale_.astype(np.float32),
                        w=w, b=np.float32(b))

    cfg = {
        "pipeline_version": f"rhythm_{args.tag}",
        "G": G,
        "ego_compensate": True,
        "grid_channels": list(F.CHANNEL_NAMES),
        "global_channels": list(F.GLOBAL_NAMES),
        "feature_layout": "per grid channel: [cell_mean, cell_std, cell_max, concentration]; "
                          "per global channel: [mean, std, p10, p90]",
        "n_features": int(X.shape[1]),
        "weights": str(wpath),
        "accept_threshold": 0.65,
        "reject_threshold": 0.35,
        "loco_auroc_estimate": round(loo, 4),
        "loco_protocol": "leave-one-clip-out over 7 clips, logistic C=0.3, class-balanced",
        "label_source": "human gold, 150 windows, 7 clips, one labeller",
        "certified": False,
        "certification_note": ("Not certified. n=150 with clip dependence cannot support an "
                               "error bound. loco_auroc_estimate is an observed value, not a guarantee."),
        "known_failures": {"pallet": "AUROC about 0.5 (chance)",
                           "pallet2": "AUROC about 0.5 (chance)",
                           "ecu_late": "AUROC about 0.63"},
        "tripwires": {"luma_mean_min": 0.10, "under_frac_max": 0.70,
                      "over_frac_max": 0.35, "frozen_frac_max": 0.95},
    }
    cpath = out / f"rhythm_{args.tag}.json"
    cpath.write_text(json.dumps(cfg, indent=1))
    sha = hashlib.sha256(cpath.read_bytes()).hexdigest()
    (out / f"rhythm_{args.tag}.json.sha256").write_text(sha + "  " + cpath.name + "\n")
    wsha = hashlib.sha256(wpath.read_bytes()).hexdigest()
    print(f"wrote {cpath}  sha256 {sha[:16]}...")
    print(f"wrote {wpath}  sha256 {wsha[:16]}...  ({wpath.stat().st_size} bytes)")
    print(f"LOCO AUROC estimate on 150 human windows: {loo:.3f}  (NOT certified)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
