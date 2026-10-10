"""Sign-robust magnitude pooling, motivated by the rhythm_analysis track.

The finding this responds to (custom_algo_data/research/rhythm_analysis/SUMMARY.md):

  * The residual-motion magnitude cue points the OPPOSITE way to the premise.
    Signed AUROC of residual-flow level is ~0.20: GOOD windows have roughly 3x
    LESS residual flow than bad ones, because ego-motion compensation here is
    translation-only, so walking/turning breaks the warp and raises residual
    everywhere. The baseline is therefore closer to a "the wearer is walking /
    hands are not dominant" detector than a "work is happening" detector.
  * `pallet` is a SIGN INVERSION, not missing signal: within pallet the magnitude
    scalar scores 0.303, i.e. good windows there have HIGHER flow than bad, the
    reverse of the other six clips. Orientation-free it reaches 0.697 within
    pallet, and pallet2 behaves like the majority (0.750).

If pallet really is a per-clip sign inversion, then a sign-robust pooling rule
should recover it WITHOUT fitting anything on the held-out clip. This script
tests that, using only label-free per-clip statistics (median / MAD of that
clip's own feature distribution), which is legal under LOCO.

Rules compared, all evaluated under LOCO + clip bootstrap:
  raw          the feature as-is, logistic on 6 training clips
  signed       negated feature (tests the direction the agent reports)
  abs_z        |z| within clip, z from the clip's OWN median/MAD (label-free)
  dist_med     |x - clip_median| (label-free)
  two_sided    fit logistic on [x, |z|] so the model can use both directions

Usage
  python eval/rhythm/sign_robust.py --out eval/results/rhythm/sign_robust.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402

from jadal.rhythm import features as F  # noqa: E402

GOLD = Path("/home/parshu/projects/jadal-qa-wt/spectral_data/manifest_gold.json")
CACHE = Path("/home/parshu/projects/jadal-qa-wt/custom_algo_data/cache/main")
G = 6


def auroc(y, p):
    y, p = np.asarray(y), np.asarray(p)
    if len(np.unique(y)) < 2:
        return float("nan")
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


def clip_bootstrap(y, p, clip, n_boot=2000, seed=20261010):
    rng = np.random.default_rng(seed)
    clips = np.unique(clip)
    idx = {c: np.where(clip == c)[0] for c in clips}
    out = []
    for _ in range(n_boot):
        pick = rng.choice(len(clips), size=len(clips), replace=True)
        sel = np.concatenate([idx[clips[i]] for i in pick])
        if len(np.unique(y[sel])) < 2:
            continue
        out.append(auroc(y[sel], p[sel]))
    return {"lo": float(np.percentile(out, 2.5)), "hi": float(np.percentile(out, 97.5)),
            "n_boot": len(out)}


def loco(X, y, clip, C=0.3):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    p = np.zeros(len(y))
    for c in np.unique(clip):
        tr, te = clip != c, clip == c
        sc = StandardScaler().fit(X[tr])
        m = LogisticRegression(max_iter=5000, C=C, class_weight="balanced").fit(sc.transform(X[tr]), y[tr])
        p[te] = m.predict_proba(sc.transform(X[te]))[:, 1]
    return p


def build():
    items = json.load(open(GOLD))["items"]
    rows, y, clip, ids = [], [], [], []
    for it in items:
        c = F.load_cache(CACHE, it["id"], G)
        if c is None:
            continue
        X = c["X"]
        fm = X[..., 4]                       # flow_mag (T,G,G)
        fc = X[..., 5]                       # flow_coh
        cells = fm.mean(axis=0).reshape(-1)
        k = max(1, int(round(0.25 * len(cells))))
        top = np.argsort(-cells)[:k]
        fmc = fm.reshape(fm.shape[0], -1)
        # a small, fixed feature set: magnitude level + coherence level + localisation
        rows.append([float(fmc[:, top].mean()), float(fmc[:, top].std()),
                     float(fm.mean()), float(cells.max()),
                     float(fc.mean()), float(fc.std()),
                     float(cells.max() / (cells.sum() + 1e-6))])
        y.append(int(it["label"]))
        clip.append(it["clip"])
        ids.append(it["id"])
    return np.nan_to_num(np.asarray(rows, np.float32)), np.array(y), np.array(clip), ids


def per_clip_transform(X, clip, mode):
    """Label-free per-clip transforms. Uses only that clip's own distribution."""
    Z = X.copy()
    for c in np.unique(clip):
        m = clip == c
        med = np.median(X[m], axis=0)
        mad = np.median(np.abs(X[m] - med), axis=0) * 1.4826 + 1e-9
        z = (X[m] - med) / mad
        if mode == "abs_z":
            Z[m] = np.abs(z)
        elif mode == "dist_med":
            Z[m] = np.abs(X[m] - med)
        elif mode == "signed":
            Z[m] = -X[m]
    return Z


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    X, y, clip, ids = build()
    res = {"label_source": "human gold, 150 windows, 7 clips, one labeller",
           "protocol": "LOCO over 7 clips + clip bootstrap 95% CI, 2000 resamples",
           "command": " ".join(sys.argv), "G": G,
           "features": ["flow_top_mean", "flow_top_std", "flow_mean", "flow_cell_max",
                        "coh_mean", "coh_std", "concentration"],
           "note": ("Per-clip transforms use ONLY that clip's own median/MAD (label-free), "
                    "which is legal under LOCO. They are not fitted on labels."),
           "runs": {}}

    variants = {
        "raw": X,
        "signed": per_clip_transform(X, clip, "signed"),
        "abs_z_within_clip": per_clip_transform(X, clip, "abs_z"),
        "dist_from_clip_median": per_clip_transform(X, clip, "dist_med"),
        "two_sided_raw_plus_absz": np.hstack([X, per_clip_transform(X, clip, "abs_z")]),
    }
    for name, Xv in variants.items():
        p = loco(Xv, y, clip)
        per = {c: round(auroc(y[clip == c], p[clip == c]), 3) for c in sorted(set(clip))}
        res["runs"][name] = {
            "auroc": round(auroc(y, p), 4),
            "ci": clip_bootstrap(y, p, clip),
            "per_clip": per,
            "n_features": int(Xv.shape[1]),
        }
        print(f"{name:24s} AUROC {res['runs'][name]['auroc']:.3f} "
              f"pallet {per['pallet']:.3f} pallet2 {per['pallet2']:.3f}", flush=True)

    # single-feature signed AUROCs (raw scalars, no model) -- the agent's key claim
    single = {}
    for j, nm in enumerate(res["features"]):
        a = auroc(y, X[:, j])
        single[nm] = {"auroc": round(a, 4), "orientation_free": round(max(a, 1 - a), 4),
                      "direction": "good>bad" if a > 0.5 else "good<bad"}
    res["single_feature_raw"] = single
    print("\nsingle raw scalars:")
    for k, v in single.items():
        print(f"  {k:20s} AUROC {v['auroc']:.3f}  orfree {v['orientation_free']:.3f}  {v['direction']}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=1)
    print("\nwrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
