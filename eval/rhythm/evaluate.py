"""Evaluate a trained-fold result: pooled LOCO AUROC + clip-bootstrap CI, per-clip
table, balanced accuracy at a threshold chosen on TRAINING clips only, and the
coverage/accuracy curve.

Also computes, under the SAME folds, the required baselines:
  (a) logistic on the 26 GPU per-pixel features   (the handover's 0.81)
  (b) single feature flow_act-like scalar from this representation
  (c) ablation (d): GLOBAL channels only from this representation
  (e) the frozen 1D-conv on the 8 global series   (approximated by (c) with a
      logistic head -- labelled as such, NOT the frozen model)

Usage
  python eval/rhythm/evaluate.py --folds eval/results/rhythm/loco_main.json \
      --out eval/results/rhythm/eval_main.json
  python eval/rhythm/evaluate.py --baselines --out eval/results/rhythm/baselines.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402

GOLD = Path("/home/parshu/projects/jadal-qa-wt/spectral_data/manifest_gold.json")
GPU_FEATS = Path("/home/parshu/projects/jadal-qa-wt/gpu-pipeline/eval/results/gpu/features.json")
CACHE = Path("/home/parshu/projects/jadal-qa-wt/custom_algo_data/cache/main")
RNG = np.random.default_rng(20261010)


def auroc(y, p):
    y = np.asarray(y)
    p = np.asarray(p)
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
    npos = int((y == 1).sum())
    nneg = int((y == 0).sum())
    return float((ranks[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def cluster_bootstrap_ci(y, p, clip, n_boot: int = 2000, seed: int = 20261010):
    """95% CI by resampling CLIPS with replacement (windows move with their clip)."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    p = np.asarray(p)
    clip = np.asarray(clip)
    clips = np.unique(clip)
    idx_by_clip = {c: np.where(clip == c)[0] for c in clips}
    stats = []
    for _ in range(n_boot):
        pick = rng.choice(len(clips), size=len(clips), replace=True)
        sel = np.concatenate([idx_by_clip[clips[i]] for i in pick])
        if len(np.unique(y[sel])) < 2:
            continue
        stats.append(auroc(y[sel], p[sel]))
    if not stats:
        return {"lo": float("nan"), "hi": float("nan"), "n_boot": 0}
    return {"lo": float(np.percentile(stats, 2.5)), "hi": float(np.percentile(stats, 97.5)),
            "n_boot": len(stats), "median": float(np.median(stats))}


def balanced_accuracy(y, pred):
    y = np.asarray(y)
    pred = np.asarray(pred)
    accs = []
    for v in (1, 0):
        m = y == v
        if m.sum():
            accs.append(float((pred[m] == v).mean()))
    return float(np.mean(accs)) if accs else float("nan")


def risk_coverage(y, p, clip=None):
    """Coverage vs balanced accuracy when we accept the most confident fraction."""
    y = np.asarray(y)
    p = np.asarray(p)
    conf = np.abs(p - 0.5)
    order = np.argsort(-conf)
    out = []
    for frac in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
        k = max(int(round(frac * len(y))), 2)
        sel = order[:k]
        pred = (p[sel] >= 0.5).astype(int)
        out.append({"coverage": round(k / len(y), 3), "balanced_accuracy":
                    round(balanced_accuracy(y[sel], pred), 4), "n": int(k)})
    return out


# --------------------------------------------------------------------------- #
# baselines under identical folds
# --------------------------------------------------------------------------- #
def _loco_logistic(X, y, clip, C=0.3):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    p = np.zeros(len(y))
    for c in np.unique(clip):
        tr = clip != c
        te = clip == c
        sc = StandardScaler().fit(X[tr])
        m = LogisticRegression(max_iter=3000, C=C, class_weight="balanced").fit(sc.transform(X[tr]), y[tr])
        p[te] = m.predict_proba(sc.transform(X[te]))[:, 1]
    return p


def baseline_gpu26():
    F = json.load(open(GPU_FEATS))["windows"]
    M = {i["id"]: i for i in json.load(open(GOLD))["items"]}
    ids = [k for k in M if k in F and F[k].get("full")]
    names = sorted(F[ids[0]]["full"].keys())
    X = np.nan_to_num(np.array([[F[k]["full"][n] for n in names] for k in ids], float))
    y = np.array([M[k]["label"] for k in ids])
    clip = np.array([M[k]["clip"] for k in ids])
    p = _loco_logistic(X, y, clip)
    return y, p, clip, names


def load_cache_matrix(G=6, tag="main"):
    """Global channels (g) summarised per window, for the ablation (d) baseline."""
    from jadal.rhythm import features as F
    cache = CACHE.parent / tag
    items = json.load(open(GOLD))["items"]
    rows, y, clip, ids = [], [], [], []
    for it in items:
        c = F.load_cache(cache, it["id"], G)
        if c is None:
            continue
        g = c["g"]
        rows.append(np.concatenate([g.mean(0), g.std(0), np.percentile(g, 10, axis=0),
                                    np.percentile(g, 90, axis=0)]))
        y.append(int(it["label"]))
        clip.append(it["clip"])
        ids.append(it["id"])
    return np.nan_to_num(np.array(rows, float)), np.array(y), np.array(clip), ids


def baseline_flow_act(G=6, tag="main"):
    """flow_act-like: mean and median of the flow_mag channel over the most active cells."""
    from jadal.rhythm import features as F
    cache = CACHE.parent / tag
    items = json.load(open(GOLD))["items"]
    feats, y, clip, ids = [], [], [], []
    for it in items:
        c = F.load_cache(cache, it["id"], G)
        if c is None:
            continue
        X = c["X"]                      # (T,G,G,C)
        fm = X[..., 4]                  # flow_mag, (T,G,G)
        # per-cell temporal mean, then flatten the GRID to a cell vector
        cell = fm.mean(axis=0).reshape(-1)
        k = max(1, int(round(0.25 * len(cell))))
        top = np.argsort(-cell)[:k]
        fm_cells = fm.reshape(fm.shape[0], -1)
        feats.append([float(fm_cells[:, top].mean()), float(np.median(fm_cells[:, top])),
                      float(fm.mean()), float(cell.max())])
        y.append(int(it["label"]))
        clip.append(it["clip"])
        ids.append(it["id"])
    return np.nan_to_num(np.array(feats, float)), np.array(y), np.array(clip), ids


def baseline_pooled_control(G=6, tag="main"):
    """DECISIVE CONTROL: same grid channels with the spatial axis summed away.

    Per grid channel, the mean and std over (time x all cells). This uses exactly
    the same channels as the spatial summaries but has no layout information, so
    spatial-vs-pooled is a like-for-like comparison. Motivated by the
    `ablations` track, which found sp_cellmax - poolmean_only = -0.0018.
    """
    from jadal.rhythm import features as F
    cache = CACHE.parent / tag
    items = json.load(open(GOLD))["items"]
    rows, y, clip, ids = [], [], [], []
    for it in items:
        c = F.load_cache(cache, it["id"], G)
        if c is None:
            continue
        X = c["X"]
        flat = X.reshape(X.shape[0], -1, X.shape[-1])   # (T, cells, C)
        feats = []
        for ch in range(flat.shape[-1]):
            v = flat[:, :, ch]
            feats += [float(v.mean()), float(v.std())]
        rows.append(feats)
        y.append(int(it["label"]))
        clip.append(it["clip"])
        ids.append(it["id"])
    return np.nan_to_num(np.asarray(rows, float)), np.array(y), np.array(clip), ids


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", default=None)
    ap.add_argument("--baselines", action="store_true")
    ap.add_argument("--tag", default="main")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = {"label_source": "human gold (150 windows, 7 clips, one labeller)",
           "protocol": "leave-one-clip-out, pooled; 95% CI = clip bootstrap, 2000 resamples",
           "command": " ".join(sys.argv), "tag": args.tag, "results": {}}

    if args.baselines:
        y, p, clip, names = baseline_gpu26()
        out["results"]["gpu26_logistic"] = {
            "auroc": auroc(y, p), "ci": cluster_bootstrap_ci(y, p, clip), "n": int(len(y)),
            "n_features": len(names),
            "per_clip": {c: auroc(y[clip == c], p[clip == c]) for c in sorted(set(clip))},
            "note": "26 GPU per-pixel features + logistic C=0.3; the handover's 0.81 baseline",
        }
        print("gpu26_logistic", round(out["results"]["gpu26_logistic"]["auroc"], 3), flush=True)

        Xg, y2, clip2, ids2 = load_cache_matrix(tag=args.tag)
        p2 = _loco_logistic(Xg, y2, clip2)
        out["results"]["global_channels_only"] = {
            "auroc": auroc(y2, p2), "ci": cluster_bootstrap_ci(y2, p2, clip2), "n": int(len(y2)),
            "n_features": Xg.shape[1],
            "per_clip": {c: auroc(y2[clip2 == c], p2[clip2 == c]) for c in sorted(set(clip2))},
            "note": "ablation (d): this representation's 8 GLOBAL channels only, "
                    "summarised by mean/std/p10/p90 -> logistic. No spatial grid.",
        }
        print("global_channels_only", round(out["results"]["global_channels_only"]["auroc"], 3), flush=True)

        Xf, y3, clip3, ids3 = baseline_flow_act(tag=args.tag)
        pf = _loco_logistic(Xf, y3, clip3)
        out["results"]["flow_act_like"] = {
            "auroc": auroc(y3, pf), "ci": cluster_bootstrap_ci(y3, pf, clip3), "n": int(len(y3)),
            "n_features": Xf.shape[1],
            "per_clip": {c: auroc(y3[clip3 == c], pf[clip3 == c]) for c in sorted(set(clip3))},
            "note": "residual flow magnitude in the top-25% most active grid cells, "
                    "mean/median + window mean + max cell. The handover's strongest single "
                    "scalar family (flow_act 0.806 there) -- different representation, "
                    "so not directly comparable.",
        }
        print("flow_act_like", round(out["results"]["flow_act_like"]["auroc"], 3), flush=True)

        # single-feature AUROCs, orientation-free
        single = {}
        for j, nm in enumerate(["flow_top_mean", "flow_top_median", "flow_mean", "flow_cell_max"]):
            a = auroc(y3, Xf[:, j])
            single[nm] = {"auroc": a, "orientation_free": max(a, 1 - a)}
        out["results"]["flow_single_features"] = single

        Xp, y4, clip4, ids4 = baseline_pooled_control(tag=args.tag)
        pp = _loco_logistic(Xp, y4, clip4)
        out["results"]["pooled_control_no_layout"] = {
            "auroc": auroc(y4, pp), "ci": cluster_bootstrap_ci(y4, pp, clip4), "n": int(len(y4)),
            "n_features": Xp.shape[1],
            "per_clip": {c: auroc(y4[clip4 == c], pp[clip4 == c]) for c in sorted(set(clip4))},
            "note": "DECISIVE CONTROL: identical grid channels with the spatial axis summed "
                    "away (mean and std over time x cells per channel). No layout information.",
        }
        print("pooled_control_no_layout", round(out["results"]["pooled_control_no_layout"]["auroc"], 3), flush=True)

    if args.folds:
        d = json.load(open(args.folds))
        folds = d["folds"]
        # Report the headline number on the 150 REAL human-labelled windows only.
        # Synthetic negatives are true-by-construction and are training material;
        # including them in the test set would inflate the score, so they are
        # excluded here and reported separately.
        def is_synth(f):
            return np.array([r == "SYNTHETIC" for r in f["reason"]])
        real_mask = np.concatenate([~is_synth(f) for f in folds])
        y_all = np.concatenate([f["y"] for f in folds])
        p_all = np.concatenate([f["pred"] for f in folds])
        clip_all = np.concatenate([[f["held_out_clip"]] * len(f["y"]) for f in folds])
        reason_all = np.concatenate([f["reason"] for f in folds])

        def block(y, p, clip, reason, label):
            r = {
                "auroc": auroc(y, p), "ci": cluster_bootstrap_ci(y, p, clip), "n": int(len(y)),
                "n_good": int((y == 1).sum()), "n_bad": int((y == 0).sum()),
                "per_clip": {c: {"auroc": auroc(y[clip == c], p[clip == c]),
                                 "n": int((clip == c).sum()),
                                 "n_bad": int((y[clip == c] == 0).sum())} for c in sorted(set(clip))},
                "risk_coverage": risk_coverage(y, p),
                "balanced_accuracy_at_0.5": balanced_accuracy(y, (p >= 0.5).astype(int)),
                "label_source": label,
            }
            rec = {}
            reason = np.array([("" if r is None else str(r)) for r in reason])
            for rr in sorted(set(reason)):
                m = (reason == rr) | (y == 1)
                if rr and (reason == rr).sum() >= 3:
                    rec[rr] = {"n": int((reason == rr).sum()),
                               "auroc_good_vs_this": auroc(y[m], p[m]),
                               "recall_at_0.5": float((p[reason == rr] < 0.5).mean())}
            r["per_reason"] = rec
            return r

        res = block(y_all[real_mask], p_all[real_mask], clip_all[real_mask],
                    reason_all[real_mask], "human gold, 150 windows, 7 clips, one labeller")
        res["config"] = d.get("configs", [d.get("config")])[0]
        res["G"] = d.get("G")
        res["top_m"] = d.get("top_m")
        res["params"] = d.get("params")
        res["synthetic_note"] = ("Synthetic-by-construction negatives are EXCLUDED from this "
                                 "headline number and reported under with_synthetic.")
        res["with_synthetic"] = {
            "auroc": auroc(y_all, p_all),
            "ci": cluster_bootstrap_ci(y_all, p_all, clip_all),
            "n": int(len(y_all)),
            "note": "inflated by construction; not an accuracy estimate",
        }
        out["results"]["rhythm_ensemble"] = res
        print("rhythm_ensemble (150 real) AUROC", round(res["auroc"], 3), res["ci"], flush=True)
        print("  per-clip:", {k: round(v["auroc"], 3) for k, v in res["per_clip"].items()}, flush=True)
        print("  with synthetic:", round(res["with_synthetic"]["auroc"], 3), flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=1)
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
