#!/usr/bin/env python3
"""Y3 -- Review-score separability via cheap cached-view ensembling.

Goal (Y3, pod jadal-qa, claim Y3): the frozen two-view PoC (pixel-26 + EG2@2fps)
leaves 75.5% of the 1,546 unlabelled Egocentric-10K windows inside the [0.4,0.6]
abstain band.  Question: do *additional cached views* (or a rank fusion instead
of a probability average) make the OOF score more separable -- i.e. shrink the
"ambiguous fraction" in the fixed [0.4,0.6] band -- at equal or better AUROC?

Honesty / protocol
------------------
* In-domain only  : 150-window IndEgo human gold (100 good / 50 bad), single
  labeller, label_source "human".  Leave-one-clip-out (LOCO) over 7 clips.
  Every per-window score is out-of-fold; the per-view regularisation C is
  chosen by an *inner* LOCO on the 6 training clips only (nested), so nothing
  is fit on the held-out clip.
* We separately reproduce the frozen-config anchor (fixed C): eg2 C=0.01,
  pixel C=0.3, equal-weight probability average -> AUROC 0.8534
  (caches/ensemble/ensemble_v1.json:full_oof_auroc_resubstitution_free).
* Out-of-domain    : the 60 human-labelled Egocentric-10K windows
  (51 good / 9 bad, caches/ensemble/ood_scores.json + labels/ood_labels.json).
  EVALUATION ONLY -- never fit on.  n_bad=9 => CIs are wide; descriptive.
* Ambiguous fraction = share of scores inside the fixed band [0.4,0.6] (the
  deployed REVIEW rule band).  Reported raw (deployed probability semantics)
  and after a LOCO Platt calibration (`amb_cal`) so that probability-average and
  rank-average fusions are on the same probability scale (fair comparison).
* Zhang et al 2009 median-stay criterion: the total-stay statistic |FAR-FRR| is
  a diagonal ROC chord, so it is *invariant* to any monotone fusion.  We report
  it at current operating points only to separate "more informative views"
  (AUROC up) from "re-calibration" (band fraction down at equal AUROC).
* No EG2@1fps / @0.5fps embeddings exist under caches/ (segments.npz is the only
  EG2 view and is 21 frames @ 0.5 s spacing = 2 fps).  Absent -> skipped.

Outputs: eval/results/exp_review/Y3.01.json, Y3.02.json
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
BUNDLE = os.environ.get("JADAL_BUNDLE", "/workspace/jadal_pod_bundle")
OUTDIR = os.path.join(ROOT, "eval", "results", "exp_review")
SEED = 7
CGRID = [0.003, 0.01, 0.03, 0.1, 0.3, 1.0]
BAND = (0.4, 0.6)

os.makedirs(OUTDIR, exist_ok=True)
rng = np.random.default_rng(SEED)


# ----------------------------------------------------------------------- data
def load_gold():
    M = {i["id"]: i for i in json.load(open(os.path.join(BUNDLE, "labels", "manifest_gold.json")))["items"]}
    z = np.load(os.path.join(BUNDLE, "caches", "frames10k", "segments.npz"), allow_pickle=True)
    ids = [str(i) for i in z["ids"]]
    y = np.array([M[i]["label"] for i in ids], int)
    clip = np.array([M[i]["clip"] for i in ids])
    reason = np.array([M[i].get("reason") for i in ids], dtype=object)

    F = json.load(open(os.path.join(BUNDLE, "caches", "gpu", "features.json")))["windows"]
    gnames = sorted(F[ids[0]]["full"].keys())
    Xpx = np.nan_to_num(np.array([[F[i]["full"][n] for n in gnames] for i in ids], float))

    # EG2 @2fps: mean over the available frames, then L2-normalised (as deployed)
    E = z["F"].astype(np.float32)
    cnt = z["counts"].astype(int)
    Xeg = np.stack([E[k, :max(1, cnt[k])].mean(0) for k in range(len(ids))])
    Xeg = Xeg / np.linalg.norm(Xeg, axis=1, keepdims=True)
    return ids, y, clip, reason, {"pixel26": Xpx, "eg2": Xeg}


def _order_ids(npz, key="ids"):
    return [str(i) for i in npz[key]]


def load_extra_views(ids):
    """Return dict name -> (X aligned to ids)."""
    out = {}
    idx = {i: k for k, i in enumerate(ids)}

    g = np.load(os.path.join(BUNDLE, "caches", "perpixel", "gold.npz"), allow_pickle=True)
    gmap = {str(i): k for k, i in enumerate(g["ids"])}
    if set(ids).issubset(gmap):
        out["perpixel121"] = np.nan_to_num(np.array([g["X"][gmap[i]] for i in ids], float))

    p = np.load(os.path.join(BUNDLE, "caches", "perpixel", "physics_feats.npz"), allow_pickle=True)
    pmap = {str(i): k for k, i in enumerate(p["ids"])}
    if set(ids).issubset(pmap):
        out["physics16"] = np.nan_to_num(np.array([p["X"][pmap[i]] for i in ids], float))

    t = np.load(os.path.join(BUNDLE, "caches", "perpixel", "theory_feats.npz"), allow_pickle=True)
    tmap = {str(i): k for k, i in enumerate(t["ids"])}
    if set(ids).issubset(tmap):
        out["theory18"] = np.nan_to_num(np.array([t["F"][tmap[i]] for i in ids], float))

    s = np.load(os.path.join(BUNDLE, "caches", "perpixel", "series_gold.npz"), allow_pickle=True)
    smap = {str(i): k for k, i in enumerate(s["ids"])}
    if set(ids).issubset(smap):
        M = np.stack([np.stack([np.asarray(s["S"][smap[i], j]).astype(float)
                                for j in range(s["S"].shape[1])])
                      for i in ids])  # (n,100,8)
        Xs = np.concatenate([M.mean(1), M.std(1)], axis=1)  # mean+std over time -> 16
        out["series16"] = np.nan_to_num(Xs)
    return out


# ------------------------------------------------------------------- scoring
def _fit_pred(Xtr, ytr, Xte, C):
    sc = StandardScaler().fit(Xtr)
    m = LogisticRegression(max_iter=4000, C=C, class_weight="balanced").fit(sc.transform(Xtr), ytr)
    return m.predict_proba(sc.transform(Xte))[:, 1]


def nested_loco(X, y, clip, Cgrid=CGRID):
    """Out-of-fold probabilities; C picked by inner LOCO on training clips only."""
    p = np.zeros(len(y))
    chosen = {}
    for c in sorted(set(clip)):
        te = clip == c
        tr = ~te
        bestC, bestA, inner = None, -1.0, None
        for C in Cgrid:
            ip = np.zeros(tr.sum())
            for c2 in sorted(set(clip[tr])):
                itr = tr & (clip != c2)
                ite = tr & (clip == c2)
                ip[clip[tr] == c2] = _fit_pred(X[itr], y[itr], X[ite], C)
            a = roc_auc_score(y[tr], ip)
            if a > bestA:
                bestA, bestC, inner = a, C, ip
        p[te] = _fit_pred(X[tr], y[tr], X[te], bestC)
        chosen[str(c)] = {"C": bestC, "inner_auc": round(float(bestA), 4)}
    return p, chosen


# ------------------------------------------------------------------- metrics
def in_band(p, lo=BAND[0], hi=BAND[1]):
    return float(np.mean((p >= lo) & (p <= hi)))


def _logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def platt_loco(score, y, clip):
    """LOCO Platt calibration of an arbitrary monotone score -> P(good)."""
    out = np.zeros_like(score, float)
    for c in sorted(set(clip)):
        te = clip == c
        tr = ~te
        lr = LogisticRegression(C=1e6, max_iter=2000).fit(_logit(score[tr]).reshape(-1, 1), y[tr])
        out[te] = lr.predict_proba(_logit(score[te]).reshape(-1, 1))[:, 1]
    return out


def boot_auroc(y, p, clip, n=2000):
    clips = sorted(set(clip))
    vals = []
    for _ in range(n):
        pick = rng.choice(len(clips), size=len(clips), replace=True)
        idx = np.concatenate([np.where(clip == clips[j])[0] for j in pick])
        if len(set(y[idx])) < 2:
            continue
        vals.append(roc_auc_score(y[idx], p[idx]))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def median_stay(y, score, w=0.1):
    """Total-stay statistic (Zhang 2009) at a +/-w band; monotone-invariant."""
    far = float(np.mean((score[y == 0] > 0.5 + w) | (score[y == 0] < 0.5 - w)))
    frr = float(np.mean((score[y == 1] > 0.5 - w) & (score[y == 1] < 0.5 + w)))
    return abs(far - frr)


# --------------------------------------------------------------- fusion methods
def fuse(views, method):
    if method == "probavg":
        P = np.stack([views[k] for k in sorted(views)], 1)
        return P.mean(1)
    if method == "logitavg":
        return 1 / (1 + np.exp(-np.stack([_logit(views[k]) for k in sorted(views)], 1).mean(1)))
    if method == "rankavg":
        R = np.stack([rankdata(views[k]) / len(views[k]) for k in sorted(views)], 1)
        return R.mean(1)
    raise ValueError(method)


# ------------------------------------------------------------------------ main
def main():
    ids, y, clip, reason, base = load_gold()
    views = dict(base)
    extra = load_extra_views(ids)
    views.update(extra)

    # --- per-view nested-LOCO OOF probabilities
    view_oof, view_meta = {}, {}
    for name, X in views.items():
        p, chosen = nested_loco(X, y, clip)
        view_oof[name] = p
        view_meta[name] = {
            "n": int(len(y)),
            "n_features": int(np.asarray(X).shape[1]),
            "label_source": "human (manifest_gold.json)",
            "protocol": "nested LOCO (outer 7 clips; inner LOCO picks C)",
            "C_by_fold": chosen,
            "auroc": round(float(roc_auc_score(y, p)), 4),
            "amb_raw": round(in_band(p), 4),
        }
        print(f"view {name:12s} dim={X.shape[1]:4d} AUROC={view_meta[name]['auroc']:.4f} band={view_meta[name]['amb_raw']:.3f}")

    # --- frozen-config anchor (fixed C, deployed semantics)
    fpx, feg = views["pixel26"], views["eg2"]
    # reproduce deployed fixed-C OOF: eg2 C=0.01, pixel C=0.3
    def fixed_loco(X, C):
        p = np.zeros(len(y))
        for c in sorted(set(clip)):
            te = clip == c
            p[te] = _fit_pred(X[~te], y[~te], X[te], C)
        return p
    px_fix, eg_fix = fixed_loco(fpx, 0.3), fixed_loco(feg, 0.01)
    ens_fix = 0.5 * px_fix + 0.5 * eg_fix

    # --- combo table
    combos = {
        "pixel26": ["pixel26"],
        "eg2": ["eg2"],
        "pixel26+eg2 (baseline)": ["pixel26", "eg2"],
        "pixel26+eg2+perpixel121": ["pixel26", "eg2", "perpixel121"],
        "pixel26+eg2+physics16": ["pixel26", "eg2", "physics16"],
        "pixel26+eg2+series16": ["pixel26", "eg2", "series16"],
        "pixel26+eg2+perpixel121+physics16": ["pixel26", "eg2", "perpixel121", "physics16"],
        "pixel26+eg2+perpixel121+physics16+theory18": ["pixel26", "eg2", "perpixel121", "physics16", "theory18"],
        "pixel26+eg2+perpixel121+physics16+series16+theory18": ["pixel26", "eg2", "perpixel121", "physics16", "series16", "theory18"],
    }
    rows = []
    for cname, members in combos.items():
        sub = {m: view_oof[m] for m in members}
        for method in ("probavg", "rankavg"):
            s = fuse(sub, method)
            cal = platt_loco(s, y, clip)
            lo, hi = boot_auroc(y, s, clip)
            rows.append({
                "combo": cname,
                "views": members,
                "fusion": method,
                "n": int(len(y)),
                "label_source": "human (manifest_gold.json, 100 good / 50 bad)",
                "protocol": "nested LOCO over 7 clips; inner-LOCO C per view",
                "auroc": round(float(roc_auc_score(y, s)), 4),
                "auroc_ci95": [round(lo, 4), round(hi, 4)],
                "amb_raw": round(in_band(s), 4),
                "amb_cal": round(in_band(cal), 4),
                "amb_raw_profileband_0.25_0.75": round(in_band(s, 0.25, 0.75), 4),
                "amb_cal_profileband_0.25_0.75": round(in_band(cal, 0.25, 0.75), 4),
                "median_stay_w0.1": round(median_stay(y, s, 0.1), 4),
            })
        print(f"combo {cname:52s} done")

    # baseline anchor row (fixed C) for reproducibility of 0.8534
    lo, hi = boot_auroc(y, ens_fix, clip)
    anchor = {
        "combo": "pixel26+eg2 fixed-C (frozen ensemble_v1 anchor)",
        "views": ["pixel26(C=0.3)", "eg2(C=0.01)"],
        "fusion": "probavg",
        "n": int(len(y)),
        "label_source": "human (manifest_gold.json)",
        "protocol": "LOCO, fixed C from ensemble_v1.json (no inner selection)",
        "auroc": round(float(roc_auc_score(y, ens_fix)), 4),
        "auroc_ci95": [round(lo, 4), round(hi, 4)],
        "amb_raw": round(in_band(ens_fix), 4),
        "amb_cal": round(in_band(platt_loco(ens_fix, y, clip)), 4),
        "amb_raw_profileband_0.25_0.75": round(in_band(ens_fix, 0.25, 0.75), 4),
        "median_stay_w0.1": round(median_stay(y, ens_fix, 0.1), 4),
    }

    # --- OOD descriptive (evaluation only, never fit on OOD).
    # The calibrator is trained on the in-domain gold OOF score of the same view
    # and merely *applied* to OOD (monotone); no OOD label ever enters a fit.
    def gold_platt(gold_score, ood_score):
        lr = LogisticRegression(C=1e6, max_iter=2000).fit(_logit(gold_score).reshape(-1, 1), y)
        return lr.predict_proba(_logit(ood_score).reshape(-1, 1))[:, 1]

    gold_probavg = fuse({k: view_oof[k] for k in ("pixel26", "eg2")}, "probavg")
    gold_rankavg = fuse({k: view_oof[k] for k in ("pixel26", "eg2")}, "rankavg")
    ood = []
    o = json.load(open(os.path.join(BUNDLE, "caches", "ensemble", "ood_scores.json")))["scores"]
    lab = {e["segment_id"]: e["verdict"] for e in
           json.load(open(os.path.join(BUNDLE, "labels", "ood_labels.json")))["history"]}
    oo = [s for s in o if s["id"] in lab]
    yod = np.array([1 if lab[s["id"]] == "good" else 0 for s in oo])
    pe = np.array([s["p_eg2"] for s in oo]); pp = np.array([s["p_pixel"] for s in oo])
    pu = np.array([s["usable_prob"] for s in oo])
    ood_specs = [("p_eg2", pe, view_oof["eg2"]), ("p_pixel", pp, view_oof["pixel26"]),
                 ("probavg (deployed usable_prob)", pu, gold_probavg),
                 ("rankavg", (rankdata(pe) + rankdata(pp)) / (2 * len(pe)), gold_rankavg)]
    for nm, s, gs in ood_specs:
        ood.append({"score": nm, "n": int(len(yod)), "n_good": int(yod.sum()), "n_bad": int((1 - yod).sum()),
                    "label_source": "human (ood_labels.json)", "protocol": "evaluation only; calibrator fit on in-domain gold OOF, applied to OOD",
                    "auroc": round(float(roc_auc_score(yod, s)), 4), "amb_raw": round(in_band(s), 4),
                    "amb_cal": round(in_band(gold_platt(gs, s)), 4)})
    # unlabelled 1,546 descriptive review share (deployed score)
    sc = []
    for line in open("/workspace/artifacts/windows_rescored.jsonl"):
        d = json.loads(line)
        if d.get("type") == "header":
            continue
        sc.append(d["score"])
    e8_review = {
        "n": len(sc), "label_source": "none (descriptive)", "protocol": "E8.01 re-score, ensemble_v1 score",
        "amb_raw": round(in_band(np.array(sc)), 4),
    }

    y3_01 = {
        "experiment": "Y3.01", "title": "Cached-view inventory + two-view baseline reproduction",
        "owner": "Y3 (claim Y3)", "device": "CPU (OMP_NUM_THREADS=2)",
        "data": "150-window IndEgo human gold (LOCO); 60-window Human-labelled Egocentric-10K OOD",
        "views_available": {k: int(np.asarray(v).shape[1]) for k, v in views.items()},
        "views_absent": ["EG2@1fps", "EG2@0.5fps (only segments.npz @2fps exists)"],
        "per_view_nested_loco": view_meta,
        "baseline_anchor_fixedC": anchor,
        "ood_descriptive": ood,
        "e8_1546_review_share": e8_review,
    }
    y3_02 = {
        "experiment": "Y3.02", "title": "Cheap view-ensembling combo sweep (LOCO)",
        "owner": "Y3 (claim Y3)", "device": "CPU (OMP_NUM_THREADS=2)",
        "band": list(BAND), "nested_loco": True,
        "label_source": "human (manifest_gold.json, 100 good / 50 bad)",
        "protocol": "nested LOCO over 7 clips; inner-LOCO C per view; probavg/rankavg fusion",
        "baseline_anchor": anchor,
        "combos": rows,
    }
    json.dump(y3_01, open(os.path.join(OUTDIR, "Y3.01.json"), "w"), indent=1)
    json.dump(y3_02, open(os.path.join(OUTDIR, "Y3.02.json"), "w"), indent=1)

    # --- ranked recommendation
    ranked = sorted(rows, key=lambda r: (r["amb_cal"], -r["auroc"]))
    print("\ncombo | fusion | AUROC [CI] | amb_raw | amb_cal")
    for r in sorted(rows, key=lambda r: (r["amb_cal"], -r["auroc"])):
        print(f"{r['combo'][:46]:46s} {r['fusion']:8s} {r['auroc']:.3f} [{r['auroc_ci95'][0]:.3f},{r['auroc_ci95'][1]:.3f}] "
              f"raw={r['amb_raw']:.3f} cal={r['amb_cal']:.3f}")
    json.dump({"ranked": ranked, "anchor": anchor, "ood": ood, "e8_1546": e8_review},
              open(os.path.join(OUTDIR, "_summary_tmp.json"), "w"), indent=1)
    print("\nwrote", OUTDIR)

    # figure
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(9, 5))
        lbl = [f"{r['combo'].replace('pixel26+','px+').replace('+','+')}\n{r['fusion']}" for r in rows]
        colors = ["#1f77b4" if r["fusion"] == "probavg" else "#d62728" for r in rows]
        ax.scatter([r["amb_cal"] for r in rows], [r["auroc"] for r in rows], c=colors, s=55)
        ax.axhline(anchor["auroc"], ls="--", c="grey", lw=1)
        ax.axvline(anchor["amb_cal"], ls="--", c="grey", lw=1)
        ax.set_xlabel("ambiguous fraction after LOCO Platt  (lower = more separable)")
        ax.set_ylabel("LOCO AUROC (higher = better)")
        ax.set_title("Y3: cheap view ensembling — ambiguous fraction vs AUROC\n(150 IndEgo gold, nested LOCO; grey = frozen 2-view anchor)")
        for r, x, yy in zip(rows, [r["amb_cal"] for r in rows], [r["auroc"] for r in rows]):
            ax.annotate(r["combo"].replace("pixel26+", "px+")[:26] + "/" + r["fusion"][:4],
                        (x, yy), fontsize=6, xytext=(3, 3), textcoords="offset points")
        fig.tight_layout()
        figpath = os.path.join(ROOT, "docs", "experiments", "exp_review", "images", "y3_separability.png")
        os.makedirs(os.path.dirname(figpath), exist_ok=True)
        fig.savefig(figpath, dpi=130)
        print("figure ->", figpath)
    except Exception as e:  # pragma: no cover
        print("figure skipped:", e)


if __name__ == "__main__":
    main()
