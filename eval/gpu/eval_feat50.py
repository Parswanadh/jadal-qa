#!/usr/bin/env python3
"""F50 — evaluate the 50-feature set vs the frozen 26-feature baseline.

Protocols (identical to the frozen baselines so the delta is honest):
  * In-domain  : leave-one-clip-out logistic (class_weight=balanced, C=0.3) over
                 the 7 IndEgo clips; clip-bootstrap 95% CI.
  * OOD        : adapted per-domain z-score recipe (z-score each domain with its
                 own mean/std; per-feature sign from IndEgo gold AUROC; equal-
                 weight mean). OOD CI is FACTORY-GROUPED bootstrap (n_boot=2000).
                 Frozen-ensemble reference = 0.566; adapted-26 baseline = 0.781.
  * Hold-out108: the SAME frozen recipe scored ONCE (no tuning).

Ablation: per-family groups + cumulative saturation for the adapted OOD score.

Reads eval/results/exp_feat50/features50.npz (from run_features50.py).
Writes eval/results/exp_feat50/{in_domain,ood,ablation,cost,holdout}.json.
Run: OMP_NUM_THREADS=2 python3 eval/gpu/eval_feat50.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)
from jadal.gpu.features import FEATURE_NAMES  # noqa: E402
from jadal.gpu.features50 import NEW_FEATURE_NAMES  # noqa: E402

BUNDLE = os.environ.get("JADAL_BUNDLE", "/workspace/jadal_pod_bundle")
OUT = os.path.join(REPO, "eval", "results", "exp_feat50")
NPZ = os.path.join(OUT, "features50.npz")
OOD_PIXEL = os.environ.get(
    "JADAL_PIXEL160",
    "/workspace/wt/m-fewshot160/eval/results/exp_fewshot/inputs/ood160_pixel.npz")
HOLD_FEATS = os.environ.get(
    "JADAL_CONFIRM_FEATS",
    "/workspace/wt/m-confirm/eval/results/exp_confirm/inputs/confirm108_features.npz")
N_BOOT = 2000
SEED = 0

# Z3 transferable subset inside the 24 new features
TRANSFER6 = ["ct_fsize_std", "mo_hand_motion_excess", "inv_clip_frac",
             "inv_laplacian_var", "inv_hist_entropy", "inv_dct_hf_lf"]
GROUPS = {
    "z3_transferable": TRANSFER6,
    "dct_bands": ["dct_band_r1", "dct_band_r2", "dct_band_r3", "dct_band_r4plus"],
    "colour": ["col_rg_ratio", "col_bg_ratio", "col_sat_std"],
    "temporal_lags": ["lag1_diff", "lag2_diff", "lag3_diff", "lag4_diff"],
    "noise_floor": ["noise_floor_med", "noise_floor_p90"],
    "exposure_clip": ["clip_low_5pct", "clip_high_99"],
    "edge_lap": ["edge_density", "lap_mean_abs"],
    "hand": ["hand_occupancy"],
}


def fast_auc(y, s):
    from scipy.stats import rankdata
    y = np.asarray(y)
    s = np.asarray(s, float)
    n_pos = int((y == 1).sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    r = rankdata(s)
    return float((r[y == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def bal_acc(y, s, thr=0.0):
    y = np.asarray(y)
    pred = (np.asarray(s) >= thr).astype(int)
    tpr = ((pred == 1) & (y == 1)).sum() / max((y == 1).sum(), 1)
    tnr = ((pred == 0) & (y == 0)).sum() / max((y == 0).sum(), 1)
    return float(0.5 * (tpr + tnr))


def group_boot(y, s, groups, n_boot=N_BOOT, seed=SEED):
    y = np.asarray(y)
    s = np.asarray(s, float)
    gs = list(groups)
    gmap = {g: i for i, g in enumerate(sorted(set(gs)))}
    gi = np.asarray([gmap[g] for g in gs])
    idx_by = [np.where(gi == k)[0] for k in range(len(gmap))]
    rng = np.random.RandomState(seed)
    vals = []
    for _ in range(n_boot):
        pick = rng.randint(0, len(idx_by), len(idx_by))
        idx = np.concatenate([idx_by[k] for k in pick])
        if len(np.unique(y[idx])) < 2:
            continue
        vals.append(fast_auc(y[idx], s[idx]))
    if not vals:
        return [None, None]
    return [round(float(np.percentile(vals, 2.5)), 4),
            round(float(np.percentile(vals, 97.5)), 4)]


def clip_boot(y, s, clips, n_boot=N_BOOT, seed=SEED):
    y = np.asarray(y)
    s = np.asarray(s, float)
    cs = np.asarray(clips)
    u = sorted(set(cs.tolist()))
    idx_by = [np.where(cs == c)[0] for c in u]
    rng = np.random.RandomState(seed)
    vals = []
    for _ in range(n_boot):
        pick = rng.randint(0, len(idx_by), len(idx_by))
        idx = np.concatenate([idx_by[k] for k in pick])
        if len(np.unique(y[idx])) < 2:
            continue
        vals.append(fast_auc(y[idx], s[idx]))
    if not vals:
        return [None, None]
    return [round(float(np.percentile(vals, 2.5)), 4),
            round(float(np.percentile(vals, 97.5)), 4)]


def paired_group_boot(y, sa, sb, groups, n_boot=N_BOOT, seed=SEED):
    y = np.asarray(y)
    sa = np.asarray(sa, float)
    sb = np.asarray(sb, float)
    gs = list(groups)
    gmap = {g: i for i, g in enumerate(sorted(set(gs)))}
    gi = np.asarray([gmap[g] for g in gs])
    idx_by = [np.where(gi == k)[0] for k in range(len(gmap))]
    rng = np.random.RandomState(seed)
    d = []
    for _ in range(n_boot):
        pick = rng.randint(0, len(idx_by), len(idx_by))
        idx = np.concatenate([idx_by[k] for k in pick])
        if len(np.unique(y[idx])) < 2:
            continue
        d.append(fast_auc(y[idx], sa[idx]) - fast_auc(y[idx], sb[idx]))
    d = np.asarray(d)
    return {"mean": round(float(d.mean()), 4),
            "ci95": [round(float(np.percentile(d, 2.5)), 4),
                     round(float(np.percentile(d, 97.5)), 4)],
            "frac_gt_0": round(float((d > 0).mean()), 4), "n_boot": int(len(d))}


def make_clf(C=0.3, balanced=True):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(StandardScaler(),
                         LogisticRegression(C=C, max_iter=4000,
                                            class_weight="balanced" if balanced else None))


def loco(X, y, clips, C=0.3):
    X = np.asarray(X, float)
    y = np.asarray(y, int)
    clips = np.asarray(clips)
    oof = np.full(len(y), np.nan)
    for held in sorted(set(clips.tolist())):
        te = np.where(clips == held)[0]
        tr = np.where(clips != held)[0]
        clf = make_clf(C)
        clf.fit(X[tr], y[tr])
        oof[te] = clf.predict_proba(X[te])[:, 1]
    return oof


def zscore(Ax, Bx, y_in, cols):
    """Per-domain z-score + IndEgo sign, equal-weight mean. Returns (s_in, s_ood, au, sg)."""
    Ax = np.asarray(Ax, float)[:, cols]
    Bx = np.asarray(Bx, float)[:, cols]
    au = np.asarray([fast_auc(y_in, Ax[:, j]) for j in range(Ax.shape[1])])
    sg = np.where(au >= 0.5, 1.0, -1.0)
    za = (Ax - Ax.mean(0)) / np.clip(Ax.std(0), 1e-9, None)
    zb = (Bx - Bx.mean(0)) / np.clip(Bx.std(0), 1e-9, None)
    return (za * sg).mean(1), (zb * sg).mean(1), au, sg


def git_sha():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                              text=True, cwd=REPO).stdout.strip()
    except Exception:
        return "unknown"


def _write(name, rep):
    os.makedirs(OUT, exist_ok=True)
    rep["git_sha"] = git_sha()
    with open(os.path.join(OUT, name), "w") as f:
        json.dump(rep, f, indent=1)
    print("wrote", os.path.join(OUT, name))


def load_all():
    z = np.load(NPZ, allow_pickle=True)
    names26 = [str(x) for x in z["names26"]]
    names_new = [str(x) for x in z["names_new"]]
    assert names26 == FEATURE_NAMES and names_new == NEW_FEATURE_NAMES
    # gold cached 26 (frozen)
    gold_items = json.load(open(os.path.join(BUNDLE, "labels", "manifest_gold.json")))["items"]
    gj = json.load(open(os.path.join(BUNDLE, "caches", "gpu", "features.json")))["windows"]
    gold_order = {it["id"]: it for it in gold_items}
    g_ids = [str(x) for x in z["gold_ids"]]
    Xg26 = np.asarray([[float(gj[i]["full"][n]) for n in FEATURE_NAMES] for i in g_ids])
    Xg_new = z["gold_fnew"]
    yg = np.asarray([int(gold_order[i]["label"]) for i in g_ids])
    cg = np.asarray([gold_order[i]["clip"] for i in g_ids])
    # ood cached 26 (frozen 10fps view)
    o = np.load(OOD_PIXEL, allow_pickle=True)
    oidx = {str(i): k for k, i in enumerate(o["ids"])}
    o_ids = [str(x) for x in z["ood_ids"]]
    Xo26 = np.asarray([o["X10"][oidx[i]] for i in o_ids])
    Xo_new = z["ood_fnew"]
    ood_lab = {w["segment_id"]: (1 if w["verdict"] == "good" else 0)
               for w in json.load(open(os.path.join(BUNDLE, "labels", "ood160_labels.json")))["windows"]}
    yo = np.asarray([ood_lab[i] for i in o_ids], int)
    fac = np.asarray([str(x) for x in z["ood_groups"]])
    # frozen ensemble scores
    sc = {}
    for fn in ("ood_scores.json", "ood_scores2.json"):
        d = json.load(open(os.path.join(BUNDLE, "caches", "ensemble", fn)))
        for x in d["scores"]:
            if "error" not in x:
                sc[x["id"]] = float(x["usable_prob"])
    frozen = np.asarray([sc.get(i, np.nan) for i in o_ids], float)
    # hold-out
    h = np.load(HOLD_FEATS, allow_pickle=True)
    hidx = {str(i): k for k, i in enumerate(h["ids"])}
    h_ids = [str(x) for x in z["hold_ids"]]
    Xh26 = np.asarray([h["Xpx"][hidx[i]] for i in h_ids])
    Xh_new = z["hold_fnew"]
    hold_lab = {x["id"]: (1 if x["verdict"] == "good" else 0) for x in
                json.load(open(os.environ.get("JADAL_HOLD108",
                 "/workspace/wt/m-confirm/eval/results/exp_confirm/inputs/windows108.json")))}
    yh = np.asarray([hold_lab[i] for i in h_ids], int)
    hfac = np.asarray([str(x) for x in z["hold_groups"]])
    return dict(names26=names26, names_new=names_new,
                Xg26=Xg26, Xg_new=Xg_new, yg=yg, cg=cg, g_ids=g_ids,
                Xo26=Xo26, Xo_new=Xo_new, yo=yo, fac=fac, o_ids=o_ids, frozen=frozen,
                Xh26=Xh26, Xh_new=Xh_new, yh=yh, hfac=hfac, h_ids=h_ids)


def main():
    d = load_all()
    sha = git_sha()
    Xg50 = np.hstack([d["Xg26"], d["Xg_new"]])
    Xo50 = np.hstack([d["Xo26"], d["Xo_new"]])
    Xh50 = np.hstack([d["Xh26"], d["Xh_new"]])
    all_names = d["names26"] + d["names_new"]
    head = {"n_gold": int(len(d["yg"])), "n_good_gold": int((d["yg"] == 1).sum()),
            "n_ood": int(len(d["yo"])), "n_good_ood": int((d["yo"] == 1).sum()),
            "n_factories_ood": int(len(set(d["fac"].tolist()))),
            "n_hold": int(len(d["yh"])), "n_good_hold": int((d["yh"] == 1).sum()),
            "device": "CPU (numpy/sklearn; decode by ffmpeg)",
            "protocol": ("In-domain = LOCO logistic C=0.3 balanced over 7 IndEgo clips, "
                         "clip-bootstrap CI. OOD = adapted per-domain z-score "
                         "(per-domain mean/std, sign from IndEgo AUROC, equal-weight mean), "
                         "FACTORY-GROUPED bootstrap CI (n_boot=2000). Hold-out108 scored once."),
            "git_sha": sha}

    # ---------------- in-domain ----------------
    ind = {}
    for tag, X in (("f26", d["Xg26"]), ("f50", Xg50)):
        oof = loco(X, d["yg"], d["cg"], C=0.3)
        ind[tag] = {"n_features": X.shape[1], "auroc": round(fast_auc(d["yg"], oof), 4),
                    "auroc_ci95_clip": clip_boot(d["yg"], oof, d["cg"]),
                    "ba": round(bal_acc(d["yg"], oof, 0.5), 4)}
    _write("in_domain.json", {**head, "experiment": "F50.01",
           "title": "In-domain LOCO: 26 vs 50 per-pixel features",
           "baseline_26_documented": 0.810, "rows": ind,
           "delta_auroc": round(ind["f50"]["auroc"] - ind["f26"]["auroc"], 4)})
    print("in-domain", {k: (v["auroc"], v["ba"]) for k, v in ind.items()})

    # ---------------- OOD adapted ----------------
    def ood_row(cols, tag):
        si, so, au, sg = zscore(Xg50, Xo50, d["yg"], cols)
        return {"tag": tag, "n_features": len(cols),
                "indego_auroc": round(fast_auc(d["yg"], si), 4),
                "ood_auroc": round(fast_auc(d["yo"], so), 4),
                "ood_ci95_factory": group_boot(d["yo"], so, d["fac"]),
                "ood_ba": round(bal_acc(d["yo"], so, 0.0), 4)}

    c26 = list(range(26))
    cnew = list(range(26, 50))
    c50 = list(range(50))
    ood = {"f26_all": ood_row(c26, "26 (frozen pixel baseline set)"),
           "f50_all": ood_row(c50, "50 (26 + 24 new)"),
           "fnew24_only": ood_row(cnew, "24 new only")}
    # documented baseline: stable8+transferable (subset of the 26) -> reproduce 0.781
    STABLE8 = ["radial_slope", "hf_ratio", "dct_energy", "luma_p05", "global_shift",
               "illum_r", "illum_b", "zncc_self"]
    s8 = [all_names.index(n) for n in STABLE8]
    t6 = [all_names.index(n) for n in TRANSFER6]
    ood["documented_baseline_26_stable8+transfer6"] = ood_row(s8 + t6,
                                                              "stable8(26)+transfer6(new)")
    # the increment: documented recipe + the remaining 18 new features
    rest18 = [j for j in cnew if all_names[j] not in TRANSFER6]
    ood["documented_recipe_plus_18new"] = ood_row(s8 + t6 + rest18,
                                                  "stable8+transfer6+18 remaining new")
    # frozen ensemble reference
    m = ~np.isnan(d["frozen"])
    ood["frozen_ensemble_v1"] = {
        "ood_auroc": round(fast_auc(d["yo"][m], d["frozen"][m]), 4),
        "ood_ci95_factory": group_boot(d["yo"][m], d["frozen"][m], d["fac"][m]),
        "ood_ba": round(bal_acc(d["yo"][m], d["frozen"][m], 0.5), 4),
        "missing": int((~m).sum())}
    # paired gain 50 vs 26
    _, so26, _, _ = zscore(Xg50, Xo50, d["yg"], c26)
    _, so50, _, _ = zscore(Xg50, Xo50, d["yg"], c50)
    gain = paired_group_boot(d["yo"], so50, so26, d["fac"])
    _write("ood.json", {**head, "experiment": "F50.02",
           "title": "OOD160 adapted per-domain z-score: 26 vs 50 features",
           "baseline_26_documented_auroc": 0.781,
           "frozen_26_documented_auroc": 0.566, "rows": ood,
           "paired_gain_50_vs_26_factory_grouped": gain})
    print("OOD", {k: v.get("ood_auroc") for k, v in ood.items()})

    # ---------------- ablation ----------------
    # leave-one-family-out (remove family from the 50) + family-only
    ab = {}
    fam_of = {}
    for g, feats in GROUPS.items():
        for n in feats:
            fam_of[n] = g
    for g in GROUPS:
        cols = [j for j in c50 if fam_of.get(all_names[j], None) != g]
        r = ood_row(cols, f"drop_{g}")
        ab[f"drop_{g}"] = r
    for g, feats in GROUPS.items():
        cols = [all_names.index(n) for n in feats]
        ab[f"only_{g}"] = ood_row(cols, f"only_{g}")
    # cumulative saturation: add groups in order of their family-only OOD AUROC
    order = sorted(GROUPS, key=lambda g: -ab[f"only_{g}"]["ood_auroc"])
    cum, cols = [], []
    for g in order:
        cols += [all_names.index(n) for n in GROUPS[g]]
        r = ood_row(list(cols), f"cum_{g}")
        r["family_added"] = g
        r["families"] = order[:order.index(g) + 1]
        cum.append(r)
    _write("ablation.json", {**head, "experiment": "F50.03",
           "title": "Ablation of the 24 new features (adapted OOD score)",
           "groups": GROUPS, "leave_one_family_out": {k: v for k, v in ab.items()
                                                       if k.startswith("drop_")},
           "family_only": {k: v for k, v in ab.items() if k.startswith("only_")},
           "cumulative_by_family_only_rank": cum})
    print("ablation drop-1 (ood auroc):",
          {k: v["ood_auroc"] for k, v in ab.items() if k.startswith("drop_")})

    # ---------------- hold-out (scored ONCE) ----------------
    # Per-domain z-score recipe: sign from IndEgo gold, each domain z-scored by its
    # OWN mean/std (the frozen O4 recipe). Applied to 26, documented-14, and 50.
    def hold_row(cols, tag):
        _, s, au, sg = zscore(Xg50, Xh50, d["yg"], cols)
        return {"tag": tag, "n_features": len(cols),
                "auroc": round(fast_auc(d["yh"], s), 4),
                "ci95_factory": group_boot(d["yh"], s, d["hfac"]),
                "ba": round(bal_acc(d["yh"], s, 0.0), 4)}
    hold = {
        "f26_all": hold_row(c26, "26 (frozen pixel baseline set)"),
        "documented_baseline_stable8+transfer6": hold_row(s8 + t6,
                                                          "stable8(26)+transfer6(new)"),
        "f50_all": hold_row(c50, "50 (26 + 24 new)"),
    }
    _write("holdout.json", {**head, "experiment": "F50.04",
           "title": "Hold-out108 scored ONCE (no tuning)",
           "documented_confirmatory_26_auroc": 0.676, "rows": hold,
           "statement": "Scored once; no feature re-selection, no threshold change, no re-fit.",
           "delta_auroc_f50_vs_f26": round(hold["f50_all"]["auroc"] - hold["f26_all"]["auroc"], 4)})
    print("hold-out", {k: v["auroc"] for k, v in hold.items()})

    print("DONE")


if __name__ == "__main__":
    main()
