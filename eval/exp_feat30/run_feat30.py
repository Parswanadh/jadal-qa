#!/usr/bin/env python3
"""F30 — per-pixel feature vector 26 -> 30: accuracy gain.

Expands the 26 E5 per-pixel features with the 4 decode-based Z3 invariants
(``inv_clip_frac``, ``inv_laplacian_var``, ``inv_hist_entropy``, ``inv_dct_hf_lf``,
see ``jadal/gpu/features30.py``) and measures the gain with the *same* recipes as
the baselines:

  * in-domain LOCO logistic (C=0.3, class_weight=balanced, per-fold
    StandardScaler) over the 150 IndEgo gold windows / 7 clips;
  * out-of-domain **adapted per-domain z-score** recipe (per-domain z-score,
    IndEgo sign, equal-weight mean) on the 160 OOD windows, factory-grouped CI;
  * frozen logistic (fit on all IndEgo) for reference;
  * per-feature ablation (26 + each single added feature, and 30 - each);
  * the 108-window round-3 hold-out scored **once** (no tuning).

All features are read from caches (no re-decode of the 26; the 4 invariants are
the cached Z3 values, whose definitions ``jadal/gpu/features30.py`` reproduces).
CPU only, fixed seeds.

Run: OMP_NUM_THREADS=2 python3 eval/exp_feat30/run_feat30.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time

import numpy as np
from scipy.stats import rankdata

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(REPO, "eval", "results", "exp_feat30")
BUNDLE = os.environ.get("JADAL_BUNDLE", "/workspace/jadal_pod_bundle")
WT = os.environ.get("JADAL_WT", "/workspace/wt")
Z3_160 = os.environ.get(
    "JADAL_Z3_160",
    os.path.join(WT, "m-adapt160", "eval", "results", "exp_adapt160",
                 "inputs", "z3_160.npz"))
PIX160 = os.environ.get(
    "JADAL_PIXEL160",
    os.path.join(WT, "m-fewshot160", "eval", "results", "exp_fewshot",
                 "inputs", "ood160_pixel.npz"))
CONFIRM108 = os.environ.get(
    "JADAL_CONFIRM108",
    os.path.join(WT, "m-confirm", "eval", "results", "exp_confirm",
                 "inputs", "confirm108_features.npz"))

N_BOOT = 2000
SEED = 0
C_LOCO = 0.3

NEW4 = ["inv_clip_frac", "inv_laplacian_var", "inv_hist_entropy", "inv_dct_hf_lf"]
STABLE8 = ["radial_slope", "hf_ratio", "dct_energy", "luma_p05", "global_shift",
           "illum_r", "illum_b", "zncc_self"]
Z3_SIX = ["ct_fsize_std", "mo_hand_motion_excess", "inv_clip_frac",
          "inv_laplacian_var", "inv_hist_entropy", "inv_dct_hf_lf"]
FEATURE_NAMES26 = [
    "temp_b0", "temp_b1", "temp_b2", "temp_b3", "frozen_frac", "flicker_idx",
    "radial_slope", "hf_ratio", "dct_energy", "pca_resid", "pca_top1_share",
    "luma_mean", "luma_p05", "luma_p50", "luma_p95", "dark_frac", "bright_frac",
    "motion_energy", "global_shift", "skin_frac", "illum_r", "illum_g", "illum_b",
    "rppg_pulse", "dup_hash_mean", "zncc_self"]
NAMES30 = FEATURE_NAMES26 + NEW4


def git_sha():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                              text=True, cwd=REPO).stdout.strip()
    except Exception:
        return "unknown"


# ---------------- metrics ----------------

def fast_auc(y, s):
    y = np.asarray(y)
    s = np.asarray(s, float)
    n_pos = int((y == 1).sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    r = rankdata(s)
    return float((r[y == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def balacc(y, s, thr=0.5):
    y = np.asarray(y)
    s = np.asarray(s, float)
    pred = (s > thr).astype(int)
    g, b = y == 1, y == 0
    tpr = float((pred[g] == 1).mean()) if g.any() else float("nan")
    tnr = float((pred[b] == 0).mean()) if b.any() else float("nan")
    return 0.5 * (tpr + tnr)


def _group_index(groups):
    gs = list(groups)
    gmap = {g: i for i, g in enumerate(sorted(set(gs)))}
    gi = np.asarray([gmap[g] for g in gs])
    return gi, [np.where(gi == k)[0] for k in range(len(gmap))]


def _ci(vals):
    if not vals:
        return [None, None]
    return [round(float(np.percentile(vals, 2.5)), 4),
            round(float(np.percentile(vals, 97.5)), 4)]


def group_boot_ci(y, s, groups, thr=None, n_boot=N_BOOT, seed=SEED):
    y, s = np.asarray(y), np.asarray(s)
    _, idx_by = _group_index(groups)
    rng = np.random.RandomState(seed)
    vals = []
    for _ in range(n_boot):
        pick = rng.randint(0, len(idx_by), len(idx_by))
        idx = np.concatenate([idx_by[k] for k in pick])
        if len(np.unique(y[idx])) < 2:
            continue
        vals.append(fast_auc(y[idx], s[idx]) if thr is None else balacc(y[idx], s[idx], thr))
    return _ci(vals)


def paired_group_boot(y, s_a, s_b, groups, n_boot=N_BOOT, seed=SEED):
    y = np.asarray(y)
    s_a, s_b = np.asarray(s_a, float), np.asarray(s_b, float)
    _, idx_by = _group_index(groups)
    rng = np.random.RandomState(seed)
    diffs = []
    for _ in range(n_boot):
        pick = rng.randint(0, len(idx_by), len(idx_by))
        idx = np.concatenate([idx_by[k] for k in pick])
        if len(np.unique(y[idx])) < 2:
            continue
        diffs.append(fast_auc(y[idx], s_a[idx]) - fast_auc(y[idx], s_b[idx]))
    diffs = np.asarray(diffs)
    if len(diffs) == 0:
        return {}
    return {"mean": round(float(diffs.mean()), 4),
            "ci95": [round(float(np.percentile(diffs, 2.5)), 4),
                     round(float(np.percentile(diffs, 97.5)), 4)],
            "frac_resamples_gt_0": round(float((diffs > 0).mean()), 4),
            "n_boot": int(len(diffs))}


# ---------------- recipes ----------------

def loco_logistic(X, y, clips, C=C_LOCO):
    """Leave-one-clip-out logistic (per-fold StandardScaler), C balanced."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    X, y = np.asarray(X, float), np.asarray(y, int)
    clips = np.asarray(clips)
    p = np.zeros(len(y))
    for c in sorted(set(clips.tolist())):
        tr, te = clips != c, clips == c
        sc = StandardScaler().fit(X[tr])
        m = LogisticRegression(max_iter=4000, C=C, class_weight="balanced")
        m.fit(sc.transform(X[tr]), y[tr])
        p[te] = m.predict_proba(sc.transform(X[te]))[:, 1]
    return p


def frozen_logistic(Xg, yg, Xo):
    """Fit on all IndEgo (StandardScaler + C balanced), score OOD."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(Xg)
    m = LogisticRegression(max_iter=4000, C=C_LOCO, class_weight="balanced")
    m.fit(sc.transform(Xg), yg)
    return m.predict_proba(sc.transform(Xo))[:, 1]


def zscore_recipe(Xg, Xo, yg, extra=None):
    """Per-domain z-score, IndEgo sign, equal-weight mean. Returns (sg, so, signs, au)."""
    Xg, Xo = np.asarray(Xg, float), np.asarray(Xo, float)
    au = np.asarray([fast_auc(yg, Xg[:, j]) for j in range(Xg.shape[1])])
    signs = np.where(au >= 0.5, 1.0, -1.0)
    zg = (Xg - Xg.mean(0)) / np.clip(Xg.std(0), 1e-9, None)
    zo = (Xo - Xo.mean(0)) / np.clip(Xo.std(0), 1e-9, None)
    sg = (zg * signs).mean(1)
    so = (zo * signs).mean(1)
    if extra is not None:  # per-domain: the hold-out z-scores with its OWN stats
        Xe = np.asarray(extra, float)
        ze = (Xe - Xe.mean(0)) / np.clip(Xe.std(0), 1e-9, None)
        return sg, so, (ze * signs).mean(1), signs, au
    return sg, so, None, signs, au


# ---------------- loaders ----------------

def load_gold():
    m = json.load(open(os.path.join(BUNDLE, "labels", "manifest_gold.json")))
    items = m["items"]
    ids = [it["id"] for it in items]
    clips = np.asarray([it["clip"] for it in items])
    y = np.asarray([int(it["label"]) for it in items])
    fj = json.load(open(os.path.join(BUNDLE, "caches", "gpu", "features.json")))["windows"]
    X26 = np.asarray([[float(fj[i]["full"][n]) for n in FEATURE_NAMES26] for i in ids])
    z = np.load(Z3_160, allow_pickle=True)
    zn = [str(x) for x in z["names"]]
    assert list(z["clips_indego"]) == list(clips) and list(z["y_indego"]) == list(y)
    Xz3 = np.asarray(z["X_indego"], float)
    return ids, clips, y, X26, Xz3, zn


def load_ood160():
    d = json.load(open(os.path.join(BUNDLE, "labels", "ood160_labels.json")))
    w = d["windows"]
    ids = [x["segment_id"] for x in w]
    y = np.asarray([1 if x["verdict"] == "good" else 0 for x in w], int)
    fac = np.asarray([x["factory"] for x in w])
    p = np.load(PIX160, allow_pickle=True)
    assert list(p["ids"]) == ids
    X26 = p["X10"].astype(float)
    z = np.load(Z3_160, allow_pickle=True)
    zn = [str(x) for x in z["names"]]
    assert list(z["ids"]) == ids
    Xz3 = z["X"].astype(float)
    return ids, y, fac, X26, Xz3, zn


def load_holdout108():
    d = json.load(open(os.path.join(BUNDLE, "labels", "round3",
                                    "ood_labels3_108_with_factory.json")))
    w = d["windows"]
    ids = [x["segment_id"] for x in w]
    y = np.asarray([1 if x["verdict"] == "good" else 0 for x in w], int)
    fac = np.asarray([x["factory"] for x in w])
    c = np.load(CONFIRM108, allow_pickle=True)
    assert list(c["ids"]) == ids, "confirm108 ids must match hold-out labels"
    X26 = c["Xpx"].astype(float)
    Xz3 = c["Xz3"].astype(float)
    zn = [str(x) for x in c["z3_names"]]
    return ids, y, fac, X26, Xz3, zn


def load_frozen160(ids):
    sc = {}
    for fn in ("ood_scores.json", "ood_scores2.json"):
        d = json.load(open(os.path.join(BUNDLE, "caches", "ensemble", fn)))
        for x in d["scores"]:
            if "error" not in x:
                sc[x["id"]] = float(x["usable_prob"])
    return np.asarray([sc[i] for i in ids], float)


# ---------------- main ----------------

def main():
    os.makedirs(OUT, exist_ok=True)
    t_start = time.perf_counter()
    sha = git_sha()

    g_ids, g_clips, gy, X26g, Xz3g, zn_g = load_gold()
    o_ids, oy, o_fac, X26o, Xz3o, zn_o = load_ood160()
    h_ids, hy, h_fac, X26h, Xz3h, zn_h = load_holdout108()
    assert zn_g == zn_o == zn_h, "Z3 feature order must match across datasets"
    zn = zn_g
    n4 = [zn.index(k) for k in NEW4]
    new_g, new_o, new_h = Xz3g[:, n4], Xz3o[:, n4], Xz3h[:, n4]

    X26g, X26o, X26h = map(np.asarray, (X26g, X26o, X26h))
    X30g = np.hstack([X26g, new_g])
    X30o = np.hstack([X26o, new_o])
    X30h = np.hstack([X26h, new_h])
    assert X30g.shape[1] == 30 and X30o.shape[1] == 30 and X30h.shape[1] == 30

    n_good, n_bad = int((oy == 1).sum()), int((oy == 0).sum())
    nf = len(set(o_fac.tolist()))
    head = {
        "experiment": "F30.01",
        "title": "Per-pixel feature vector 26 -> 30 (4 Z3 transferable invariants)",
        "n_features": 30,
        "added_features": NEW4,
        "feature_names30": NAMES30,
        "data": {
            "gold": {"n": len(gy), "n_good": int((gy == 1).sum()),
                     "n_bad": int((gy == 0).sum()), "clips": len(set(g_clips.tolist())),
                     "label_source": "human manifest_gold.json (single labeller)"},
            "ood160": {"n": len(oy), "n_good": n_good, "n_bad": n_bad,
                       "factories": nf,
                       "label_source": "human ood160_labels.json (138 good/22 bad, 47 factories)"},
            "holdout108": {"n": len(hy), "n_good": int((hy == 1).sum()),
                           "n_bad": int((hy == 0).sum()),
                           "factories": len(set(h_fac.tolist())),
                           "label_source": "human round3/ood_labels3_108_with_factory.json"},
        },
        "protocol": ("in-domain = LOCO logistic C=0.3 balanced (per-fold StandardScaler), "
                     "clip-bootstrap CI; OOD = adapted per-domain z-score (IndEgo sign, "
                     "equal-weight mean) with FACTORY-GROUPED bootstrap (n_boot=2000, seed=0); "
                     "hold-out scored once, no tuning"),
        "device": "CPU (numpy/sklearn; cached features, no decode)",
        "git_sha": sha,
        "command": "OMP_NUM_THREADS=2 python3 eval/exp_feat30/run_feat30.py",
    }

    res = {"head": head}

    # ---- 1. in-domain LOCO logistic, 26 vs 30 (+ablations) ----
    def loco_block(X, name):
        p = loco_logistic(X, gy, g_clips)
        return {"features": name, "n": len(gy),
                "auroc": round(fast_auc(gy, p), 4),
                "auroc_ci95_clip": group_boot_ci(gy, p, g_clips),
                "bal_acc": round(balacc(gy, p, 0.5), 4),
                "bal_acc_ci95_clip": group_boot_ci(gy, p, g_clips, thr=0.5),
                "proba": [round(float(v), 6) for v in p]}

    loco26 = loco_block(X26g, "26")
    loco30 = loco_block(X30g, "30")
    abl_in = {}
    for j, k in enumerate(NEW4):
        Xj = np.hstack([X26g, new_g[:, [j]]])
        b = loco_block(Xj, f"26+{k}")
        abl_in[f"add_{k}"] = {"auroc": b["auroc"], "bal_acc": b["bal_acc"],
                              "gain_vs_26": round(b["auroc"] - loco26["auroc"], 4)}
    drop_in = {}
    for j, k in enumerate(NEW4):
        cols = [c for c in range(30) if c != 26 + j]
        b = loco_block(X30g[:, cols], f"30-{k}")
        drop_in[f"drop_{k}"] = {"auroc": b["auroc"], "bal_acc": b["bal_acc"],
                                "loss_vs_30": round(loco30["auroc"] - b["auroc"], 4)}
    res["indomain_loco"] = {
        "f26": {k: v for k, v in loco26.items() if k != "proba"},
        "f30": {k: v for k, v in loco30.items() if k != "proba"},
        "gain_30_vs_26": round(loco30["auroc"] - loco26["auroc"], 4),
        "gain_30_vs_26_ci95_clip": group_boot_ci(
            gy, np.asarray(loco30["proba"]) - np.asarray(loco26["proba"]), g_clips),
        "ablation_add_one": abl_in,
        "ablation_drop_one": drop_in,
        "note": "C=0.3 class_weight=balanced; LOCO over 7 IndEgo clips",
    }

    # ---- 2. OOD adapted per-domain z-score, 26 vs 30 (+ablations) ----
    def z_block(Xg, Xo, name, extra=None):
        sg, so, se, signs, au = zscore_recipe(Xg, Xo, gy, extra=extra)
        out = {"features": name, "n": len(so),
               "indego_auroc": round(fast_auc(gy, sg), 4),
               "indego_ci95_clip": group_boot_ci(gy, sg, g_clips),
               "ood_auroc": round(fast_auc(oy, so), 4),
               "ood_ci95_grouped": group_boot_ci(oy, so, o_fac),
               "ood_bal_acc": round(balacc(oy, so, 0.0), 4),
               "ood_bal_acc_ci95_grouped": group_boot_ci(oy, so, o_fac, thr=0.0),
               "indego_signs": {n: int(s) for n, s in zip(name, signs)}}
        if se is not None:
            out["holdout_auroc"] = round(fast_auc(hy, se), 4)
            out["holdout_ci95_grouped"] = group_boot_ci(hy, se, h_fac)
            out["holdout_bal_acc"] = round(balacc(hy, se, 0.0), 4)
            out["holdout_bal_acc_ci95_grouped"] = group_boot_ci(hy, se, h_fac, thr=0.0)
        return out, so

    z26, so26 = z_block(X26g, X26o, FEATURE_NAMES26, extra=X26h)
    z30, so30 = z_block(X30g, X30o, NAMES30, extra=X30h)
    abl_ood = {}
    for j, k in enumerate(NEW4):
        Xj_g = np.hstack([X26g, new_g[:, [j]]])
        Xj_o = np.hstack([X26o, new_o[:, [j]]])
        Xj_h = np.hstack([X26h, new_h[:, [j]]])
        b, _ = z_block(Xj_g, Xj_o, f"26+{k}", extra=Xj_h)
        abl_ood[f"add_{k}"] = {"ood_auroc": b["ood_auroc"], "ood_bal_acc": b["ood_bal_acc"],
                               "holdout_auroc": b["holdout_auroc"],
                               "gain_ood_vs_26": round(b["ood_auroc"] - z26["ood_auroc"], 4)}
    drop_ood = {}
    for j, k in enumerate(NEW4):
        cols = [c for c in range(30) if c != 26 + j]
        b, _ = z_block(X30g[:, cols], X30o[:, cols], f"30-{k}", extra=X30h[:, cols])
        drop_ood[f"drop_{k}"] = {"ood_auroc": b["ood_auroc"], "ood_bal_acc": b["ood_bal_acc"],
                                 "holdout_auroc": b["holdout_auroc"],
                                 "loss_ood_vs_30": round(z30["ood_auroc"] - b["ood_auroc"], 4)}
    res["ood_adapted_zscore"] = {
        "f26": z26, "f30": z30,
        "gain_30_vs_26_ood": round(z30["ood_auroc"] - z26["ood_auroc"], 4),
        "gain_30_vs_26_ood_paired_grouped": paired_group_boot(oy, so30, so26, o_fac),
        "gain_30_vs_26_holdout": round(z30["holdout_auroc"] - z26["holdout_auroc"], 4),
        "ablation_add_one": abl_ood,
        "ablation_drop_one": drop_ood,
        "note": ("per-domain z-score (gold stats for gold, OOD-160 stats for OOD, "
                 "same OOD stats for the 108 hold-out), IndEgo sign, equal-weight mean"),
    }

    # ---- 3. frozen logistic (fit all IndEgo) ----
    def frozen_block(Xg, Xo, Xh, name):
        po = frozen_logistic(Xg, gy, Xo)
        ph = frozen_logistic(Xg, gy, Xh)
        return {"features": name,
                "ood_auroc": round(fast_auc(oy, po), 4),
                "ood_ci95_grouped": group_boot_ci(oy, po, o_fac),
                "ood_bal_acc": round(balacc(oy, po, 0.5), 4),
                "holdout_auroc": round(fast_auc(hy, ph), 4),
                "holdout_ci95_grouped": group_boot_ci(hy, ph, h_fac),
                "holdout_bal_acc": round(balacc(hy, ph, 0.5), 4)}, po

    fr26, po_fr26 = frozen_block(X26g, X26o, X26h, "26")
    fr30, po_fr30 = frozen_block(X30g, X30o, X30h, "30")
    res["ood_frozen_logistic"] = {
        "f26": fr26, "f30": fr30,
        "gain_30_vs_26_ood": round(fr30["ood_auroc"] - fr26["ood_auroc"], 4),
        "gain_30_vs_26_ood_paired_grouped": paired_group_boot(oy, po_fr30, po_fr26, o_fac),
        "gain_30_vs_26_holdout": round(fr30["holdout_auroc"] - fr26["holdout_auroc"], 4),
    }

    # ---- 4. baseline reproduction (all14 = stable8 + z3six) ----
    def subset(Xg, Xo, Xh, names, allnames):
        gi = [allnames.index(n) for n in names]
        return Xg[:, gi], Xo[:, gi], Xh[:, gi]
    s8g, s8o, s8h = subset(X26g, X26o, X26h, STABLE8, FEATURE_NAMES26)
    z6g, z6o, z6h = subset(Xz3g, Xz3o, Xz3h, Z3_SIX, zn)
    A14g, A14o, A14h = np.hstack([s8g, z6g]), np.hstack([s8o, z6o]), np.hstack([s8h, z6h])
    z14, _ = z_block(A14g, A14o, STABLE8 + Z3_SIX, extra=A14h)
    res["baseline_reproduce_all14_zscore"] = {
        "features": STABLE8 + Z3_SIX,
        "ood_auroc": z14["ood_auroc"], "ood_ci95_grouped": z14["ood_ci95_grouped"],
        "ood_bal_acc": z14["ood_bal_acc"],
        "holdout_auroc": z14["holdout_auroc"],
        "holdout_ci95_grouped": z14["holdout_ci95_grouped"],
        "holdout_bal_acc": z14["holdout_bal_acc"],
        "expected_pooled_ood": 0.781, "expected_holdout": 0.676,
    }

    # per-pixel-only transferable subset: stable8 + the 4 added invariants
    # (== all14 minus the two non-per-pixel Z3 features ct_fsize_std,
    #  mo_hand_motion_excess).
    X12g, X12o, X12h = (np.hstack([s8g, new_g]), np.hstack([s8o, new_o]),
                        np.hstack([s8h, new_h]))
    z12, _ = z_block(X12g, X12o, STABLE8 + NEW4, extra=X12h)
    res["adapted_zscore_stable8_plus_new4"] = z12
    # the 4 added invariants alone
    z4, _ = z_block(new_g, new_o, NEW4, extra=new_h)
    res["adapted_zscore_new4_only"] = z4

    # frozen ensemble (cited baseline)
    frozen = load_frozen160(o_ids)
    res["frozen_ensemble_v1_ood160"] = {
        "ood_auroc": round(fast_auc(oy, frozen), 4),
        "ood_ci95_grouped": group_boot_ci(oy, frozen, o_fac),
        "note": "frozen two-view ensemble usable_prob (no fitting); expected 0.566",
    }

    # paired gain of 30-adapted vs frozen
    res["ood_adapted_zscore"]["gain_30_vs_frozen_paired_grouped"] = paired_group_boot(
        oy, so30, frozen, o_fac)

    # ---- 5. per-feature OOD AUROC of the 4 new features (transfer sign) ----
    rows = []
    for j, k in enumerate(NEW4):
        ai = fast_auc(gy, new_g[:, j])
        ao = fast_auc(oy, new_o[:, j])
        ah = fast_auc(hy, new_h[:, j])
        rows.append({"feature": k, "indego_auroc": round(ai, 4),
                     "ood160_auroc": round(ao, 4),
                     "ood160_ci95_grouped": group_boot_ci(oy, new_o[:, j], o_fac),
                     "holdout108_auroc": round(ah, 4),
                     "sign_indego": "high-good" if ai >= 0.5 else "low-good",
                     "sign_ood": "high-good" if ao >= 0.5 else "low-good"})
    res["new_feature_transfer"] = rows

    res["head"]["wall_s"] = round(time.perf_counter() - t_start, 2)

    json.dump(res, open(os.path.join(OUT, "F30.01.json"), "w"), indent=1)

    # ---- stdout ----
    print(f"gold n={len(gy)} clips={len(set(g_clips.tolist()))} | "
          f"ood n={len(oy)} good={n_good} bad={n_bad} fac={nf} | holdout n={len(hy)}")
    print(f"IN-DOMAIN LOCO 26: AUROC {loco26['auroc']:.4f} {loco26['auroc_ci95_clip']} "
          f"BA {loco26['bal_acc']:.4f}")
    print(f"IN-DOMAIN LOCO 30: AUROC {loco30['auroc']:.4f} {loco30['auroc_ci95_clip']} "
          f"BA {loco30['bal_acc']:.4f}  gain {loco30['auroc']-loco26['auroc']:+.4f}")
    print("IN-DOMAIN add-one gains:", {k: v["gain_vs_26"] for k, v in abl_in.items()})
    print(f"OOD adapted-z 26: {z26['ood_auroc']:.4f} {z26['ood_ci95_grouped']} "
          f"BA {z26['ood_bal_acc']:.4f}")
    print(f"OOD adapted-z 30: {z30['ood_auroc']:.4f} {z30['ood_ci95_grouped']} "
          f"BA {z30['ood_bal_acc']:.4f}  gain {z30['ood_auroc']-z26['ood_auroc']:+.4f}")
    print("OOD add-one gains:", {k: v["gain_ood_vs_26"] for k, v in abl_ood.items()})
    print(f"HOLD-OUT 108 adapted-z 26: {z26['holdout_auroc']:.4f} | 30: {z30['holdout_auroc']:.4f}")
    print(f"baseline all14 zscore: ood {z14['ood_auroc']:.4f} holdout {z14['holdout_auroc']:.4f}")
    print(f"frozen ensemble ood: {res['frozen_ensemble_v1_ood160']['ood_auroc']:.4f}")
    print(f"wall {res['head']['wall_s']}s")


if __name__ == "__main__":
    main()
