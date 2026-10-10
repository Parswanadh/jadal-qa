#!/usr/bin/env python3
"""Z3 (new-features) re-run on the 160 OOD labels.

Same protocol as `eval/exp_features/run_eval.py` (Z3.01-Z3.03, PR #127), but the
out-of-domain set is the merged 160 human-labelled Egocentric-10K windows
(138 good / 22 bad, 47 factories) instead of the old 60 (51 good / 9 bad), and
the primary CI is the *factory-grouped* bootstrap (honest; windows within a
factory are correlated) with the window bootstrap kept for continuity.

Inputs (all read-only, precomputed by the O/Z agents):
  * $JADAL_BUNDLE/labels/ood160_labels.json           (160 human labels)
  * /workspace/wt/m-adapt160/.../inputs/z3_160.npz    (20 Z3 candidates, IndEgo+OOD160)
  * /workspace/wt/m-fewshot160/.../ood160_pixel.npz   (26 pixel feats, OOD160, 10fps)
  * $JADAL_BUNDLE/labels/manifest_gold.json + caches/gpu/features.json (IndEgo 150)
  * $JADAL_BUNDLE/caches/ensemble/ood_scores{,2}.json (frozen scores, 160)

Outputs eval/results/exp_features160/Z3.01_160.json, Z3.02_160.json, Z3.03_160.json.
CPU only, fixed seeds. Run: OMP_NUM_THREADS=2 python3 eval/exp_features160/run_z3_160.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np
from scipy.stats import rankdata

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(REPO, "eval", "results", "exp_features160")
BUNDLE = os.environ.get("JADAL_BUNDLE", "/workspace/jadal_pod_bundle")
Z3_160 = os.environ.get(
    "JADAL_Z3_160",
    "/workspace/wt/m-adapt160/eval/results/exp_adapt160/inputs/z3_160.npz")
PIXEL160 = os.environ.get(
    "JADAL_PIXEL160",
    "/workspace/wt/m-fewshot160/eval/results/exp_fewshot/inputs/ood160_pixel.npz")
DEV = "CPU (numpy/sklearn; decode done by the O/Z agents, no GPU here)"
N_BOOT = 2000
SEED = 0

STABLE8 = ["radial_slope", "hf_ratio", "dct_energy", "luma_p05", "global_shift",
           "illum_r", "illum_b", "zncc_self"]
FEATURE_NAMES26 = [
    "temp_b0", "temp_b1", "temp_b2", "temp_b3", "frozen_frac", "flicker_idx",
    "radial_slope", "hf_ratio", "dct_energy", "pca_resid", "pca_top1_share",
    "luma_mean", "luma_p05", "luma_p50", "luma_p95", "dark_frac", "bright_frac",
    "motion_energy", "global_shift", "skin_frac", "illum_r", "illum_g", "illum_b",
    "rppg_pulse", "dup_hash_mean", "zncc_self"]


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


def _group_index(groups):
    gs = list(groups)
    gmap = {g: i for i, g in enumerate(sorted(set(gs)))}
    gi = np.asarray([gmap[g] for g in gs])
    return gi, [np.where(gi == k)[0] for k in range(len(gmap))]


def group_boot_ci(y, s, groups, n_boot=N_BOOT, seed=SEED):
    y = np.asarray(y)
    s = np.asarray(s)
    _, idx_by = _group_index(groups)
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


def window_boot_ci(y, s, n_boot=N_BOOT, seed=SEED):
    y = np.asarray(y)
    s = np.asarray(s)
    rng = np.random.RandomState(seed)
    vals = []
    for _ in range(n_boot):
        idx = rng.randint(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        vals.append(fast_auc(y[idx], s[idx]))
    if not vals:
        return [None, None]
    return [round(float(np.percentile(vals, 2.5)), 4),
            round(float(np.percentile(vals, 97.5)), 4)]


def paired_group_boot(y, s_a, s_b, groups, n_boot=N_BOOT, seed=SEED):
    y = np.asarray(y)
    s_a = np.asarray(s_a, float)
    s_b = np.asarray(s_b, float)
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


def classify_transfer(a, b, tol=0.05):
    da, db = abs(a - 0.5), abs(b - 0.5)
    if max(da, db) < tol:
        return "chance"
    return "stable" if (a - 0.5) * (b - 0.5) > 0 else "flip"


# ---------------- sklearn helpers (verbatim protocol from Z3/eflib) ----------------

def make_clf(C=1.0, balanced=True):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=C, max_iter=3000, class_weight="balanced" if balanced else None))


def loco_cv(X, y, clips, C_grid=(0.01, 0.1, 1.0, 10.0), seed=0):
    X = np.asarray(X, float)
    y = np.asarray(y, int)
    clips = np.asarray(clips)
    oof = np.full(len(y), np.nan)
    for held in sorted(set(clips.tolist())):
        te = np.where(clips == held)[0]
        tr = np.where(clips != held)[0]
        best_c, best = C_grid[0], -1.0
        for c in C_grid:
            aucs = []
            tclips = clips[tr]
            for h2 in sorted(set(tclips.tolist())):
                itr = tr[tclips != h2]
                ite = tr[tclips == h2]
                if len(np.unique(y[ite])) < 2 or len(np.unique(y[itr])) < 2:
                    continue
                clf = make_clf(c)
                clf.fit(X[itr], y[itr])
                try:
                    aucs.append(fast_auc(y[ite], clf.predict_proba(X[ite])[:, 1]))
                except ValueError:
                    pass
            m = float(np.mean(aucs)) if aucs else 0.5
            if m > best:
                best, best_c = m, c
        clf = make_clf(best_c)
        clf.fit(X[tr], y[tr])
        oof[te] = clf.predict_proba(X[te])[:, 1]
    return oof


def grouped_cv(X, y, groups, C_grid=(0.01, 0.1, 1.0, 10.0), n_splits=5, seed=0):
    from sklearn.model_selection import StratifiedGroupKFold
    X = np.asarray(X, float)
    y = np.asarray(y, int)
    groups = np.asarray(groups)
    oof = np.full(len(y), np.nan)
    folds = []
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for k, (tr, te) in enumerate(sgkf.split(X, y, groups)):
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            folds.append({"fold": k, "skipped": True,
                          "n_train": int(len(tr)), "n_bad_train": int((y[tr] == 0).sum()),
                          "n_test": int(len(te)), "n_bad_test": int((y[te] == 0).sum())})
            continue
        best_c, best = C_grid[0], -1.0
        inner = StratifiedGroupKFold(n_splits=min(3, max(2, len(set(groups[tr].tolist())) // 2)),
                                     shuffle=True, random_state=seed + 1)
        for c in C_grid:
            aucs = []
            for itr, ite in inner.split(X[tr], y[tr], groups[tr]):
                if len(np.unique(y[tr][itr])) < 2 or len(np.unique(y[tr][ite])) < 2:
                    continue
                clf = make_clf(c)
                clf.fit(X[tr][itr], y[tr][itr])
                try:
                    aucs.append(fast_auc(y[tr][ite], clf.predict_proba(X[tr][ite])[:, 1]))
                except ValueError:
                    pass
            m = float(np.mean(aucs)) if aucs else 0.5
            if m > best:
                best, best_c = m, c
        clf = make_clf(best_c)
        clf.fit(X[tr], y[tr])
        oof[te] = clf.predict_proba(X[te])[:, 1]
        folds.append({"fold": k, "skipped": False, "C": best_c,
                      "n_train": int(len(tr)), "n_bad_train": int((y[tr] == 0).sum()),
                      "n_test": int(len(te)), "n_bad_test": int((y[te] == 0).sum())})
    return oof, folds


# ---------------- loaders ----------------

def load_labels160():
    d = json.load(open(os.path.join(BUNDLE, "labels", "ood160_labels.json")))
    w = d["windows"]
    return {"ids": [x["segment_id"] for x in w],
            "y": np.asarray([1 if x["verdict"] == "good" else 0 for x in w], int),
            "batch": np.asarray([x["batch"] for x in w]),
            "factory": np.asarray([x["factory"] for x in w]),
            "n_good": d["n_good"], "n_bad": d["n_bad"], "n": d["n"]}


def load_z3_160():
    z = np.load(Z3_160, allow_pickle=True)
    return {"ids": [str(x) for x in z["ids"]], "y": z["y"].astype(int),
            "batch": np.asarray([str(x) for x in z["batch"]]),
            "factory": np.asarray([str(x) for x in z["factory"]]),
            "names": [str(x) for x in z["names"]], "X": z["X"].astype(float),
            "X_indego": z["X_indego"].astype(float), "y_indego": z["y_indego"].astype(int),
            "clips_indego": np.asarray([str(x) for x in z["clips_indego"]])}


def load_pixel160():
    z = np.load(PIXEL160, allow_pickle=True)
    return {"ids": [str(x) for x in z["ids"]], "names": [str(x) for x in z["names"]],
            "X10": z["X10"].astype(float)}


def load_gold():
    m = json.load(open(os.path.join(BUNDLE, "labels", "manifest_gold.json")))
    items = m["items"]
    ids = [it["id"] for it in items]
    clips = np.asarray([it["clip"] for it in items])
    y = np.asarray([int(it["label"]) for it in items])
    fj = json.load(open(os.path.join(BUNDLE, "caches", "gpu", "features.json")))["windows"]
    X = np.asarray([[float(fj[i]["full"][n]) for n in FEATURE_NAMES26] for i in ids])
    return {"ids": ids, "clips": clips, "y": y, "X26": X,
            "labels_sha256": m.get("labels_sha256")}


def load_frozen160(ids):
    sc = {}
    for fn in ("ood_scores.json", "ood_scores2.json"):
        d = json.load(open(os.path.join(BUNDLE, "caches", "ensemble", fn)))
        for x in d["scores"]:
            if "error" not in x:
                sc[x["id"]] = float(x["usable_prob"])
    missing = [i for i in ids if i not in sc]
    return np.asarray([sc[i] for i in ids], float), missing


def align(ids_src, X_src, ids_tgt):
    m = {i: k for k, i in enumerate(ids_src)}
    return np.asarray([X_src[m[i]] for i in ids_tgt], float)


# ---------------- combos ----------------

def zscore_combo(Ax, Bx, y_in):
    au = np.asarray([fast_auc(y_in, Ax[:, j]) for j in range(Ax.shape[1])])
    sg = np.where(au >= 0.5, 1.0, -1.0)
    za = (Ax - Ax.mean(0)) / np.clip(Ax.std(0), 1e-9, None)
    zb = (Bx - Bx.mean(0)) / np.clip(Bx.std(0), 1e-9, None)
    return (za * sg).mean(1), (zb * sg).mean(1), au, sg


def main():
    os.makedirs(OUT, exist_ok=True)
    lab = load_labels160()
    z3 = load_z3_160()
    pix = load_pixel160()
    gold = load_gold()
    sha = git_sha()

    ids, y, batch, fac = lab["ids"], lab["y"], lab["batch"], lab["factory"]
    assert list(z3["ids"]) == ids, "z3_160 ids must align with labels160"
    assert list(pix["ids"]) == ids, "ood160_pixel ids must align with labels160"
    n_good, n_bad = int((y == 1).sum()), int((y == 0).sum())
    n_fac = len(set(fac.tolist()))
    names = z3["names"]
    Xi, yi, ci = z3["X_indego"], z3["y_indego"], z3["clips_indego"]
    Xo = z3["X"]
    X26o = align(pix["ids"], pix["X10"], ids)
    X26g = gold["X26"]
    yg = gold["y"]

    head = {
        "n": len(ids), "n_good": n_good, "n_bad": n_bad, "n_factories": n_fac,
        "n_indego": int(len(yi)), "n_good_indego": int((yi == 1).sum()),
        "n_bad_indego": int((yi == 0).sum()),
        "label_source": ("human (IndEgo manifest_gold n=150 + Egocentric-10K "
                         "ood160_labels.json n=160: 138 good / 22 bad, 47 factories; "
                         "merge of q1=60 + q2=100, single labeller)"),
        "protocol": ("signed AUROC per feature, label 1 = usable/good; IndEgo CI = "
                     "clip-bootstrap (7 clips); OOD-160 CI = FACTORY-GROUPED bootstrap "
                     "(honest, n_boot=2000, seed=0), window-bootstrap kept as the "
                     "optimistic comparison. E5.03 class rule."),
        "supersedes": {"branch": "m/new-features", "pr": 127, "n_ood": 60},
        "device": DEV, "git_sha": sha,
        "command": "OMP_NUM_THREADS=2 python3 eval/exp_features160/run_z3_160.py",
    }

    # ---------------- Z3.01 per-feature table ----------------
    def feat_row(name, xg, xo):
        a_i = round(fast_auc(yg, xg), 4)
        a_o = round(fast_auc(y, xo), 4)
        return {"feature": name, "indego_auroc": a_i,
                "indego_ci95": group_boot_ci(yg, xg, ci),
                "ood160_auroc": a_o,
                "ood160_ci95_factory_grouped": group_boot_ci(y, xo, fac),
                "ood160_ci95_window_optimistic": window_boot_ci(y, xo),
                "class_160": classify_transfer(a_i, a_o),
                "indego_sign": "high-usable" if a_i >= 0.5 else "low-usable",
                "ood_sign": "high-usable" if a_o >= 0.5 else "low-usable"}
    rows = [feat_row(n, Xi[:, j], Xo[:, j]) for j, n in enumerate(names)]
    counts = {}
    for r in rows:
        counts[r["class_160"]] = counts.get(r["class_160"], 0) + 1
    z301 = {**head, "experiment": "Z3.01_160",
            "title": "Z3 candidate transfer table on OOD-160",
            "counts_160": counts, "table": rows,
            "counts_60_old": {"stable": 9, "flip": 11, "chance": 0}}
    json.dump(z301, open(os.path.join(OUT, "Z3.01_160.json"), "w"), indent=1)

    # ---------------- Z3.02 family verdicts ----------------
    by = {r["feature"]: r for r in rows}
    families = {
        "compression_telemetry": ["ct_bitrate", "ct_fsize_mean", "ct_fsize_cv",
                                  "ct_iframe_ratio", "ct_fsize_entropy", "ct_iframe_frac",
                                  "ct_fsize_std"],
        "affine_ego_motion": ["mo_trans_resid", "mo_affine_resid", "mo_affine_gain",
                              "mo_hand_trans_resid", "mo_hand_affine_resid",
                              "mo_hand_motion_excess", "mo_motion_frac"],
        "invariants": ["inv_dct_hf_lf", "inv_noise_snr", "inv_clip_frac",
                       "inv_hist_entropy", "inv_dhash_dist", "inv_laplacian_var"],
    }
    famrep = {}
    for f, feats in families.items():
        cls = [by[x]["class_160"] for x in feats]
        famrep[f] = {"n": len(feats), "counts": {c: cls.count(c) for c in set(cls)},
                     "stable_features": [x for x in feats if by[x]["class_160"] == "stable"],
                     "flip_features": [x for x in feats if by[x]["class_160"] == "flip"]}
    z302 = {**head, "experiment": "Z3.02_160",
            "title": "Z3 family verdicts on OOD-160", "families": famrep,
            "verdicts_160": {
                "compression": "relative frame-size variability (ct_fsize_std) stable; "
                               "absolute bitrate/frame-size flip (as at n=60)",
                "affine": "raw motion residuals (translation-only and full-affine, global "
                          "and hand-restricted) flip; only hand-region motion EXCESS is stable",
                "invariants": "exposure-clipping / Laplacian sharpness / histogram entropy "
                              "stable; noise-SNR and dHash flip"}}
    json.dump(z302, open(os.path.join(OUT, "Z3.02_160.json"), "w"), indent=1)

    # ---------------- Z3.03 combined sets ----------------
    n26 = FEATURE_NAMES26
    s8_idx = [n26.index(s) for s in STABLE8]
    transfer_cols = [j for j, n in enumerate(names)
                     if by[n]["class_160"] == "stable"
                     and min(abs(by[n]["indego_auroc"] - 0.5),
                             abs(by[n]["ood160_auroc"] - 0.5)) >= 0.03]
    transfer_names = [names[j] for j in transfer_cols]

    X8g, X8o = X26g[:, s8_idx], X26o[:, s8_idx]
    XTg, XTo = Xi[:, transfer_cols], Xo[:, transfer_cols]
    XCg = np.hstack([X8g, XTg])
    XCo = np.hstack([X8o, XTo])

    def combined_eval(Ax, Bx, cols, label_names):
        oof = loco_cv(Ax, yi, ci)
        a_i = fast_auc(yi, oof)
        clf = make_clf(1.0)
        clf.fit(Ax, yi)
        po = clf.predict_proba(Bx)[:, 1]
        a_o = fast_auc(y, po)
        oof_o, folds = grouped_cv(Bx, y, fac)
        a_of = fast_auc(y, oof_o)
        return {"n_features": len(cols), "features": label_names,
                "indego_loco_auroc": round(a_i, 4),
                "indego_loco_ci95": group_boot_ci(yi, oof, ci),
                "ood160_zeroshot_auroc": round(a_o, 4),
                "ood160_zeroshot_ci95_factory_grouped": group_boot_ci(y, po, fac),
                "ood160_fewshot_groupcv_auroc": round(a_of, 4),
                "ood160_fewshot_groupcv_ci95_factory_grouped": group_boot_ci(y, oof_o, fac),
                "fold_min_n_bad_train": min(f["n_bad_train"] for f in folds
                                            if not f["skipped"]) if folds else None}

    sets = {
        "in_domain26": (X26g, X26o, list(range(26)), n26),
        "stable8_only": (X8g, X8o, s8_idx, STABLE8),
        "transferable_only": (XTg, XTo, transfer_cols, transfer_names),
        "stable8+transferable": (XCg, XCo, list(range(XCg.shape[1])),
                                 STABLE8 + transfer_names),
    }
    combo = {}
    for k, (Ax, Bx, cols, ln) in sets.items():
        if Ax.shape[1] == 0:
            continue
        combo[k] = combined_eval(Ax, Bx, cols, ln)

    # z-score variants
    def zrow(Ax, Bx, label_names):
        si, so, au, sg = zscore_combo(Ax, Bx, yi)
        return {"n_features": Ax.shape[1], "features": label_names,
                "indego_auroc": round(fast_auc(yi, si), 4),
                "indego_ci95": group_boot_ci(yi, si, ci),
                "ood160_auroc": round(fast_auc(y, so), 4),
                "ood160_ci95_factory_grouped": group_boot_ci(y, so, fac),
                "ood160_ci95_window_optimistic": window_boot_ci(y, so),
                "method": "per-domain z-score, sign=IndEgo, equal-weight mean "
                          "(unsupervised but for the per-feature sign)",
                "indego_feature_auroc": {n: round(float(a), 4) for n, a in zip(label_names, au)},
                "indego_signs": {n: int(s) for n, s in zip(label_names, sg)}}
    combo["zscore_transferable_only"] = zrow(XTg, XTo, transfer_names)
    combo["zscore_stable8+transferable"] = zrow(XCg, XCo, STABLE8 + transfer_names)

    # frozen baseline
    frozen, frozen_missing = load_frozen160(ids)
    combo["frozen_ensemble_v1"] = {
        "n_features": None, "features": None,
        "ood160_auroc": round(fast_auc(y, frozen), 4),
        "ood160_ci95_factory_grouped": group_boot_ci(y, frozen, fac),
        "ood160_ci95_window_optimistic": window_boot_ci(y, frozen),
        "missing_scores": frozen_missing,
        "note": "frozen two-view usable_prob (no fitting)"}

    # paired grouped gain vs frozen for the z-score combos
    _, so6, _, _ = zscore_combo(XTg, XTo, yi)
    _, so14, _, _ = zscore_combo(XCg, XCo, yi)
    gains = {
        "zscore_transferable_only": paired_group_boot(y, so6, frozen, fac),
        "zscore_stable8+transferable": paired_group_boot(y, so14, frozen, fac),
    }

    z303 = {**head, "experiment": "Z3.03_160",
            "title": "Z3 combined transferable sets on OOD-160",
            "stable8": STABLE8,
            "transferable_subset_160": transfer_names,
            "selection_rule": "candidate class==stable AND min(|AUROC-0.5|) >= 0.03 in both domains",
            "combined": combo,
            "paired_factory_grouped_gain_vs_frozen": gains}
    json.dump(z303, open(os.path.join(OUT, "Z3.03_160.json"), "w"), indent=1)

    # ---------------- stdout ----------------
    print(f"n={len(ids)} good={n_good} bad={n_bad} factories={n_fac} "
          f"(old n_ood=60, n_bad=9)")
    print(f"{'feature':<24}{'IndEgo':>8}{'OOD160':>8}  class160")
    for r in rows:
        print(f"{r['feature']:<24}{r['indego_auroc']:>8.3f}{r['ood160_auroc']:>8.3f}"
              f"  {r['class_160']}")
    print("counts_160:", counts, "(old 60: 9 stable / 11 flip / 0 chance)")
    print("transferable subset 160:", transfer_names)
    print(f"{'recipe':<30}{'IndEgo':>8}{'OOD0':>8}{'OODfew':>9}{'grpCI':>18}")
    for k, v in combo.items():
        ai = v.get("indego_loco_auroc", v.get("indego_auroc"))
        ao = v.get("ood160_zeroshot_auroc", v.get("ood160_auroc"))
        af = v.get("ood160_fewshot_groupcv_auroc", float("nan"))
        ci = v.get("ood160_ci95_factory_grouped")
        print(f"{k:<30}{(ai if ai is not None else float('nan')):>8.3f}{ao:>8.3f}"
              f"{(af if af == af else float('nan')):>9.3f}{str(ci):>18}")
    print("paired gain vs frozen:", gains)


if __name__ == "__main__":
    main()
