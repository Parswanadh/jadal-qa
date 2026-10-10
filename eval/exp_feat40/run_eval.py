"""F40 evaluation: 26 -> 40 per-pixel features.

In-domain: leave-one-clip-out (LOCO) logistic (C=0.3, balanced) over the 7 IndEgo
clips + clip-bootstrap 95% CI, AUROC + balanced accuracy @ 0.5.

OOD: the adapted per-domain-z-score recipe (O4): per-feature sign from the 150
IndEgo gold windows, unsupervised per-domain z-score, equal-weight mean,
threshold 0.0, factory-grouped bootstrap 95% CI. Sets: the frozen baseline
(stable-8 + Z3-6), the 26, the 40, and the expanded transferable set.

Hold-out 108: scored ONCE, no tuning (same recipe).

Ablation: leave-one-feature-out over the 14 added features (in-domain LOCO +
adapted recipe).

CPU only. Reads caches + feat40_new8_*.npz. Writes eval/results/exp_feat40/*.json
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)
BUNDLE = os.environ.get("JADAL_BUNDLE", "/workspace/jadal_pod_bundle")
OUT = os.path.join(REPO, "eval", "results", "exp_feat40")
SEED, N_BOOT = 0, 2000

F26 = ["temp_b0", "temp_b1", "temp_b2", "temp_b3", "frozen_frac", "flicker_idx",
       "radial_slope", "hf_ratio", "dct_energy", "pca_resid", "pca_top1_share",
       "luma_mean", "luma_p05", "luma_p50", "luma_p95", "dark_frac", "bright_frac",
       "motion_energy", "global_shift", "skin_frac", "illum_r", "illum_g", "illum_b",
       "rppg_pulse", "dup_hash_mean", "zncc_self"]
Z3_6 = ["ct_fsize_std", "mo_hand_motion_excess", "inv_clip_frac",
        "inv_laplacian_var", "inv_hist_entropy", "inv_dct_hf_lf"]
STABLE8 = ["radial_slope", "hf_ratio", "dct_energy", "luma_p05", "global_shift",
           "illum_r", "illum_b", "zncc_self"]
FRESH8 = ["spec_b1", "spec_b2", "spec_b3", "col_rg_ratio", "col_bg_ratio",
          "temp_lag2", "temp_lag3", "noise_floor"]
NEW14 = Z3_6 + FRESH8
F40 = F26 + NEW14


# ---------------- metrics ----------------
def fast_auc(y, s):
    y = np.asarray(y); s = np.asarray(s, float)
    n_pos = int((y == 1).sum()); n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    r = rankdata(s)
    return float((r[y == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def balacc(y, s, thr=0.5):
    y = np.asarray(y); pred = (np.asarray(s) > thr).astype(int)
    g, b = y == 1, y == 0
    tpr = float((pred[g] == 1).mean()) if g.any() else float("nan")
    tnr = float((pred[b] == 0).mean()) if b.any() else float("nan")
    return 0.5 * (tpr + tnr)


def group_boot_ci(y, s, groups, fn, n_boot=N_BOOT, seed=SEED):
    y = np.asarray(y); s = np.asarray(s)
    gs = sorted(set(groups)); idx = [np.where(np.asarray(groups) == g)[0] for g in gs]
    rng = np.random.RandomState(seed); vals = []
    for _ in range(n_boot):
        pick = rng.randint(0, len(idx), len(idx))
        ii = np.concatenate([idx[k] for k in pick])
        if len(np.unique(y[ii])) < 2:
            continue
        vals.append(fn(y[ii], s[ii]))
    if not vals:
        return [None, None]
    return [round(float(np.percentile(vals, 2.5)), 4),
            round(float(np.percentile(vals, 97.5)), 4)]


def paired_group_boot(y, s_a, s_b, groups, n_boot=N_BOOT, seed=SEED):
    y = np.asarray(y); gs = sorted(set(groups))
    idx = [np.where(np.asarray(groups) == g)[0] for g in gs]
    rng = np.random.RandomState(seed); d = []
    for _ in range(n_boot):
        pick = rng.randint(0, len(idx), len(idx))
        ii = np.concatenate([idx[k] for k in pick])
        if len(np.unique(y[ii])) < 2:
            continue
        d.append(fast_auc(y[ii], np.asarray(s_a)[ii]) - fast_auc(y[ii], np.asarray(s_b)[ii]))
    d = np.asarray(d)
    if len(d) == 0:
        return {}
    return {"mean": round(float(d.mean()), 4),
            "ci95": [round(float(np.percentile(d, 2.5)), 4),
                     round(float(np.percentile(d, 97.5)), 4)],
            "frac_gt_0": round(float((d > 0).mean()), 4), "n_boot": int(len(d))}


def indego_signs(Xg, yg):
    au = np.asarray([fast_auc(yg, Xg[:, j]) for j in range(Xg.shape[1])])
    return np.where(au >= 0.5, 1.0, -1.0), au


def zscore(X):
    return (X - X.mean(0)) / np.clip(X.std(0), 1e-9, None)


# ---------------- data ----------------
def load_all():
    gold = json.load(open(os.path.join(BUNDLE, "labels", "manifest_gold.json")))["items"]
    gids = [it["id"] for it in gold]
    gy = np.asarray([int(it["label"]) for it in gold])
    gclip = np.asarray([it["clip"] for it in gold])
    fj = json.load(open(os.path.join(BUNDLE, "caches", "gpu", "features.json")))["windows"]
    gX26 = np.asarray([[float(fj[i]["full"][n]) for n in F26] for i in gids])
    z = np.load(os.path.join("/workspace/wt/m-adapt160/eval/results/exp_adapt160",
                             "inputs", "z3_160.npz"), allow_pickle=True)
    gz3 = np.asarray(z["X_indego"], float)
    gz3n = [str(x) for x in z["names"]]

    o = json.load(open(os.path.join(BUNDLE, "labels", "ood160_labels.json")))["windows"]
    oids = [x["segment_id"] for x in o]
    oy = np.asarray([1 if x["verdict"] == "good" else 0 for x in o])
    ofac = np.asarray([x["factory"] for x in o])
    p = np.load(os.path.join("/workspace/wt/m-fewshot160/eval/results/exp_fewshot",
                             "inputs", "ood160_pixel.npz"), allow_pickle=True)
    assert list(p["ids"]) == oids
    oX26 = np.asarray(p["X10"], float)
    assert list(z["ids"]) == oids
    oz3 = np.asarray(z["X"], float)

    c = np.load(os.path.join("/workspace/wt/m-confirm/eval/results/exp_confirm",
                             "inputs", "confirm108_features.npz"), allow_pickle=True)
    hids = [str(x) for x in c["ids"]]
    hy = np.asarray(c["y"], int)
    hfac = np.asarray([str(x) for x in c["factory"]])
    hX26 = np.asarray(c["Xpx"], float)
    hz3 = np.asarray(c["Xz3"], float)
    hz3n = [str(x) for x in c["z3_names"]]
    assert list(c["pix_names"]) == F26

    # fresh 8 (computed by run_features40.py)
    def fresh(name):
        d = np.load(os.path.join(OUT, f"feat40_new8_{name}.npz"), allow_pickle=True)
        names = [str(x) for x in d["names"]]
        assert names == FRESH8, names
        ids = [str(x) for x in d["ids"]]
        return ids, np.asarray(d["X"], float)

    gfi, gX8 = fresh("gold"); ofi, oX8 = fresh("ood"); hfi, hX8 = fresh("holdout")

    def reindex(fids, X, target):
        m = {i: k for k, i in enumerate(fids)}
        return np.asarray([X[m[i]] for i in target], float)

    gX8 = reindex(gfi, gX8, gids)
    oX8 = reindex(ofi, oX8, oids)
    hX8 = reindex(hfi, hX8, hids)

    return {
        "gold": {"ids": gids, "y": gy, "clip": gclip,
                 "X26": gX26, "z3": gz3, "z3n": gz3n, "X8": gX8},
        "ood": {"ids": oids, "y": oy, "fac": ofac,
                "X26": oX26, "z3": oz3, "z3n": [str(x) for x in z["names"]], "X8": oX8},
        "holdout": {"ids": hids, "y": hy, "fac": hfac,
                    "X26": hX26, "z3": hz3, "z3n": hz3n, "X8": hX8},
    }


def mat(dset, names):
    """Assemble a feature matrix for the given ordered names."""
    cols = []
    zmap = {n: dset["z3"][:, dset["z3n"].index(n)] for n in Z3_6}
    x8map = {n: dset["X8"][:, FRESH8.index(n)] for n in FRESH8}
    x26map = {n: dset["X26"][:, F26.index(n)] for n in F26}
    for n in names:
        if n in x26map:
            cols.append(x26map[n])
        elif n in zmap:
            cols.append(zmap[n])
        elif n in x8map:
            cols.append(x8map[n])
        else:
            raise KeyError(n)
    return np.stack(cols, axis=1)


# ---------------- in-domain LOCO ----------------
def loco_auc(gold, names):
    X = mat(gold, names)
    y, clip = gold["y"], gold["clip"]
    p = np.zeros(len(y))
    for c in set(clip):
        tr, te = clip != c, clip == c
        sc = StandardScaler().fit(X[tr])
        m = LogisticRegression(max_iter=3000, C=0.3,
                               class_weight="balanced").fit(sc.transform(X[tr]), y[tr])
        p[te] = m.predict_proba(sc.transform(X[te]))[:, 1]
    auc = fast_auc(y, p)
    ci = group_boot_ci(y, p, clip, fast_auc)
    per_clip = {c: (round(fast_auc(y[clip == c], p[clip == c]), 3)
                    if 0 < y[clip == c].sum() < (clip == c).sum() else None)
                for c in sorted(set(clip))}
    return {"auroc": round(auc, 4), "ci95": ci, "ba": round(balacc(y, p, 0.5), 4),
            "per_clip": per_clip, "n": int(len(y)), "n_bad": int((y == 0).sum()),
            "probs": p.tolist()}


# ---------------- adapted recipe ----------------
def adapted(gold, tgt, names):
    Xg, Xo = mat(gold, names), mat(tgt, names)
    signs, _ = indego_signs(Xg, gold["y"])
    score = (zscore(Xo) * signs).mean(axis=1)
    auc = fast_auc(tgt["y"], score)
    ci = group_boot_ci(tgt["y"], score, tgt["fac"], fast_auc)
    return {"auroc": round(auc, 4), "ci95": ci,
            "ba": round(balacc(tgt["y"], score, 0.0), 4),
            "n": int(len(tgt["y"])), "n_bad": int((tgt["y"] == 0).sum()),
            "n_groups": int(len(set(tgt["fac"]))),
            "score": score.tolist(), "signs": signs.tolist()}


def main():
    os.makedirs(OUT, exist_ok=True)
    d = load_all()
    res = {"protocol": {
        "in_domain": "LOCO over 7 IndEgo clips, logistic C=0.3 class_weight=balanced, "
                     "per-fold StandardScaler; clip-bootstrap 95% CI (2000)",
        "ood": "adapted per-domain-z-score: IndEgo signs, unsupervised target z-score, "
               "equal-weight mean, threshold 0.0; factory-grouped bootstrap 95% CI (2000)",
        "holdout": "round-3 108, scored ONCE, no tuning",
        "label_sources": {"gold": "human (manifest_gold)", "ood": "human (ood160_labels)",
                          "holdout": "human (round3)"},
        "seed": SEED, "n_boot": N_BOOT},
        "sets": {}}

    # in-domain
    res["sets"]["in_domain_26"] = loco_auc(d["gold"], F26)
    res["sets"]["in_domain_40"] = loco_auc(d["gold"], F40)

    # adapted recipes
    recipe_sets = {
        "frozen_all14(stable8+z3_6)": STABLE8 + Z3_6,
        "all26": F26,
        "all40": F40,
        "expanded22(stable8+z3_6+new8)": STABLE8 + Z3_6 + FRESH8,
    }
    for tgt_name in ("ood", "holdout"):
        tgt = d[tgt_name]
        for rname, names in recipe_sets.items():
            res["sets"][f"{tgt_name}_{rname}"] = adapted(d["gold"], tgt, names)

    # ablation: leave-one-feature-out over the 14 added features (in-domain LOCO)
    base40 = res["sets"]["in_domain_40"]["auroc"]
    abl = {}
    for n in NEW14:
        names = [x for x in F40 if x != n]
        a = loco_auc(d["gold"], names)["auroc"]
        abl[n] = {"auroc_without": a, "delta_vs_40": round(base40 - a, 4)}
    # groups
    groups = {"z3_6": Z3_6, "fresh8": FRESH8,
              "spectral3": ["spec_b1", "spec_b2", "spec_b3"],
              "colour2": ["col_rg_ratio", "col_bg_ratio"],
              "temporal2": ["temp_lag2", "temp_lag3"],
              "noise1": ["noise_floor"]}
    for gname, g in groups.items():
        names = [x for x in F40 if x not in g]
        a = loco_auc(d["gold"], names)["auroc"]
        abl[f"DROP_{gname}"] = {"auroc_without": a, "delta_vs_40": round(base40 - a, 4)}
    res["ablation_in_domain"] = abl

    # ablation: adapted recipe on OOD, drop each new feature from all40
    ood_base = res["sets"]["ood_all40"]["auroc"]
    oabl = {}
    for n in NEW14:
        names = [x for x in F40 if x != n]
        a = adapted(d["gold"], d["ood"], names)["auroc"]
        oabl[n] = {"auroc_without": a, "delta_vs_40": round(ood_base - a, 4)}
    res["ablation_ood_all40"] = oabl

    # paired delta 40 vs 26 (OOD, adapted)
    s40 = np.asarray(res["sets"]["ood_all40"]["score"])
    s26 = np.asarray(res["sets"]["ood_all26"]["score"])
    res["paired_ood_40_minus_26"] = paired_group_boot(d["ood"]["y"], s40, s26, d["ood"]["fac"])

    # Z3 fresh-vs-cache consistency is documented separately (extractor computes all 14);
    # here we record which sources were used.
    res["feature_sets"] = {"F26": F26, "Z3_6": Z3_6, "STABLE8": STABLE8,
                           "FRESH8": FRESH8, "NEW14": NEW14, "F40": F40}
    try:
        res["extract_cost"] = json.load(open(os.path.join(OUT, "extract_cost.json")))
    except FileNotFoundError:
        res["extract_cost"] = None
    json.dump(res, open(os.path.join(OUT, "eval_feat40.json"), "w"), indent=2)
    print("wrote", os.path.join(OUT, "eval_feat40.json"))
    for k in ("in_domain_26", "in_domain_40", "ood_frozen_all14(stable8+z3_6)",
              "ood_all40", "holdout_frozen_all14(stable8+z3_6)", "holdout_all40"):
        v = res["sets"][k]
        print(f"  {k:38s} AUROC {v['auroc']:.4f} {v['ci95']} BA {v['ba']:.4f}")


if __name__ == "__main__":
    main()
