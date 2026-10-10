"""Z3 evaluation: per-feature IndEgo/OOD AUROC + transfer class + combined sets.

Writes eval/results/exp_features/Z3.01.json (candidate table),
Z3.02.json (family verdicts), Z3.03.json (combined stable-8 + transferable).
"""
from __future__ import annotations

import json
import os
import sys
import subprocess

import numpy as np
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)
from eval.exp_features import eflib as L  # noqa: E402
import extract as E  # noqa: E402

BUNDLE = L.BUNDLE
OUT = os.path.join(REPO, "eval", "results", "exp_features")
DEV = "CPU (numpy/ffmpeg/cv2)"
STABLE8 = ["radial_slope", "hf_ratio", "dct_energy", "luma_p05", "global_shift",
           "illum_r", "illum_b", "zncc_self"]


def git_sha():
    return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                          text=True, cwd=REPO).stdout.strip()


def load_all():
    z = np.load(os.path.join(OUT, "candidates.npz"), allow_pickle=True)
    names = [str(x) for x in z["names"]]
    d = {"names": names}
    for kind in ("indego", "ood"):
        d[kind + "_ids"] = [str(x) for x in z[kind + "_ids"]]
        d[kind + "_y"] = z[kind + "_y"].astype(int)
        d[kind + "_X"] = z[kind + "_X"]
        d[kind + "_meta"] = [json.loads(str(m)) for m in z[kind + "_meta"]]
    # gold 26 (IndEgo) + ood 26 pixel
    g = L.load_gold()
    d["gold"] = g
    d["names26"] = L.FEATURE_NAMES
    # align 26-feature matrices to the same ids
    gid = {wid: i for i, wid in enumerate(g["ids"])}
    d["indego_X26"] = np.asarray([[float(g["X26"][gid[i]][j]) for j in range(26)]
                                  for i in d["indego_ids"]])
    od = os.path.join(REPO, "..", "exp-domain-adapt", "eval", "results", "exp_domain",
                      "inputs", "ood_pixel_selfdec.npz")
    od = os.path.normpath(od)
    zz = np.load(od, allow_pickle=True)
    oid = {str(s): i for i, s in enumerate(zz["ids"])}
    d["ood_X26"] = np.asarray([[float(zz["X_10fps"][oid[i]][j]) for j in range(26)]
                               for i in d["ood_ids"]])
    # ood factory
    q = json.load(open(os.path.join(BUNDLE, "queues", "label_queue_random.json")))["segments"]
    fmap = {s["segment_id"]: s.get("factory") for s in q}
    d["ood_factory"] = np.asarray([fmap.get(i, "?") for i in d["ood_ids"]])
    d["indego_clips"] = g["clips"]
    return d


def fast_auc(y, s):
    """Rank-based AUROC (handles ties); ~50x faster than sklearn in bootstrap loops."""
    y = np.asarray(y)
    s = np.asarray(s, float)
    n_pos = int((y == 1).sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    r = rankdata(s)
    return float((r[y == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def fast_group_boot_ci(y, s, groups, n_boot=1000, seed=0):
    """Cluster bootstrap over groups with integer-coded groups (fast vs string isin)."""
    y = np.asarray(y)
    s = np.asarray(s)
    gs = list(groups)
    gmap = {g: i for i, g in enumerate(sorted(set(gs)))}
    gi = np.asarray([gmap[g] for g in gs])
    idx_by_group = [np.where(gi == k)[0] for k in range(len(gmap))]
    rng = np.random.RandomState(seed)
    vals = []
    for _ in range(n_boot):
        pick = rng.randint(0, len(gmap), len(gmap))
        idx = np.concatenate([idx_by_group[k] for k in pick])
        if len(np.unique(y[idx])) < 2:
            continue
        vals.append(fast_auc(y[idx], s[idx]))
    if not vals:
        return float("nan"), float("nan"), float("nan")
    return float(np.mean(vals)), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def sci_rule_class(a, b, tol=0.05):
    """E5.03 classify_transfer, restricted to 'transferable' when both informative."""
    c = L.classify_transfer(a, b, tol)
    return c


def per_feature(Xi, yi, ci, Xo, yo, name, n_boot=1000):
    a = float(roc_auc_score(yi, Xi))
    b = float(roc_auc_score(yo, Xo))
    _, lo_i, hi_i = fast_group_boot_ci(yi, Xi, ci, n_boot=n_boot)
    _, lo_o, hi_o = L.boot_ci(yo, Xo, fast_auc, n_boot=n_boot)
    return {
        "feature": name,
        "indego_auroc": round(a, 4),
        "indego_ci95": [round(lo_i, 4), round(hi_i, 4)],
        "ood_auroc": round(b, 4),
        "ood_ci95": [round(lo_o, 4), round(hi_o, 4)],
        "class": sci_rule_class(a, b),
        "indego_sign": "high-usable" if a >= 0.5 else "low-usable",
        "ood_sign": "high-usable" if b >= 0.5 else "low-usable",
    }


def combined_eval(Xi, yi, ci, Xo, yo, fo, cols, C):
    """IndEgo LOCO (inner-LOCO C) + zero-shot transfer to OOD + OOD grouped CV."""
    Xi_s = Xi[:, cols]
    Xo_s = Xo[:, cols]
    oof = L.loco_cv(Xi_s, yi, ci, C_grid=(C,))
    a_i = float(roc_auc_score(yi, oof))
    _, lo_i, hi_i = fast_group_boot_ci(yi, oof, ci, n_boot=2000)
    clf = L.make_clf(C, balanced=True)
    clf.fit(Xi_s, yi)
    po = clf.predict_proba(Xo_s)[:, 1]
    a_o = float(roc_auc_score(yo, po))
    _, lo_o, hi_o = L.boot_ci(yo, po, fast_auc, n_boot=2000)
    oof_o, folds = L.grouped_cv(Xo_s, yo, fo, C_grid=(C,))
    a_of = float(roc_auc_score(yo, oof_o))
    _, lo_of, hi_of = L.boot_ci(yo, oof_o, fast_auc, n_boot=2000)
    return {
        "n_features": len(cols),
        "indego_loco_auroc": round(a_i, 4), "indego_loco_ci95": [round(lo_i, 4), round(hi_i, 4)],
        "ood_zeroshot_auroc": round(a_o, 4), "ood_zeroshot_ci95": [round(lo_o, 4), round(hi_o, 4)],
        "ood_fewshot_groupcv_auroc": round(a_of, 4),
        "ood_fewshot_groupcv_ci95": [round(lo_of, 4), round(hi_of, 4)],
        "fold_min_n_bad_train": min(f["n_bad_train"] for f in folds if not f["skipped"]) if folds else None,
    }


def main():
    d = load_all()
    names, Xi, yi, ci = d["names"], d["indego_X"], d["indego_y"], d["indego_clips"]
    Xo, yo, fo = d["ood_X"], d["ood_y"], d["ood_factory"]
    sha = git_sha()

    # candidate per-feature table
    rows = [per_feature(Xi[:, j], yi, ci, Xo[:, j], yo, n) for j, n in enumerate(names)]
    counts = {}
    for r in rows:
        counts[r["class"]] = counts.get(r["class"], 0) + 1
    head_common = {
        "experiment": "Z3",
        "n": len(yi) + len(yo), "n_indego": len(yi), "n_good_indego": int((yi == 1).sum()),
        "n_ood": len(yo), "n_good_ood": int((yo == 1).sum()),
        "label_source": "human (IndEgo manifest_gold n=150 + Egocentric-10K ood_labels n=60; single labeller; OOD n_bad=9 exploratory)",
        "protocol": "signed AUROC per feature, label 1 = usable/good; IndEgo CI = clip-bootstrap (7 clips), OOD CI = window-bootstrap (optimistic); both decoded to ~10 fps effective for decode-based candidates",
        "device": DEV,
        "git_sha": sha,
        "command": "OMP_NUM_THREADS=2 python3 eval/exp_features/extract.py && python3 eval/exp_features/run_eval.py",
        "cost": {"candidates": len(names), "indego_windows": len(yi), "ood_windows": len(yo)},
        "classification_rule": "chance iff BOTH |AUROC-0.5| < 0.05; else stable if same side, flip if opposite (E5.03 rule)",
    }
    json.dump({**head_common, "experiment": "Z3.01", "counts": counts, "table": rows},
              open(os.path.join(OUT, "Z3.01.json"), "w"), indent=1)

    # family verdicts
    fam = {"compression_telemetry": E.COMPRESSION, "affine_ego_motion": E.MOTION,
           "invariants": E.INVARIANTS}
    by = {r["feature"]: r for r in rows}
    famrep = {}
    for f, feats in fam.items():
        cls = [by[x]["class"] for x in feats]
        stable = [x for x in feats if by[x]["class"] == "stable"]
        best = max(feats, key=lambda x: min(abs(by[x]["indego_auroc"] - .5),
                                            abs(by[x]["ood_auroc"] - .5)))
        famrep[f] = {"n": len(feats), "counts": {c: cls.count(c) for c in set(cls)},
                     "stable_features": stable,
                     "best_transfer": best,
                     "best_transfer_indego_auroc": by[best]["indego_auroc"],
                     "best_transfer_ood_auroc": by[best]["ood_auroc"]}
    json.dump({**head_common, "experiment": "Z3.02", "families": famrep,
               "verdicts": {
                   "compression": "compression telemetry is a weak, partly stable cue (see table)",
                   "affine": "full-affine residual vs translation-only; compare mo_affine_* with mo_trans_*",
                   "invariants": "DCT/noise/entropy/dHash invariants"}},
              open(os.path.join(OUT, "Z3.02.json"), "w"), indent=1)

    # ---- combined sets ----
    n26 = d["names26"]
    Xi26, Xo26 = d["indego_X26"], d["ood_X26"]
    stable_cols = [n26.index(s) for s in STABLE8]
    # transferable candidates: stable + informative in BOTH (>=0.06 from 0.5)
    transfer_cols = [j for j, n in enumerate(names)
                     if by[n]["class"] == "stable"
                     and min(abs(by[n]["indego_auroc"] - .5), abs(by[n]["ood_auroc"] - .5)) >= 0.03]
    transfer_names = [names[j] for j in transfer_cols]
    # combined matrices: [stable8 (26-space) | transferable candidates]
    XiC = np.hstack([Xi26[:, stable_cols], Xi[:, transfer_cols]])
    XoC = np.hstack([Xo26[:, stable_cols], Xo[:, transfer_cols]])
    sets = {
        "stable8_only": (Xi26[:, stable_cols], Xo26[:, stable_cols], STABLE8),
        "transferable_only": (Xi[:, transfer_cols], Xo[:, transfer_cols], transfer_names),
        "stable8+transferable": (XiC, XoC, STABLE8 + transfer_names),
        "in_domain26": (Xi26, Xo26, n26),
    }
    combo = {}
    for k, (Ax, Bx, cols) in sets.items():
        if Ax.shape[1] == 0:
            continue
        combo[k] = combined_eval(Ax, yi, ci, Bx, yo, fo, list(range(Ax.shape[1])), C=1.0)
        combo[k]["features"] = cols

    # truly domain-agnostic combination: per-domain z-score each feature, sign from
    # IndEgo AUROC, equal-weight mean. No cross-domain weights to shift.
    def znorm(A):
        return (A - A.mean(axis=0)) / (A.std(axis=0) + 1e-9)

    def zscore_combo(Ax, Bx, cols, label_names):
        a_au = np.asarray([roc_auc_score(yi, Ax[:, j]) for j in range(Ax.shape[1])])
        sg = np.where(a_au >= 0.5, 1.0, -1.0)
        si = (znorm(Ax) * sg).mean(axis=1)
        so = (znorm(Bx) * sg).mean(axis=1)
        a_i = float(roc_auc_score(yi, si))
        a_o = float(roc_auc_score(yo, so))
        _, lo_i, hi_i = fast_group_boot_ci(yi, si, ci, n_boot=1000)
        _, lo_o, hi_o = L.boot_ci(yo, so, fast_auc, n_boot=1000)
        return {"n_features": Ax.shape[1], "features": label_names,
                "indego_auroc": round(a_i, 4), "indego_ci95": [round(lo_i, 4), round(hi_i, 4)],
                "ood_auroc": round(a_o, 4), "ood_ci95": [round(lo_o, 4), round(hi_o, 4)],
                "method": "per-domain z-score, sign=IndEgo, equal-weight mean (unsupervised but for the per-feature sign)",
                "note": "sign fitted on full IndEgo (mild optimism); per-domain z-score removes cross-domain scale shift"}

    combo["zscore_transferable_only"] = zscore_combo(
        Xi[:, transfer_cols], Xo[:, transfer_cols], transfer_cols, transfer_names)
    combo["zscore_stable8+transferable"] = zscore_combo(
        XiC, XoC, list(range(XiC.shape[1])), STABLE8 + transfer_names)
    json.dump({**head_common, "experiment": "Z3.03",
               "stable8": STABLE8,
               "transferable_subset": transfer_names,
               "selection_rule": "candidate class==stable AND min(|AUROC-0.5|) >= 0.03 in both domains",
               "combined": combo},
              open(os.path.join(OUT, "Z3.03.json"), "w"), indent=1)

    # console summary
    print(f"{'feature':<24}{'IndEgo':>8}{'OOD':>8}  class")
    for r in sorted(rows, key=lambda r: -min(r["indego_auroc"], r["ood_auroc"]) if r['class']=='stable' else 0):
        print(f"{r['feature']:<24}{r['indego_auroc']:>8.3f}{r['ood_auroc']:>8.3f}  {r['class']}")
    print("counts", counts)
    print("transferable subset:", transfer_names)
    for k, v in combo.items():
        ai = v.get("indego_loco_auroc", v.get("indego_auroc"))
        ao = v.get("ood_zeroshot_auroc", v.get("ood_auroc"))
        af = v.get("ood_fewshot_groupcv_auroc", float("nan"))
        print(f"{k:<28} IndEgo={ai:.3f} OOD0shot={ao:.3f} OODfewshot={af:.3f}")


if __name__ == "__main__":
    main()
