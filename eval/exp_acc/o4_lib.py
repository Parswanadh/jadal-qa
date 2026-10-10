#!/usr/bin/env python3
"""O4: confirm the Z3 transferable-feature adaptation on the 160 OOD labels.

Recipes compared on the full 160 human-labelled Egocentric-10K windows
(138 good / 22 bad, 47 factories, q1=60 + q2=100):

  (a) Z3 6 transferable features, per-domain z-score, IndEgo-signed, equal mean:
      ct_fsize_std, mo_hand_motion_excess, inv_clip_frac, inv_laplacian_var,
      inv_hist_entropy, inv_dct_hf_lf
  (b) E5 stable-8 + the same 6, same z-score rule.
  (c) Z1 sharpness-3 head (radial_slope, hf_ratio, dct_energy): logistic fit on
      IndEgo (C by inner LOCO), zero-shot transfer; plus the z-score variant of
      the same 3 features for a like-for-like comparison.
  (d) frozen ensemble_v1 `usable_prob` (baseline 0.566 on 160).

Evaluation only. The only fitted quantities are (i) the per-feature SIGN taken
from the 150 IndEgo gold windows and (ii) for (c) the logistic head on IndEgo.
The per-domain z-score is unsupervised (target mean/std, no OOD labels).

Metrics: AUROC + balanced accuracy, factory-grouped bootstrap 95% CI (honest)
and window bootstrap (optimistic). Reports q1 vs q2.

Outputs eval/results/exp_adapt160/O4.01.json, O4.02.json.
Run: OMP_NUM_THREADS=2 python3 eval/exp_adapt160/run_adapt160.py
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
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

BUNDLE = os.environ.get("JADAL_BUNDLE", "jadal_pod_bundle")
OUT = os.path.join(REPO, "eval", "results", "exp_adapt160")
INPUTS = os.path.join(OUT, "inputs")
PIXEL160 = os.environ.get(
    "JADAL_PIXEL160",
    os.path.join(os.environ.get("JADAL_WT", "wt"), 'm-fewshot160/eval/results/exp_fewshot/inputs/ood160_pixel.npz'))
N_BOOT = 2000
SEED = 0
DEV = "CPU (numpy/sklearn; ffmpeg/cv2 decode only, no GPU)"

Z3_SIX = ["ct_fsize_std", "mo_hand_motion_excess", "inv_clip_frac",
          "inv_laplacian_var", "inv_hist_entropy", "inv_dct_hf_lf"]
STABLE8 = ["radial_slope", "hf_ratio", "dct_energy", "luma_p05", "global_shift",
           "illum_r", "illum_b", "zncc_self"]
SHARP3 = ["radial_slope", "hf_ratio", "dct_energy"]
FEATURE_NAMES26 = [
    "temp_b0", "temp_b1", "temp_b2", "temp_b3", "frozen_frac", "flicker_idx",
    "radial_slope", "hf_ratio", "dct_energy", "pca_resid", "pca_top1_share",
    "luma_mean", "luma_p05", "luma_p50", "luma_p95", "dark_frac", "bright_frac",
    "motion_energy", "global_shift", "skin_frac", "illum_r", "illum_g", "illum_b",
    "rppg_pulse", "dup_hash_mean", "zncc_self"]
C_GRID = (0.01, 0.1, 1.0, 10.0)


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


def balacc(y, s, thr=0.0):
    y = np.asarray(y)
    s = np.asarray(s, float)
    pred = (s > thr).astype(int)
    g = y == 1
    b = y == 0
    tpr = float((pred[g] == 1).mean()) if g.any() else float("nan")
    tnr = float((pred[b] == 0).mean()) if b.any() else float("nan")
    return 0.5 * (tpr + tnr)


def _group_index(groups):
    gs = list(groups)
    gmap = {g: i for i, g in enumerate(sorted(set(gs)))}
    gi = np.asarray([gmap[g] for g in gs])
    idx = [np.where(gi == k)[0] for k in range(len(gmap))]
    return gi, idx


def group_boot_ci(y, s, groups, fn, thr=None, n_boot=N_BOOT, seed=SEED):
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
        vals.append(fn(y[idx], s[idx]) if thr is None else fn(y[idx], s[idx], thr))
    if not vals:
        return [None, None]
    return [round(float(np.percentile(vals, 2.5)), 4),
            round(float(np.percentile(vals, 97.5)), 4)]


def window_boot_ci(y, s, fn, thr=None, n_boot=N_BOOT, seed=SEED):
    y = np.asarray(y)
    s = np.asarray(s)
    rng = np.random.RandomState(seed)
    vals = []
    for _ in range(n_boot):
        idx = rng.randint(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        vals.append(fn(y[idx], s[idx]) if thr is None else fn(y[idx], s[idx], thr))
    if not vals:
        return [None, None]
    return [round(float(np.percentile(vals, 2.5)), 4),
            round(float(np.percentile(vals, 97.5)), 4)]


def paired_group_boot(y, s_a, s_b, groups, n_boot=N_BOOT, seed=SEED):
    """Factory-grouped bootstrap of AUROC(a) - AUROC(b), same resamples."""
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


# ---------------- loaders ----------------

def load_labels160():
    d = json.load(open(os.path.join(BUNDLE, "labels", "ood160_labels.json")))
    w = d["windows"]
    return {"ids": [x["segment_id"] for x in w],
            "y": np.asarray([1 if x["verdict"] == "good" else 0 for x in w], int),
            "batch": np.asarray([x["batch"] for x in w]),
            "factory": np.asarray([x["factory"] for x in w])}


def load_z3_160():
    z = np.load(os.path.join(INPUTS, "z3_160.npz"), allow_pickle=True)
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
    return {"ids": ids, "clips": clips, "y": y, "X26": X}


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


# ---------------- z-score combo ----------------

def combo_scores(Xg, Xo, yg):
    """IndEgo-signed, per-domain z-score, equal-weight mean.

    sign = +1 if IndEgo AUROC >= 0.5 else -1; z-score each domain by its own
    (unlabelled) mean/std.
    """
    au = np.asarray([fast_auc(yg, Xg[:, j]) for j in range(Xg.shape[1])])
    sg = np.where(au >= 0.5, 1.0, -1.0)
    zg = (Xg - Xg.mean(0)) / np.clip(Xg.std(0), 1e-9, None)
    zo = (Xo - Xo.mean(0)) / np.clip(Xo.std(0), 1e-9, None)
    return (zg * sg).mean(1), (zo * sg).mean(1), au, sg


# ---------------- Z1 head ----------------

def _make_clf(C):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(StandardScaler(),
                         LogisticRegression(C=C, max_iter=5000, class_weight="balanced"))


def z1_head(Xg, yg, clips, Xo):
    """LOCO OOF on IndEgo (C by inner LOCO) + zero-shot OOD probs (head on all gold)."""
    Xg = np.asarray(Xg, float)
    oof = np.full(len(yg), np.nan)
    for held in sorted(set(clips.tolist())):
        te = np.where(clips == held)[0]
        tr = np.where(clips != held)[0]
        best_c, best = C_GRID[0], -1.0
        for c in C_GRID:
            aucs = []
            for h2 in sorted(set(clips[tr].tolist())):
                itr = tr[clips[tr] != h2]
                ite = tr[clips[tr] == h2]
                if len(np.unique(yg[itr])) < 2 or len(np.unique(yg[ite])) < 2:
                    continue
                clf = _make_clf(c)
                clf.fit(Xg[itr], yg[itr])
                aucs.append(fast_auc(yg[ite], clf.predict_proba(Xg[ite])[:, 1]))
            m = float(np.mean(aucs)) if aucs else 0.5
            if m > best:
                best, best_c = m, c
        clf = _make_clf(best_c)
        clf.fit(Xg[tr], yg[tr])
        oof[te] = clf.predict_proba(Xg[te])[:, 1]
    clf = _make_clf(1.0)
    clf.fit(Xg, yg)
    return oof, clf.predict_proba(np.asarray(Xo, float))[:, 1]


# ---------------- report helpers ----------------

def score_block(name, method, y_ood, s_ood, groups, thr, n_features, features,
                y_in=None, s_in=None, extra=None):
    block = {
        "recipe": name, "method": method, "n_features": n_features, "features": features,
        "threshold": thr,
        "ood_auroc": round(fast_auc(y_ood, s_ood), 4),
        "ood_balacc": round(balacc(y_ood, s_ood, thr), 4),
        "ood_auroc_ci95_window_bootstrap_optimistic":
            window_boot_ci(y_ood, s_ood, fast_auc),
        "ood_auroc_ci95_factory_grouped_honest":
            group_boot_ci(y_ood, s_ood, groups, fast_auc),
        "ood_balacc_ci95_factory_grouped_honest":
            group_boot_ci(y_ood, s_ood, groups, balacc, thr=thr),
    }
    if y_in is not None and s_in is not None:
        block["indego_auroc"] = round(fast_auc(y_in, s_in), 4)
        block["indego_balacc"] = round(balacc(y_in, s_in, 0.0 if thr == 0 else 0.5), 4)
    if extra:
        block.update(extra)
    return block


def batch_split(y, s, batch, groups, thr):
    out = {}
    for b in sorted(set(batch.tolist())):
        m = batch == b
        out[b] = {
            "n": int(m.sum()), "n_good": int((y[m] == 1).sum()),
            "n_bad": int((y[m] == 0).sum()),
            "auroc": round(fast_auc(y[m], s[m]), 4),
            "balacc": round(balacc(y[m], s[m], thr), 4),
            "auroc_ci95_factory_grouped_honest":
                group_boot_ci(y[m], s[m], groups[m], fast_auc),
        }
    return out


def main():
    lab = load_labels160()
    z3 = load_z3_160()
    pix = load_pixel160()
    gold = load_gold()
    sha = git_sha()

    ids = lab["ids"]
    y = lab["y"]
    batch = lab["batch"]
    fac = lab["factory"]
    assert list(z3["ids"]) == ids, "z3 ids align to labels160"
    n_good, n_bad = int((y == 1).sum()), int((y == 0).sum())

    # ---- matrices aligned to the 160 ----
    Xz3 = z3["X"]  # (160,20) Z3 candidates
    Z3i = z3["X_indego"]  # (150,20)
    yi = z3["y_indego"]
    clips = z3["clips_indego"]
    X26o = align(pix["ids"], pix["X10"], ids)  # (160,26) pixel-26 10fps
    X26g = gold["X26"]  # (150,26)
    yg = gold["y"]

    z6_idx = [z3["names"].index(n) for n in Z3_SIX]
    s8_idx = [FEATURE_NAMES26.index(n) for n in STABLE8]
    sharp_idx = [FEATURE_NAMES26.index(n) for n in SHARP3]

    X6o, X6g = Xz3[:, z6_idx], Z3i[:, z6_idx]
    X8o, X8g = X26o[:, s8_idx], X26g[:, s8_idx]
    XS3o, XS3g = X26o[:, sharp_idx], X26g[:, sharp_idx]
    X14o = np.hstack([X8o, X6o])
    X14g = np.hstack([X8g, X6g])

    blocks = []
    # (a) Z3 6
    si_a, so_a, au_a, sg_a = combo_scores(X6g, X6o, yi)
    blocks.append(score_block(
        "(a) Z3 transferable-6", "per-domain z-score, IndEgo-signed, equal-weight mean",
        y, so_a, fac, 0.0, 6, Z3_SIX, y_in=yi, s_in=si_a,
        extra={"indego_feature_auroc": {n: round(float(a), 4) for n, a in zip(Z3_SIX, au_a)},
               "indego_signs": {n: int(s) for n, s in zip(Z3_SIX, sg_a)}}))
    blocks[-1]["ood_batch_split"] = batch_split(y, so_a, batch, fac, 0.0)

    # (b) stable-8 + 6
    si_b, so_b, au_b, sg_b = combo_scores(X14g, X14o, yg)
    names14 = STABLE8 + Z3_SIX
    blocks.append(score_block(
        "(b) E5 stable-8 + Z3 transferable-6", "per-domain z-score, IndEgo-signed, equal-weight mean",
        y, so_b, fac, 0.0, 14, names14, y_in=yg, s_in=si_b,
        extra={"indego_feature_auroc": {n: round(float(a), 4) for n, a in zip(names14, au_b)},
               "indego_signs": {n: int(s) for n, s in zip(names14, sg_b)}}))
    blocks[-1]["ood_batch_split"] = batch_split(y, so_b, batch, fac, 0.0)

    # (c) Z1 sharpness-3 head (trained on IndEgo)
    oof_s3, ood_s3 = z1_head(XS3g, yg, gold["clips"], XS3o)
    blocks.append(score_block(
        "(c) Z1 sharpness-3 head", "logistic on IndEgo (C by inner LOCO), zero-shot to OOD",
        y, ood_s3, fac, 0.5, 3, SHARP3, y_in=yg, s_in=oof_s3))
    blocks[-1]["ood_batch_split"] = batch_split(y, ood_s3, batch, fac, 0.5)

    # (c2) sharpness-3 z-score variant (like-for-like with (a)/(b))
    si_c2, so_c2, au_c2, sg_c2 = combo_scores(XS3g, XS3o, yg)
    blocks.append(score_block(
        "(c2) sharpness-3 z-score", "per-domain z-score, IndEgo-signed, equal-weight mean",
        y, so_c2, fac, 0.0, 3, SHARP3, y_in=yg, s_in=si_c2,
        extra={"indego_feature_auroc": {n: round(float(a), 4) for n, a in zip(SHARP3, au_c2)},
               "indego_signs": {n: int(s) for n, s in zip(SHARP3, sg_c2)}}))
    blocks[-1]["ood_batch_split"] = batch_split(y, so_c2, batch, fac, 0.0)

    # (d) frozen baseline
    frozen, frozen_missing = load_frozen160(ids)
    blocks.append(score_block(
        "(d) frozen ensemble_v1", "frozen two-view usable_prob (no fitting)", y, frozen,
        fac, 0.5, None, None,
        extra={"frozen_config_sha256": json.load(open(os.path.join(
            BUNDLE, "caches", "ensemble", "ood_scores.json")))["config_sha256"],
            "missing_scores": frozen_missing}))
    blocks[-1]["ood_batch_split"] = batch_split(y, frozen, batch, fac, 0.5)

    # paired (same-resample) factory-grouped bootstrap of AUROC(recipe) - AUROC(frozen)
    gains = {}
    for nm, sc in (("(a) Z3 transferable-6", so_a),
                   ("(b) E5 stable-8 + Z3 transferable-6", so_b),
                   ("(c) Z1 sharpness-3 head", ood_s3),
                   ("(c2) sharpness-3 z-score", so_c2)):
        gains[nm] = paired_group_boot(y, sc, frozen, fac)

    o401 = {
        "experiment": "O4.01",
        "title": "Confirm Z3 transferable-feature adaptation on the 160 OOD labels",
        "n": len(ids), "n_good": n_good, "n_bad": n_bad,
        "n_factories": int(len(set(fac.tolist()))),
        "n_q1": int((batch == "q1").sum()), "n_q2": int((batch == "q2").sum()),
        "label_source": ("human (labels/ood160_labels.json; merge of ood_labels.json q1=60 "
                         "and ood_labels2.json q2=100, single labeller; 138 good / 22 bad, "
                         "47 factories)"),
        "protocol": ("evaluation only. Per-domain z-score = target-domain mean/std "
                     "(unsupervised); sign from the 150 IndEgo gold windows. "
                     "Factory-grouped bootstrap CI (n_boot=2000, seed=0) is the honest CI; "
                     "window bootstrap is optimistic (correlated within factory)."),
        "threshold_rule": ("balanced accuracy at a fixed natural threshold: 0 for the "
                           "z-score combos (symmetric signed mean), 0.5 for probability "
                           "heads / frozen ensemble. No OOD threshold tuning."),
        "device": DEV, "git_sha": sha,
        "command": ("OMP_NUM_THREADS=2 python3 eval/exp_adapt160/extract160.py && "
                    "OMP_NUM_THREADS=2 python3 eval/exp_adapt160/run_adapt160.py"),
        "gain_vs_frozen_paired_factory_grouped_bootstrap": gains,
        "blocks": blocks,
    }
    os.makedirs(OUT, exist_ok=True)
    json.dump(o401, open(os.path.join(OUT, "O4.01.json"), "w"), indent=1)

    # ---- O4.02: per-feature transfer table on the 160 ----
    def feat_row(name, xg, xo):
        a_i = round(fast_auc(yg, xg), 4)
        a_o = round(fast_auc(y, xo), 4)
        return {"feature": name, "indego_auroc": a_i,
                "indego_ci95": group_boot_ci(yg, xg, clips, fast_auc),
                "ood160_auroc": a_o,
                "ood160_ci95_factory_grouped": group_boot_ci(y, xo, fac, fast_auc),
                "indego_sign": "high-usable" if a_i >= 0.5 else "low-usable",
                "ood_sign": "high-usable" if a_o >= 0.5 else "low-usable",
                "class": ("stable" if (a_i - 0.5) * (a_o - 0.5) > 0 else "flip")
                         if max(abs(a_i - 0.5), abs(a_o - 0.5)) >= 0.05 else "chance"}
    rows6 = [feat_row(n, X6g[:, j], X6o[:, j]) for j, n in enumerate(Z3_SIX)]
    rows8 = [feat_row(n, X8g[:, j], X8o[:, j]) for j, n in enumerate(STABLE8)]
    rows_all = [feat_row(n, Z3i[:, j], Xz3[:, j]) for j, n in enumerate(z3["names"])]
    counts = {}
    for r in rows_all:
        counts[r["class"]] = counts.get(r["class"], 0) + 1
    o402 = {
        "experiment": "O4.02",
        "title": "Per-feature IndEgo vs OOD-160 transfer",
        "n_ood": len(ids), "n_bad_ood": n_bad,
        "label_source": o401["label_source"], "protocol": o401["protocol"],
        "device": DEV, "git_sha": sha,
        "z3_six": rows6, "stable8": rows8, "z3_all20": rows_all,
        "z3_all20_class_counts": counts,
        "note": ("OOD-160 CI is factory-grouped (honest); IndEgo CI is clip-bootstrap "
                 "(7 clips). Class rule: chance iff both |AUROC-0.5|<0.05, else "
                 "stable/flip by side (E5.03 rule)."),
    }
    json.dump(o402, open(os.path.join(OUT, "O4.02.json"), "w"), indent=1)

    # ---- stdout ----
    print(f"n={len(ids)} good={n_good} bad={n_bad} factories={o401['n_factories']}")
    print(f"{'recipe':38s}{'OOD AUROC':>11}{'grp CI':>18}{'OOD BA':>9}{'IndEgo':>9}")
    for b in blocks:
        ci = b["ood_auroc_ci95_factory_grouped_honest"]
        ind = b.get("indego_auroc")
        print(f"{b['recipe']:38s}{b['ood_auroc']:>11.3f}{str(ci):>18}{b['ood_balacc']:>9.3f}"
              f"{(ind if ind is not None else float('nan')):>9.3f}")
    print("\nq1 vs q2 (per-batch z-score for the combos):")
    for b in blocks:
        bs = b["ood_batch_split"]
        print(f"  {b['recipe']:38s} q1 {bs['q1']['auroc']:.3f} (n_bad={bs['q1']['n_bad']})"
              f"  q2 {bs['q2']['auroc']:.3f} (n_bad={bs['q2']['n_bad']})")
    print("\nper-feature (Z3 six | IndEgo | OOD160):")
    for r in rows6:
        print(f"  {r['feature']:24s} {r['indego_auroc']:>7.3f} {r['ood160_auroc']:>7.3f}  {r['class']}")
    print("\nper-feature (stable-8):")
    for r in rows8:
        print(f"  {r['feature']:24s} {r['indego_auroc']:>7.3f} {r['ood160_auroc']:>7.3f}  {r['class']}")


if __name__ == "__main__":
    main()
