#!/usr/bin/env python3
"""A2 — incremental value of NEW full-resolution signals.

Measures each new signal group (flow, hand, full-resolution per-pixel), alone and
added on top of the 26-feature + EG2 frozen ensemble, with
  * in-domain leave-one-clip-out (LOCO) AUROC on IndEgo gold (150), and
  * OOD AUROC with factory-grouped 95% CI on Egocentric-10K (160),
plus paired grouped-bootstrap gains vs the ensemble baseline.

Baseline = the frozen 26-feature + EG2 two-view ensemble
(gold OOF p_ens, OOD usable_prob). CPU only. No labels edited.

Writes eval/results/exp_acc/A2.01.json (alone), A2.02.json (incremental),
A2.03.json (best combined).
Run: OMP_NUM_THREADS=2 python3 eval/exp_acc/run_acc_signals.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time

import numpy as np
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
BUNDLE = os.environ.get("JADAL_BUNDLE", "/workspace/jadal_pod_bundle")
INP = os.path.join(ROOT, "eval", "results", "exp_acc", "inputs")
OUT = os.path.join(ROOT, "eval", "results", "exp_acc")
N_BOOT = 2000
SEED = 0
DEV = "CPU (numpy/sklearn; MediaPipe HandLandmarker + Farneback flow extracted separately)"

GROUPS = {
    "flow": ["flow_mag_mean", "flow_mag_p90", "flow_mag_p99", "flow_mag_spatial_std",
             "flow_active_frac", "flow_div_abs_mean", "flow_div_abs_p90",
             "flow_curl_abs_mean", "flow_curl_abs_p90", "flow_stability",
             "flow_static_frac", "flow_mag_temporal_std", "flow_peak_frac"],
    "hand": ["hand_present_frac", "hand_motion_frac", "hand_occupancy_mean",
             "hand_n_mean", "hand_flow_excess_mean"],
    "fullres": ["fr_lap_var_mean", "fr_lap_var_p10", "fr_sharp_temporal_cv",
                "fr_noise_mean", "fr_noise_p90", "fr_hf_band_frac"],
}
# direction in which a higher value means BAD (set after sign fit on gold; these
# are the expected physics directions, refit below anyway)
NAMES26 = ["temp_b0", "temp_b1", "temp_b2", "temp_b3", "frozen_frac", "flicker_idx",
           "radial_slope", "hf_ratio", "dct_energy", "pca_resid", "pca_top1_share",
           "luma_mean", "luma_p05", "luma_p50", "luma_p95", "dark_frac", "bright_frac",
           "motion_energy", "global_shift", "skin_frac", "illum_r", "illum_g", "illum_b",
           "rppg_pulse", "dup_hash_mean", "zncc_self"]


def git_sha():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                              text=True, cwd=ROOT).stdout.strip()
    except Exception:
        return "unknown"


# ------------------------------------------------------------------ metrics
def auc(y, s):
    """bad-score AUROC: higher score = more likely bad (y==0)."""
    return float(roc_auc_score(1 - np.asarray(y), np.asarray(s, float)))


def _groups_idx(groups):
    ug = sorted(set(np.asarray(groups).tolist()))
    return ug, {g: np.where(np.asarray(groups) == g)[0] for g in ug}


def grouped_boot(y, s, groups, n_boot=N_BOOT, seed=SEED):
    ug, idx = _groups_idx(groups)
    rng = np.random.RandomState(seed)
    vals = []
    for _ in range(n_boot):
        pick = rng.randint(0, len(ug), len(ug))
        ii = np.concatenate([idx[ug[p]] for p in pick])
        if len(np.unique(np.asarray(y)[ii])) < 2:
            continue
        vals.append(auc(np.asarray(y)[ii], np.asarray(s)[ii]))
    if not vals:
        return [None, None]
    return [round(float(np.percentile(vals, 2.5)), 4),
            round(float(np.percentile(vals, 97.5)), 4)]


def paired_grouped_boot(y, s_a, s_b, groups, n_boot=N_BOOT, seed=SEED):
    ug, idx = _groups_idx(groups)
    rng = np.random.RandomState(seed)
    diffs = []
    for _ in range(n_boot):
        pick = rng.randint(0, len(ug), len(ug))
        ii = np.concatenate([idx[ug[p]] for p in pick])
        if len(np.unique(np.asarray(y)[ii])) < 2:
            continue
        diffs.append(auc(np.asarray(y)[ii], np.asarray(s_a)[ii])
                     - auc(np.asarray(y)[ii], np.asarray(s_b)[ii]))
    diffs = np.asarray(diffs)
    if len(diffs) == 0:
        return {}
    return {"mean": round(float(diffs.mean()), 4),
            "ci95": [round(float(np.percentile(diffs, 2.5)), 4),
                     round(float(np.percentile(diffs, 97.5)), 4)],
            "frac_gt0": round(float((diffs > 0).mean()), 4)}


def rank01(x):
    x = np.asarray(x, float)
    return (rankdata(x) - 1.0) / max(len(x) - 1, 1)


# ------------------------------------------------------------------ loaders
def load_gold():
    m = json.load(open(os.path.join(BUNDLE, "labels", "manifest_gold.json")))
    items = m["items"]
    ids = [it["id"] for it in items]
    y = np.asarray([int(it["label"]) for it in items])
    clips = np.asarray([it["clip"] for it in items])
    reasons = np.asarray([it.get("reason", "") or "" for it in items])
    W = json.load(open(os.path.join(BUNDLE, "caches", "gpu", "features.json")))["windows"]
    F26 = np.asarray([[float(W[i]["full"][n]) for n in NAMES26] for i in ids], float)
    oof = np.load(os.path.join(BUNDLE, "caches", "ensemble", "oof_probs.npz"),
                  allow_pickle=True)
    omap = {str(k): j for j, k in enumerate(oof["ids"])}
    ens = np.asarray([float(oof["p_ens"][omap[i]]) for i in ids])
    sig = np.load(os.path.join(INP, "signals_gold.npz"), allow_pickle=True)
    smap = {str(k): j for j, k in enumerate(sig["ids"])}
    snames = [str(n) for n in sig["names"]]
    S = np.asarray(sig["X"], float)[[smap[i] for i in ids]]
    return dict(ids=ids, y=y, clips=clips, reasons=reasons, F26=F26,
                ens=ens, S=S, snames=snames)


def load_ood():
    d = json.load(open(os.path.join(BUNDLE, "labels", "ood160_labels.json")))
    w = d["windows"]
    ids = [x["segment_id"] for x in w]
    y = np.asarray([1 if x["verdict"] == "good" else 0 for x in w])
    factory = np.asarray([x["factory"] for x in w])
    reasons = np.asarray([x.get("reason", "") or "" for x in w])
    probs = {}
    for fn in ("ood_scores.json", "ood_scores2.json"):
        dd = json.load(open(os.path.join(BUNDLE, "caches", "ensemble", fn)))
        for s in dd["scores"]:
            if "error" not in s:
                probs[s["id"]] = float(s["usable_prob"])
    ens = np.asarray([probs[i] for i in ids])
    sig = np.load(os.path.join(INP, "signals_ood.npz"), allow_pickle=True)
    smap = {str(k): j for j, k in enumerate(sig["ids"])}
    snames = [str(n) for n in sig["names"]]
    S = np.asarray(sig["X"], float)[[smap[i] for i in ids]]
    return dict(ids=ids, y=y, factory=factory, reasons=reasons, ens=ens,
                S=S, snames=snames)


# ------------------------------------------------------------------ model helpers
def _impute_std(train, test):
    med = np.nanmedian(train, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    tr = np.where(np.isfinite(train), train, med)
    te = np.where(np.isfinite(test), test, med)
    sc = StandardScaler().fit(tr)
    return sc.transform(tr), sc.transform(te)


def loco_logistic(X, y, clips, C=1.0):
    """OOF P(good) by logistic, leave-one-clip-out. Returns score (higher=good)."""
    p = np.full(len(y), np.nan)
    for c in sorted(set(np.asarray(clips).tolist())):
        te = np.where(np.asarray(clips) == c)[0]
        tr = np.where(np.asarray(clips) != c)[0]
        if len(np.unique(np.asarray(y)[tr])) < 2:
            continue
        Xtr, Xte = _impute_std(X[tr], X[te])
        clf = LogisticRegression(max_iter=3000, C=C)
        clf.fit(Xtr, y[tr])
        p[te] = clf.predict_proba(Xte)[:, 1]
    return p


def fit_predict(Xtr, ytr, Xte, C=1.0):
    Xa, Xb = _impute_std(Xtr, Xte)
    clf = LogisticRegression(max_iter=3000, C=C)
    clf.fit(Xa, ytr)
    return clf.predict_proba(Xb)[:, 1]


def cols(S, snames, group):
    return S[:, [snames.index(n) for n in group]]


# ------------------------------------------------------------------ main
def main():
    t0 = time.perf_counter()
    gold = load_gold()
    ood = load_ood()
    sha = git_sha()
    gy, oy = gold["y"], ood["y"]
    gS, oS = gold["S"], ood["S"]
    snames = gold["snames"]
    assert snames == ood["snames"], "signal name mismatch"

    # baseline bad-scores
    base_g = -gold["ens"]      # higher = bad
    base_o = -ood["ens"]
    base_g_auc = round(auc(gy, base_g), 4)
    base_o_auc = round(auc(oy, base_o), 4)

    # ---- A2.01: each new signal alone (sign fit on gold, applied to OOD) ----
    alone = {"experiment": "A2.01",
             "title": "Each new full-res signal alone (bad-score AUROC)",
             "device": DEV, "git_sha": sha,
             "baseline_ensemble": {"gold_loco_auroc": base_g_auc, "ood_auroc": base_o_auc,
                                   "ood_auroc_ci95_grouped": grouped_boot(oy, base_o, ood["factory"])},
             "note": "sign (direction of 'higher=bad') is fit on gold AUROC then applied to OOD",
             "signals": {}}
    sign = {}
    for n in snames:
        xg = gS[:, snames.index(n)]
        au = auc(gy, xg) if np.isfinite(xg).all() else None
        # choose sign on gold so higher = bad
        s = 1.0 if (au is not None and au >= 0.5) else -1.0
        sign[n] = s
        gb = s * xg
        ob = s * oS[:, snames.index(n)]
        alone["signals"][n] = {
            "group": next((g for g, ns in GROUPS.items() if n in ns), "other"),
            "gold_auroc": round(auc(gy, gb), 4),
            "ood_auroc": round(auc(oy, ob), 4),
            "ood_auroc_ci95_grouped": grouped_boot(oy, ob, ood["factory"]),
        }
    json.dump(alone, open(os.path.join(OUT, "A2.01.json"), "w"), indent=1, sort_keys=True)
    print("A2.01 done", round(time.perf_counter() - t0, 1), "s")

    # ---- A2.02: incremental on top of the 26-feature + EG2 ensemble ----
    inc = {"experiment": "A2.02",
           "title": "Incremental value added on top of the 26-feature + EG2 ensemble",
           "device": DEV, "git_sha": sha,
           "method": ("baseline feature = frozen ensemble good-score (gold OOF p_ens, OOD "
                      "usable_prob); +group = logistic on [ens] + group features, LOCO on "
                      "gold / train-all-gold -> OOD. gold '26+ens+group' adds the 26 cached "
                      "per-pixel features. gain = paired grouped bootstrap vs baseline."),
           "baseline": {"gold_auroc": base_g_auc, "ood_auroc": base_o_auc},
           "models": {}}

    def eval_model(name, gX, oX, ref_g, ref_o):
        gp = loco_logistic(gX, gy, gold["clips"])          # P(good)
        op = fit_predict(gX, gy, oX)                       # P(good) on OOD
        gb, ob = -gp, -op
        row = {"gold_auroc": round(auc(gy, gb), 4),
               "ood_auroc": round(auc(oy, ob), 4),
               "ood_auroc_ci95_grouped": grouped_boot(oy, ob, ood["factory"]),
               "gain_vs_baseline_gold_paired": paired_grouped_boot(gy, gb, ref_g, gold["clips"]),
               "gain_vs_baseline_ood_paired": paired_grouped_boot(oy, ob, ref_o, ood["factory"])}
        inc["models"][name] = row
        return row

    # baseline model (ens alone as a logistic-free score) for paired refs
    for gname, gcols in GROUPS.items():
        gX = np.column_stack([gold["ens"], cols(gS, snames, gcols)])
        oX = np.column_stack([ood["ens"], cols(oS, snames, gcols)])
        eval_model(f"ens+{gname}", gX, oX, base_g, base_o)
        # with the 26 features too (gold-only metric is meaningful; OOD too)
        gX26 = np.column_stack([gold["F26"], gold["ens"], cols(gS, snames, gcols)])
        oX26 = np.column_stack([np.zeros((len(oy), 26)), ood["ens"], cols(oS, snames, gcols)])
        # for OOD we cannot build the 26 features; skip OOD for 26+ models
        gp = loco_logistic(gX26, gy, gold["clips"])
        inc["models"][f"26+ens+{gname}"] = {
            "gold_auroc": round(auc(gy, -gp), 4),
            "note": "in-domain only (26-feature view unavailable on OOD in this bundle)"}
    json.dump(inc, open(os.path.join(OUT, "A2.02.json"), "w"), indent=1, sort_keys=True)
    print("A2.02 done", round(time.perf_counter() - t0, 1), "s")

    # ---- A2.03: best combined ----
    allcols = sum(GROUPS.values(), [])
    best = {"experiment": "A2.03",
            "title": "Best combined signal set vs the ensemble baseline",
            "device": DEV, "git_sha": sha,
            "candidates": {}}
    combos = {
        "ens (baseline)": ([], base_g, base_o),
        "ens+flow": (GROUPS["flow"], None, None),
        "ens+hand": (GROUPS["hand"], None, None),
        "ens+fullres": (GROUPS["fullres"], None, None),
        "ens+flow+hand": (GROUPS["flow"] + GROUPS["hand"], None, None),
        "ens+flow+hand+fullres": (allcols, None, None),
    }
    for name, (gc, rg, ro) in combos.items():
        if not gc:
            best["candidates"][name] = {"gold_auroc": base_g_auc, "ood_auroc": base_o_auc,
                                        "ood_auroc_ci95_grouped": grouped_boot(oy, base_o, ood["factory"])}
            continue
        gX = np.column_stack([gold["ens"], cols(gS, snames, gc)])
        oX = np.column_stack([ood["ens"], cols(oS, snames, gc)])
        gp = loco_logistic(gX, gy, gold["clips"])
        op = fit_predict(gX, gy, oX)
        best["candidates"][name] = {
            "gold_auroc": round(auc(gy, -gp), 4),
            "ood_auroc": round(auc(oy, -op), 4),
            "ood_auroc_ci95_grouped": grouped_boot(oy, -op, ood["factory"]),
            "gain_vs_baseline_gold_paired": paired_grouped_boot(gy, -gp, base_g, gold["clips"]),
            "gain_vs_baseline_ood_paired": paired_grouped_boot(oy, -op, base_o, ood["factory"])}
    # mean-domain ranking
    best["mean_domain_auroc"] = {n: round((v["gold_auroc"] + v["ood_auroc"]) / 2, 4)
                                 for n, v in best["candidates"].items()}
    json.dump(best, open(os.path.join(OUT, "A2.03.json"), "w"), indent=1, sort_keys=True)
    print("A2.03 done", round(time.perf_counter() - t0, 1), "s")

    # compact stdout
    print(json.dumps({
        "baseline": {"gold": base_g_auc, "ood": base_o_auc},
        "incremental": {k: {"gold": v["gold_auroc"], "ood": v.get("ood_auroc"),
                            "g_gain": v.get("gain_vs_baseline_gold_paired", {}).get("mean"),
                            "o_gain": v.get("gain_vs_baseline_ood_paired", {}).get("mean")}
                        for k, v in inc["models"].items()},
        "combined": best["mean_domain_auroc"],
    }, indent=1))


if __name__ == "__main__":
    main()
