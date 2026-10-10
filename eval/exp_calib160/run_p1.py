#!/usr/bin/env python3
"""P1: calibrate the adapted (Z3/O4) OOD score on the 160 human labels.

Recipes calibrated: (a) Z3 transferable-6 and (b) stable-8 + transferable-6,
recomputed exactly per O4 (IndEgo-signed per-domain z-score, equal mean).

Protocol: leave-factory-out (LFO) grouped CV over the 47 factories. For each
held-out factory, fit Platt / isotonic / beta calibrators on the other 46
factories' (adapted score -> P(bad)) and predict the held-out windows. All
reported calibration metrics and the band frontier use these pooled
out-of-fold (OOF) predictions, so no factory is ever fit-and-evaluated.

Outputs eval/results/exp_calib160/P1.01.json + figure inputs.
Run: OMP_NUM_THREADS=2 python3 eval/exp_calib160/run_p1.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np
from scipy.optimize import minimize
from scipy.stats import rankdata
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)

BUNDLE = os.environ.get("JADAL_BUNDLE", "/workspace/jadal_pod_bundle")
ADAPT_INPUTS = os.environ.get(
    "JADAL_ADAPT_INPUTS",
    "/workspace/wt/m-adapt160/eval/results/exp_adapt160/inputs")
PIXEL160 = os.environ.get(
    "JADAL_PIXEL160",
    "/workspace/wt/m-fewshot160/eval/results/exp_fewshot/inputs/ood160_pixel.npz")
OUT = os.path.join(REPO, "eval", "results", "exp_calib160")
SEED = 11
N_BOOT = 2000
DEV = "CPU (numpy/sklearn/scipy; no GPU)"

Z3_SIX = ["ct_fsize_std", "mo_hand_motion_excess", "inv_clip_frac",
          "inv_laplacian_var", "inv_hist_entropy", "inv_dct_hf_lf"]
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


# ---------------- O4-exact score recomputation ----------------

def fast_auc(y, s):
    y = np.asarray(y)
    s = np.asarray(s, float)
    n_pos = int((y == 1).sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    r = rankdata(s)
    return float((r[y == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def combo_scores(Xg, Xo, yg):
    au = np.asarray([fast_auc(yg, Xg[:, j]) for j in range(Xg.shape[1])])
    sg = np.where(au >= 0.5, 1.0, -1.0)
    zg = (Xg - Xg.mean(0)) / np.clip(Xg.std(0), 1e-9, None)
    zo = (Xo - Xo.mean(0)) / np.clip(Xo.std(0), 1e-9, None)
    return (zg * sg).mean(1), (zo * sg).mean(1), au, sg


def load_labels160():
    d = json.load(open(os.path.join(BUNDLE, "labels", "ood160_labels.json")))
    w = d["windows"]
    return {"ids": [x["segment_id"] for x in w],
            "y_good": np.asarray([1 if x["verdict"] == "good" else 0 for x in w], int),
            "batch": np.asarray([x["batch"] for x in w]),
            "factory": np.asarray([x["factory"] for x in w])}


def load_frozen160(ids):
    sc = {}
    for fn in ("ood_scores.json", "ood_scores2.json"):
        d = json.load(open(os.path.join(BUNDLE, "caches", "ensemble", fn)))
        for x in d["scores"]:
            if "error" not in x:
                sc[x["id"]] = float(x["usable_prob"])
    return np.asarray([sc[i] for i in ids], float)


# ---------------- calibration ----------------

def fit_platt(s_tr, yb_tr):
    clf = LogisticRegression(C=1.0, max_iter=5000)
    clf.fit(np.asarray(s_tr, float).reshape(-1, 1), yb_tr)
    return lambda s: clf.predict_proba(np.asarray(s, float).reshape(-1, 1))[:, 1]


def fit_isotonic(s_tr, yb_tr):
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(np.asarray(s_tr, float), yb_tr)
    return lambda s: np.clip(iso.predict(np.asarray(s, float)), 0.0, 1.0)


def _sig(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, float)))


def fit_beta(s_tr, yb_tr):
    """Beta calibration (Kull et al. 2017) on fixed-sigmoid-mapped scores.

    q = sigmoid(s) in (0,1); logit(p) = a*ln q - b*ln(1-q) + c, a,b >= 0,
    fit by log-loss (L-BFGS-B, fixed inits -> deterministic).
    """
    EPS = 1e-6
    q = np.clip(_sig(s_tr), EPS, 1 - EPS)
    lq, l1q = np.log(q), np.log(1 - q)
    yb = np.asarray(yb_tr, float)

    def nll(t):
        a, b, c = t
        m = a * lq - b * l1q + c
        # log-loss of sigmoid(m) vs yb, stable form
        return float(np.mean(np.maximum(m, 0) - m * yb + np.log1p(np.exp(-np.abs(m)))))

    best, best_v = None, np.inf
    for init in ((1.0, 1.0, 0.0), (0.5, 0.5, 0.0), (2.0, 2.0, 0.0), (1.0, 0.5, 0.0)):
        r = minimize(nll, np.array(init), method="L-BFGS-B",
                     bounds=[(0, None), (0, None), (None, None)],
                     options={"maxiter": 500})
        if r.fun < best_v:
            best_v, best = float(r.fun), r.x
    a, b, c = (float(v) for v in best)

    def pred(s):
        qq = np.clip(_sig(s), EPS, 1 - EPS)
        return _sig(a * np.log(qq) - b * np.log(1 - qq) + c)

    return pred, {"a": round(a, 4), "b": round(b, 4), "c": round(c, 4)}


def lfo_oof(s_good, yb, fac):
    """Leave-factory-out OOF calibrated P(bad) for the three methods.

    Orientation is fixed BEFORE seeing OOD labels: the O4 adapted score is
    good-oriented (higher = more usable, IndEgo signs), so the bad-oriented
    input t = -s_good is calibrated with increasing=True conventions.
    """
    t = -np.asarray(s_good, float)
    facs = sorted(set(fac.tolist()))
    oof = {m: np.full(len(t), np.nan) for m in ("platt", "isotonic", "beta")}
    beta_params = {}
    for held in facs:
        te = np.where(fac == held)[0]
        tr = np.where(fac != held)[0]
        assert len(np.unique(yb[tr])) == 2, f"degenerate train fold {held}"
        oof["platt"][te] = fit_platt(t[tr], yb[tr])(t[te])
        oof["isotonic"][te] = fit_isotonic(t[tr], yb[tr])(t[te])
        fn, pr = fit_beta(t[tr], yb[tr])
        oof["beta"][te] = fn(t[te])
        beta_params[str(held)] = pr
    return oof, beta_params


# ---------------- metrics ----------------

def brier(yb, p):
    return float(np.mean((np.asarray(p, float) - yb) ** 2))


def logloss(yb, p):
    p = np.clip(np.asarray(p, float), 1e-12, 1 - 1e-12)
    return float(-np.mean(yb * np.log(p) + (1 - yb) * np.log(1 - p)))


def _group_index(groups):
    gmap = {g: i for i, g in enumerate(sorted(set(groups.tolist())))}
    gi = np.asarray([gmap[g] for g in groups])
    return [np.where(gi == k)[0] for k in range(len(gmap))]


def group_boot_ci(yb, p, groups, fn, n_boot=N_BOOT, seed=SEED):
    idx_by = _group_index(groups)
    rng = np.random.RandomState(seed)
    vals = []
    for _ in range(n_boot):
        idx = np.concatenate([idx_by[k] for k in
                              rng.randint(0, len(idx_by), len(idx_by))])
        if len(np.unique(yb[idx])) < 2:
            continue
        vals.append(fn(yb[idx], p[idx]))
    return [round(float(np.percentile(vals, 2.5)), 4),
            round(float(np.percentile(vals, 97.5)), 4)]


def reliability(yb, p, n_bins=10):
    yb = np.asarray(yb, float)
    p = np.asarray(p, float)
    edges = np.linspace(0, 1, n_bins + 1)
    out = []
    for i in range(n_bins):
        m = (p > edges[i]) & (p <= edges[i + 1] if i < n_bins - 1 else p <= edges[i + 1] + 1e-12)
        if i == 0:
            m = (p >= edges[0]) & (p <= edges[1])
        out.append({"bin": [round(float(edges[i]), 3), round(float(edges[i + 1]), 3)],
                    "n": int(m.sum()),
                    "mean_pred": round(float(p[m].mean()), 4) if m.any() else None,
                    "emp_bad_rate": round(float(yb[m].mean()), 4) if m.any() else None})
    return out


def balacc_decided(y_good, p_good, w):
    """Symmetric band half-width w on P(good): accept>=.5+w, reject<=.5-w."""
    y_good = np.asarray(y_good)
    p_good = np.asarray(p_good, float)
    accept = p_good >= 0.5 + w
    reject = p_good <= 0.5 - w
    decided = accept | reject
    pred = np.where(accept, 1, np.where(reject, 0, -1))
    if decided.sum() == 0:
        return {"review_rate": 1.0, "n_decided": 0, "ba": None,
                "false_accept": None, "false_reject": None}
    yy, pp = y_good[decided], pred[decided]
    g, b = yy == 1, yy == 0
    tpr = float((pp[g] == 1).mean()) if g.any() else float("nan")
    tnr = float((pp[b] == 0).mean()) if b.any() else float("nan")
    ba = 0.5 * (tpr + tnr)
    fa = float((pp[b] == 1).mean()) if b.any() else None  # accepted | decided bad
    fr = float((pp[g] == 0).mean()) if g.any() else None
    return {"review_rate": round(float(1 - decided.mean()), 4),
            "n_decided": int(decided.sum()),
            "ba": round(float(ba), 4),
            "err": round(float(1 - ba), 4),
            "false_accept": round(float(fa), 4) if fa is not None else None,
            "false_reject": round(float(fr), 4) if fr is not None else None}


def frontier(y_good, p_good, fac, grid):
    rows = []
    for w in grid:
        m = balacc_decided(y_good, p_good, w)
        dec = np.abs(np.asarray(p_good, float) - 0.5) > w
        ci = None
        if dec.sum() > 0 and len(np.unique(y_good[dec])) == 2:
            # grouped bootstrap of decided-BA at this w
            idx_by = _group_index(fac)
            rng = np.random.RandomState(SEED)
            vals = []
            yy = np.asarray(y_good)
            pp = np.asarray(p_good, float)
            for _ in range(N_BOOT):
                idx = np.concatenate([idx_by[k] for k in
                                      rng.randint(0, len(idx_by), len(idx_by))])
                d2 = np.abs(pp[idx] - 0.5) > w
                if d2.sum() == 0 or len(np.unique(yy[idx][d2])) < 2:
                    continue
                r = balacc_decided(yy[idx], pp[idx], w)
                vals.append(r["ba"])
            if vals:
                ci = [round(float(np.percentile(vals, 2.5)), 4),
                      round(float(np.percentile(vals, 97.5)), 4)]
        m["ba_ci95_factory_grouped"] = ci
        m["w"] = round(float(w), 3)
        rows.append(m)
    return rows


def main():
    lab = load_labels160()
    ids, y_good, fac, batch = lab["ids"], lab["y_good"], lab["factory"], lab["batch"]
    yb = (1 - y_good).astype(int)
    n_good, n_bad = int((y_good == 1).sum()), int((y_good == 0).sum())

    z = np.load(os.path.join(ADAPT_INPUTS, "z3_160.npz"), allow_pickle=True)
    assert [str(x) for x in z["ids"]] == ids, "z3 ids must align to labels160"
    names = [str(x) for x in z["names"]]
    X, Xi, yi = z["X"].astype(float), z["X_indego"].astype(float), z["y_indego"].astype(int)
    pix = np.load(PIXEL160, allow_pickle=True)
    pm = {str(i): k for k, i in enumerate([str(x) for x in pix["ids"]])}
    X26o = np.asarray([pix["X10"][pm[i]] for i in ids], float)
    fj = json.load(open(os.path.join(BUNDLE, "caches", "gpu", "features.json")))["windows"]
    m_gold = json.load(open(os.path.join(BUNDLE, "labels", "manifest_gold.json")))["items"]
    gids = [it["id"] for it in m_gold]
    yg = np.asarray([int(it["label"]) for it in m_gold])
    X26g = np.asarray([[float(fj[i]["full"][n]) for n in FEATURE_NAMES26] for i in gids])
    frozen = load_frozen160(ids)

    z6 = [names.index(n) for n in Z3_SIX]
    s8 = [FEATURE_NAMES26.index(n) for n in STABLE8]
    _, s_a, au_a, sg_a = combo_scores(Xi[:, z6], X[:, z6], yi)
    _, s_b, au_b, sg_b = combo_scores(
        np.hstack([X26g[:, s8], Xi[:, z6]]), np.hstack([X26o[:, s8], X[:, z6]]), yg)

    # sanity: reproduce O4 headline AUROCs
    assert abs(fast_auc(y_good, s_a) - 0.7592) < 0.002, fast_auc(y_good, s_a)
    assert abs(fast_auc(y_good, s_b) - 0.781) < 0.005, fast_auc(y_good, s_b)

    base_rate = float(yb.mean())
    results = {"recipes": {}}
    OOF_PBADS = {}
    GRID = [round(0.01 * i, 2) for i in range(31)]  # w = 0..0.30
    for tag, s in (("(a) Z3 transferable-6", s_a),
                   ("(b) stable-8 + transferable-6", s_b)):
        oof, beta_params = lfo_oof(s, yb, fac)
        OOF_PBADS[tag] = {m: np.asarray(p, float) for m, p in oof.items()}
        cal = {}
        for m, p in oof.items():
            c = {
                "brier": round(brier(yb, p), 4),
                "brier_ci95_factory_grouped": group_boot_ci(yb, p, fac, brier),
                "logloss": round(logloss(yb, p), 4),
                "logloss_ci95_factory_grouped": group_boot_ci(yb, p, fac, logloss),
                "auroc_rank": round(fast_auc(yb, p), 4),
                "reliability_deciles": reliability(yb, p),
                "p_bad_max": round(float(p.max()), 4),
                "p_bad_median_bad": round(float(np.median(p[yb == 1])), 4),
                "p_bad_median_good": round(float(np.median(p[yb == 0])), 4),
                "frac_bad_above_50": round(float((p[yb == 1] > 0.5).mean()), 4),
                "frac_good_below_50": round(float((p[yb == 0] < 0.5).mean()), 4),
            }
            cal[m] = c
        cal["base_rate"] = {"brier": round(brier(yb, np.full_like(yb, base_rate, dtype=float)), 4),
                            "logloss": round(logloss(yb, np.full_like(yb, base_rate, dtype=float)), 4)}
        cal["uncal_sigmoid"] = {
            "brier": round(brier(yb, _sig(s)), 4),
            "logloss": round(logloss(yb, _sig(s)), 4)}
        # frontier on calibrated P(good) = 1 - P(bad), per method
        fronts = {}
        for m, p in oof.items():
            fronts[m] = frontier(y_good, 1 - p, fac, GRID)
        # operating point: min OOD err s.t. review <= 60%
        ops = {}
        for m, rows in fronts.items():
            elig = [r for r in rows if r["review_rate"] <= 0.60 and r["ba"] is not None]
            ops[m] = min(elig, key=lambda r: (r["err"], -r["w"])) if elig else None
        results["recipes"][tag] = {
            "auroc_raw": round(fast_auc(y_good, s), 4),
            "score_min": round(float(s.min()), 4), "score_max": round(float(s.max()), 4),
            "oof_calibration": cal,
            "frontiers": fronts,
            "operating_points_review_le_60": ops,
            "lfo_beta_params_median": {
                k: round(float(np.median([p[k] for p in beta_params.values()])), 4)
                for k in ("a", "b", "c")},
        }

    # apples-to-apples frozen frontier on the SAME 160 (band on usable_prob)
    fgrid = frontier(y_good, frozen, fac, GRID)
    felig = [r for r in fgrid if r["review_rate"] <= 0.60 and r["ba"] is not None]
    results["frozen_same160"] = {
        "frontier": fgrid,
        "operating_point_review_le_60": min(felig, key=lambda r: (r["err"], -r["w"])),
        "auroc": round(fast_auc(y_good, frozen), 4),
    }

    # illustrative ONLY: single-threshold accept/reject on OOF P(bad) at
    # theory-derived cost cutoffs 1/(1+C). Threshold VALUES come from cost
    # theory, but any choice among them on this test set is selection-biased.
    # NOT an operating recommendation; explains what the symmetric band lacks.
    ill = {}
    for tag, pbads in OOF_PBADS.items():
        ill[tag] = {}
        for m, p in pbads.items():
            rows = []
            for C, cut in ((1, 0.5), (3, 0.25), (4, 0.2), (5, 1 / 6)):
                pred = np.where(np.asarray(p, float) >= cut, 0, 1)  # 0=reject,1=accept
                g, b = y_good == 1, y_good == 0
                tpr = float((pred[g] == 1).mean())
                tnr = float((pred[b] == 0).mean())
                rows.append({"miss_cost_C": C, "cut_bad_ge": round(cut, 4),
                             "ba": round(0.5 * (tpr + tnr), 4),
                             "false_accept": round(float((pred[b] == 1).mean()), 4),
                             "false_reject": round(float((pred[g] == 0).mean()), 4)})
            ill[tag][m] = rows
    results["illustrative_cost_thresholds"] = ill

    out = {
        "experiment": "P1.01",
        "title": "Calibrate the adapted OOD score on 160 labels + joint review/error frontier",
        "n": len(ids), "n_good": n_good, "n_bad": n_bad,
        "n_factories": int(len(set(fac.tolist()))),
        "label_source": "human (labels/ood160_labels.json; 138 good / 22 bad, 47 factories, single labeller)",
        "protocol": ("leave-factory-out grouped CV: calibrator fit on 46 factories, "
                     "evaluated on the held-out factory; pooled OOF predictions. "
                     "Factory-grouped bootstrap CI (n_boot=2000, seed=11)."),
        "adapted_scores": ("recomputed exactly per O4 (IndEgo-signed per-domain z-score, "
                           "equal mean); O4 AUROCs reproduced to <0.005."),
        "device": DEV, "git_sha": git_sha(),
        "command": "OMP_NUM_THREADS=2 python3 eval/exp_calib160/run_p1.py",
        "base_bad_rate": round(base_rate, 4),
        **results,
    }
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "P1.01.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps({k: v for k, v in out.items() if k not in ("recipes", "frozen_same160")}, indent=1))
    for tag, r in results["recipes"].items():
        print(tag, {m: {k: v for k, v in c.items() if k != "reliability_deciles"}
                     for m, c in r["oof_calibration"].items() if isinstance(c, dict) and "brier" in c})
        print("  op:", {m: o for m, o in r["operating_points_review_le_60"].items()})
    print("frozen_same160 op:", results["frozen_same160"]["operating_point_review_le_60"])


if __name__ == "__main__":
    main()
