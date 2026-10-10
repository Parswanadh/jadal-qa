"""Nested LOCO of the two-view ensemble (EG2 mean-embedding + 26 per-pixel feats).

Protocol (all selection INSIDE training clips of each outer fold):
- Outer: leave-one-clip-out over 7 clips (n=150 human gold windows).
- Inner: leave-one-clip-out over the 6 training clips; C grid
  {0.003,0.01,0.03,0.1,0.3,1} per view, picked by inner-OOF AUROC.
- Fusion: equal-weight probability average (no learned weights); rank-average
  reported too (global ranks of OOF probs, no within-clip ranking).
- Calibration: Platt (sigmoid) + isotonic fitted on inner-OOF train probs,
  applied to the held-out clip. Reported for reference; ranking metrics use
  uncalibrated probs.
- Thresholds: operating points chosen on inner-OOF train probs at 5% and 10%
  false-reject on training-good windows; applied to held-out clip.
- Uncertainty: clip-bootstrap 95% CI (2,000 resamples, seed 0).
- Same-fold single-view baselines with the same nested-C procedure.

Outputs eval/results/ensemble/nested_loco.json (+ summary printed).
Deterministic: fixed seeds, lbfgs, OMP_NUM_THREADS=2 expected.
"""
import json
import os

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")

import numpy as np
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.isotonic import IsotonicRegression

GRID = [0.003, 0.01, 0.03, 0.1, 0.3, 1.0]
BOOT = 2000
SEED = 0

MANIFEST = "/home/parshu/projects/jadal-qa-wt/spectral_data/manifest_gold.json"
FEATS = "/home/parshu/projects/jadal-qa-wt/gpu-pipeline/eval/results/gpu/features.json"
SEG = "/home/parshu/projects/jadal-qa-wt/frames10k/eval/results/frames10k/human_segments/segments.npz"


def load():
    man = json.load(open(MANIFEST))
    M = {i["id"]: i for i in man["items"]}
    F = json.load(open(FEATS))["windows"]
    z = np.load(SEG, allow_pickle=True)
    ids = [str(i) for i in z["ids"]]
    clip = np.array([M[i]["clip"] for i in ids])
    y = np.array([M[i]["label"] for i in ids], dtype=int)
    reason = np.array([M[i].get("reason", "") for i in ids])
    E = z["F"].astype(np.float64)
    cnt = z["counts"].astype(int)
    Xe = np.stack([E[k, :max(1, cnt[k])].mean(0) for k in range(len(ids))])
    Xe /= np.maximum(np.linalg.norm(Xe, axis=1, keepdims=True), 1e-12)
    names = sorted(F[ids[0]]["full"])
    Xg = np.nan_to_num(np.array([[F[i]["full"][n] for n in names] for i in ids], float))
    return ids, clip, y, reason, Xe, Xg, names, man


def fit_predict(Xtr, ytr, Xte, C):
    sc = StandardScaler()
    m = LogisticRegression(max_iter=5000, C=C, class_weight="balanced")
    m.fit(sc.fit_transform(Xtr), ytr)
    return m.predict_proba(sc.transform(Xte))[:, 1], sc, m


def inner_oof(X, y, clip, train_idx, C):
    """Inner LOCO OOF probs over the training clips for a fixed C."""
    tclips = sorted(set(clip[train_idx]))
    p = np.zeros(len(train_idx))
    for c in tclips:
        te = train_idx[clip[train_idx] == c]
        tr = train_idx[clip[train_idx] != c]
        pp, _, _ = fit_predict(X[tr], y[tr], X[te], C)
        pos = np.searchsorted(train_idx, te)
        p[pos] = pp
    return p


def pick_C(X, y, clip, train_idx):
    aucs = {}
    for C in GRID:
        p = inner_oof(X, y, clip, train_idx, C)
        try:
            aucs[C] = roc_auc_score(y[train_idx], p)
        except ValueError:
            aucs[C] = 0.5
    best = max(GRID, key=lambda c: (aucs[c], -GRID.index(c)))
    return best, aucs


def platt_fit(p, y):
    m = LogisticRegression(max_iter=5000)
    m.fit(np.asarray(p).reshape(-1, 1), y)
    return m


def main():
    rng = np.random.default_rng(SEED)
    ids, clip, y, reason, Xe, Xg, names, man = load()
    clips = sorted(set(clip))
    n = len(y)

    pe = np.zeros(n)  # nested OOF: EG2 view
    pg = np.zeros(n)  # nested OOF: per-pixel view
    pe_cal = np.zeros(n)  # Platt-calibrated ensemble
    ie_cal = np.zeros(n)  # isotonic-calibrated ensemble
    thr05 = np.zeros(n, dtype=int)  # verdict at 5%-FR train threshold
    thr10 = np.zeros(n, dtype=int)
    t05_hist, t10_hist = [], []
    chosen = {}

    for c in clips:
        te = np.where(clip == c)[0]
        tr = np.where(clip != c)[0]
        Ce, aucse = pick_C(Xe, y, clip, tr)
        Cg, aucsg = pick_C(Xg, y, clip, tr)
        chosen[c] = {"C_eg2": Ce, "C_px": Cg,
                     "inner_auc_eg2": aucse, "inner_auc_px": aucsg}
        pre, _, _ = fit_predict(Xe[tr], y[tr], Xe[te], Ce)
        prg, _, _ = fit_predict(Xg[tr], y[tr], Xg[te], Cg)
        pe[te], pg[te] = pre, prg
        # train-OOF probs with chosen C -> calibration + thresholds (train only)
        oe = inner_oof(Xe, y, clip, tr, Ce)
        og = inner_oof(Xg, y, clip, tr, Cg)
        oens = (oe + og) / 2
        ytr = y[tr]
        pm = platt_fit(oens, ytr)
        pe_cal[te] = pm.predict_proba(((pre + prg) / 2).reshape(-1, 1))[:, 1]
        ir = IsotonicRegression(out_of_bounds="clip").fit(oens, ytr)
        ie_cal[te] = ir.predict((pre + prg) / 2)
        good = oens[ytr == 1]
        t05 = float(np.quantile(good, 0.05))
        t10 = float(np.quantile(good, 0.10))
        t05_hist.append(t05)
        t10_hist.append(t10)
        ens_te = (pre + prg) / 2
        thr05[te] = (ens_te > t05).astype(int)
        thr10[te] = (ens_te > t10).astype(int)

    ensp = (pe + pg) / 2
    ensr = (rankdata(pe) + rankdata(pg)) / 2 / len(pe)

    def boot(p, B=BOOT):
        out = []
        for _ in range(B):
            cs = rng.choice(clips, len(clips))
            idx = np.concatenate([np.where(clip == c)[0] for c in cs])
            if len(set(y[idx])) < 2:
                continue
            out.append(roc_auc_score(y[idx], p[idx]))
        out = np.asarray(out)
        return [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))]

    def bal(pred, sel=None):
        s = np.arange(n) if sel is None else np.asarray(sel)
        pr = np.asarray(pred)[s]
        yy = y[s]
        return float(((pr[yy == 1] == 1).mean() + (pr[yy == 0] == 0).mean()) / 2)

    def acc(pred, sel=None):
        s = np.arange(n) if sel is None else np.asarray(sel)
        return float((np.asarray(pred)[s] == y[s]).mean())

    res = {"n": n, "clips": clips, "C_grid": GRID,
           "protocol": ("outer LOCO (7 clips); inner LOCO on 6 train clips picks C; "
                        "Platt/isotonic + thresholds fit on inner-OOF train probs only; "
                        "equal-weight probability average, no learned stacking"),
           "label_source": "human gold (single labeller)",
           "labels_sha256": man.get("labels_sha256", ""),
           "chosen_C": chosen,
           "views": {},
           "gates": {"AUROC>=0.85": None, "CI_low>=0.72": None, "no_clip<0.60": None}}
    for tag, p in [("eg2_nested", pe), ("pixel_nested", pg),
                   ("ensemble_probavg", ensp), ("ensemble_rankavg", ensr),
                   ("ensemble_platt", pe_cal), ("ensemble_isotonic", ie_cal)]:
        lo, hi = boot(p)
        res["views"][tag] = {"auroc": float(roc_auc_score(y, p)),
                             "ci95": [lo, hi],
                             "balacc@0.5": bal((p > 0.5).astype(int))}
    # operating points from frozen-train thresholds
    for tag, pred in [("frozen_5pctFR_thresh", thr05), ("frozen_10pctFR_thresh", thr10)]:
        fr = int(((pred == 0) & (y == 1)).sum())
        res["views"][tag] = {"balacc": bal(pred), "acc": acc(pred),
                             "good_wrongly_rejected": [fr, int((y == 1).sum())],
                             "mean_thresh": float(np.mean(t05_hist if "5" in tag else t10_hist))}
    # per-clip table (prob-average + singles)
    res["per_clip"] = {}
    for c in clips:
        idx = np.where(clip == c)[0]
        row = {"n": int(len(idx)), "n_good": int(y[idx].sum()), "n_bad": int((y[idx] == 0).sum())}
        for tag, p in [("ensemble", ensp), ("eg2", pe), ("pixel", pg)]:
            try:
                row[tag + "_auroc"] = float(roc_auc_score(y[idx], p[idx]))
            except ValueError:
                row[tag + "_auroc"] = None
        row["balacc@0.5_ens"] = bal((ensp > 0.5).astype(int), idx)
        res["per_clip"][c] = row
    # per-reason recall at 0.5 (bad correctly rejected) + one-vs-rest AUROC
    res["per_reason"] = {}
    for r in ["NO_ACTIVE_WORK", "HANDS_NOT_VISIBLE", "DUPLICATE", "TOO_DARK", "UNDECODABLE"]:
        idx = np.where(reason == r)[0]
        sel = np.concatenate([np.where(y == 1)[0], idx])
        try:
            ovr = float(roc_auc_score(y[sel], ensp[sel]))
        except ValueError:
            ovr = None
        res["per_reason"][r] = {"n_bad": int(len(idx)),
                                "recall_reject@0.5": float((ensp[idx] <= 0.5).mean()) if len(idx) else None,
                                "ovr_auroc": ovr}
    # coverage-vs-balanced-accuracy curve (confidence = |p-0.5|), with bootstrap CIs
    conf = np.abs(ensp - 0.5)
    order = np.argsort(-conf)
    res["coverage_curve"] = []
    for cov in (1.0, 0.8, 0.6, 0.5):
        k = int(cov * n)
        s = order[:k]
        pr = (ensp[s] > 0.5).astype(int)
        yy = y[s]
        b = float(((pr[yy == 1] == 1).mean() + (pr[yy == 0] == 0).mean()) / 2)
        boots = []
        for _ in range(BOOT):
            bs = rng.choice(s, len(s), replace=True)
            if len(set(y[bs])) < 2:
                continue
            pp, yb = ensp[bs], y[bs]
            prb = (pp > 0.5).astype(int)
            boots.append(((prb[yb == 1] == 1).mean() + (prb[yb == 0] == 0).mean()) / 2)
        boots = np.asarray(boots)
        fr = int(((ensp[s] <= 0.5) & (y[s] == 1)).sum())
        res["coverage_curve"].append({
            "coverage": cov, "k": k, "balacc": b,
            "balacc_ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
            "good_wrongly_rejected": [fr, int((y[s] == 1).sum())]})
    # gates
    e = res["views"]["ensemble_probavg"]
    res["gates"]["AUROC>=0.85"] = bool(e["auroc"] >= 0.85)
    res["gates"]["CI_low>=0.72"] = bool(e["ci95"][0] >= 0.72)
    minclip = min(v["ensemble_auroc"] for v in res["per_clip"].values() if v["ensemble_auroc"] is not None)
    res["gates"]["no_clip<0.60"] = bool(minclip >= 0.60)
    res["gates"]["min_clip_auroc"] = float(minclip)

    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "ensemble")
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "nested_loco.json"), "w") as f:
        json.dump(res, f, indent=1)
    print(f"n={n} clips={len(clips)}")
    for tag, v in res["views"].items():
        if "auroc" in v:
            print(f"{tag:20s} AUROC {v['auroc']:.3f} [{v['ci95'][0]:.3f},{v['ci95'][1]:.3f}] bal@0.5 {v['balacc@0.5']:.3f}")
        else:
            print(f"{tag:20s} balacc {v['balacc']:.3f} rej-good {v['good_wrongly_rejected']} mean_thr {v['mean_thresh']:.3f}")
    print("per-clip:", {c: res["per_clip"][c]["ensemble_auroc"] for c in clips})
    print("chosen C:", {c: (chosen[c]["C_eg2"], chosen[c]["C_px"]) for c in clips})
    print("gates:", res["gates"])
    print("coverage:", [(r["coverage"], round(r["balacc"], 3), r["good_wrongly_rejected"]) for r in res["coverage_curve"]])


if __name__ == "__main__":
    main()
