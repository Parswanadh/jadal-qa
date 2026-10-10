#!/usr/bin/env python3
"""N5 — refit/recalibrate the decision heads for the optimized (sparse) pipeline.

Reads
  frozen gold features : $BUNDLE/caches/gpu/features.json (pixel, full sampling)
                         $BUNDLE/caches/frames10k/segments.npz (EG2, 2 fps)
  sparse gold features : eval/results/exp_recal/sparse_gold.npz
  frozen config        : configs/ensemble_v1.json (C grid, shipped head, threshold/band)
  frozen OOD scores    : eval/results/ensemble/ood_scores{,2}.json
  sparse OOD features  : eval/results/exp_recal/sparse_ood160.npz
  OOD labels           : $BUNDLE/labels/ood160_labels.json

Writes
  eval/results/exp_recal/N5.01.json  (gold: frozen vs sparse LOCO, refit head, thresholds)
  eval/results/exp_recal/N5.02.json  (OOD: frozen vs sparse-refit, factory-grouped CI)

Protocol mirrors eval/ensemble_nested_loco.py: outer LOCO over 7 clips, inner LOCO
picks C from the grid, StandardScaler + LogisticRegression(max_iter=5000,
class_weight='balanced'), equal-weight probability average, clip-bootstrap CI.
"""
from __future__ import annotations

import json
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "2")

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", ".."))
BUNDLE = os.environ.get("JADAL_BUNDLE", "/workspace/jadal_pod_bundle")
CONFIG = os.path.join(REPO, "configs", "ensemble_v1.json")
OUTDIR = os.path.join(REPO, "eval", "results", "exp_recal")
GRID = [0.003, 0.01, 0.03, 0.1, 0.3, 1.0]
BOOT = 2000
SEED = 0
MAX_ITER = 5000


# ------------------------------------------------------------------ load
def load_frozen_gold(cfg):
    F = json.load(open(os.path.join(BUNDLE, "caches", "gpu", "features.json")))["windows"]
    z = np.load(os.path.join(BUNDLE, "caches", "frames10k", "segments.npz"),
                allow_pickle=True)
    ids = [str(i) for i in z["ids"]]
    y = np.array(z["labels"]).astype(int)
    clip = np.array([str(c) for c in z["clips"]])
    E = z["F"].astype(np.float64)
    cnt = z["counts"].astype(int)
    Xe = np.stack([E[k, :max(1, cnt[k])].mean(0) for k in range(len(ids))])
    Xe /= np.maximum(np.linalg.norm(Xe, axis=1, keepdims=True), 1e-12)
    names = cfg["views"]["pixel"]["feature_names"]
    Xg = np.nan_to_num(np.array([[F[i]["full"][n] for n in names] for i in ids], float))
    return ids, y, clip, Xe, Xg, names


def load_sparse(path):
    z = np.load(path, allow_pickle=True)
    ids = [str(i) for i in z["ids"]]
    return {"ids": ids, "y": np.array(z["labels"]).astype(int),
            "clip": np.array([str(c) for c in z["clips"]]),
            "Xe": np.asarray(z["X_eg2"], dtype=np.float64),
            "Xg": np.asarray(z["X_pixel"], dtype=np.float64),
            "npix": np.asarray(z["n_frames_pixel"]),
            "neg": np.asarray(z["n_frames_eg2"]),
            "fps": np.asarray(z["fps_pixel"])}


def align(ids_ref, sp):
    idx = [sp["ids"].index(i) for i in ids_ref]
    return {"Xe": sp["Xe"][idx], "Xg": sp["Xg"][idx], "y": sp["y"][idx],
            "clip": sp["clip"][idx], "npix": sp["npix"][idx], "neg": sp["neg"][idx],
            "fps": sp["fps"][idx]}


# ------------------------------------------------------------------ heads
def fit_predict(Xtr, ytr, Xte, C):
    sc = StandardScaler()
    m = LogisticRegression(max_iter=MAX_ITER, C=C, class_weight="balanced")
    m.fit(sc.fit_transform(Xtr), ytr)
    return m.predict_proba(sc.transform(Xte))[:, 1], sc, m


def fit_head(X, y, C):
    sc = StandardScaler()
    m = LogisticRegression(max_iter=MAX_ITER, C=C, class_weight="balanced")
    m.fit(sc.fit_transform(X), y)
    return {"mean": sc.mean_.tolist(), "scale": sc.scale_.tolist(),
            "coef": m.coef_[0].tolist(), "intercept": float(m.intercept_[0]), "C": C}


def head_prob(X, h):
    z = (np.asarray(X, float) - np.array(h["mean"])) / np.array(h["scale"])
    return 1.0 / (1.0 + np.exp(-(z @ np.array(h["coef"]) + h["intercept"])))


def inner_oof(X, y, clip, train_idx, C):
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


def nested_loco(Xe, Xg, y, clip):
    n = len(y)
    pe, pg = np.zeros(n), np.zeros(n)
    chosen = {}
    for c in sorted(set(clip)):
        te = np.where(clip == c)[0]
        tr = np.where(clip != c)[0]
        Ce, _ = pick_C(Xe, y, clip, tr)
        Cg, _ = pick_C(Xg, y, clip, tr)
        chosen[c] = {"C_eg2": Ce, "C_px": Cg}
        pre, _, _ = fit_predict(Xe[tr], y[tr], Xe[te], Ce)
        prg, _, _ = fit_predict(Xg[tr], y[tr], Xg[te], Cg)
        pe[te], pg[te] = pre, prg
    return pe, pg, chosen


def fixed_loco(Xe, Xg, y, clip, Ce, Cg):
    pe, pg = np.zeros(len(y)), np.zeros(len(y))
    for c in sorted(set(clip)):
        te = np.where(clip == c)[0]
        tr = np.where(clip != c)[0]
        pe[te], _, _ = fit_predict(Xe[tr], y[tr], Xe[te], Ce)
        pg[te], _, _ = fit_predict(Xg[tr], y[tr], Xg[te], Cg)
    return pe, pg


# ------------------------------------------------------------------ metrics
def clip_boot(y, p, clip, B=BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    names = sorted(set(clip))
    out = []
    for _ in range(B):
        cs = rng.choice(names, len(names))
        idx = np.concatenate([np.where(clip == c)[0] for c in cs])
        if len(set(y[idx])) < 2:
            continue
        out.append(roc_auc_score(y[idx], p[idx]))
    out = np.asarray(out)
    return [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))]


def group_boot(y, p, groups, B=BOOT, seed=SEED):
    """Group (factory) bootstrap CI: resample groups with replacement."""
    rng = np.random.default_rng(seed)
    names = sorted(set(groups))
    out = []
    for _ in range(B):
        gs = rng.choice(names, len(names))
        idx = np.concatenate([np.where(groups == g)[0] for g in gs])
        if len(set(y[idx])) < 2:
            continue
        out.append(roc_auc_score(y[idx], p[idx]))
    out = np.asarray(out)
    return [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))]


def balacc(y, pred, sel=None):
    s = np.arange(len(y)) if sel is None else np.asarray(sel)
    pr, yy = np.asarray(pred)[s], y[s]
    return float(((pr[yy == 1] == 1).mean() + (pr[yy == 0] == 0).mean()) / 2)


def op_point(p, y, lo, hi):
    """3-way verdict: accept p>hi, reject p<lo, review in [lo,hi]."""
    v = np.where(p > hi, "accept", np.where(p < lo, "reject", "review"))
    nrev = int((v == "review").sum())
    sel = v != "review"
    pred = (v[sel] == "accept").astype(int)
    yy = y[sel]
    ba = ((pred[yy == 1] == 1).mean() + (pred[yy == 0] == 0).mean()) / 2
    return {"accept": int((v == "accept").sum()), "reject": int((v == "reject").sum()),
            "review": nrev, "review_share": round(nrev / len(p), 4),
            "confident_n": int(sel.sum()),
            "confident_balacc": round(float(ba), 5),
            "confident_acc": round(float((pred == yy).mean()), 5)}


def derive_band(p, y, target_review_share):
    """Pick (lo,hi) with review share ~= target and max confident balanced accuracy."""
    best = None
    for c in np.quantile(p, np.linspace(0.2, 0.8, 25)):
        for w in np.linspace(0.02, 0.30, 60):
            lo, hi = c - w, c + w
            r = (p >= lo) & (p <= hi)
            rs = r.mean()
            if abs(rs - target_review_share) > 0.02:
                continue
            sel = ~r
            if sel.sum() < 20 or len(set(y[sel])) < 2:
                continue
            pred = (p[sel] > hi).astype(int)
            yy = y[sel]
            ba = ((pred[yy == 1] == 1).mean() + (pred[yy == 0] == 0).mean()) / 2
            cand = {"lo": round(float(lo), 4), "hi": round(float(hi), 4),
                    "review_share": round(float(rs), 4), "confident_balacc": round(float(ba), 5)}
            if best is None or (ba, -abs(rs - target_review_share)) > \
                    (best["confident_balacc"], -abs(best["review_share"] - target_review_share)):
                best = cand
    return best


# ------------------------------------------------------------------ main
def main():
    cfg = json.load(open(CONFIG))
    ids, y, clip, Xe_f, Xg_f, names = load_frozen_gold(cfg)
    sp = load_sparse(os.path.join(OUTDIR, "sparse_gold.npz"))
    a = align(ids, sp)
    assert np.array_equal(a["y"], y) and np.array_equal(a["clip"], clip), "gold align mismatch"

    frozen_Ce, frozen_Cg = float(cfg["views"]["eg2"]["C"]), float(cfg["views"]["pixel"]["C"])
    res = {"experiment": "N5.01", "n": int(len(y)), "n_good": int(y.sum()),
           "n_bad": int((1 - y).sum()), "clips": sorted(set(clip.tolist())),
           "label_source": "human gold (single labeller)",
           "labels_sha256": json.load(open(os.path.join(BUNDLE, "labels", "manifest_gold.json"))).get("labels_sha256", ""),
           "protocol": ("outer LOCO (7 clips); inner LOCO picks C from {0.003..1}; "
                        "StandardScaler+LogisticRegression(max_iter=5000,balanced); "
                        "equal-weight prob average; clip-bootstrap B=2000 seed0"),
           "device": "A40",
           "frozen_config": {"C_eg2": frozen_Ce, "C_px": frozen_Cg,
                             "threshold": cfg["threshold"], "abstain_band": cfg["abstain_band"],
                             "declared_abstain_stats": cfg.get("abstain_band_oof_stats"),
                             "threshold_5pctFR": cfg.get("threshold_5pctFR_fullOOF"),
                             "threshold_10pctFR": cfg.get("threshold_10pctFR_fullOOF")},
           "sparse_sampling": {"pixel_fps": float(np.median(a["fps"])),
                               "pixel_frames_p50": int(np.median(a["npix"])),
                               "eg2_frames_p50": int(np.median(a["neg"])),
                               "observed_s": 4.0,
                               "code_path": "eval/exp_latency_fast/run_fast.py decode_three_branch_cfg + run_e8.encode_eg2"}}

    # ---- nested LOCO: frozen features vs sparse features
    pe_f, pg_f, chosen_f = nested_loco(Xe_f, Xg_f, y, clip)
    pe_s, pg_s, chosen_s = nested_loco(a["Xe"], a["Xg"], y, clip)
    ens_f, ens_s = (pe_f + pg_f) / 2, (pe_s + pg_s) / 2

    # fixed-C LOCO (shipped C) for both
    fe_f, fg_f = fixed_loco(Xe_f, Xg_f, y, clip, frozen_Ce, frozen_Cg)
    fe_s, fg_s = fixed_loco(a["Xe"], a["Xg"], y, clip, frozen_Ce, frozen_Cg)
    fens_f, fens_s = (fe_f + fg_f) / 2, (fe_s + fg_s) / 2

    def view(p, tag):
        return {"auroc": round(float(roc_auc_score(y, p)), 4),
                "ci95": [round(x, 4) for x in clip_boot(y, p, clip)],
                "balacc@0.5": round(balacc(y, (p > 0.5).astype(int)), 4)}

    res["loco"] = {
        "frozen_features": {"eg2": view(pe_f, "e"), "pixel": view(pg_f, "p"),
                            "ensemble_probavg": view(ens_f, "ens"),
                            "chosen_C": chosen_f},
        "sparse_features_nestedC": {"eg2": view(pe_s, "e"), "pixel": view(pg_s, "p"),
                                    "ensemble_probavg": view(ens_s, "ens"),
                                    "chosen_C": chosen_s},
        "frozen_features_fixedC": {"eg2": view(fe_f, "e"), "pixel": view(fg_f, "p"),
                                   "ensemble_probavg": view(fens_f, "ens")},
        "sparse_features_fixedC": {"eg2": view(fe_s, "e"), "pixel": view(fg_s, "p"),
                                   "ensemble_probavg": view(fens_s, "ens")},
    }

    # ---- score shift: frozen heads applied to sparse features (LOCO fixed-C)
    p_eg2_frozen_on_sparse = head_prob(a["Xe"], cfg["views"]["eg2"])
    p_px_frozen_on_sparse = head_prob(a["Xg"], cfg["views"]["pixel"])
    p_frozen_on_sparse = (p_eg2_frozen_on_sparse + p_px_frozen_on_sparse) / 2
    res["score_shift_gold"] = {
        "frozen_heads_on_frozen_features_mean": round(float(ens_f.mean()), 4),
        "frozen_heads_on_sparse_features_mean": round(float(p_frozen_on_sparse.mean()), 4),
        "delta_mean": round(float(p_frozen_on_sparse.mean() - ens_f.mean()), 4),
        "note": "resubstitution frozen heads on sparse features vs nested-OOF frozen features",
    }

    # ---- operating points on gold (band [0.4,0.6], the frozen abstain band)
    lo0, hi0 = float(cfg["abstain_band"][0]), float(cfg["abstain_band"][1])
    res["operating_point_gold"] = {
        "frozen_features_frozen_band": op_point(ens_f, y, lo0, hi0),
        "sparse_frozen_heads_frozen_band": op_point(p_frozen_on_sparse, y, lo0, hi0),
        "sparse_refit_frozen_band": op_point(ens_s, y, lo0, hi0),
    }
    target_rs = res["operating_point_gold"]["frozen_features_frozen_band"]["review_share"]
    band = derive_band(ens_s, y, target_rs)
    res["operating_point_gold"]["sparse_refit_derived_band"] = {
        **(band or {}), **(op_point(ens_s, y, band["lo"], band["hi"]) if band else {})}
    res["operating_point_gold"]["target_review_share"] = target_rs
    res["operating_point_gold"]["target_confident_balacc"] = \
        res["operating_point_gold"]["frozen_features_frozen_band"]["confident_balacc"]

    # ---- corpus-style profile operating points (the L3 verdict path) + 3-way agreement
    def vdist(p, acc, rej):
        v = np.where(p >= acc, "accept", np.where(p <= rej, "reject", "review"))
        return {"accept": int((v == "accept").sum()), "reject": int((v == "reject").sum()),
                "review": int((v == "review").sum()),
                "review_share": round(float((v == "review").mean()), 4)}
    prof = {}
    for name, (acc, rej) in {"default": (0.8, 0.2), "factory_1080p30": (0.75, 0.25)}.items():
        prof[name] = {"frozen_features": vdist(ens_f, acc, rej),
                      "sparse_frozen_heads": vdist(p_frozen_on_sparse, acc, rej),
                      "sparse_refit": vdist(ens_s, acc, rej)}
    res["operating_point_profile_thresholds_gold"] = prof

    def vagree(a, b):
        return round(float((a == b).mean()), 4)
    va_frozen = np.where(ens_f > hi0, "accept", np.where(ens_f < lo0, "reject", "review"))
    va_sparse_frozen = np.where(p_frozen_on_sparse > hi0, "accept",
                                np.where(p_frozen_on_sparse < lo0, "reject", "review"))
    va_sparse_refit = np.where(ens_s > hi0, "accept", np.where(ens_s < lo0, "reject", "review"))
    res["verdict_agreement_gold"] = {
        "operating_point": f"band [{lo0},{hi0}]",
        "frozen_vs_sparse_frozen_heads": vagree(va_frozen, va_sparse_frozen),
        "frozen_vs_sparse_refit": vagree(va_frozen, va_sparse_refit),
        "note": "3-way accept/reject/review agreement on the 150 gold windows; "
                "the L3 corpus blocker was 0.5433 with the frozen heads",
    }

    # ---- re-derive 5%/10% FR thresholds (config style) on sparse OOF
    def fr_thr(p, q):
        return round(float(np.quantile(p[y == 1], q)), 4)
    res["sparse_fr_thresholds"] = {"threshold_5pctFR": fr_thr(ens_s, 0.05),
                                   "threshold_10pctFR": fr_thr(ens_s, 0.10),
                                   "frozen_threshold_5pctFR": cfg.get("threshold_5pctFR_fullOOF"),
                                   "frozen_threshold_10pctFR": cfg.get("threshold_10pctFR_fullOOF")}

    # ---- shipped sparse heads (fit on all 150, frozen C) + sparse-refit nested C
    Ce_s = max(set(c["C_eg2"] for c in chosen_s.values()), key=lambda c: [c["C_eg2"] for c in chosen_s.values()].count(c))
    Cg_s = max(set(c["C_px"] for c in chosen_s.values()), key=lambda c: [c["C_px"] for c in chosen_s.values()].count(c))
    head_eg2 = fit_head(a["Xe"], y, frozen_Ce)
    head_px = fit_head(a["Xg"], y, frozen_Cg)
    res["refit_heads"] = {
        "chosen_C_mode_sparse": {"C_eg2": Ce_s, "C_px": Cg_s},
        "eg2": head_eg2, "pixel": head_px,
        "note": "fit on all 150 gold sparse features with the FROZEN C (0.003/0.3) to keep everything else identical; coefficients replace the frozen head for the optimized pipeline",
    }

    # resubstitution op point of the shipped refit heads
    p_eg2_ship = head_prob(a["Xe"], head_eg2)
    p_px_ship = head_prob(a["Xg"], head_px)
    p_ship = (p_eg2_ship + p_px_ship) / 2
    res["operating_point_gold"]["sparse_refit_shipped_frozen_band"] = op_point(p_ship, y, lo0, hi0)
    if band:
        res["operating_point_gold"]["sparse_refit_shipped_derived_band"] = \
            op_point(p_ship, y, band["lo"], band["hi"])
    res["ship"] = {
        "head_eg2_C": frozen_Ce, "head_px_C": frozen_Cg,
        "threshold": cfg["threshold"],
        "abstain_band": band if band else cfg["abstain_band"],
        "threshold_5pctFR": res["sparse_fr_thresholds"]["threshold_5pctFR"],
        "threshold_10pctFR": res["sparse_fr_thresholds"]["threshold_10pctFR"],
    }

    os.makedirs(OUTDIR, exist_ok=True)
    json.dump(res, open(os.path.join(OUTDIR, "N5.01.json"), "w"), indent=1)
    np.savez(os.path.join(OUTDIR, "oof_gold.npz"), ids=np.asarray(ids), y=y, clip=clip,
             ens_frozen=ens_f, ens_sparse=ens_s, pe_f=pe_f, pg_f=pg_f,
             pe_s=pe_s, pg_s=pg_s, frozen_heads_on_sparse=p_frozen_on_sparse,
             shipped_sparse=p_ship)

    print("=== N5.01 gold ===")
    for k, v in res["loco"].items():
        e = v["ensemble_probavg"]
        print(f"{k:28s} ens AUROC {e['auroc']:.4f} {e['ci95']} bal@0.5 {e['balacc@0.5']:.4f}")
    print("score shift:", res["score_shift_gold"])
    print("op frozen:", res["operating_point_gold"]["frozen_features_frozen_band"])
    print("op sparse frozen-heads:", res["operating_point_gold"]["sparse_frozen_heads_frozen_band"])
    print("op sparse refit frozen-band:", res["operating_point_gold"]["sparse_refit_frozen_band"])
    print("op sparse refit derived-band:", res["operating_point_gold"]["sparse_refit_derived_band"])
    print("profile thresholds:", json.dumps(res["operating_point_profile_thresholds_gold"], indent=1))
    print("agreement:", res["verdict_agreement_gold"])
    print("ship:", res["ship"])
    return res


if __name__ == "__main__":
    main()
