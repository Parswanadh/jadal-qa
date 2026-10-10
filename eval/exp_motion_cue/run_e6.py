#!/usr/bin/env python3
"""E6 — motion cue: is the strongest cue just ego-motion magnitude, and can a
per-clip sign be fixed at inference time without labels?

Sub-experiments (results JSON per id under eval/results/exp_motion_cue/):
  E6.01  conditioning of the residual-motion cue on ego-motion magnitude
         (terciles + Spearman partial correlation)
  E6.02  per-clip sign calibration from UNLABELLED windows only (LOCO)
  E6.03  out-of-domain (60 human-labelled Egocentric-10K windows, n_bad=9)
  E6.04  robustness of the conditioning (quantiles vs fixed bins, cue variants)

The cue under test is the residual-motion magnitude after translation-only
ego-motion compensation, recomputed from video by motion_cue.py (the cached
scalars `shift_px`/`diff_e` are NOT that quantity; see docs/reports/).

Deterministic: fixed analysis size 160x90, fixed subsampling, no randomness in
extraction; bootstrap resampling uses a fixed seed. CPU only, <= 3 workers.

Run:
  JADAL_BUNDLE=/workspace/jadal_pod_bundle python3 eval/exp_motion_cue/run_e6.py \
      --experiments E6.01,E6.02,E6.03,E6.04 --workers 3
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))
from motion_cue import decode_gray, cue_series, extract_windows  # noqa: E402

BUNDLE = os.environ.get("JADAL_BUNDLE", "/workspace/jadal_pod_bundle")
OUT = ROOT / "eval" / "results" / "exp_motion_cue"
CUE_INDEGO = OUT / "cue_indego.npz"
CUE_OOD = OUT / "cue_ood.npz"
DEVICE = "A40 (CPU-only feature extraction; GPU idle)"
SEED = 0
BOOT = 2000

CUE_PRIMARY = "resid_mean"          # declared in advance: mean residual motion per frame
CUE_VARIANTS = ["resid_mean", "resid_med", "resid_p90", "resid_std"]


# ---------------------------------------------------------------- helpers ---
def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                       text=True).strip()
    except Exception:
        return "unknown"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def auroc(y, s) -> float:
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y)
    if len(set(y.tolist())) < 2:
        return float("nan")
    return float(roc_auc_score(y, s))


def clip_bootstrap_ci(y, s, clips, B=BOOT, seed=SEED):
    """95% CI by resampling CLIPS with replacement (2,000 draws, skip single-class draws)."""
    rng = np.random.default_rng(seed)
    names = sorted(set(np.asarray(clips).tolist()))
    vals = []
    for _ in range(B):
        cs = rng.choice(names, len(names))
        idx = np.concatenate([np.where(np.asarray(clips) == c)[0] for c in cs])
        yb, sb = np.asarray(y)[idx], np.asarray(s)[idx]
        if len(set(yb.tolist())) < 2:
            continue
        vals.append(auroc(yb, sb))
    if not vals:
        return float("nan"), float("nan")
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def window_bootstrap_ci(y, s, B=BOOT, seed=SEED):
    """95% CI by resampling WINDOWS (optimistic; OOD windows are one per factory)."""
    rng = np.random.default_rng(seed)
    y, s = np.asarray(y), np.asarray(s)
    vals = []
    for _ in range(B):
        idx = rng.integers(0, len(y), len(y))
        if len(set(y[idx].tolist())) < 2:
            continue
        vals.append(auroc(y[idx], s[idx]))
    if not vals:
        return float("nan"), float("nan")
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def spearman_ci(x, y, clips, B=BOOT, seed=SEED):
    from scipy.stats import spearmanr
    r = float(spearmanr(x, y)[0])
    rng = np.random.default_rng(seed)
    names = sorted(set(np.asarray(clips).tolist()))
    vals = []
    for _ in range(B):
        cs = rng.choice(names, len(names))
        idx = np.concatenate([np.where(np.asarray(clips) == c)[0] for c in cs])
        xb, yb = np.asarray(x)[idx], np.asarray(y)[idx]
        if len(set(yb.tolist())) < 2 or len(set(xb.tolist())) < 2:
            continue
        vals.append(float(spearmanr(xb, yb)[0]))
    if not vals:
        return r, float("nan"), float("nan")
    return r, float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def spearman_partial_ci(cue, y, z, clips, B=BOOT, seed=SEED):
    """Spearman partial correlation of cue and label controlling for z (ego-motion)."""
    from scipy.stats import spearmanr

    def _p(c, yy, zz):
        r_xy = spearmanr(c, yy)[0]
        r_zy = spearmanr(zz, yy)[0]
        r_xz = spearmanr(c, zz)[0]
        den = np.sqrt((1 - r_zy ** 2) * (1 - r_xz ** 2))
        return float((r_xy - r_zy * r_xz) / den) if den > 0 else float("nan")

    r = _p(cue, y, z)
    rng = np.random.default_rng(seed)
    names = sorted(set(np.asarray(clips).tolist()))
    vals = []
    for _ in range(B):
        cs = rng.choice(names, len(names))
        idx = np.concatenate([np.where(np.asarray(clips) == c)[0] for c in cs])
        cb, yb, zb = np.asarray(cue)[idx], np.asarray(y)[idx], np.asarray(z)[idx]
        if len(set(yb.tolist())) < 2 or len(set(cb.tolist())) < 2:
            continue
        vals.append(_p(cb, yb, zb))
    vals = [v for v in vals if np.isfinite(v)]
    if not vals:
        return r, float("nan"), float("nan")
    return r, float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def zscore_within(v, clips):
    v = np.asarray(v, float)
    out = np.zeros_like(v)
    for c in sorted(set(np.asarray(clips).tolist())):
        m = np.asarray(clips) == c
        sd = v[m].std()
        out[m] = (v[m] - v[m].mean()) / sd if sd > 0 else 0.0
    return out


def save_json(path: Path, obj: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, sort_keys=False)
    print(f"wrote {path.relative_to(ROOT)}")


# ------------------------------------------------------------------- data ---
def load_indego():
    man = json.load(open(f"{BUNDLE}/labels/manifest_gold.json"))["items"]
    ids = [i["id"] for i in man]
    y = np.array([i["label"] for i in man])
    clips = np.array([i["clip"] for i in man])
    reasons = np.array([i.get("reason") or "" for i in man])
    d = np.load(f"{BUNDLE}/caches/perpixel/series_gold.npz", allow_pickle=True)
    sids = [str(x) for x in d["ids"]]
    S = d["S"]
    assert sids == ids, "series_gold order != manifest order"
    cached = {
        "shift_px_mean": S[:, :, 4].astype(float).mean(1),
        "shift_px_med": np.median(S[:, :, 4].astype(float), 1),
        "diff_e_mean": S[:, :, 2].astype(float).mean(1),
    }
    f = json.load(open(f"{BUNDLE}/caches/gpu/features.json"))["windows"]
    cached["motion_energy"] = np.array([f[i]["full"]["motion_energy"] for i in ids], float)
    cached["global_shift"] = np.array([f[i]["full"]["global_shift"] for i in ids], float)
    return man, ids, y, clips, reasons, cached


def ensure_cue_indego(man, workers, force=False):
    if CUE_INDEGO.exists() and not force:
        d = np.load(CUE_INDEGO, allow_pickle=True)
        return {k: d[k] for k in d.files}
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = [{"id": it["id"], "path": it["path"], "start_s": it["start_s"], "stride": 1}
            for it in man]
    t0 = time.time()
    res = extract_windows(jobs, workers=workers)
    wall = time.time() - t0
    ids = np.array([r["id"] for r in res])
    np.savez_compressed(CUE_INDEGO, ids=ids,
                        **{k: np.array([r[k] for r in res], float)
                           for k in res[0] if k != "id"},
                        wall_s=np.array([wall]))
    return {k: np.load(CUE_INDEGO, allow_pickle=True)[k] for k in
            np.load(CUE_INDEGO, allow_pickle=True).files}


def ensure_cue_ood(workers, force=False):
    if CUE_OOD.exists() and not force:
        return {k: np.load(CUE_OOD, allow_pickle=True)[k] for k in
                np.load(CUE_OOD, allow_pickle=True).files}
    OUT.mkdir(parents=True, exist_ok=True)
    q = json.load(open(f"{BUNDLE}/queues/label_queue_random.json"))["segments"]
    jobs = [{"id": s["segment_id"], "path": s["path"], "start_s": s.get("start_s", 0.0),
             "stride": 3, "n_frames": 100} for s in q]
    t0 = time.time()
    res = extract_windows(jobs, workers=workers)
    wall = time.time() - t0
    ids = np.array([r["id"] for r in res])
    np.savez_compressed(CUE_OOD, ids=ids,
                        **{k: np.array([r[k] for r in res], float)
                           for k in res[0] if k != "id"},
                        wall_s=np.array([wall]))
    return {k: np.load(CUE_OOD, allow_pickle=True)[k] for k in
            np.load(CUE_OOD, allow_pickle=True).files}


def load_ood_labels():
    o = json.load(open(f"{BUNDLE}/labels/ood_labels.json"))["latest"]
    q = json.load(open(f"{BUNDLE}/queues/label_queue_random.json"))["segments"]
    src = {s["segment_id"]: s.get("source_path") for s in q}
    return o, src


# -------------------------------------------------------------- E6.01 ------
def run_e601(cue, y, clips, cached, meta) -> dict:
    """Conditioning: does the residual cue survive within ego-motion strata?"""
    res = {"id": "E6.01"}
    ego = cached["shift_px_mean"]                     # ego-motion magnitude (task-specified)
    res["ego_definition"] = "per-window mean of series_gold.npz channel shift_px (phase-correlation shift vs first frame)"
    res["cue_definition"] = ("per-window mean residual |I_t - warp(I_{t-1})| after "
                             "translation-only phase-correlation compensation, 160x90, every frame")

    rows = {}
    for name in CUE_VARIANTS:
        v = cue[name]
        o = {"spearman_cue_label": {}, "spearman_cue_ego": {}, "partial_cue_label_given_ego": {}}
        r, lo, hi = spearman_ci(v, y, clips)
        o["spearman_cue_label"] = {"r": r, "ci95": [lo, hi]}
        r2, lo2, hi2 = spearman_ci(v, ego, clips)
        o["spearman_cue_ego"] = {"r": r2, "ci95": [lo2, hi2]}
        rp, lop, hip = spearman_partial_ci(v, y, ego, clips)
        o["partial_cue_label_given_ego"] = {"r": rp, "ci95": [lop, hip]}
        o["auroc_oriented_pooled"] = auroc(y, -v)      # -cue: low residual = usable
        rows[name] = o
    res["cue_rows"] = rows

    # reference rows for the cached scalars: is the strongest cached motion scalar <=> ego-motion?
    ref = {}
    for name in ["diff_e_mean", "motion_energy", "global_shift"]:
        v = cached[name]
        r, lo, hi = spearman_ci(v, y, clips)
        r2, lo2, hi2 = spearman_ci(v, ego, clips)
        rp, lop, hip = spearman_partial_ci(v, y, ego, clips)
        ref[name] = {"auroc_oriented_pooled": auroc(y, -v),
                     "spearman_cue_label": {"r": r, "ci95": [lo, hi]},
                     "spearman_cue_ego": {"r": r2, "ci95": [lo2, hi2]},
                     "partial_cue_label_given_ego": {"r": rp, "ci95": [lop, hip]}}
    ref["shift_px_mean (ego motion itself)"] = {
        "auroc_oriented_pooled": auroc(y, -ego),
        "note": "this IS the conditioning variable"}
    res["cached_scalar_controls"] = ref

    # terciles of ego-motion magnitude, within-tercile AUROC
    q1, q2 = np.quantile(ego, [1 / 3, 2 / 3])
    terc = np.digitize(ego, [q1, q2])
    res["ego_tercile_edges_px"] = [float(q1), float(q2)]
    tinfo = {}
    for k in range(3):
        m = terc == k
        t = {"n": int(m.sum()), "n_good": int(y[m].sum()), "n_bad": int((1 - y[m]).sum()),
             "ego_range_px": [float(ego[m].min()), float(ego[m].max())], "cue": {}}
        for name in CUE_VARIANTS:
            a = auroc(y[m], -cue[name][m])
            lo, hi = clip_bootstrap_ci(y[m], -cue[name][m], clips[m])
            t["cue"][name] = {"auroc_oriented": a, "ci95": [lo, hi]}
        for name in ["diff_e_mean", "motion_energy"]:
            t["cue"][name] = {"auroc_oriented": auroc(y[m], -cached[name][m])}
        tinfo[f"tercile_{k}"] = t
    res["within_ego_terciles"] = tinfo

    res["n"] = int(len(y))
    res["n_good"] = int(y.sum())
    res["n_bad"] = int((1 - y).sum())
    res["label_source"] = "human (IndEgo gold, one labeller)"
    res["protocol"] = "in-domain: 150 windows, 7 clips; within-stratum AUROC + Spearman partial correlation (clip-bootstrap 95% CI, 2000 draws)"
    res["ci95"] = {"method": "clip bootstrap", "draws": BOOT, "seed": SEED}
    res.update(meta)
    return res


# -------------------------------------------------------------- E6.02 ------
def calibrate_abs_z(z):
    """Label-free: score = -|z| (extreme cue values in EITHER direction -> bad)."""
    return -np.abs(z)


def calibrate_majority(z):
    """Label-free: pick the per-clip sign so the majority (>50%) of the clip's
    windows score usable (prior: usable windows are the majority in this corpus)."""
    s = -z                                     # canonical orientation: low residual = usable
    if np.mean(s > 0) < 0.5:                   # <-- uses NO labels, only the clip's own values
        s = -s
    return s


def run_e602(cue, y, clips, meta, variants=CUE_VARIANTS) -> dict:
    """Per-clip sign calibration from unlabelled windows only, scored LOCO."""
    names = sorted(set(clips.tolist()))
    out = {"id": "E6.02", "protocol": (
        "IndEgo 150 windows, 7 clips; leave-one-clip-out direction (chosen on the 6 "
        "training clips), per-clip sign recalibration performed at inference on the "
        "held-out clip's OWN unlabelled window values only; pooled AUROC with "
        "clip-bootstrap 95% CI (2000 draws).")}
    per_variant = {}
    for name in variants:
        v = cue[name]
        raw = -v                                            # canonical global orientation
        # LOCO: direction is re-estimated on the 6 training clips (sign of pooled AUROC)
        loco_raw = np.zeros(len(y))
        loco_maj = np.zeros(len(y))
        loco_absz = np.zeros(len(y))
        per_clip = {}
        for c in names:
            te = clips == c
            tr = ~te
            a_tr = auroc(y[tr], raw[tr])
            d = 1.0 if a_tr >= 0.5 else -1.0                 # training-clip direction
            s = d * raw[te]
            loco_raw[te] = s
            # label-free per-clip recalibration on the held-out clip's own values (no labels)
            zc = (s - s.mean()) / (s.std() if s.std() > 0 else 1.0)
            loco_absz[te] = -np.abs(zc)
            sm = s.copy()
            if np.mean(sm > 0) < 0.5:
                sm = -sm
            loco_maj[te] = sm
            per_clip[c] = {
                "n": int(te.sum()), "n_good": int(y[te].sum()), "n_bad": int((1 - y[te]).sum()),
                "auroc_raw_oriented": auroc(y[te], raw[te]),
                "auroc_loco_raw": auroc(y[te], s),
                "auroc_loco_abs_z": auroc(y[te], loco_absz[te]),
                "auroc_loco_majority_prior": auroc(y[te], loco_maj[te]),
                "sign_chosen_by_majority_prior": float(np.sign(sm.mean())),
            }
        variant = {
            "pooled_auroc_raw_global_orientation": auroc(y, raw),
            "pooled_auroc_loco_raw": auroc(y, loco_raw),
            "pooled_auroc_loco_abs_z": auroc(y, loco_absz),
            "pooled_auroc_loco_majority_prior": auroc(y, loco_maj),
            "ci95_loco_raw": list(clip_bootstrap_ci(y, loco_raw, clips)),
            "ci95_loco_abs_z": list(clip_bootstrap_ci(y, loco_absz, clips)),
            "ci95_loco_majority_prior": list(clip_bootstrap_ci(y, loco_maj, clips)),
            "per_clip": per_clip,
        }
        per_variant[name] = variant
    out["variants"] = per_variant
    p = per_variant[CUE_PRIMARY]
    out["headline"] = {"cue": CUE_PRIMARY, **{k: p[k] for k in p if k != "per_clip"}}
    out["n"] = int(len(y))
    out["n_good"] = int(y.sum())
    out["n_bad"] = int((1 - y).sum())
    out["label_source"] = "human (IndEgo gold, one labeller)"
    out["ci95"] = {"method": "clip bootstrap", "draws": BOOT, "seed": SEED}
    out.update(meta)
    return out


# -------------------------------------------------------------- E6.03 ------
def run_e603(cue_ood, ood_ids, y_ood, src, cue_indego, y_ind, clips_ind, meta) -> dict:
    """Out-of-domain: 60 human-labelled Egocentric-10K windows (n_bad = 9)."""
    res = {"id": "E6.03", "n": int(len(y_ood)), "n_good": int((y_ood == 1).sum()),
           "n_bad": int((y_ood == 0).sum()),
           "label_source": "human (Egocentric-10K random windows, one labeller; n_bad=9 -> EXPLORATORY)"}
    # frozen IndEgo direction: sign that orients the cue on IndEgo (no OOD labels used)
    dir_ind = {}
    for name in CUE_VARIANTS:
        a = auroc(y_ind, -cue_indego[name])
        dir_ind[name] = 1.0 if a >= 0.5 else -1.0
    rows = {}
    for name in CUE_VARIANTS:
        s = dir_ind[name] * (-cue_ood[name])
        a = auroc(y_ood, s)
        lo, hi = window_bootstrap_ci(y_ood, s)
        from scipy.stats import spearmanr
        rows[name] = {
            "auroc_frozen_indego_direction": a, "ci95_window_optimistic": [lo, hi],
            "spearman_cue_label": float(spearmanr(cue_ood[name], y_ood)[0]),
            "auroc_within_ood_best_orientation": max(auroc(y_ood, -cue_ood[name]),
                                                     auroc(y_ood, cue_ood[name])),
            "indego_direction": dir_ind[name],
        }
    res["cue_rows"] = rows
    res["per_clip_calibration"] = ("NOT APPLICABLE: the 60 OOD windows come from 47+ distinct "
                                   "source videos, almost all with a single labelled window, so a "
                                   "per-source sign cannot be estimated from that source's own windows.")
    res["source_data"] = ("cue computed on queues/label_queue_random.json `path` (480p/30 fps, 10 s "
                          "transcode; effective ~10 fps via stride 3), i.e. the same file the human "
                          "labelled. The 1080p/30 fps `source_path` originals exist; not used here.")
    res["protocol"] = ("out-of-domain: 60 human-labelled Egocentric-10K windows scored with the IndEgo-frozen "
                       "cue direction; window-bootstrap 95% CI (optimistic: one window per factory); "
                       "n_bad=9 -> exploratory, no accuracy claim")
    res["device"] = DEVICE
    res["ci95"] = {"method": "window bootstrap (optimistic)", "draws": BOOT, "seed": SEED}
    res.update(meta)
    return res


# -------------------------------------------------------------- E6.04 ------
def strata_auroc(v, y, ego, edges) -> dict:
    idx = np.digitize(ego, edges)
    out = {}
    for k in range(len(edges) + 1):
        m = idx == k
        out[f"stratum_{k}"] = {
            "n": int(m.sum()), "n_good": int(y[m].sum()), "n_bad": int((1 - y[m]).sum()),
            "ego_range_px": [float(ego[m].min()), float(ego[m].max())] if m.sum() else None,
            "auroc_oriented": auroc(y[m], -v[m]),
        }
    return out


def run_e604(cue, y, clips, cached, meta) -> dict:
    """Robustness of the conditioning: quantiles vs fixed bins, cue variants."""
    ego = cached["shift_px_mean"]
    res = {"id": "E6.04",
           "protocol": ("IndEgo 150 windows, 7 clips; within-stratum AUROC of the residual cue under "
                        "different stratification rules (quantile vs fixed ego-motion bins) and cue "
                        "variants; stability of the Spearman partial correlation (clip bootstrap CI).")}
    schemes = {
        "quantile_terciles": np.quantile(ego, [1 / 3, 2 / 3]),
        "quantile_quintiles": np.quantile(ego, [1 / 5, 2 / 5, 3 / 5, 4 / 5]),
        "fixed_bins_px_5_15": np.array([5.0, 15.0]),
        "fixed_bins_px_1_5_20": np.array([1.0, 5.0, 20.0]),
    }
    out = {}
    for sname, edges in schemes.items():
        rows = {}
        for vname in CUE_VARIANTS:
            st = strata_auroc(cue[vname], y, ego, edges)
            vals = [s["auroc_oriented"] for s in st.values()
                    if s["n_bad"] >= 3 and np.isfinite(s["auroc_oriented"])]
            rows[vname] = {
                "strata": st,
                "strata_with_ge3_bad": len(vals),
                "min_auroc": float(min(vals)) if vals else None,
                "max_auroc": float(max(vals)) if vals else None,
                "mean_auroc": float(np.mean(vals)) if vals else None,
                "std_auroc": float(np.std(vals)) if vals else None,
                "partial_spearman_given_ego": spearman_partial_ci(cue[vname], y, ego, clips)[0],
            }
        out[sname] = {"edges_px": edges.tolist(), "variants": rows}
    res["stratification_schemes"] = out

    # fixed-direction control: does the cue add anything on top of ego-motion?
    # LOCO logistic(y ~ cue) vs logistic(y ~ ego) vs logistic(y ~ cue + ego) on the same folds
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    names = sorted(set(clips.tolist()))
    def loco_lr(X):
        p = np.zeros(len(y))
        for c in names:
            tr, te = clips != c, clips == c
            sc = StandardScaler().fit(X[tr])
            m = LogisticRegression(max_iter=4000, class_weight="balanced").fit(sc.transform(X[tr]), y[tr])
            p[te] = m.predict_proba(sc.transform(X[te]))[:, 1]
        return p
    cue_p = cue[CUE_PRIMARY]
    res["loco_logistic_controls"] = {
        "cue_only": {"auroc": auroc(y, loco_lr(cue_p[:, None])),
                     "ci95": list(clip_bootstrap_ci(y, loco_lr(cue_p[:, None]), clips))},
        "ego_only": {"auroc": auroc(y, loco_lr(ego[:, None]))},
        "cue_plus_ego": {"auroc": auroc(y, loco_lr(np.c_[cue_p, ego]))},
    }
    res["n"] = int(len(y))
    res["label_source"] = "human (IndEgo gold, one labeller)"
    res["ci95"] = {"method": "clip bootstrap", "draws": BOOT, "seed": SEED}
    res.update(meta)
    return res


# ------------------------------------------------------------------ main ---
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiments", default="E6.01,E6.02,E6.03,E6.04")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--force-extract", action="store_true")
    args = ap.parse_args()
    want = set(args.experiments.split(","))

    t0 = time.time()
    man, ids, y, clips, reasons, cached = load_indego()
    cue_npz = ensure_cue_indego(man, args.workers, args.force_extract)
    cue = {k: cue_npz[k] for k in CUE_VARIANTS}
    assert [str(x) for x in cue_npz["ids"]] == ids
    wall_ind = float(cue_npz["wall_s"][0])

    meta = {
        "device": DEVICE,
        "command": "JADAL_BUNDLE=$JADAL_BUNDLE python3 eval/exp_motion_cue/run_e6.py --experiments " + args.experiments + f" --workers {args.workers}",
        "git_sha": git_sha(),
        "cost": {
            "frames_per_window": 100,
            "windows": 150,
            "extraction_wall_s_indego": round(wall_ind, 1),
            "ms_per_window": round(1000 * wall_ind / 150, 1),
            "workers": args.workers,
            "vram_mb": 0,
            "cue_cache": str(CUE_INDEGO.relative_to(ROOT)),
            "cue_cache_sha256": sha256_file(CUE_INDEGO),
        },
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    if "E6.01" in want:
        save_json(OUT / "E6.01.json", run_e601(cue, y, clips, cached, meta))
    if "E6.02" in want:
        save_json(OUT / "E6.02.json", run_e602(cue, y, clips, meta))
    if "E6.04" in want:
        save_json(OUT / "E6.04.json", run_e604(cue, y, clips, cached, meta))
    if "E6.03" in want:
        ood = ensure_cue_ood(args.workers, args.force_extract)
        ood_ids = np.array([str(x) for x in ood["ids"]])
        labels, src = load_ood_labels()
        missing = [i for i in ood_ids if i not in labels]
        y_ood = np.array([1 if labels[i]["verdict"] == "good" else 0 for i in ood_ids])
        cue_ood = {k: ood[k] for k in CUE_VARIANTS}
        m3 = dict(meta)
        m3["cost"] = dict(meta["cost"])
        m3["cost"]["extraction_wall_s_ood"] = round(float(ood["wall_s"][0]), 1)
        m3["cost"]["ms_per_window_ood"] = round(1000 * float(ood["wall_s"][0]) / len(ood_ids), 1)
        m3["cost"]["cue_cache_ood"] = str(CUE_OOD.relative_to(ROOT))
        m3["cost"]["cue_cache_ood_sha256"] = sha256_file(CUE_OOD)
        j = run_e603(cue_ood, ood_ids, y_ood, src, cue, y, clips, m3)
        j["missing_labels"] = missing
        save_json(OUT / "E6.03.json", j)

    print(f"done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
