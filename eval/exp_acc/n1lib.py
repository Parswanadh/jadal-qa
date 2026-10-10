"""N1 shared helpers: data assembly, metrics, grouped CV, bootstrap.

CPU only (numpy + sklearn). Fixed seeds. Every number carries n / label_source /
protocol / CI. Reuses frozen artefacts (no re-decode):
  * 160 OOD labels  : bundle labels/ood160_labels.json
  * 160 OOD Z3-20   : m/adapt160 eval/results/exp_adapt160/inputs/z3_160.npz
  * 160 OOD pixel26 : m/fewshot160 eval/results/exp_fewshot/inputs/ood160_pixel.npz
  * 150 IndEgo gold : bundle caches/gpu/features.json (26 pixel) + z3_160.npz (Z3-20)
  * 1546 unlabelled : $JADAL_POD_ROOT/artifacts/windows_rescored.jsonl (26 pixel, no labels)
"""
from __future__ import annotations

import json
import os

import numpy as np
from scipy.stats import rankdata

BUNDLE = os.environ.get("JADAL_BUNDLE", "jadal_pod_bundle")
WT = os.environ.get("JADAL_WT", "wt")
Z3_160 = os.path.join(WT, "m-adapt160", "eval", "results", "exp_adapt160",
                      "inputs", "z3_160.npz")
PIX160 = os.path.join(WT, "m-fewshot160", "eval", "results", "exp_fewshot",
                      "inputs", "ood160_pixel.npz")
UNL_JSONL = os.path.join(os.environ.get("JADAL_ARTIFACTS", "artifacts"), 'windows_rescored.jsonl')

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

SEED = 0
N_BOOT = 2000


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


# ---------------- data ----------------

def load_ood():
    d = json.load(open(os.path.join(BUNDLE, "labels", "ood160_labels.json")))
    w = d["windows"]
    ids = [x["segment_id"] for x in w]
    y = np.asarray([1 if x["verdict"] == "good" else 0 for x in w], int)
    batch = np.asarray([x["batch"] for x in w])
    fac = np.asarray([x["factory"] for x in w])
    z = np.load(Z3_160, allow_pickle=True)
    p = np.load(PIX160, allow_pickle=True)
    assert list(z["ids"]) == ids and list(p["ids"]) == ids
    znames = [str(x) for x in z["names"]]
    pnames = [str(x) for x in p["names"]]
    return {"ids": ids, "y": y, "batch": batch, "fac": fac,
            "Xz3": z["X"].astype(float), "znames": znames,
            "Xpix": p["X10"].astype(float), "pnames": pnames}


def load_gold():
    m = json.load(open(os.path.join(BUNDLE, "labels", "manifest_gold.json")))
    items = m["items"]
    ids = [it["id"] for it in items]
    clips = np.asarray([it["clip"] for it in items])
    y = np.asarray([int(it["label"]) for it in items])
    fj = json.load(open(os.path.join(BUNDLE, "caches", "gpu", "features.json")))["windows"]
    X26 = np.asarray([[float(fj[i]["full"][n]) for n in FEATURE_NAMES26] for i in ids])
    z = np.load(Z3_160, allow_pickle=True)
    # X_indego rows are in manifest_gold order (indego_ids == gold ids).
    Xz3 = np.asarray(z["X_indego"], float)
    assert list(z["clips_indego"]) == list(clips)
    assert list(z["y_indego"]) == list(y)
    return {"ids": ids, "clips": clips, "y": y, "Xpix": X26,
            "Xz3": Xz3, "znames": [str(x) for x in z["names"]]}


def load_unlabelled():
    fac, clip, rows = [], [], []
    with open(UNL_JSONL) as f:
        header = json.loads(f.readline())
        for line in f:
            d = json.loads(line)
            fac.append(d["factory"])
            clip.append(d["clip"])
            rows.append(d)
    X26 = np.asarray([[float(r["features"].get(n, np.nan)) for n in FEATURE_NAMES26]
                      for r in rows])
    return {"fac": np.asarray(fac), "clip": np.asarray(clip),
            "ids": np.asarray([r["window_id"] for r in rows]),
            "Xpix": X26,
            "p_eg2": np.asarray([float(r["features"].get("p_eg2", np.nan)) for r in rows]),
            "p_pixel": np.asarray([float(r["features"].get("p_pixel", np.nan)) for r in rows]),
            "score": np.asarray([float(r.get("score", np.nan)) for r in rows]),
            "header": header}


def build_transfer(ood, gold, feature_set):
    """Return aligned matrices for a named transferable feature set.

    feature_set in {'z3six','stable8','all14','sharp3'}.
    Xo (n_ood, d), Xg (n_gold, d), names.
    """
    if feature_set == "z3six":
        names = Z3_SIX
        oi = [ood["znames"].index(n) for n in names]
        gi = [gold["znames"].index(n) for n in names]
        return ood["Xz3"][:, oi], gold["Xz3"][:, gi], names
    if feature_set == "sharp3":
        names = STABLE8[:3]
        oi = [ood["pnames"].index(n) for n in names]
        gi = [FEATURE_NAMES26.index(n) for n in names]
        return ood["Xpix"][:, oi], gold["Xpix"][:, gi], names
    if feature_set == "stable8":
        names = STABLE8
        oi = [ood["pnames"].index(n) for n in names]
        gi = [FEATURE_NAMES26.index(n) for n in names]
        return ood["Xpix"][:, oi], gold["Xpix"][:, gi], names
    if feature_set == "all14":
        names = STABLE8 + Z3_SIX
        Xo8, Xg8, _ = build_transfer(ood, gold, "stable8")
        Xo6, Xg6, _ = build_transfer(ood, gold, "z3six")
        return np.hstack([Xo8, Xo6]), np.hstack([Xg8, Xg6]), names
    raise ValueError(feature_set)


def indego_signs(Xg, yg):
    au = np.asarray([fast_auc(yg, Xg[:, j]) for j in range(Xg.shape[1])])
    return np.where(au >= 0.5, 1.0, -1.0), au


def zscore(X, mean=None, std=None):
    if mean is None:
        mean = X.mean(0)
    if std is None:
        std = X.std(0)
    return (X - mean) / np.clip(std, 1e-9, None)


def factory_zscore(Xo, fac_o, X_unl, fac_unl, fallback_mean, fallback_std):
    """Per-factory z-score: for each OOD window use the unlabelled windows of the
    SAME factory (unsupervised) to standardise. Falls back to global stats."""
    out = np.empty_like(Xo, float)
    for k, f in enumerate(fac_o):
        m = fac_unl == f
        if m.sum() >= 5:
            mu = X_unl[m].mean(0)
            sd = X_unl[m].std(0)
        else:
            mu, sd = fallback_mean, fallback_std
        out[k] = (Xo[k] - mu) / np.clip(sd, 1e-9, None)
    return out
