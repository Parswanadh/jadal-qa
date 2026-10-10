#!/usr/bin/env python3
"""Direction 7 prototype: spectral-graph loop/duplicate detection + learning-free hand-activity proxy.

CPU-only, deterministic, read-only wrt source videos.
Outputs per-window features + LOCO evaluation + dup-pair evaluation.

Usage:
    OMP_NUM_THREADS=2 /home/parshu/envs/jadal/bin/python \
        eval/lab/graphhands/proto_graphhands.py [--out DIR] [--n-frames 20] [--seed 0]

House rules respected: no GPU, numpy/scipy/cv2 only (+sklearn for AUROC),
OMP threads set by caller, read-only sampling of source media.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone

import cv2
import numpy as np

try:
    from sklearn.metrics import roc_auc_score
except Exception:  # pragma: no cover
    roc_auc_score = None

GOLD_MANIFEST = "/home/parshu/projects/jadal-qa-wt/spectral_data/manifest_gold.json"
DUP_MANIFEST = "/home/parshu/projects/jadal-qa-wt/spectral_data/manifest_dups.json"

N_GRAPH = 20          # frames sampled per 10 s gold window (-> 0.5 s spacing)
N_DUP = 10            # frames sampled per dup file
SMALL = 96            # working resolution (square) for flow / skin
LUMA = 16             # downscaled luma resolution for similarity graph
ROI_TOP = 0.5         # lower-frame ROI: y in [0.5, 1.0]
FROZEN_THR = 0.995    # adjacent-frame cosine above this => frozen link
NCOMP_REL = 1e-3      # relative eigenvalue threshold for near-zero count
BOOT_B = 2000         # clip-bootstrap replicates


# ----------------------------------------------------------------------------
# frame sampling (deterministic: fixed timestamps, sequential decode)
# ----------------------------------------------------------------------------
def sample_frames(path: str, start_s: float, end_s: float, n: int, size: int = SMALL):
    """Return (n, size, size, 3) BGR uint8 frames sampled uniformly in [start,end]."""
    ts = np.linspace(start_s, end_s, n + 2)[1:-1] if n > 0 else np.array([])
    # NOTE: linspace endpoint handling: use closed interval then drop exact ends
    ts = np.linspace(start_s + 0.05, end_s - 0.05, n)
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 25.0
    frames = []
    for t in ts:
        cap.set(cv2.CAP_PROP_POS_MSEC, float(t * 1000.0))
        ok, fr = cap.read()
        if not ok or fr is None:
            cap.release()
            return None
        fr = cv2.resize(fr, (size, size), interpolation=cv2.INTER_AREA)
        frames.append(fr)
    cap.release()
    if len(frames) != n:
        return None
    return np.stack(frames)


def sample_file_frames(path: str, n: int, size: int = SMALL):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        # fall back to time-based sampling of up to 10 s
        cap.release()
        return sample_frames(path, 0.0, 10.0, n, size)
    idx = np.linspace(0, max(total - 1, 0), n).round().astype(int)
    frames = []
    for i in idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, fr = cap.read()
        if not ok or fr is None:
            cap.release()
            return None
        frames.append(cv2.resize(fr, (size, size), interpolation=cv2.INTER_AREA))
    cap.release()
    return np.stack(frames)


# ----------------------------------------------------------------------------
# (a) frame-similarity graph features
# ----------------------------------------------------------------------------
def dhash64(gray9x8: np.ndarray) -> int:
    d = (gray9x8[:, 1:] > gray9x8[:, :-1]).astype(np.uint8).ravel()
    h = 0
    for b in d:
        h = (h << 1) | int(b)
    return h


def frame_descriptors(frames: np.ndarray):
    """Luma vectors (LUMA x LUMA, float) + dHash ints for each frame."""
    gray = np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames])
    small = np.stack([cv2.resize(g, (LUMA, LUMA), interpolation=cv2.INTER_AREA)
                      for g in gray]).astype(np.float32)
    wide = np.stack([cv2.resize(g, (9, 8), interpolation=cv2.INTER_AREA)
                     for g in gray])
    hashes = [dhash64(w) for w in wide]
    return small.reshape(len(frames), -1), hashes


def cosine_sim_matrix(X: np.ndarray) -> np.ndarray:
    Xc = X - X.mean(axis=1, keepdims=True)
    nrm = np.linalg.norm(Xc, axis=1, keepdims=True)
    nrm[nrm == 0] = 1.0
    Xn = Xc / nrm
    S = Xn @ Xn.T
    return np.clip(S, -1.0, 1.0)


def hamming_sim_matrix(hashes) -> np.ndarray:
    n = len(hashes)
    S = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            ham = bin(hashes[i] ^ hashes[j]).count("1")
            s = 1.0 - ham / 64.0
            S[i, j] = S[j, i] = s
    return S


def spectral_features(S: np.ndarray) -> dict:
    n = S.shape[0]
    W = (S + 1.0) / 2.0
    np.fill_diagonal(W, 0.0)
    d = W.sum(axis=1)
    L = np.diag(d) - W
    lam = np.linalg.eigvalsh(L)
    lam = np.clip(lam, 0.0, None)
    lmax = float(lam[-1]) if lam[-1] > 0 else 1.0
    ncomp = int((lam < NCOMP_REL * lmax).sum())
    gap = float(lam[1] / lmax) if n > 1 else 0.0  # algebraic connectivity / scale
    return {
        "lam0": float(lam[0]), "lam1": float(lam[1]),
        "lam2": float(lam[2]) if n > 2 else 0.0,
        "lam_max": lmax, "n_near_zero": ncomp, "spectral_gap": gap,
    }


def loop_features(S: np.ndarray) -> dict:
    n = S.shape[0]
    iu = np.triu_indices(n, k=1)
    off = S[iu]
    adj = np.diag(S, k=1)
    frozen = float((adj > FROZEN_THR).mean())
    prof = {}
    for k in range(1, n - 1):
        prof[k] = float(np.diag(S, k=k).mean())
    lags = np.array(sorted(prof))
    vals = np.array([prof[k] for k in lags])
    med = float(np.median(vals))
    cands = [(k, prof[k]) for k in prof if k >= 3]
    if cands:
        bk, bv = max(cands, key=lambda kv: kv[1])
    else:
        bk, bv = 1, prof.get(1, 0.0)
    return {
        "mean_offdiag": float(off.mean()), "std_offdiag": float(off.std()),
        "mean_lag1": float(prof.get(1, 0.0)),
        "frozen_frac": frozen,
        "loop_strength": float(bv - med), "loop_period_lag": int(bk),
        "loop_peak": float(bv),
    }


# ----------------------------------------------------------------------------
# (b) learning-free hand-activity proxy
# ----------------------------------------------------------------------------
def skin_mask_bgr(frames: np.ndarray) -> np.ndarray:
    """YCrCb skin-chromaticity locus mask (bool, per pixel).

    Classic locus from the Kakumanu et al. (2007) survey family:
    Y > 40, 133 <= Cr <= 173, 77 <= Cb <= 127.
    """
    ycrcb = np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2YCrCb) for f in frames])
    Y = ycrcb[..., 0].astype(np.int16)
    Cr = ycrcb[..., 1].astype(np.int16)
    Cb = ycrcb[..., 2].astype(np.int16)
    return (Y > 40) & (Cr >= 133) & (Cr <= 173) & (Cb >= 77) & (Cb <= 127)


def hand_activity(frames: np.ndarray) -> dict:
    n, h, w, _ = frames.shape
    y0 = int(h * ROI_TOP)
    gray = np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames])
    skin = skin_mask_bgr(frames)  # (n,h,w) bool
    hann = cv2.createHanningWindow((w, h), cv2.CV_64F)
    resids, weighted, covs = [], [], []
    shifts = []
    for t in range(1, n):
        prev = gray[t - 1].astype(np.float64)
        cur = gray[t].astype(np.float64)
        try:
            shift, _resp = cv2.phaseCorrelate(prev * hann / 255.0, cur * hann / 255.0)
            dx, dy = float(shift[0]), float(shift[1])
        except cv2.error:
            dx, dy = 0.0, 0.0
        shifts.append([dx, dy])
        M = np.float32([[1, 0, -dx], [0, 1, -dy]])
        aligned = cv2.warpAffine(prev.astype(np.float32), M, (w, h),
                                 flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REPLICATE)
        res = np.abs(cur.astype(np.float32) - aligned)
        roi = res[y0:, :]
        m_union = (skin[t - 1][y0:, :] | skin[t][y0:, :]).astype(np.float32)
        resids.append(float(roi.mean()))
        weighted.append(float((roi * m_union).mean()))
        covs.append(float(m_union.mean()))
    return {
        "resid_raw": float(np.mean(resids)) if resids else 0.0,
        "hand_activity": float(np.mean(weighted)) if weighted else 0.0,
        "skin_coverage": float(np.mean(covs)) if covs else 0.0,
        "mean_shift_px": float(np.mean(np.hypot(*np.array(shifts).T))) if shifts else 0.0,
    }


# ----------------------------------------------------------------------------
# evaluation helpers
# ----------------------------------------------------------------------------
def auc(y, s) -> float:
    y = np.asarray(y); s = np.asarray(s)
    m = np.isfinite(s)
    y, s = y[m], s[m]
    if len(np.unique(y)) < 2 or len(y) < 3:
        return float("nan")
    if roc_auc_score is None:
        # Mann-Whitney fallback
        o = np.argsort(s, kind="stable")
        ys = y[o]
        r = np.argsort(o, kind="stable").astype(float) + 1
        n1 = ys.sum(); n0 = len(ys) - n1
        return float((r[ys == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
    return float(roc_auc_score(y, s))


def clip_bootstrap_ci(y, s, clips, b: int, seed: int):
    rng = np.random.default_rng(seed)
    y = np.asarray(y); s = np.asarray(s); clips = np.asarray(clips)
    uclips = sorted(set(clips.tolist()))
    base = auc(y, s)
    reps = []
    for _ in range(b):
        pick = rng.choice(uclips, size=len(uclips), replace=True)
        idx = np.concatenate([np.where(clips == c)[0] for c in pick])
        reps.append(auc(y[idx], s[idx]))
    reps = np.array([r for r in reps if np.isfinite(r)])
    lo, hi = float(np.percentile(reps, 2.5)), float(np.percentile(reps, 97.5))
    return base, lo, hi, float(np.mean(reps))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="eval/lab/graphhands/results")
    ap.add_argument("--n-frames", type=int, default=N_GRAPH)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--boot", type=int, default=BOOT_B)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    t_all = time.time()

    gold = json.load(open(GOLD_MANIFEST))["items"]
    rows = []
    costs = []
    fails = []
    for it in gold:
        t0 = time.time()
        fr = sample_frames(it["path"], it["start_s"], it["end_s"], args.n_frames)
        if fr is None:
            fails.append(it["id"])
            rows.append({"id": it["id"], "clip": it["clip"], "label": it["label"],
                         "reason": it["reason"], "ok": False})
            continue
        X, hh = frame_descriptors(fr)
        S = cosine_sim_matrix(X)
        Sh = hamming_sim_matrix(hh)
        feat = {}
        feat.update(spectral_features(S))
        feat.update(loop_features(S))
        feat.update({f"h_{k}": v for k, v in loop_features(Sh).items()})
        feat.update(hand_activity(fr))
        costs.append((time.time() - t0) * 1000.0)
        rows.append({"id": it["id"], "clip": it["clip"], "label": it["label"],
                     "reason": it["reason"], "ok": True, **feat})

    # determinism self-check: recompute first ok window, require bitwise-close
    det = {"checked": False, "max_abs_diff": None, "pass": False}
    first_ok = next((r for r in rows if r.get("ok")), None)
    if first_ok is not None:
        src = next(i for i in gold if i["id"] == first_ok["id"])
        fr2 = sample_frames(src["path"], src["start_s"], src["end_s"], args.n_frames)
        X2, hh2 = frame_descriptors(fr2)
        S2 = cosine_sim_matrix(X2)
        keys = [k for k in first_ok if k not in
                ("id", "clip", "label", "reason", "ok")]
        f2 = {}
        f2.update(spectral_features(S2))
        f2.update(loop_features(S2))
        f2.update({f"h_{k}": v for k, v in loop_features(hamming_sim_matrix(hh2)).items()})
        f2.update(hand_activity(fr2))
        diffs = [abs(float(first_ok[k]) - float(f2[k])) for k in f2]
        det = {"checked": True, "max_abs_diff": float(max(diffs)),
               "pass": bool(max(diffs) < 1e-9)}

    ok_rows = [r for r in rows if r.get("ok")]
    y = np.array([r["label"] for r in ok_rows])
    clips = np.array([r["clip"] for r in ok_rows])

    # score orientations: higher score => usable(1), except frozen/loop/offdiag (bad-positive, negated)
    usable_scores = ["hand_activity", "skin_coverage", "resid_raw"]
    bad_scores = ["frozen_frac", "mean_offdiag", "loop_strength", "loop_peak",
                  "h_frozen_frac", "h_mean_offdiag", "n_near_zero", "spectral_gap"]

    loco = {}
    for name in usable_scores + bad_scores:
        s = np.array([float(r.get(name, np.nan)) for r in ok_rows])
        if name in bad_scores:
            s = -s  # convert to usable-positive for AUROC
        base, lo, hi, mrep = clip_bootstrap_ci(y, s, clips, args.boot, args.seed)
        per_clip = {}
        for c in sorted(set(clips.tolist())):
            m = clips == c
            per_clip[c] = {"n": int(m.sum()), "n_pos": int(y[m].sum()),
                           "auc_usable": auc(y[m], s[m])}
        loco[name] = {"orientation": "higher=usable" + (" (negated bad-positive)" if name in bad_scores else ""),
                      "auc_pooled": base, "ci95_lo": lo, "ci95_hi": hi,
                      "boot_mean": mrep, "per_clip": per_clip}

    # per-reason descriptives
    from collections import defaultdict
    desc = defaultdict(list)
    for r in ok_rows:
        key = "GOOD" if r["label"] == 1 else (r["reason"] or "BAD_UNKNOWN")
        desc[key].append(r)
    feat_keys = ["frozen_frac", "mean_offdiag", "loop_strength", "n_near_zero",
                 "spectral_gap", "hand_activity", "resid_raw", "skin_coverage",
                 "mean_shift_px"]
    descriptives = {}
    for k, rs in sorted(desc.items()):
        d = {"n": len(rs)}
        for fk in feat_keys:
            v = np.array([float(r[fk]) for r in rs])
            d[fk] = {"mean": float(v.mean()), "sd": float(v.std()),
                     "median": float(np.median(v))}
        descriptives[k] = d

    # ---- constructed dup pairs ----
    dm = json.load(open(DUP_MANIFEST))
    files = dm["files"]
    sig_cache = {}

    def sig(key):
        if key not in sig_cache:
            fr = sample_file_frames(files[key], N_DUP)
            if fr is None:
                sig_cache[key] = None
            else:
                X, hh = frame_descriptors(fr)
                sig_cache[key] = (X, hh)
        return sig_cache[key]

    pair_rows = []
    for p in dm["pairs"]:
        A, B = sig(p["a"]), sig(p["b"])
        if A is None or B is None:
            continue
        Xa, ha = A
        Xb, hb = B
        # cosine cross-similarity, mean row-max
        na = Xa - Xa.mean(axis=1, keepdims=True)
        nb = Xb - Xb.mean(axis=1, keepdims=True)
        na /= np.maximum(np.linalg.norm(na, axis=1, keepdims=True), 1e-12)
        nb /= np.maximum(np.linalg.norm(nb, axis=1, keepdims=True), 1e-12)
        C = na @ nb.T
        dup_cos = float(np.maximum(C.max(axis=1).mean(), C.max(axis=0).mean()))
        # dhash: min hamming over cross pairs -> similarity
        best = 64
        for i in ha:
            for j in hb:
                h = bin(i ^ j).count("1")
                if h < best:
                    best = h
        dup_dhash = float(1.0 - best / 64.0)
        pair_rows.append({"a": p["a"], "b": p["b"], "kind": p["kind"],
                          "is_dup": bool(p["is_duplicate"]),
                          "dup_cos": dup_cos, "dup_dhash": dup_dhash})

    yp = np.array([int(r["is_dup"]) for r in pair_rows])
    dup_eval = {}
    for name in ("dup_cos", "dup_dhash"):
        s = np.array([r[name] for r in pair_rows])
        rng = np.random.default_rng(args.seed)
        reps = []
        for _ in range(args.boot):
            idx = rng.integers(0, len(s), len(s))
            a = auc(yp[idx], s[idx])
            if np.isfinite(a):
                reps.append(a)
        dup_eval[name] = {"auc": auc(yp, s),
                          "ci95_lo": float(np.percentile(reps, 2.5)),
                          "ci95_hi": float(np.percentile(reps, 97.5)),
                          "n_pairs": len(s), "n_dup": int(yp.sum())}
    by_kind = {}
    for r in pair_rows:
        by_kind.setdefault(r["kind"], []).append(r)
    kind_desc = {k: {"n": len(v),
                     "dup_cos_mean": float(np.mean([x["dup_cos"] for x in v])),
                     "dup_dhash_mean": float(np.mean([x["dup_dhash"] for x in v]))}
                 for k, v in sorted(by_kind.items())}

    total_min = (time.time() - t_all) / 60.0
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "n_gold": len(gold), "n_ok": len(ok_rows), "n_fail": fails,
        "n_frames_per_window": args.n_frames,
        "working_resolution": SMALL, "luma_resolution": LUMA,
        "cost_ms_per_window_mean": float(np.mean(costs)) if costs else None,
        "cost_ms_per_window_median": float(np.median(costs)) if costs else None,
        "runtime_total_min": total_min,
        "determinism_selfcheck": det,
        "seed": args.seed, "bootstrap_reps": args.boot,
        "label_note": "gold label 1=usable/0=bad; AUROC reported usable-positive",
    }

    with open(os.path.join(args.out, "window_features.json"), "w") as f:
        json.dump(rows, f)
    with open(os.path.join(args.out, "loco.json"), "w") as f:
        json.dump({"summary": summary, "loco": loco,
                   "descriptives": descriptives}, f, indent=1)
    with open(os.path.join(args.out, "dup_pairs.json"), "w") as f:
        json.dump({"dup_eval": dup_eval, "by_kind": kind_desc,
                   "pairs": pair_rows}, f, indent=1)

    print(json.dumps({"summary": summary,
                      "loco_auc": {k: round(v["auc_pooled"], 4) for k, v in loco.items()},
                      "dup_eval": dup_eval}, indent=1))


if __name__ == "__main__":
    main()
