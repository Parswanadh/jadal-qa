#!/usr/bin/env python3
"""Direction 5: SCATTERING / wavelet-modulus + phase-correlation + optimal-transport cues.

Training-free, CPU-only, deterministic prototype. No learned weights.

Per 10-s window (or per dup file), decode a small deterministic frame series
(cv2, read-only, fixed seek/stride/cap, gray, width 160; 64x64 copy for FFT):

(a) Wavelet-modulus texture/blur statistics (1-level Haar DWT via pywt,
    i.e. a 1-scale scattering-like modulus mean, no learning):
      SHARP_H = mean_t mean(|cH|+|cV|+|cD|)/3   (high-freq modulus energy)
      SHARP_R = mean_t high/low, low = mean|cA| (relative high-freq energy)
      LAPL    = mean_t Var(Laplacian)           (classical sharpness ref)
    Quality hypothesis: higher = sharper = more usable (label 1).

(b) Phase-correlation (Kuglin & Hines 1975) between consecutive kept frames:
      R = F1*conj(F2)/(|F1||F2|+eps); pc = |ifft2(R)|
      PC_PEAK  = mean peak value  (frozen/duplicate cue: ~1 at zero shift)
      PC_FROZ  = fraction with peak>0.4 and shift<=1px
      PC_SHIFT = median shift magnitude in px (ego-motion estimator)

(c) Optimal-transport deduplication:
      per-window pooled luma histogram (32 bins) -> 1-D Wasserstein distance
      (scipy.stats.wasserstein_distance on bin centers; exact 1-D OT, which is
      also the closed form behind each sliced-Wasserstein projection) between
      window pairs. Lower = more likely duplicate.
      Secondary: 1-D Wasserstein between frame-mean series.

Evaluation: LOCO AUROC + clip-bootstrap 95% CI (quality scores); dup-pair
precision/recall/F1 at a held-out threshold (even/odd pair split); descriptive
stats on the 6 real gold DUPLICATE labels; ms/window cost; deterministic /
streaming / hashable verdicts.

CPU only, OMP_NUM_THREADS=2, sequential (low-RAM machine).
"""
import hashlib
import json
import os
import time

import cv2
import numpy as np
import pywt
from scipy.stats import wasserstein_distance

GOLD = "/home/parshu/projects/jadal-qa-wt/spectral_data/manifest_gold.json"
DUPS = "/home/parshu/projects/jadal-qa-wt/spectral_data/manifest_dups.json"
OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

W160 = 160          # luma width for wavelet/histVN
W64 = 64            # luma width for phase correlation
STRIDE = 5          # keep every 5th decoded frame
MAX_KEEP = 12       # cap kept frames per window/file
NBINS = 32
PC_FROZ_PEAK = 0.4
PC_FROZ_SHIFT = 1.0
BOOT_ITERS = 1000
BOOT_SEED = 0


# ---------------- decode ----------------
def decode_series(path, start_s=0.0, end_s=None):
    """Deterministic read-only decode. Returns (frames160, frames64, fps, err)."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None, None, 0.0, "undecodable"
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 10.0
    if start_s > 0:
        cap.set(cv2.CAP_PROP_POS_MSEC, start_s * 1000.0)
    keep160, keep64 = [], []
    idx = 0
    while len(keep160) < MAX_KEEP:
        ok, fr = cap.read()
        if not ok:
            break
        if end_s is not None and (cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0) > end_s:
            break
        if idx % STRIDE == 0:
            g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
            h, w = g.shape
            s = W160 / max(w, 1)
            small = cv2.resize(g, (W160, max(1, int(h * s))), interpolation=cv2.INTER_AREA)
            tiny = cv2.resize(g, (W64, W64), interpolation=cv2.INTER_AREA)
            keep160.append(small)
            keep64.append(tiny)
        idx += 1
    cap.release()
    if not keep160:
        return None, None, fps, "no_frames"
    return keep160, keep64, fps, None


# ---------------- (a) wavelet modulus ----------------
def wavelet_frame_stats(g):
    x = np.asarray(g, dtype=np.float64)
    cA, (cH, cV, cD) = pywt.dwt2(x, "haar")
    high = (np.mean(np.abs(cH)) + np.mean(np.abs(cV)) + np.mean(np.abs(cD))) / 3.0
    low = float(np.mean(np.abs(cA))) + 1e-9
    lap = cv2.Laplacian(np.asarray(g), cv2.CV_64F).var()
    return high, high / low, float(lap)


# ---------------- (b) phase correlation ----------------
def phase_corr(a, b):
    A = np.fft.fft2(np.asarray(a, dtype=np.float64))
    B = np.fft.fft2(np.asarray(b, dtype=np.float64))
    R = A * np.conj(B)
    R = R / (np.abs(R) + 1e-12)
    pc = np.abs(np.fft.ifft2(R))
    iy, ix = np.unravel_index(int(np.argmax(pc)), pc.shape)
    peak = float(pc[iy, ix])
    H, W = pc.shape
    dx = min(ix, W - ix)
    dy = min(iy, H - iy)
    shift = float(np.hypot(dx, dy))
    return peak, shift


# ---------------- window features ----------------
def window_features(path, start_s=0.0, end_s=None):
    t0 = time.perf_counter()
    f160, f64, fps, err = decode_series(path, start_s, end_s)
    if err:
        return {"error": err, "cost_ms": (time.perf_counter() - t0) * 1000.0}
    hs, rs, ls = [], [], []
    means = []
    for g in f160:
        h, r, l = wavelet_frame_stats(g)
        hs.append(h)
        rs.append(r)
        ls.append(l)
        means.append(float(np.mean(g)))
    peaks, shifts = [], []
    for a, b in zip(f64[:-1], f64[1:]):
        p, s = phase_corr(a, b)
        peaks.append(p)
        shifts.append(s)
    peaks = peaks or [0.0]
    shifts = shifts or [0.0]
    hist, _ = np.histogram(np.concatenate([f.ravel() for f in f160]), bins=NBINS, range=(0, 255))
    hist = hist.astype(np.float64)
    hist /= hist.sum()
    out = {
        "n_frames": len(f160),
        "fps_native": fps,
        "sharp_h": float(np.mean(hs)),
        "sharp_r": float(np.mean(rs)),
        "lapl": float(np.mean(ls)),
        "pc_peak": float(np.mean(peaks)),
        "pc_froz": float(np.mean([(p > PC_FROZ_PEAK and s <= PC_FROZ_SHIFT) for p, s in zip(peaks, shifts)])),
        "pc_shift": float(np.median(shifts)),
        "frame_means": means,
        "hist": hist.tolist(),
        "cost_ms": (time.perf_counter() - t0) * 1000.0,
    }
    return out


def wdist_hist(h1, h2):
    centers = np.linspace(0, 255, NBINS)
    return float(wasserstein_distance(centers, centers, np.asarray(h1), np.asarray(h2)))


def wdist_means(m1, m2):
    return float(wasserstein_distance(np.asarray(m1), np.asarray(m2)))


def auroc(scores, labels):
    scores = np.asarray(scores)
    labels = np.asarray(labels)
    if len(np.unique(labels)) < 2:
        return float("nan")
    order = np.argsort(scores)
    lab = labels[order]
    n1 = lab.sum()
    n0 = len(lab) - n1
    ranks = np.argsort(np.argsort(scores)) + 1.0
    return float((ranks[labels == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    gold = json.load(open(GOLD))["items"]
    dm = json.load(open(DUPS))

    # ---- gold windows ----
    winfeats, winrows = {}, []
    for it in gold:
        f = window_features(it["path"], it["start_s"], it["end_s"])
        f["id"] = it["id"]
        f["clip"] = it["clip"]
        f["label"] = it["label"]
        f["reason"] = it.get("reason")
        winfeats[it["id"]] = f
        winrows.append(f)
    json.dump(winrows, open(os.path.join(OUTDIR, "gold_features.json"), "w"))

    scores_def = [("sharp_h", True), ("sharp_r", True), ("lapl", True),
                  ("pc_peak", False), ("pc_froz", False), ("pc_shift", False)]
    clips = sorted(set(r["clip"] for r in winrows))
    rng = np.random.RandomState(BOOT_SEED)
    loco = {}
    for name, higher_usable in scores_def:
        s = np.array([r.get(name, float("nan")) for r in winrows])
        y = np.array([r["label"] for r in winrows])
        ok = ~np.isnan(s)
        s, y = s[ok], y[ok]
        ids = [winrows[i]["id"] for i in np.where(ok)[0]]
        cl = np.array([winrows[i]["clip"] for i in np.where(ok)[0]])
        pooled = auroc(s if higher_usable else -s, y)
        per_fold = {}
        for c in clips:
            m = cl == c
            if m.sum() == 0:
                per_fold[c] = None
                continue
            per_fold[c] = auroc((s if higher_usable else -s)[m], y[m])
        valid = [v for v in per_fold.values() if v is not None and not np.isnan(v)]
        mean_fold = float(np.mean(valid)) if valid else float("nan")
        # clip bootstrap CI for pooled AUROC
        boots = []
        clip_list = sorted(set(cl.tolist()))
        for _ in range(BOOT_ITERS):
            samp = rng.choice(clip_list, size=len(clip_list), replace=True)
            idx = [i for i, c in enumerate(cl) if c in set(samp)]
            # proper bootstrap: resample with multiplicity
            ii = []
            for c in samp:
                ii += [i for i, cc in enumerate(cl) if cc == c]
            boots.append(auroc((s if higher_usable else -s)[ii], y[ii]))
        boots = np.array([b for b in boots if not np.isnan(b)])
        ci = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))] if len(boots) else [float("nan")] * 2
        loco[name] = {"pooled_auroc": pooled, "mean_fold_auroc": mean_fold,
                      "n_valid_folds": len(valid), "per_fold": per_fold,
                      "ci95": ci, "n": int(ok.sum())}
    json.dump(loco, open(os.path.join(OUTDIR, "loco_quality.json"), "w"), indent=1)

    # ---- dup files (feature per unique file, computed once) ----
    filefeat = {}
    for key, path in dm["files"].items():
        filefeat[key] = window_features(path, 0.0, None)
    json.dump({k: v for k, v in filefeat.items()},
              open(os.path.join(OUTDIR, "dup_features.json"), "w"))

    pairs = dm["pairs"]
    pdists = []
    for i, p in enumerate(pairs):
        fa, fb = filefeat[p["a"]], filefeat[p["b"]]
        if "error" in fa or "error" in fb:
            d, dmns = None, None
        else:
            d = wdist_hist(fa["hist"], fb["hist"])
            dmns = wdist_means(fa["frame_means"], fb["frame_means"])
        pdists.append({"idx": i, "a": p["a"], "b": p["b"], "is_dup": p["is_duplicate"],
                       "kind": p.get("kind"), "w_hist": d, "w_means": dmns})
    json.dump(pdists, open(os.path.join(OUTDIR, "dup_distances.json"), "w"))

    # held-out threshold: train on even idx, test on odd idx
    def f1_at(thr, rows):
        tp = sum(1 for r in rows if r["w_hist"] is not None and r["w_hist"] <= thr and r["is_dup"])
        fp = sum(1 for r in rows if r["w_hist"] is not None and r["w_hist"] <= thr and not r["is_dup"])
        fn = sum(1 for r in rows if r["w_hist"] is not None and r["w_hist"] > thr and r["is_dup"])
        p = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * p * rc / (p + rc) if p + rc else 0.0
        return p, rc, f1

    train = [r for r in pdists if r["idx"] % 2 == 0 and r["w_hist"] is not None]
    test = [r for r in pdists if r["idx"] % 2 == 1 and r["w_hist"] is not None]
    cands = sorted(set(r["w_hist"] for r in train))
    grid = [(cands[i] + cands[i + 1]) / 2 for i in range(len(cands) - 1)] + cands
    best = max(grid, key=lambda t: (f1_at(t, train)[2], -t))
    ptr, rtr, ftr = f1_at(best, train)
    pte, rte, fte = f1_at(best, test)
    # per-kind recall on test
    kinds = {}
    for r in test:
        kinds.setdefault(r["kind"], []).append(r)
    kind_recall = {k: float(np.mean([x["w_hist"] <= best for x in v])) for k, v in kinds.items()}
    dup_eval = {"threshold": best, "train": {"n": len(train), "p": ptr, "r": rtr, "f1": ftr},
                "test": {"n": len(test), "p": pte, "r": rte, "f1": fte},
                "kind_recall_test": kind_recall}
    json.dump(dup_eval, open(os.path.join(OUTDIR, "dup_eval.json"), "w"), indent=1)

    # ---- 6 real gold DUPLICATEs, descriptive ----
    dup_ids = [r["id"] for r in winrows if (r.get("reason") or "") == "DUPLICATE"]
    dd = []
    for i in range(len(dup_ids)):
        for j in range(i + 1, len(dup_ids)):
            dd.append(wdist_hist(winfeats[dup_ids[i]]["hist"], winfeats[dup_ids[j]]["hist"]))
    brng = np.random.RandomState(1)
    others = [r["id"] for r in winrows if r["id"] not in dup_ids]
    bg = []
    for _ in range(200):
        a, b = brng.choice(others, size=2, replace=False)
        bg.append(wdist_hist(winfeats[a]["hist"], winfeats[b]["hist"]))
    # dup-dup AUC: dup-dup pairs as positives (lower distance), bg as negatives
    alls = np.array(dd + bg)
    lab = np.array([1] * len(dd) + [0] * len(bg))
    dupdup_auc = auroc(-alls, lab)
    json.dump({"dup_ids": dup_ids, "n_dupdup": len(dd),
               "dupdup_median": float(np.median(dd)) if dd else None,
               "dupdup_min": float(np.min(dd)) if dd else None,
               "bg_median": float(np.median(bg)), "bg_p05": float(np.percentile(bg, 5)),
               "dupdup_auc_vs_bg": dupdup_auc,
               "constructed_dup_test_median_pos": float(np.median([r["w_hist"] for r in test if r["is_dup"]])),
               "constructed_dup_test_median_neg": float(np.median([r["w_hist"] for r in test if not r["is_dup"]]))},
              open(os.path.join(OUTDIR, "gold_dup_descriptive.json"), "w"), indent=1)

    # ---- cost ----
    costs = [r["cost_ms"] for r in winrows if "cost_ms" in r] + \
            [v["cost_ms"] for v in filefeat.values() if "cost_ms" in v]
    costs = np.array(costs)
    json.dump({"n_windows": len(costs), "mean_ms": float(costs.mean()),
               "p50_ms": float(np.percentile(costs, 50)), "p95_ms": float(np.percentile(costs, 95)),
               "max_ms": float(costs.max()), "omp": os.environ.get("OMP_NUM_THREADS"),
               "single_core": True},
              open(os.path.join(OUTDIR, "cost.json"), "w"), indent=1)

    # ---- hashable verdict sample ----
    h = hashlib.sha256()
    for r in winrows:
        if "error" not in r:
            h.update(f"{r['id']}|{r['sharp_h']:.6f}|{r['pc_peak']:.6f}|{json.dumps(r['hist'])[:64]}".encode())
    print(json.dumps({"loco": loco, "dup_eval": dup_eval, "dupdup_auc": dupdup_auc,
                      "cost_mean_ms": float(costs.mean()), "audit_sha": h.hexdigest()[:16]}, indent=1))


if __name__ == "__main__":
    main()
