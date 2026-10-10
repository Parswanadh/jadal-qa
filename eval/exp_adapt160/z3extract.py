"""Z3 candidate domain-agnostic feature extraction for IndEgo + Egocentric-10K.

Three candidate families, computed on the SAME 10-s windows:
  A. compression telemetry (ffprobe frame packets; no full decode)
  B. affine ego-motion residual (fixes the translation-only flaw)
  C. illumination / scale-space invariants

Compression telemetry is computed in the parent from a per-file frame table
(ffprobe `-read_intervals` proved unreliable, so we probe whole clips once and
slice by pts_time). Decode-based candidates run in 2 worker processes.

Outputs eval/results/exp_features/candidates.npz.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)
# Pod path from the environment only (no tracked default, review finding 6).
BUNDLE = os.environ.get("JADAL_BUNDLE")
OUT = os.path.join(REPO, "eval", "results", "exp_features")
os.makedirs(OUT, exist_ok=True)

COMPRESSION = ["ct_bitrate", "ct_fsize_mean", "ct_fsize_cv", "ct_iframe_ratio",
               "ct_fsize_entropy", "ct_iframe_frac", "ct_fsize_std"]
MOTION = ["mo_trans_resid", "mo_affine_resid", "mo_affine_gain",
          "mo_hand_trans_resid", "mo_hand_affine_resid", "mo_hand_motion_excess",
          "mo_motion_frac"]
INVARIANTS = ["inv_dct_hf_lf", "inv_noise_snr", "inv_clip_frac",
              "inv_hist_entropy", "inv_dhash_dist", "inv_laplacian_var"]
CAND = COMPRESSION + MOTION + INVARIANTS


# ---------------- A. compression telemetry ----------------

def probe_full(path):
    """Whole-file per-frame (pts_time, pkt_size, pict_type) via one ffprobe call."""
    cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0",
           "-show_entries", "frame=pts_time,pkt_size,pict_type", "-of", "csv=p=0", path]
    try:
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=180)
    except subprocess.TimeoutExpired:
        return None
    pts, sz, ty = [], [], []
    for line in p.stdout.decode("utf-8", "replace").splitlines():
        parts = line.split(",")
        if len(parts) < 3:
            continue
        try:
            pts.append(float(parts[0]))
            s = parts[1]
            sz.append(float(s) if s else np.nan)
        except ValueError:
            continue
        ty.append(parts[2])
    if not pts:
        return None
    return np.asarray(pts), np.asarray(sz, float), np.asarray(ty)


def compression_from_table(tbl, start_s, end_s):
    if tbl is None:
        return None
    pts, sizes, types = tbl
    m = (pts >= float(start_s) - 1e-6) & (pts < float(end_s) - 1e-6)
    sizes = sizes[m]
    types = types[m]
    if len(sizes) < 3 or np.isnan(sizes).any():
        return None
    dur = float(end_s) - float(start_s)
    i_sz = sizes[types == "I"]
    p_sz = sizes[types != "I"]
    if len(p_sz) == 0:
        p_sz = sizes
    if len(i_sz) == 0:
        i_sz = sizes[:1]
    total = float(sizes.sum())
    bitrate = total * 8.0 / max(dur, 1e-6)
    p = sizes / total
    entropy = float(-(p * np.log(p + 1e-12)).sum())
    return {
        "ct_bitrate": bitrate,
        "ct_fsize_mean": float(sizes.mean()),
        "ct_fsize_std": float(sizes.std()),
        "ct_fsize_cv": float(sizes.std() / (sizes.mean() + 1e-9)),
        "ct_iframe_ratio": float(i_sz.mean() / (p_sz.mean() + 1e-9)),
        "ct_fsize_entropy": entropy,
        "ct_iframe_frac": float(len(i_sz) / len(sizes)),
    }


# ---------------- B + C. decode-based ----------------

def _skin_mask(rgb):
    import cv2
    u8 = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    ycc = cv2.cvtColor(u8, cv2.COLOR_RGB2YCrCb)
    y, cr, cb = ycc[..., 0], ycc[..., 1], ycc[..., 2]
    return (cr >= 133) & (cr <= 173) & (cb >= 77) & (cb <= 127) & (y > 35)


def _pair_residuals(g0, g1):
    import cv2
    f0 = g0.astype(np.float32)
    f1 = g1.astype(np.float32)
    try:
        (dx, dy), _ = cv2.phaseCorrelate(f0, f1)
        Mt = np.array([[1, 0, float(dx)], [0, 1, float(dy)]], np.float32)
        gt = cv2.warpAffine(f0, Mt, (f0.shape[1], f0.shape[0]),
                            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        trans = float(np.abs(gt - f1).mean())
    except Exception:
        trans = float(np.abs(f0 - f1).mean())
    affine = trans
    try:
        u0 = (np.clip(g0, 0, 1) * 255).astype(np.uint8)
        u1 = (np.clip(g1, 0, 1) * 255).astype(np.uint8)
        pts0 = cv2.goodFeaturesToTrack(u0, maxCorners=250, qualityLevel=0.01, minDistance=4)
        if pts0 is not None and len(pts0) >= 8:
            pts1, st, _ = cv2.calcOpticalFlowPyrLK(u0, u1, pts0, None,
                                                   winSize=(15, 15), maxLevel=3)
            good = st.ravel() == 1
            if good.sum() >= 8:
                M, _ = cv2.estimateAffinePartial2D(pts0[good], pts1[good],
                                                   method=cv2.RANSAC, ransacReprojThreshold=3.0)
                if M is not None and np.isfinite(M).all():
                    gw = cv2.warpAffine(f0, M.astype(np.float32), (f0.shape[1], f0.shape[0]),
                                        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                    affine = float(np.abs(gw - f1).mean())
    except Exception:
        pass
    return trans, affine


def _invariants_step(rgb):
    import cv2
    g = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    out = {}
    x = cv2.resize(rgb, (64, 64), interpolation=cv2.INTER_AREA)
    xg = 0.299 * x[..., 0] + 0.587 * x[..., 1] + 0.114 * x[..., 2]
    d = cv2.dct(xg.astype(np.float32))
    yy, xx = np.mgrid[0:64, 0:64]
    r = np.sqrt((yy - 32) ** 2 + (xx - 32) ** 2)
    out["inv_dct_hf_lf"] = float((d[r > 20] ** 2).sum() / ((d[r < 8] ** 2).sum() + 1e-9))
    lap = cv2.Laplacian(g.astype(np.float32), cv2.CV_32F)
    nvar = float(np.median(np.abs(lap - np.median(lap))) + 1e-6) ** 2
    out["inv_laplacian_var"] = float(lap.var())
    out["inv_noise_snr"] = float(lap.var() / (nvar * 4.0 + 1e-9))
    out["inv_clip_frac"] = float(((g <= 2 / 255.0) | (g >= 253 / 255.0)).mean())
    h, _ = np.histogram(g, bins=32, range=(0, 1))
    p = h / max(h.sum(), 1)
    out["inv_hist_entropy"] = float(-(p[p > 0] * np.log(p[p > 0])).sum())
    return out, g


def _dhash(g):
    import cv2
    small = cv2.resize(g, (9, 8), interpolation=cv2.INTER_AREA)
    return (small[:, 1:] > small[:, :-1]).ravel()


def decode_features(path, start_s, end_s, target_fps=10.0, w=160, h=90):
    import cv2
    cv2.setNumThreads(1)
    from jadal.ensemble.scorer import decode_pixel_window
    luma, rgb, native_fps = decode_pixel_window(path, start_s, end_s, w, h)
    if native_fps > target_fps + 1:
        step = max(1, int(round(native_fps / target_fps)))
        luma, rgb = luma[::step], rgb[::step]
    T = len(luma)
    if T < 3:
        return None
    out = {}
    sk = _skin_mask(rgb[T // 2])
    out["_skin_frac"] = float(sk.mean())
    trans_l, aff_l, htrans_l, haff_l, motion_l = [], [], [], [], []
    inv_acc = {k: [] for k in INVARIANTS if k != "inv_dhash_dist"}
    dhash_l = []
    prev_hash = None
    for t in range(T - 1):
        g0, g1 = luma[t], luma[t + 1]
        motion_l.append(float(np.abs(g0 - g1).mean()))
        tr, af = _pair_residuals(g0, g1)
        trans_l.append(tr)
        aff_l.append(af)
        if sk.sum() >= 20:
            d = np.abs(g0 - g1)
            htrans_l.append(float(d[sk].mean()))
            haff_l.append(float(af * (float(d[sk].mean()) / (tr + 1e-9))))
        inv, g = _invariants_step(rgb[t])
        for k, v in inv.items():
            inv_acc[k].append(v)
        hsh = _dhash(g)
        if prev_hash is not None:
            dhash_l.append(float(np.count_nonzero(hsh != prev_hash)))
        prev_hash = hsh
    out["mo_trans_resid"] = float(np.mean(trans_l))
    out["mo_affine_resid"] = float(np.mean(aff_l))
    out["mo_affine_gain"] = float(np.mean(trans_l) - np.mean(aff_l))
    out["mo_hand_trans_resid"] = float(np.mean(htrans_l)) if htrans_l else 0.0
    out["mo_hand_affine_resid"] = float(np.mean(haff_l)) if haff_l else 0.0
    out["mo_hand_motion_excess"] = float(np.mean(htrans_l) - np.mean(trans_l)) if htrans_l else 0.0
    out["mo_motion_frac"] = float(np.mean(motion_l))
    for k, v in inv_acc.items():
        out[k] = float(np.mean(v))
    out["inv_dhash_dist"] = float(np.mean(dhash_l)) if dhash_l else 0.0
    out["_fps"] = float((T - 1) / max(end_s - start_s, 1e-6))
    out["_T"] = int(T)
    return out


def decode_job(job):
    kind, wid, path, start_s, end_s, label = job
    res = {"kind": kind, "id": wid, "label": int(label), "error": None}
    try:
        d = decode_features(path, start_s, end_s)
        if d is None:
            raise RuntimeError("too few frames")
        res.update(d)
    except Exception as e:
        res["error"] = f"decode:{e}"[:200]
    for k in CAND:
        res.setdefault(k, np.nan)
    return res


def build_jobs():
    if not BUNDLE:
        raise SystemExit(
            "JADAL_BUNDLE is not set and no pod path is hard-coded; export it, e.g. "
            "export JADAL_BUNDLE=/path/on/pod")
    jobs = []
    gold = json.load(open(os.path.join(BUNDLE, "labels", "manifest_gold.json")))["items"]
    for it in gold:
        jobs.append(("indego", it["id"], it["path"], it["start_s"], it["end_s"], it["label"]))
    lab = json.load(open(os.path.join(BUNDLE, "labels", "ood_labels.json")))["latest"]
    q = json.load(open(os.path.join(BUNDLE, "queues", "label_queue_random.json")))["segments"]
    for s in q:
        v = lab.get(s["segment_id"])
        if v is None:
            continue
        jobs.append(("ood", s["segment_id"], s["path"], s["start_s"], s["end_s"],
                     1 if v["verdict"] == "good" else 0))
    return jobs


def main():
    import multiprocessing as mp
    import time
    jobs = build_jobs()
    print(f"jobs: {len(jobs)} (indego={sum(j[0]=='indego' for j in jobs)}, "
          f"ood={sum(j[0]=='ood' for j in jobs)})", flush=True)

    # compression: probe unique paths once
    t0 = time.time()
    uniq = {}
    for j in jobs:
        uniq.setdefault(j[2], j[3:5])
    print(f"probing {len(uniq)} unique clips ...", flush=True)
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=2) as ex:
        tables = dict(zip(uniq.keys(), ex.map(probe_full, uniq.keys())))
    print(f"probe done {time.time()-t0:.1f}s", flush=True)

    with mp.Pool(2) as pool:
        rows = pool.map(decode_job, jobs, chunksize=4)

    # attach compression from parent tables (keyed by path -> need path per row)
    path_by = {(j[0], j[1]): (j[2], j[3], j[4]) for j in jobs}
    for r in rows:
        path, s0, s1 = path_by[(r["kind"], r["id"])]
        ct = compression_from_table(tables.get(path), s0, s1)
        if ct is None:
            for k in COMPRESSION:
                r[k] = np.nan
        else:
            r.update(ct)

    names = CAND
    out = {}
    for kind in ("indego", "ood"):
        rr = [r for r in rows if r["kind"] == kind]
        out[f"{kind}_ids"] = np.asarray([r["id"] for r in rr])
        out[f"{kind}_y"] = np.asarray([r["label"] for r in rr], int)
        out[f"{kind}_X"] = np.asarray([[float(r.get(n, np.nan)) for n in names] for r in rr], float)
        out[f"{kind}_meta"] = np.asarray([json.dumps({k: v for k, v in r.items()
                                                       if k.startswith("_") or k == "error"}) for r in rr])
        errs = [r for r in rr if r.get("error")]
        print(f"{kind}: n={len(rr)} good={int((out[kind+'_y']==1).sum())} errors={len(errs)}", flush=True)
        if errs:
            print("  sample error:", errs[0]["error"], flush=True)
    out["names"] = np.asarray(names)
    np.savez_compressed(os.path.join(OUT, "candidates.npz"), **out)
    print("wrote", os.path.join(OUT, "candidates.npz"), flush=True)


if __name__ == "__main__":
    main()
