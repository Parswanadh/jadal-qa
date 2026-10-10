#!/usr/bin/env python3
"""A2 — extract NEW full-resolution signals (all frames) for gold + OOD160.

Signals (none of these exist in the cached 160x90 feature set):
  flow.*    dense optical flow (Farneback, all consecutive frames) magnitude /
            divergence / curl statistics + flow stability + active area.
            Flow is computed on the full-res decode downscaled to FLOW_W wide
            (320p) = ~4x the 160x90 baseline pixel density.
  hand.*    MediaPipe HandLandmarker (Tasks API, CPU) on ~100 sampled frames /
            window: hand-present fraction, hand-motion fraction, occupancy,
            mean #hands, hand-region flow excess.
  fr.*      full-resolution (native 480x480 / 854x480) per-pixel temporal
            features: Laplacian sharpness, noise floor, high-freq band energy.

Decodes with ffmpeg (NVDEC chain if available, CPU fallback). CPU only in
steady state (no GPU lock); one process per window, OMP_NUM_THREADS=1.

Writes eval/results/exp_acc/inputs/signals_gold.npz, signals_ood.npz
and a cost log eval/results/exp_acc/inputs/signals_cost.json.
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import subprocess
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
BUNDLE = os.environ.get("JADAL_BUNDLE", "/workspace/jadal_pod_bundle")
INP = os.path.join(ROOT, "eval", "results", "exp_acc", "inputs")
HAND_MODEL = os.environ.get("HAND_MODEL", "/tmp/hand_landmarker.task")

FLOW_W = 256          # flow compute width (full-res decode -> 256p)
HAND_STRIDE = None    # set per-domain to hit ~60 calls/window
N_HAND_TARGET = 60

cv2.setNumThreads(1)


# ------------------------------------------------------------------ decode
def decode_uint8(path, start_s, dur_s, timeout_s=180):
    """Decode [start_s, start_s+dur_s) at native res, rgb24 uint8. (T,H,W,3)."""
    cmd = ["ffmpeg", "-hide_banner", "-v", "error", "-nostdin",
           "-ss", str(start_s), "-t", str(dur_s), "-i", path,
           "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                       stdin=subprocess.DEVNULL, timeout=timeout_s)
    if p.returncode != 0 or len(p.stdout) < 3:
        raise RuntimeError(f"ffmpeg rc={p.returncode}: {p.stderr[-300:].decode('utf-8', 'replace')}")
    # probe size from first frame via ffprobe
    probe = subprocess.run(["ffprobe", "-hide_banner", "-v", "error", "-select_streams",
                            "v:0", "-show_entries", "stream=width,height", "-of",
                            "csv=s=x:p=0", path], capture_output=True, text=True, timeout=30)
    w, h = [int(v) for v in probe.stdout.strip().split("x")]
    fs = w * h * 3
    n = len(p.stdout) // fs
    if n == 0:
        raise RuntimeError(f"no frames from {path}@{start_s}")
    arr = np.frombuffer(p.stdout[:n * fs], dtype=np.uint8).reshape(n, h, w, 3)
    return arr, n / max(dur_s, 1e-9)


def luma_u8(rgb):
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


# ------------------------------------------------------------------ flow
def _flow_stats(flow):
    u, v = flow[..., 0], flow[..., 1]
    mag = np.sqrt(u * u + v * v)
    # divergence / curl via central differences
    du = np.gradient(u)
    dv = np.gradient(v)
    div = du[1] + dv[0]
    curl = dv[1] - du[0]
    return mag, div, curl


def flow_features(luma_frames):
    """luma_frames: list of uint8 full-res luma. Farneback on FLOW_W-wide."""
    T = len(luma_frames)
    small = []
    for f in luma_frames:
        h, w = f.shape
        nw = min(FLOW_W, w)
        nh = max(8, int(round(h * nw / w)))
        small.append(cv2.resize(f, (nw, nh), interpolation=cv2.INTER_AREA))
    mags, divs, curls, means, actives = [], [], [], [], []
    magmaps = []
    prev = small[0]
    for i in range(1, T):
        cur = small[i]
        fl = cv2.calcOpticalFlowFarneback(prev, cur, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        m, d, c = _flow_stats(fl)
        mags.append(m); divs.append(np.abs(d)); curls.append(np.abs(c))
        magmaps.append(m.astype(np.float32))
        means.append(float(m.mean())); actives.append(float((m > 1.0).mean()))
        prev = cur
    if not mags:
        return {}, []
    M = np.concatenate([m.ravel() for m in mags])
    D = np.concatenate([d.ravel() for d in divs])
    C = np.concatenate([c.ravel() for c in curls])
    means = np.asarray(means)
    # per-frame spatial coefficient of variation = std/mean
    cv_t = np.asarray([float(m.std() / (m.mean() + 1e-6)) for m in mags])
    return {
        "flow_mag_mean": float(M.mean()),
        "flow_mag_p90": float(np.percentile(M, 90)),
        "flow_mag_p99": float(np.percentile(M, 99)),
        "flow_mag_spatial_std": float(np.mean([float(m.std()) for m in mags])),
        "flow_active_frac": float(np.mean(actives)),
        "flow_div_abs_mean": float(D.mean()),
        "flow_div_abs_p90": float(np.percentile(D, 90)),
        "flow_curl_abs_mean": float(C.mean()),
        "flow_curl_abs_p90": float(np.percentile(C, 90)),
        "flow_stability": float(1.0 / (1.0 + cv_t.mean())),      # 1 = coherent
        "flow_static_frac": float((means < 0.35).mean()),
        "flow_mag_temporal_std": float(means.std()),
        "flow_peak_frac": float((means > max(2.0 * float(np.median(means)), 0.5)).mean()),
    }, magmaps


# ------------------------------------------------------------------ hands
def _new_hand_landmarker():
    from mediapipe.tasks import python as mpp
    from mediapipe.tasks.python import vision
    opt = vision.HandLandmarkerOptions(
        base_options=mpp.BaseOptions(model_asset_path=HAND_MODEL),
        running_mode=vision.RunningMode.VIDEO, num_hands=2,
        min_hand_detection_confidence=0.3, min_tracking_confidence=0.3)
    return vision.HandLandmarker.create_from_options(opt)


def hand_features(rgb, luma_frames, flow_mag_map, stride):
    """MediaPipe Hands on sampled frames + hand-region flow excess."""
    from mediapipe import Image, ImageFormat
    lm = _new_hand_landmarker()
    T = rgb.shape[0]
    idx = list(range(0, T, stride))
    present = 0
    occ = []
    nh = []
    hmot = 0
    exc = []
    for k, i in enumerate(idx):
        im = Image(image_format=ImageFormat.SRGB, data=np.ascontiguousarray(rgb[i]))
        res = lm.detect_for_video(im, int(i * 1000 / 10.0))
        hands = res.hand_landmarks or []
        nh.append(len(hands))
        if not hands:
            continue
        present += 1
        H, W = rgb.shape[1], rgb.shape[2]
        xs, ys = [], []
        for hd in hands:
            for p in hd:
                xs.append(p.x); ys.append(p.y)
        x0 = int(max(0, min(xs) * W)); x1 = int(min(W, max(xs) * W))
        y0 = int(max(0, min(ys) * H)); y1 = int(min(H, max(ys) * H))
        occ.append(max(0, (x1 - x0)) * max(0, (y1 - y0)) / (W * H))
        # hand-region flow excess using the nearest flow map
        if flow_mag_map is not None and 0 < i < len(flow_mag_map) + 1:
            fm = flow_mag_map[min(i - 1, len(flow_mag_map) - 1)]
            fh, fw = fm.shape
            sx, sy = fw / W, fh / H
            rx0, rx1 = int(x0 * sx), max(int(x0 * sx) + 1, int(x1 * sx))
            ry0, ry1 = int(y0 * sy), max(int(y0 * sy) + 1, int(y1 * sy))
            reg = fm[ry0:ry1, rx0:rx1]
            if reg.size > 0:
                exc.append(float(reg.mean() - fm.mean()))
                if float(reg.mean()) > max(1.5 * float(fm.mean()), 0.5):
                    hmot += 1
    out = {
        "hand_present_frac": present / max(len(idx), 1),
        "hand_motion_frac": hmot / max(len(idx), 1),
        "hand_occupancy_mean": float(np.mean(occ)) if occ else 0.0,
        "hand_n_mean": float(np.mean(nh)) if nh else 0.0,
        "hand_flow_excess_mean": float(np.mean(exc)) if exc else 0.0,
    }
    lm.close()
    return out


# ------------------------------------------------------------------ full-res per-pixel
def fullres_features(luma_frames):
    lap, noise, hf = [], [], []
    for f in luma_frames:
        lap.append(float(cv2.Laplacian(f, cv2.CV_64F).var()))
        # noise: MAD of a 3x3 median-filter residual on a centre crop
        h, w = f.shape
        c = f[h // 4:3 * h // 4, w // 4:3 * w // 4]
        med = cv2.medianBlur(c, 3)
        noise.append(float(np.median(np.abs(c.astype(np.float32) - med.astype(np.float32)))))
        # high-freq band energy ratio via 2D FFT of a 256x256 centre patch
        p = f[h // 2 - 128:h // 2 + 128, w // 2 - 128:w // 2 + 128]
        if p.shape == (256, 256):
            F = np.fft.fftshift(np.abs(np.fft.fft2(p.astype(np.float32) - p.mean())) ** 2)
            yy, xx = np.mgrid[-128:128, -128:128]
            r = np.sqrt(yy ** 2 + xx ** 2) / 128.0
            hf.append(float(F[r > 0.5].sum() / (F[r > 0].sum() + 1e-9)))
    lap = np.asarray(lap); noise = np.asarray(noise); hf = np.asarray(hf)
    return {
        "fr_lap_var_mean": float(lap.mean()),
        "fr_lap_var_p10": float(np.percentile(lap, 10)),
        "fr_sharp_temporal_cv": float(lap.std() / (lap.mean() + 1e-6)),
        "fr_noise_mean": float(noise.mean()),
        "fr_noise_p90": float(np.percentile(noise, 90)),
        "fr_hf_band_frac": float(hf.mean()) if hf.size else 0.0,
    }


# ------------------------------------------------------------------ per window
def process(item):
    wid = item["id"]
    t0 = time.perf_counter()
    cost = {}
    try:
        t = time.perf_counter()
        rgb, fps = decode_uint8(item["path"], item["start_s"], item["dur_s"])
        cost["decode_ms"] = round((time.perf_counter() - t) * 1000, 1)
        T = rgb.shape[0]
        luma = [luma_u8(rgb[i]) for i in range(T)]
        t = time.perf_counter()
        fl, maps = flow_features(luma)
        cost["flow_ms"] = round((time.perf_counter() - t) * 1000, 1)
        stride = item.get("hand_stride", 3)
        t = time.perf_counter()
        hf = hand_features(rgb, luma, maps, stride)
        cost["hand_ms"] = round((time.perf_counter() - t) * 1000, 1)
        t = time.perf_counter()
        ff = fullres_features(luma)
        cost["fullres_ms"] = round((time.perf_counter() - t) * 1000, 1)
        out = {**fl, **hf, **ff}
        out["_frames"] = int(T)
        out["_fps"] = round(float(fps), 2)
        cost["total_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        return wid, out, cost
    except Exception as e:
        return wid, {"error": str(e)[:300]}, cost


# ------------------------------------------------------------------ main
def load_items(domain):
    if domain == "gold":
        m = json.load(open(os.path.join(BUNDLE, "labels", "manifest_gold.json")))
        items = []
        for it in m["items"]:
            p = it["path"].replace("/home/parshu/projects/awing/data/",
                                   os.path.join(BUNDLE, "data") + "/")
            items.append({"id": it["id"], "path": p, "start_s": it["start_s"],
                          "dur_s": round(it["end_s"] - it["start_s"], 2),
                          "hand_stride": 2})
        return items
    d = json.load(open(os.path.join(BUNDLE, "labels", "ood160_labels.json")))
    items = []
    for w in d["windows"]:
        p = os.path.join(BUNDLE, w["video"])
        items.append({"id": w["segment_id"], "path": p, "start_s": 0.0,
                      "dur_s": 10.0, "hand_stride": 6})
    return items


def main(domain="gold", jobs=6, limit=None):
    items = load_items(domain)
    if limit:
        items = items[:limit]
    t0 = time.perf_counter()
    with mp.Pool(min(jobs, len(items))) as pool:
        rows = pool.map(process, items)
    ids = [r[0] for r in rows]
    feats = [r[1] for r in rows]
    costs = [r[2] for r in rows]
    names = sorted({k for f in feats if "error" not in f for k in f if not k.startswith("_")})
    X = np.array([[float(f.get(n, np.nan)) for n in names] for f in feats], dtype=np.float32)
    os.makedirs(INP, exist_ok=True)
    np.savez(os.path.join(INP, f"signals_{domain}.npz"),
             ids=np.asarray(ids), X=X, names=np.asarray(names),
             frames=np.asarray([f.get("_frames", 0) for f in feats]))
    wall = time.perf_counter() - t0
    errs = sum(1 for f in feats if "error" in f)
    tot = np.array([c.get("total_ms", 0) for c in costs])
    summary = {"domain": domain, "n": len(rows), "errors": errs,
               "wall_s": round(wall, 1),
               "ms_per_window_mean": round(float(tot.mean()), 1) if len(tot) else None,
               "ms_per_window_median": round(float(np.median(tot)), 1) if len(tot) else None,
               "decode_ms_median": round(float(np.median([c.get("decode_ms", 0) for c in costs])), 1),
               "flow_ms_median": round(float(np.median([c.get("flow_ms", 0) for c in costs])), 1),
               "hand_ms_median": round(float(np.median([c.get("hand_ms", 0) for c in costs])), 1),
               "fullres_ms_median": round(float(np.median([c.get("fullres_ms", 0) for c in costs])), 1)}
    json.dump(summary, open(os.path.join(INP, f"signals_cost_{domain}.json"), "w"), indent=1)
    print(json.dumps(summary))
    for i, f in enumerate(feats):
        if "error" in f:
            print("ERR", ids[i], f["error"])
    return summary


if __name__ == "__main__":
    dom = sys.argv[1] if len(sys.argv) > 1 else "gold"
    lim = int(sys.argv[2]) if len(sys.argv) > 2 else None
    jobs = int(sys.argv[3]) if len(sys.argv) > 3 else 6
    main(dom, jobs, lim)
