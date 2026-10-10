"""Arm B spectral feature extractor (RGB, whole-video, CPU-only, deterministic).

Decode: ONE ffmpeg subprocess per segment (fast input seek + accurate output
trim), scaled in C to the tier working resolution, rgb24 rawvideo pipe.
MV counts come from the same process's stderr (-flags2 +export_mvs).
Frame-size stats come from a decode-free ffprobe -show_packets pass.
Every decoded frame in [start_s, end_s) updates streaming accumulators.

Tiers: T0 = 64x64, T1 = 160x160 (source here is square 480x480; the
contract's 64x36/160x90 assumed 16:9 — same pixel budgets).

extract(video_path, start_s, end_s, tier='T1') -> dict[str, float]
Deterministic: fixed decode order, float64 reductions, no randomness.
"""

from __future__ import annotations

import json
import subprocess
from collections import deque

import cv2
import numpy as np
import pywt

TIERS = {"T0": {"size": 64}, "T1": {"size": 160}}

FAMILIES = {
    "spatial_fft": ["spec_slope", "spec_hfratio", "spec_hfenergy", "spec_total"],
    "spatial_dct": ["dct_hfenergy", "dct_dcmean"],
    "wavelet": ["wav_LH", "wav_HL", "wav_HH"],
    "basic": ["lap_var", "sobel_energy", "brightness", "contrast",
              "under_frac", "over_frac"],
    "temporal": ["temp_lo", "temp_mid", "temp_hi", "temp_frozen_frac",
                 "temp_flicker", "temp_period", "motion_energy", "motion_std"],
    "chroma": ["cr_mean", "cb_mean", "cr_std", "cb_std", "opp_rg_mean",
               "opp_rg_std", "colorfulness"],
    "skin": ["skin_frac_mean", "skin_frac_std", "skin_centroid_motion",
             "skin_maxfrac"],
    "compressed": ["mv_per_frame", "i_size", "p_size_mean", "pb_size_ratio",
                   "frame_size_std", "p_count_frac"],
}

ALL_FEATURES = [f for fam in FAMILIES.values() for f in fam]


# ---------------------------------------------------------------- decode

def decode_pipe(path: str, start_s: float, end_s: float, size: int):
    """Single ffmpeg pass -> (frames[N,H,W,3] float64 in [0,1], mv_per_frame).

    Frames are EXACTLY [start_s, end_s) (fast input seek to start-3s, then
    accurate output trim). MV rate covers [seek_keyframe, end_s), i.e. up to
    ~3 s of pre-roll included; it is a rate so the bias is negligible.
    """
    dur = end_s - start_s
    pre = max(0.0, start_s - 3.0)
    off = start_s - pre
    p = subprocess.run(
        ["ffmpeg", "-hide_banner", "-v", "debug", "-flags2", "+export_mvs",
         "-ss", str(pre), "-i", path,
         "-ss", str(off), "-t", str(dur),
         "-map", "0:v:0", "-vf", f"scale={size}:{size}",
         "-pix_fmt", "rgb24", "-f", "rawvideo", "-"],
        capture_output=True, timeout=300)
    px = len(p.stdout) // (size * size * 3)
    if px == 0:
        raise ValueError(f"no frames decoded in [{start_s},{end_s}) of {path}")
    frames = np.frombuffer(p.stdout, dtype=np.uint8,
                           count=px * size * size * 3)
    frames = frames.reshape(px, size, size, 3).astype(np.float64) / 255.0
    n_mv, n_fr = 0, 0
    for line in p.stderr.decode("utf-8", "replace").splitlines():
        if " MVs info to frame " in line and "Adding " in line:
            try:
                n_mv += int(line.split("Adding ")[1].split(" MVs")[0])
                n_fr += 1
            except (IndexError, ValueError):
                pass
    mv_rate = float(n_mv / n_fr) if n_fr else 0.0
    return frames, mv_rate


def compressed_stats(path: str, start_s: float, end_s: float,
                     fps_hint: float = 10.0) -> dict:
    """Decode-free ffprobe packet sizes (key vs non-key) for the segment.

    Numeric per-block MV *magnitudes* are not exposed by the ffmpeg 6.1 CLI
    (only per-frame MV counts, captured in decode_pipe); motion magnitude
    comes from pixel frame-diffs. Documented limitation.
    """
    out = {"i_size": 0.0, "p_size_mean": 0.0, "pb_size_ratio": 0.0,
           "frame_size_std": 0.0, "p_count_frac": 0.0}
    n_exp = max(1, int(round((end_s - start_s) * fps_hint)) + 60)
    try:
        p = subprocess.run(
            ["ffprobe", "-v", "error", "-show_packets", "-select_streams",
             "v:0", "-read_intervals", f"{start_s}%+#{(n_exp)}",
             "-show_entries", "packet=size,flags,pts_time",
             "-of", "json", path],
            capture_output=True, text=True, timeout=120)
        pkts = json.loads(p.stdout).get("packets", [])
        key, nonkey = [], []
        for f in pkts:
            try:
                ts = float(f.get("pts_time", start_s))
            except (TypeError, ValueError):
                ts = start_s
            if not (start_s - 0.5 <= ts < end_s):
                continue
            try:
                s = int(f.get("size", 0))
            except (TypeError, ValueError):
                continue
            if "K" in f.get("flags", ""):
                key.append(s)
            else:
                nonkey.append(s)
        allp = np.asarray(key + nonkey, dtype=np.float64)
        if len(allp):
            out["frame_size_std"] = float(allp.std() / (allp.mean() + 1e-9))
            out["i_size"] = float(np.mean(np.asarray(key or [0], dtype=np.float64)))
            out["p_size_mean"] = float(np.mean(np.asarray(nonkey or [0], dtype=np.float64)))
            out["pb_size_ratio"] = float(out["p_size_mean"] / (out["i_size"] + 1e-9))
            out["p_count_frac"] = float(len(nonkey) / len(allp))
    except Exception:
        pass
    return out


# ------------------------------------------------------------- accumulator

class FrameAccumulator:
    """Streaming per-frame accumulators; O(window) memory via ring buffer."""

    def __init__(self, size: int, window: int = 400):
        self.size = size
        self.n = 0
        self.spec_pow = None
        self.acc = {k: 0.0 for k in
                    ["lap", "sob", "bri", "con", "under", "over", "dcthf",
                     "dctdc", "wLH", "wHL", "wHH", "cr_m", "cb_m", "cr_s",
                     "cb_s", "rg_m", "rg_s", "cful", "skin", "skin2", "mot"]}
        self.prev = None
        self.diff_list: list[float] = []
        self.mean_list: list[float] = []
        self.ring: deque = deque(maxlen=window)
        self.cx: list[float] = []
        self.cy: list[float] = []
        self.skin_list: list[float] = []

    def update(self, rgb: np.ndarray):
        """Feed one frame, float64 HxWx3 in [0,1] at working resolution."""
        size = self.size
        gray = (0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] +
                0.114 * rgb[:, :, 2])
        r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
        # YCrCb (BT.601) without extra resize: computed from rgb directly
        cr = 0.5 * (r - gray) / ((1 - 0.299) + 1e-12) + 0.5
        cb = 0.5 * (b - gray) / ((1 - 0.114) + 1e-12) + 0.5

        F = np.fft.rfft2(gray)
        power = (F.real ** 2 + F.conj().imag ** 2).astype(np.float64)
        self.spec_pow = power if self.spec_pow is None else self.spec_pow + power

        D = cv2.dct(gray.astype(np.float32)).astype(np.float64)
        h8, w8 = min(8, D.shape[0]), min(8, D.shape[1])
        hf = D.copy()
        hf[:h8, :w8] = 0.0
        a = self.acc
        a["dcthf"] += float(np.mean(np.abs(hf)))
        a["dctdc"] += float(abs(D[0, 0]))

        cA, (cH, cV, cD) = pywt.dwt2(gray, "db1")
        e = float(cA.size)
        a["wLH"] += float(np.sum(cH ** 2) / e)
        a["wHL"] += float(np.sum(cV ** 2) / e)
        a["wHH"] += float(np.sum(cD ** 2) / e)

        a["lap"] += float(cv2.Laplacian(gray, cv2.CV_64F).var())
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
        a["sob"] += float(np.mean(gx * gx + gy * gy))
        a["bri"] += float(gray.mean())
        a["con"] += float(gray.std())
        a["under"] += float(np.mean(gray < 0.05))
        a["over"] += float(np.mean(gray > 0.95))

        a["cr_m"] += float(cr.mean())
        a["cb_m"] += float(cb.mean())
        a["cr_s"] += float(cr.std())
        a["cb_s"] += float(cb.std())
        rg = r - g
        a["rg_m"] += float(rg.mean())
        a["rg_s"] += float(rg.std())
        rb, gb = r - b, g - b
        a["cful"] += float(np.sqrt(rb.std() ** 2 + gb.std() ** 2) +
                           0.3 * np.sqrt(rb.mean() ** 2 + gb.mean() ** 2))

        skin = ((cr * 255 >= 133) & (cr * 255 <= 173) &
                (cb * 255 >= 77) & (cb * 255 <= 127)).astype(np.float64)
        sf = float(skin.mean())
        self.skin_list.append(sf)
        a["skin"] += sf
        a["skin2"] += sf * sf
        ys, xs = np.nonzero(skin)
        if len(xs):
            self.cx.append(float(xs.mean()) / size)
            self.cy.append(float(ys.mean()) / size)
        else:
            self.cx.append(0.5)
            self.cy.append(0.5)

        if self.prev is not None:
            d = float(np.mean(np.abs(gray - self.prev)))
            self.diff_list.append(d)
            a["mot"] += d
        self.prev = gray
        self.mean_list.append(float(gray.mean()))
        self.ring.append(gray.astype(np.float32))
        self.n += 1

    def running_features(self, fps_hint: float = 10.0) -> dict:
        """Feature dict from frames seen so far (temporal FFT over window)."""
        n = self.n
        if n == 0:
            return {k: 0.0 for k in ALL_FEATURES if k != "mv_per_frame"}
        feats: dict[str, float] = {}
        inv = 1.0 / n
        h, w = self.spec_pow.shape
        yy, xx = np.mgrid[0:h, 0:w]
        rr = np.sqrt(((xx - w // 2) / (w / 2)) ** 2 +
                     ((yy - h // 2) / (h / 2)) ** 2)
        rbin = np.clip((rr * (min(h, w) // 2)).astype(np.int64), 0, min(h, w) // 2)
        pw = (self.spec_pow * inv).ravel().astype(np.float64)
        sums = np.bincount(rbin.ravel(), weights=pw)
        cnts = np.bincount(rbin.ravel())
        prof = (sums / np.maximum(cnts, 1))[1:]
        freqs = np.arange(1, len(prof) + 1, dtype=np.float64) / max(len(prof), 1)
        tot = float(prof.sum()) + 1e-12
        feats["spec_slope"] = float(np.polyfit(np.log1p(freqs), np.log1p(prof / tot), 1)[0])
        feats["spec_hfratio"] = float(prof[freqs > 0.5].sum() / tot)
        feats["spec_hfenergy"] = float(np.log1p(prof[freqs > 0.5].sum()))
        feats["spec_total"] = float(np.log1p(tot))

        a = self.acc
        feats["dct_hfenergy"] = float(np.log1p(a["dcthf"] * inv))
        feats["dct_dcmean"] = float(a["dctdc"] * inv)
        wtot = a["wLH"] + a["wHL"] + a["wHH"] + 1e-12
        feats["wav_LH"] = float(a["wLH"] / wtot)
        feats["wav_HL"] = float(a["wHL"] / wtot)
        feats["wav_HH"] = float(a["wHH"] / wtot)

        feats["lap_var"] = float(np.log1p(a["lap"] * inv))
        feats["sobel_energy"] = float(np.log1p(a["sob"] * inv))
        feats["brightness"] = float(a["bri"] * inv)
        feats["contrast"] = float(a["con"] * inv)
        feats["under_frac"] = float(a["under"] * inv)
        feats["over_frac"] = float(a["over"] * inv)

        cube = np.stack(list(self.ring), axis=0).astype(np.float64)
        T = cube.shape[0]
        spec = np.fft.rfft(cube, axis=0)
        mag2 = spec.real ** 2 + spec.conj().imag ** 2
        fr_t = np.fft.rfftfreq(T, d=1.0 / fps_hint)
        e_tot = mag2[1:].sum(axis=0) + 1e-12

        def band(lo, hi):
            m = (fr_t[1:] >= lo) & (fr_t[1:] < hi)
            if not m.any():
                return np.zeros_like(e_tot)
            return mag2[1:][m].sum(axis=0) / e_tot

        feats["temp_lo"] = float(band(0.15, 1.0).mean())
        feats["temp_mid"] = float(band(1.0, 3.0).mean())
        feats["temp_hi"] = float(band(3.0, 8.0).mean())
        feats["temp_frozen_frac"] = float(np.mean(cube.var(axis=0) < 1e-6))
        feats["temp_flicker"] = float((mag2[-1] / e_tot).mean()) if T > 2 else 0.0
        ms = np.asarray(self.mean_list[-len(self.ring):] or [0.0], dtype=np.float64)
        ms0 = ms - ms.mean()
        if float((ms0 ** 2).sum()) > 1e-12 and T > 4:
            ac = np.correlate(ms0, ms0, mode="full")[T - 1:]
            ac = ac / (ac[0] + 1e-12)
            feats["temp_period"] = float(ac[1:].max()) if len(ac) > 1 else 0.0
        else:
            feats["temp_period"] = 0.0
        feats["motion_energy"] = float(np.log1p(a["mot"] / max(n - 1, 1)))
        feats["motion_std"] = float(np.std(np.asarray(self.diff_list or [0.0],
                                                      dtype=np.float64)))

        feats["cr_mean"] = float(a["cr_m"] * inv)
        feats["cb_mean"] = float(a["cb_m"] * inv)
        feats["cr_std"] = float(a["cr_s"] * inv)
        feats["cb_std"] = float(a["cb_s"] * inv)
        feats["opp_rg_mean"] = float(a["rg_m"] * inv)
        feats["opp_rg_std"] = float(a["rg_s"] * inv)
        feats["colorfulness"] = float(a["cful"] * inv)

        sm = float(a["skin"] * inv)
        feats["skin_frac_mean"] = sm
        feats["skin_frac_std"] = float(np.sqrt(max(a["skin2"] * inv - sm ** 2, 0.0)))
        cx = np.asarray(self.cx, dtype=np.float64)
        cy = np.asarray(self.cy, dtype=np.float64)
        feats["skin_centroid_motion"] = float(cx.std() + cy.std())
        feats["skin_maxfrac"] = float(max(self.skin_list))
        return feats


# ---------------------------------------------------------------- extract

def extract(video_path: str, start_s: float, end_s: float,
            tier: str = "T1") -> dict:
    size = TIERS[tier]["size"]
    frames, mv_rate = decode_pipe(video_path, start_s, end_s, size)
    acc = FrameAccumulator(size, window=frames.shape[0] + 1)
    for i in range(frames.shape[0]):
        acc.update(frames[i])
    fps_hint = frames.shape[0] / max(end_s - start_s, 1e-9)
    feats = acc.running_features(fps_hint=fps_hint)
    feats["mv_per_frame"] = mv_rate
    feats.update(compressed_stats(video_path, start_s, end_s,
                                 fps_hint=fps_hint))
    return {k: float(v) for k, v in feats.items()}
