#!/usr/bin/env python3
"""Confirmatory one-shot prep: features on the 108 round-3 hold-out windows.

Reuses the FROZEN extractor code verbatim:
  * Z3 candidates  : m/adapt160 eval/exp_adapt160/z3extract.py (read-only import)
  * pixel-26       : jadal.gpu.features.cpu_window_features (repo, frozen)
  * transcode      : the queue-2 recipe (480p, 10 fps, libx264 veryfast crf 26,
                     no audio, 10 s from source_start_s). Verified to reproduce
                     the frozen q2 features to ~4 decimals on r000.

NO fitting, NO selection, NO threshold changes. Extraction only.

Run: OMP_NUM_THREADS=2 python3 eval/exp_confirm/prep_confirm.py
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)
Z3_DIR = os.environ.get("JADAL_Z3_DIR", "/workspace/wt/m-adapt160/eval/exp_adapt160")
sys.path.insert(0, Z3_DIR)
import z3extract as E  # noqa: E402  (frozen Z3 extractor)

INPUTS = os.path.join(REPO, "eval", "results", "exp_confirm", "inputs")
WINDOWS = os.path.join(INPUTS, "windows108.json")
CLIPS = os.environ.get("JADAL_CONFIRM_CLIPS", "/workspace/scratch/c1/q3_clips")
STABLE8 = ["radial_slope", "hf_ratio", "dct_energy", "luma_p05", "global_shift",
           "illum_r", "illum_b", "zncc_self"]


def transcode(it):
    """Queue-2 transcode recipe (verified exact vs the frozen q2 clips)."""
    os.makedirs(CLIPS, exist_ok=True)
    out = os.path.join(CLIPS, it["id"] + ".mp4")
    if os.path.exists(out) and os.path.getsize(out) > 0:
        return out
    st = float(it["start_s"])
    pre = max(0.0, st - 5.0)
    skip = st - pre
    r = subprocess.run(
        ["ffmpeg", "-hide_banner", "-v", "error", "-y", "-ss", str(pre),
         "-i", it["path"], "-ss", str(skip), "-t", "10",
         "-vf", "scale=854:480,fps=10", "-c:v", "libx264", "-preset", "veryfast",
         "-crf", "26", "-an", out],
        capture_output=True, text=True, timeout=900)
    if r.returncode != 0:
        raise RuntimeError(f"transcode rc={r.returncode}: {r.stderr[-300:]}")
    return out


def _one(it):
    res = {"id": it["id"], "error": None}
    try:
        clip = transcode(it)
        # compression telemetry (probe the 10 s transcode)
        tbl = E.probe_full(clip)
        ct = E.compression_from_table(tbl, 0.0, 10.0)
        if ct is None:
            for k in E.COMPRESSION:
                res[k] = np.nan
        else:
            res.update(ct)
        # decode-based Z3 candidates
        d = E.decode_features(clip, 0.0, 10.0)
        if d is None:
            raise RuntimeError("z3 too few frames")
        res.update(d)
        # pixel-26 (frozen recipe: 160x90 rgb24, 10 fps effective)
        from jadal.ensemble.scorer import decode_pixel_window
        from jadal.gpu.features import cpu_window_features
        luma, rgb, fps = decode_pixel_window(clip, 0.0, 10.0, 160, 90)
        if len(luma) < 3:
            raise RuntimeError("pixel too few frames")
        step = max(1, int(round(fps / 10.0)))
        luma, rgb = luma[::step], rgb[::step]
        f = cpu_window_features(luma, rgb, fps=10.0)
        for k in f:
            res["px_" + k] = float(f[k])
        res["_T"] = int(len(luma))
        res["_fps"] = float(fps)
    except Exception as e:
        res["error"] = f"prep:{e}"[:300]
    for k in E.CAND:
        res.setdefault(k, np.nan)
    return res


def main():
    items = json.load(open(WINDOWS))
    with mp.Pool(2) as pool:
        rows = pool.map(_one, items, chunksize=2)
    # align back to input order
    by = {r["id"]: r for r in rows}
    rows = [by[it["id"]] for it in items]
    errs = [r["id"] for r in rows if r.get("error")]
    z3_names = E.CAND
    Xz3 = np.asarray([[float(r.get(n, np.nan)) for n in z3_names] for r in rows], float)
    pix_names = ["temp_b0", "temp_b1", "temp_b2", "temp_b3", "frozen_frac",
                 "flicker_idx", "radial_slope", "hf_ratio", "dct_energy",
                 "pca_resid", "pca_top1_share", "luma_mean", "luma_p05", "luma_p50",
                 "luma_p95", "dark_frac", "bright_frac", "motion_energy", "global_shift",
                 "skin_frac", "illum_r", "illum_g", "illum_b", "rppg_pulse",
                 "dup_hash_mean", "zncc_self"]
    Xpx = np.asarray([[float(r.get("px_" + n, np.nan)) for n in pix_names] for r in rows], float)
    out = {
        "ids": np.asarray([it["id"] for it in items]),
        "y": np.asarray([1 if it["verdict"] == "good" else 0 for it in items], int),
        "factory": np.asarray([it["factory"] for it in items]),
        "reason": np.asarray([it.get("reason", "") for it in items]),
        "z3_names": np.asarray(z3_names),
        "Xz3": Xz3,
        "pix_names": np.asarray(pix_names),
        "Xpx": Xpx,
        "T": np.asarray([int(r.get("_T", 0)) for r in rows]),
        "native_fps": np.asarray([float(r.get("_fps", 0.0)) for r in rows]),
        "errors": np.asarray(errs),
    }
    os.makedirs(INPUTS, exist_ok=True)
    path = os.path.join(INPUTS, "confirm108_features.npz")
    np.savez_compressed(path, **out)
    ok = ~np.isnan(Xz3).any(1) & ~np.isnan(Xpx).any(1)
    print(f"wrote {path}: complete {int(ok.sum())}/{len(items)}; errors={errs[:5]}")
    print(f"good={int(out['y'].sum())} bad={int((1-out['y']).sum())}")


if __name__ == "__main__":
    main()
