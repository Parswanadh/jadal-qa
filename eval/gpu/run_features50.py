#!/usr/bin/env python3
"""F50 — extract the 24 NEW per-pixel features on gold + OOD160 + hold-out108.

Reuses the frozen 26-feature extractor (`jadal.gpu.features.cpu_window_features`)
for validation only; the 26 columns themselves are taken from the frozen caches.
Computes the 24 new columns (`jadal.gpu.features50.cpu_new_features`).

Decode protocol (matches the frozen 26-feature caches):
  * gold    : manifest_gold.json, 160x90 rgb24, native fps (IndEgo mp4_480 = 10 fps)
  * OOD160  : ood160_labels.json, 160x90 rgb24, stepped to 10 fps-effective
  * holdout : windows108.json q3 transcodes, 160x90 rgb24, stepped to 10 fps
ct_fsize_std uses one ffprobe `-show_packets` over the same window.

Run:  OMP_NUM_THREADS=2 python3 eval/gpu/run_features50.py [--jobs 6]
Writes eval/results/exp_feat50/features50.npz  {gold_*, ood_*, hold_*}
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)
from eval.gpu.decode import decode_window  # noqa: E402
from jadal.gpu.features import FEATURE_NAMES, cpu_window_features  # noqa: E402
from jadal.gpu.features50 import NEW_FEATURE_NAMES, cpu_new_features  # noqa: E402

BUNDLE = os.environ.get("JADAL_BUNDLE", "/workspace/jadal_pod_bundle")
OUT = os.path.join(REPO, "eval", "results", "exp_feat50")
GOLD = os.path.join(BUNDLE, "labels", "manifest_gold.json")
OOD = os.path.join(BUNDLE, "labels", "ood160_labels.json")
HOLD = os.environ.get("JADAL_HOLD108",
                      "/workspace/wt/m-confirm/eval/results/exp_confirm/inputs/windows108.json")
Q3 = os.environ.get("JADAL_Q3_CLIPS", "/workspace/scratch/c1/q3_clips")
W, H = 160, 90
_VALIDATE26 = False


def _remap_gold(path):
    """manifest paths point at the laptop tree; remap onto the pod bundle."""
    marker = "data/indego/"
    i = path.find(marker)
    return os.path.join(BUNDLE, "data", "indego", path[i + len(marker):]) if i >= 0 else path


def fsize_std(path, start, dur):
    """std/mean of per-frame compressed packet sizes over [start,start+dur)."""
    if start > 0:
        args = ["-read_intervals", f"{start}%+{dur}"]
    else:
        args = []
    p = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0"] + args +
                       ["-show_entries", "packet=size", "-of", "csv=p=0", path],
                       capture_output=True, text=True, timeout=120)
    sz = np.asarray([float(x) for x in p.stdout.split() if x.strip()], float)
    if len(sz) < 2 or sz.mean() <= 0:
        return None
    return float(sz.std() / sz.mean())


def decode_10(path, start, dur):
    rgb, fps = decode_window(path, start, dur, W, H, "rgb24")
    luma = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    step = max(1, int(round(fps / 10.0)))
    if step > 1:
        rgb, luma = rgb[::step], luma[::step]
    return luma, rgb


def _one(job):
    wid, path, start, dur, full_fps = job
    try:
        luma, rgb = decode_10(path, start, dur)
        if len(luma) < 3:
            return wid, {"error": "too few frames"}
        fps = float(full_fps if full_fps is not None else 10.0)
        fnew = cpu_new_features(luma, rgb, fps=fps,
                                fsize_std=fsize_std(path, start, dur))
        out = {"fnew": {k: float(fnew[k]) for k in NEW_FEATURE_NAMES},
               "T": int(len(luma))}
        if _VALIDATE26:
            f26 = cpu_window_features(luma, rgb, fps=fps)
            out["f26"] = {k: float(f26[k]) for k in FEATURE_NAMES}
        return wid, out
    except Exception as e:  # noqa: BLE001
        return wid, {"error": f"{e}"[:200]}


def build_jobs():
    jobs, meta = [], []
    # gold
    gold = json.load(open(GOLD))["items"]
    for it in gold:
        dur = round(it["end_s"] - it["start_s"], 2)
        jobs.append((it["id"], _remap_gold(it["path"]), it["start_s"], dur, 10.0))
        meta.append(("gold", it["id"], it["clip"], int(it["label"])))
    # ood160
    ood = json.load(open(OOD))["windows"]
    for x in ood:
        p = os.path.join(BUNDLE, x["video"])
        jobs.append((x["segment_id"], p, 0.0, 10.0, 10.0))
        meta.append(("ood", x["segment_id"], x["factory"],
                     1 if x["verdict"] == "good" else 0))
    # hold-out 108 (q3 transcodes)
    hold = json.load(open(HOLD))
    for x in hold:
        p = os.path.join(Q3, x["id"] + ".mp4")
        jobs.append((x["id"], p, 0.0, 10.0, 10.0))
        meta.append(("hold", x["id"], x["factory"],
                     1 if x["verdict"] == "good" else 0))
    return jobs, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--validate26", action="store_true",
                    help="also recompute the frozen 26 (slow, for a sanity check)")
    args = ap.parse_args()
    global _VALIDATE26
    _VALIDATE26 = args.validate26
    jobs, meta = build_jobs()
    print(f"windows: {len(jobs)} (gold={sum(1 for m in meta if m[0]=='gold')} "
          f"ood={sum(1 for m in meta if m[0]=='ood')} "
          f"hold={sum(1 for m in meta if m[0]=='hold')})", flush=True)
    t0 = time.perf_counter()
    with mp.Pool(min(args.jobs, len(jobs))) as pool:
        rows = dict(pool.map(_one, jobs))
    wall = time.perf_counter() - t0
    errs = [m[1] for m in meta if "error" in rows[m[1]]]
    print(f"decoded in {wall:.1f}s; errors={len(errs)} {errs[:5]}", flush=True)

    os.makedirs(OUT, exist_ok=True)
    payload = {}
    for split in ("gold", "ood", "hold"):
        sub = [m for m in meta if m[0] == split and "error" not in rows[m[1]]]
        payload[f"{split}_ids"] = np.asarray([m[1] for m in sub])
        payload[f"{split}_groups"] = np.asarray([m[2] for m in sub])
        payload[f"{split}_y"] = np.asarray([m[3] for m in sub], int)
        if _VALIDATE26:
            payload[f"{split}_f26"] = np.asarray(
                [[rows[m[1]]["f26"][k] for k in FEATURE_NAMES] for m in sub], float)
        payload[f"{split}_fnew"] = np.asarray(
            [[rows[m[1]]["fnew"][k] for k in NEW_FEATURE_NAMES] for m in sub], float)
        payload[f"{split}_T"] = np.asarray([rows[m[1]]["T"] for m in sub], int)
    payload["names26"] = np.asarray(FEATURE_NAMES)
    payload["names_new"] = np.asarray(NEW_FEATURE_NAMES)
    payload["wall_s"] = wall
    payload["n_jobs"] = args.jobs
    np.savez_compressed(os.path.join(OUT, "features50.npz"), **payload)
    print(f"wrote {os.path.join(OUT, 'features50.npz')}", flush=True)


if __name__ == "__main__":
    main()
