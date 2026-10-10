#!/usr/bin/env python3
"""LB1-decode, part 3: does the decoder swap change a VERDICT?

Parts 1-2 measure pixel differences. This part answers the question that
actually matters: feed the SAME window through the frozen ensemble's
decode-dependent view (pixel: ffmpeg CPU 160x90 rgb24 -> cpu_window_features
-> logistic head) once from a CPU decode and once from an NVDEC decode, and
report the change in the pixel-view probability.

The eg2 view is embedding-model based and is not exercised here (it samples
frames through cv2, a different decode path); only the pixel view depends on
the ffmpeg decode path.

Usage:
  OMP_NUM_THREADS=2 flock -w 3600 /workspace/state/gpu.slot2.lock \
      python3 eval/scripts/decode_scorer_delta.py --out <json> [--merge-into <json>]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, REPO)

from jadal.gpu.features import cpu_window_features  # noqa: E402
from jadal.ensemble.scorer import head_prob, load_config  # noqa: E402

INDEGO_480 = ("/workspace/jadal_pod_bundle/data/indego/1_Assembly/mp4_480/"
              "User_15_410_0411_2_480.mp4")
EGO10K_1080 = ("/workspace/jadal_pod_bundle/data/ego10k_scale/clips/factory_051/"
               "worker_005/factory051_worker005_00167.mp4")

W, H = 160, 90


def decode_window(clip, ss, dur, pre):
    """Mirror jadal/ensemble/scorer.py:decode_pixel_window, with an optional
    NVDEC pre-chain (replay.py's exact pre-chain)."""
    cmd = (["ffmpeg", "-hide_banner", "-v", "error", "-nostdin"] + pre
           + ["-ss", str(float(ss)), "-t", str(dur), "-i", clip,
              "-vf", f"scale={W}:{H},format=rgb24",
              "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"])
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                       stdin=subprocess.DEVNULL, timeout=300)
    if p.returncode != 0:
        return None, None, p.stderr[-400:].decode("utf-8", "replace"), cmd
    px = W * H * 3
    n = len(p.stdout) // px
    if n == 0:
        return None, None, "zero frames", cmd
    rgb = (np.frombuffer(p.stdout[: n * px], dtype=np.uint8)
           .astype(np.float64) / 255.0).reshape(n, H, W, 3)
    luma = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    return luma, rgb, None, cmd


def feats_for(clip, ss, dur, pre, cfg):
    luma, rgb, err, cmd = decode_window(clip, ss, dur, pre)
    if err:
        return {"ok": False, "err": err, "cmd": " ".join(cmd)}
    fps = min(len(luma) / float(dur), 30.0)
    f = cpu_window_features(luma, rgb, fps=fps)
    order = cfg["views"]["pixel"]["feature_names"]
    x = np.array([float(f[k]) for k in order], dtype=np.float64)
    return {"ok": True, "n_frames": int(len(luma)), "fps": float(fps),
            "feats": {k: float(v) for k, v in f.items()}, "x": x,
            "p_pixel": head_prob(x, cfg["views"]["pixel"]),
            "cmd": " ".join(cmd)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--merge-into", default=None)
    ap.add_argument("--dur", type=float, default=10.0)
    a = ap.parse_args()

    cfg, cfg_path = load_config()

    plan = [
        ("ego10k_1080", EGO10K_1080, [30.0, 60.0, 90.0],
         ["-hwaccel", "cuda", "-c:v", "hevc_cuvid"]),
        ("indego_480", INDEGO_480, [30.0, 90.0, 150.0],
         ["-hwaccel", "cuda", "-c:v", "h264_cuvid"]),
    ]

    out = {
        "schema": "jadal.pod_verify.decode_scorer_delta/1",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config": cfg_path,
        "question": "does swapping CPU decode -> NVDEC change the frozen "
                    "pixel-view probability for the same window?",
        "work_res": f"{W}x{H}", "window_s": a.dur,
        "windows": {},
    }

    for tag, clip, starts, pre in plan:
        for ss in starts:
            key = f"{tag}@{ss:g}s"
            cpu = feats_for(clip, ss, a.dur, [], cfg)
            nv = feats_for(clip, ss, a.dur, pre, cfg)
            rec = {"clip": clip, "ss": ss, "dur_s": a.dur,
                   "cpu": {k: v for k, v in cpu.items() if k != "x"},
                   "nvdec": {k: v for k, v in nv.items() if k != "x"}}
            if cpu.get("ok") and nv.get("ok"):
                order = cfg["views"]["pixel"]["feature_names"]
                d = np.abs(cpu["x"] - nv["x"])
                worst = order[int(np.argmax(d))]
                scale = np.maximum(np.abs(cpu["x"]), 1e-9)
                rec["feature_delta"] = {
                    "n_features": len(order),
                    "max_abs_delta": float(d.max()),
                    "max_abs_delta_feature": worst,
                    "max_rel_delta": float((d / scale).max()),
                    "n_features_changed_gt_1e-3": int((d > 1e-3).sum()),
                }
                rec["p_pixel_cpu"] = round(cpu["p_pixel"], 6)
                rec["p_pixel_nvdec"] = round(nv["p_pixel"], 6)
                rec["p_pixel_abs_delta"] = round(abs(cpu["p_pixel"] - nv["p_pixel"]), 6)
                rec["cpu_margin_to_0.5"] = round(abs(cpu["p_pixel"] - 0.5), 6)
                rec["verdict_flip_at_0.5"] = bool(
                    (cpu["p_pixel"] >= 0.5) != (nv["p_pixel"] >= 0.5))
                rec["frame_count_equal"] = bool(cpu["n_frames"] == nv["n_frames"])
            out["windows"][key] = rec
            print(f"{key}: p_cpu={rec.get('p_pixel_cpu')} "
                  f"p_nv={rec.get('p_pixel_nvdec')} "
                  f"dp={rec.get('p_pixel_abs_delta')} "
                  f"flip={rec.get('verdict_flip_at_0.5')}", flush=True)

    dps = [r["p_pixel_abs_delta"] for r in out["windows"].values()
           if "p_pixel_abs_delta" in r]
    out["summary"] = {
        "n_windows": len(out["windows"]),
        "max_p_pixel_abs_delta": max(dps) if dps else None,
        "n_verdict_flips_at_0.5": sum(
            bool(r.get("verdict_flip_at_0.5")) for r in out["windows"].values()),
        "all_frame_counts_equal": all(
            r.get("frame_count_equal", False) for r in out["windows"].values()),
    }
    with open(a.out, "w") as f:
        json.dump(out, f, indent=2)
    print("summary:", json.dumps(out["summary"]))
    print(f"wrote {a.out}")

    if a.merge_into and os.path.exists(a.merge_into):
        with open(a.merge_into) as f:
            main_json = json.load(f)
        main_json["scorer_delta"] = out
        with open(a.merge_into, "w") as f:
            json.dump(main_json, f, indent=2)
        print(f"merged into {a.merge_into}")


if __name__ == "__main__":
    main()
