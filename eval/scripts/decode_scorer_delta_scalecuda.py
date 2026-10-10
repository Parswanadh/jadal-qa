#!/usr/bin/env python3
"""LB1-decode, part 6: verdict impact of the scale_cuda/nv12 decode path.

Part 3 showed the cuvid path (jadal/poc_stream/replay.py) is bit-exact with
the CPU decode and leaves the frozen pixel-view probability untouched.

Part 5 showed the OTHER in-repo NVDEC path (eval/gpu/decode.py: hwaccel +
scale_cuda + numpy nv12->rgb) differs from the CPU reference by MAE 5-10
(PSNR 21-29) at the working resolution -- larger than the frame-to-frame
MAE of the same span. This script measures what that does to the frozen
pixel-view probability, on the same six windows as part 3.
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
sys.path.insert(0, HERE)
REPO = os.path.normpath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, REPO)

from jadal.gpu.features import cpu_window_features  # noqa: E402
from jadal.ensemble.scorer import head_prob, load_config  # noqa: E402
from decode_pipeline_parity import nv12_to_rgb, sanitize  # noqa: E402
from decode_scorer_delta import INDEGO_480, EGO10K_1080, feats_for, W, H  # noqa: E402


def nv12_feats(clip, ss, dur, cfg):
    """eval/gpu/decode.py NVDEC_CHAIN: decode+scale on GPU, download nv12."""
    vf = (f"scale_cuda={W}:{H}:format=nv12,hwdownload,format=nv12,"
          f"fps={int(round(30 if 'ego10k' in clip else 10))}")
    cmd = (["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin",
            "-hwaccel", "cuda", "-hwaccel_output_format", "cuda",
            "-ss", f"{ss:.3f}", "-t", f"{dur:.3f}", "-i", clip,
            "-vf", vf, "-f", "rawvideo", "-pix_fmt", "nv12", "-vsync", "0",
            "pipe:1"])
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                       stdin=subprocess.DEVNULL, timeout=300)
    fs = W * H * 3 // 2
    n = len(p.stdout) // fs
    if p.returncode != 0 or n == 0:
        return {"ok": False, "rc": p.returncode,
                "err": p.stderr[-300:].decode("utf-8", "replace"),
                "cmd": " ".join(cmd)}
    raw = np.frombuffer(p.stdout[: n * fs], np.uint8).reshape(n, H * 3 // 2, W)
    rgb = np.empty((n, H, W, 3))
    for i in range(n):
        rgb[i] = nv12_to_rgb(raw[i, :H, :],
                             raw[i, H:, :].reshape(H // 2, W // 2, 2))
    luma = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    fps = min(n / float(dur), 30.0)
    f = cpu_window_features(luma, rgb, fps=fps)
    order = cfg["views"]["pixel"]["feature_names"]
    x = np.array([float(f[k]) for k in order], dtype=np.float64)
    return {"ok": True, "n_frames": n, "fps": float(fps),
            "x": x, "p_pixel": head_prob(x, cfg["views"]["pixel"]),
            "cmd": " ".join(cmd)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--merge-into", default=None)
    ap.add_argument("--dur", type=float, default=10.0)
    a = ap.parse_args()

    cfg, cfg_path = load_config()
    out = {"schema": "jadal.pod_verify.decode_scorer_delta_scalecuda/1",
           "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "config": cfg_path,
           "question": "does the eval/gpu/decode.py scale_cuda+nv12 path change "
                       "the frozen pixel-view probability?",
           "windows": {}}

    plan = [("ego10k_1080", EGO10K_1080, [30.0, 60.0, 90.0]),
            ("indego_480", INDEGO_480, [30.0, 90.0, 150.0])]
    for tag, clip, starts in plan:
        for ss in starts:
            key = f"{tag}@{ss:g}s"
            cpu = feats_for(clip, ss, a.dur, [], cfg)
            svc = nv12_feats(clip, ss, a.dur, cfg)
            rec = {"clip": clip, "ss": ss, "dur_s": a.dur,
                   "cpu": {k: v for k, v in cpu.items() if k != "x"},
                   "scale_cuda": {k: v for k, v in svc.items() if k != "x"}}
            if cpu.get("ok") and svc.get("ok"):
                d = np.abs(cpu["x"] - svc["x"])
                order = cfg["views"]["pixel"]["feature_names"]
                rec["feature_delta"] = {
                    "max_abs_delta": float(d.max()),
                    "max_abs_delta_feature": order[int(np.argmax(d))],
                    "n_features_changed_gt_1e-3": int((d > 1e-3).sum()),
                }
                rec["p_pixel_cpu"] = round(cpu["p_pixel"], 6)
                rec["p_pixel_scale_cuda"] = round(svc["p_pixel"], 6)
                rec["p_pixel_abs_delta"] = round(
                    abs(cpu["p_pixel"] - svc["p_pixel"]), 6)
                rec["verdict_flip_at_0.5"] = bool(
                    (cpu["p_pixel"] >= 0.5) != (svc["p_pixel"] >= 0.5))
                rec["frame_count_equal"] = bool(cpu["n_frames"] == svc["n_frames"])
            out["windows"][key] = rec
            print(f"{key}: p_cpu={rec.get('p_pixel_cpu')} "
                  f"p_svc={rec.get('p_pixel_scale_cuda')} "
                  f"dp={rec.get('p_pixel_abs_delta')} "
                  f"flip={rec.get('verdict_flip_at_0.5')}", flush=True)

    dps = [r["p_pixel_abs_delta"] for r in out["windows"].values()
           if "p_pixel_abs_delta" in r]
    out["summary"] = {
        "n_windows": len(out["windows"]),
        "max_p_pixel_abs_delta": max(dps) if dps else None,
        "mean_p_pixel_abs_delta": round(float(np.mean(dps)), 6) if dps else None,
        "n_verdict_flips_at_0.5": sum(
            bool(r.get("verdict_flip_at_0.5")) for r in out["windows"].values()),
        "all_frame_counts_equal": all(
            r.get("frame_count_equal", False) for r in out["windows"].values()),
    }
    out = sanitize(out)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=2)
    print("summary:", json.dumps(out["summary"]))
    if a.merge_into and os.path.exists(a.merge_into):
        with open(a.merge_into) as f:
            mj = json.load(f)
        mj["scorer_delta_scale_cuda_path"] = out
        with open(a.merge_into, "w") as f:
            json.dump(mj, f, indent=2)
        print(f"merged into {a.merge_into}")


if __name__ == "__main__":
    main()
