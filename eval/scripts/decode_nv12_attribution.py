#!/usr/bin/env python3
"""LB1-decode, part 5: attribute the difference in the nv12/scale_cuda path.

Part 2 found:
  NVDEC (cuvid) + CPU swscale  ==  CPU decode + CPU swscale   (bit-exact)
  NVDEC + scale_cuda + numpy nv12->rgb  !=  CPU reference      (MAE 16-22)

That gap can come from three different places, and they have very different
consequences, so measure them separately on the same 2 s span:

  c1  cpu bilinear vs cpu bicubic       -> resampler choice
  c2  cpu nv12(numpy rgb) vs bgr24      -> the hand-rolled nv12->rgb conversion
                                           (eval/gpu/decode.py hardcodes BT.601
                                           limited range; swscale uses the
                                           stream's matrix / resolution default)
  c3  cuvid+bilinear vs cpu+bilinear    -> decoder, resampler matched (expect 0)
  c4  scale_cuda+numpy vs cpu bilinear  -> residual GPU-resize effect
      +numpy (same conversion on both sides)
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from decode_bench_a40 import INDEGO_480, EGO10K_1080  # noqa: E402
from decode_pipeline_parity import nv12_to_rgb, sanitize  # noqa: E402

W, H = 160, 90


def dec(clip, vf, pix_fmt, pre, ss, dur):
    cmd = (["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin"] + pre
           + ["-ss", f"{ss:.3f}", "-t", f"{dur:.3f}", "-i", clip,
              "-vf", vf, "-f", "rawvideo", "-pix_fmt", pix_fmt,
              "-vsync", "0", "pipe:1"])
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                       stdin=subprocess.DEVNULL, timeout=600)
    npx = (W * H * 3) if pix_fmt == "bgr24" else (W * H * 3 // 2)
    n = len(p.stdout) // npx
    out = {"rc": p.returncode, "frames": n, "cmd": " ".join(cmd)}
    if p.returncode != 0:
        out["stderr_tail"] = p.stderr[-400:].decode("utf-8", "replace")
        return None, out
    if pix_fmt == "bgr24":
        arr = np.frombuffer(p.stdout[: n * npx], np.uint8).reshape(n, H, W, 3)
    else:
        raw = np.frombuffer(p.stdout[: n * npx], np.uint8).reshape(n, H * 3 // 2, W)
        rgb = np.empty((n, H, W, 3))
        for i in range(n):
            rgb[i] = nv12_to_rgb(raw[i, :H, :],
                                 raw[i, H:, :].reshape(H // 2, W // 2, 2))
        arr = np.clip(np.round(rgb * 255.0), 0, 255).astype(np.uint8)
    return arr, out


def m(a, b):
    n = min(len(a), len(b))
    d = np.abs(a[:n].astype(np.int16) - b[:n].astype(np.int16))
    mse = float((d.astype(np.float64) ** 2).mean())
    return {"frames_compared": int(n), "mae": round(float(d.mean()), 5),
            "max_abs_diff": int(d.max()),
            "psnr_db": (None if mse == 0 else round(float(10 * np.log10(255.0 ** 2 / mse)), 2)),
            "pct_gt8": round(float((d > 8).mean()) * 100, 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--merge-into", default=None)
    ap.add_argument("--ss", type=float, default=30.0)
    ap.add_argument("--dur", type=float, default=2.0)
    a = ap.parse_args()

    res = {"schema": "jadal.pod_verify.decode_nv12_attribution/1",
           "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "work_res": f"{W}x{H}", "ss": a.ss, "dur_s": a.dur,
           "clip_color_metadata": {}, "cases": {}}

    for tag, clip, fps in (("indego_480", INDEGO_480, 10),
                           ("ego10k_1080", EGO10K_1080, 30)):
        pr = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
             "stream=color_space,color_primaries,color_transfer,color_range",
             "-of", "json", clip], stdout=subprocess.PIPE)
        res["clip_color_metadata"][tag] = json.loads(pr.stdout)["streams"][0]

        cuvid = "h264_cuvid" if "indego" in clip else "hevc_cuvid"
        runs = {
            "cpu_bicubic_bgr24": (f"scale={W}:{H}:flags=bicubic,fps={fps},format=bgr24",
                                  "bgr24", []),
            "cpu_bilinear_bgr24": (f"scale={W}:{H}:flags=bilinear,fps={fps},format=bgr24",
                                   "bgr24", []),
            "cpu_bicubic_nv12": (f"scale={W}:{H}:flags=bicubic,fps={fps},format=nv12",
                                 "nv12", []),
            "cpu_bilinear_nv12": (f"scale={W}:{H}:flags=bilinear,fps={fps},format=nv12",
                                  "nv12", []),
            "cuvid_bilinear_bgr24": (f"scale={W}:{H}:flags=bilinear,fps={fps},format=bgr24",
                                     "bgr24", ["-hwaccel", "cuda", "-c:v", cuvid]),
            "cuda_scale_nv12": (f"scale_cuda={W}:{H}:format=nv12,hwdownload,format=nv12,fps={fps}",
                                "nv12", ["-hwaccel", "cuda",
                                         "-hwaccel_output_format", "cuda"]),
        }
        frames, meta = {}, {}
        for name, (vf, pf, pre) in runs.items():
            arr, info = dec(clip, vf, pf, pre, a.ss, a.dur)
            frames[name] = arr
            meta[name] = info
            print(f"{tag}/{name}: rc={info['rc']} frames={info['frames']}", flush=True)

        # `-pix_fmt bgr24` bytes are B,G,R; nv12_to_rgb returns R,G,B. Flip the
        # CPU reference before any cross-conversion comparison (c1/c3 compare
        # like with like and are unaffected either way).
        bgr_to_rgb = {k: (v[..., ::-1] if v is not None else None)
                      for k, v in frames.items() if k.endswith("bgr24")}
        frames_rgb = dict(frames)
        frames_rgb.update(bgr_to_rgb)

        c = {"decodes": meta,
             "channel_order_note": "bgr24 outputs flipped to RGB for "
                                   "comparisons against nv12_to_rgb output"}
        c["c1_resampler_bicubic_vs_bilinear"] = m(frames["cpu_bicubic_bgr24"],
                                                  frames["cpu_bilinear_bgr24"])
        c["c2_conversion_nv12numpy_vs_swscale_bgr24_same_scale"] = m(
            frames_rgb["cpu_bicubic_nv12"], frames_rgb["cpu_bicubic_bgr24"])
        c["c3_decoder_cuvid_vs_cpu_matched_bilinear"] = m(
            frames["cuvid_bilinear_bgr24"], frames["cpu_bilinear_bgr24"])
        c["c4_scale_cuda_vs_cpu_bilinear_same_conversion"] = m(
            frames["cuda_scale_nv12"], frames["cpu_bilinear_nv12"])
        c["c5_scale_cuda_vs_cpu_bicubic_same_conversion"] = m(
            frames["cuda_scale_nv12"], frames["cpu_bicubic_nv12"])
        c["c6_scale_cuda_vs_cpu_bilinear_bgr24_endtoend"] = m(
            frames_rgb["cuda_scale_nv12"], frames_rgb["cpu_bilinear_bgr24"])
        res["cases"][tag] = c

    res = sanitize(res)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"wrote {a.out}")
    if a.merge_into and os.path.exists(a.merge_into):
        with open(a.merge_into) as f:
            mj = json.load(f)
        mj["nv12_path_attribution"] = res
        with open(a.merge_into, "w") as f:
            json.dump(mj, f, indent=2)
        print(f"merged into {a.merge_into}")


if __name__ == "__main__":
    main()
