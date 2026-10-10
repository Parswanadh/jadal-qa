#!/usr/bin/env python3
"""LB1-decode, part 2: parity at the *working resolution the verdict path reads*.

Part 1 compares full-resolution yuv420p output. But the pipeline never scores
full-res frames: it decodes to a small working resolution first.

  jadal/poc_stream/replay.py  : -hwaccel cuda -c:v hevc_cuvid -ss S -t T -i clip
                                -vf scale=W:H,fps=F -f rawvideo -pix_fmt bgr24
                                (NVDEC decode + CPU swscale)
  eval/gpu/decode.py          : -hwaccel cuda -hwaccel_output_format cuda ...
                                -vf scale_cuda=W:H:format=nv12,hwdownload,
                                format=nv12 -pix_fmt nv12
                                (NVDEC decode + GPU scale, then numpy nv12->rgb)

So this script measures, for the same span:
  * CPU  scale=W:H -> bgr24        (the reference the frozen scorer was built on)
  * NVDEC(cuvid) + CPU scale -> bgr24   (replay.py path)
  * NVDEC + scale_cuda -> nv12 -> rgb   (eval/gpu/decode.py path)
and reports frame counts + MAE/PSNR against the CPU reference.

--merge-into <part1.json> adds the result under "pipeline_parity".
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
from decode_bench_a40 import INDEGO_480, EGO10K_1080, probe, run  # noqa: E402


def decode_pipe(clip, vf, pix_fmt, pre, ss, dur, svc=None):
    """Raw video goes through a pipe, so stdout must stay BYTES: the generic
    run() helper decodes stdout to str and would corrupt the frames."""
    cmd = (["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin"] + pre
           + ["-ss", f"{ss:.3f}", "-t", f"{dur:.3f}", "-i", clip,
              "-vf", vf, "-f", "rawvideo", "-pix_fmt", pix_fmt,
              "-vsync", "0", "pipe:1"])
    t0 = time.time()
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                       stdin=subprocess.DEVNULL, timeout=600)
    return {"cmd": cmd, "rc": p.returncode, "wall_s": time.time() - t0,
            "stdout": p.stdout,
            "stderr": p.stderr.decode("utf-8", "replace")}


def nv12_to_rgb(y, uv):
    u = uv[..., 0].repeat(2, axis=0).repeat(2, axis=1).astype(np.float64)
    v = uv[..., 1].repeat(2, axis=0).repeat(2, axis=1).astype(np.float64)
    c = y.astype(np.float64) - 16.0
    d, e = u - 128.0, v - 128.0
    r = 1.164 * c + 1.596 * e
    g = 1.164 * c - 0.391 * d - 0.813 * e
    b = 1.164 * c + 2.018 * d
    return np.stack([r, g, b], axis=-1).clip(0, 255) / 255.0


def metrics(a, b):
    """a, b: (N,H,W,3) uint8 arrays."""
    n = min(len(a), len(b))
    A = a[:n].astype(np.int16)
    B = b[:n].astype(np.int16)
    d = np.abs(A - B)
    mse = float((d.astype(np.float64) ** 2).mean())
    psnr = float("inf") if mse == 0 else 10.0 * np.log10(255.0 ** 2 / mse)
    # per-frame mean abs diff: how many frames cross a 1-LSB mean perturbation
    pf = d.reshape(n, -1).mean(axis=1)
    return {
        "frames_compared": int(n),
        "mae": round(float(d.mean()), 6),
        "max_abs_diff": int(d.max()),
        "psnr_db": (None if psnr == float("inf") else round(psnr, 3)),
        "psnr_infinite": psnr == float("inf"),
        "frac_frames_bitexact": round(float((pf == 0).mean()), 4),
        "max_per_frame_mae": round(float(pf.max()), 6),
        "pct_pixels_diff_gt2": round(float((d > 2).mean()) * 100, 6),
        "pct_pixels_diff_gt8": round(float((d > 8).mean()) * 100, 6),
    }


def case(clip, w, h, fps_int, ss, dur, tag):
    res = {"clip": clip, "work_res": f"{w}x{h}", "ss": ss, "dur_s": dur,
           "extract_fps": fps_int}
    vf_cpu = f"scale={w}:{h}:flags=bicubic,fps={fps_int}"
    vf_gpu = (f"scale_cuda={w}:{h}:format=nv12,hwdownload,format=nv12,"
              f"fps={fps_int}")

    cuvid = "hevc_cuvid" if _codec(clip) == "hevc" else "h264_cuvid"
    runs = {
        "cpu_scale_bgr24": (vf_cpu, "bgr24", []),
        "nvdec_cuvid_scale_bgr24": (vf_cpu, "bgr24",
                                    ["-hwaccel", "cuda", "-c:v", cuvid]),
        "nvdec_scale_cuda_nv12": (vf_gpu, "nv12",
                                  ["-hwaccel", "cuda",
                                   "-hwaccel_output_format", "cuda"]),
    }
    frames = {}
    for name, (vf, pf, pre) in runs.items():
        r = decode_pipe(clip, vf, pf, pre, ss, dur)
        npx = w * h * (3 if pf == "bgr24" else 3) // (1 if pf == "bgr24" else 2)
        n = int(len(r["stdout"]) // npx)
        res[name] = {"rc": r["rc"], "frames": n, "wall_s": round(r["wall_s"], 3),
                     "cmd": " ".join(r["cmd"])}
        if r["rc"] != 0:
            res[name]["stderr_tail"] = r["stderr"][-500:]
            frames[name] = None
            continue
        if pf == "bgr24":
            arr = np.frombuffer(r["stdout"][: int(n * npx)], np.uint8)
            frames[name] = arr.reshape(n, h, w, 3)
        else:
            raw = np.frombuffer(r["stdout"][: int(n * npx)], np.uint8)
            raw = raw.reshape(n, h * 3 // 2, w)
            rgb = np.empty((n, h, w, 3))
            for i in range(n):
                y = raw[i, :h, :]
                uv = raw[i, h:, :].reshape(h // 2, w // 2, 2)
                rgb[i] = nv12_to_rgb(y, uv)
            frames[name] = np.clip(np.round(rgb * 255.0), 0, 255).astype(np.uint8)

    ref = frames.get("cpu_scale_bgr24")
    if ref is None:
        res["verdict"] = "NOT COMPARABLE: CPU reference decode failed"
        return res
    for name in ("nvdec_cuvid_scale_bgr24", "nvdec_scale_cuda_nv12"):
        f = frames.get(name)
        if f is None:
            res[f"{name}_vs_cpu"] = "NOT COMPARABLE (decode failed)"
            continue
        same_n = (len(f) == len(ref))
        res[f"{name}_vs_cpu"] = metrics(ref, f)
        res[f"{name}_vs_cpu"]["frame_count_equal"] = bool(same_n)

    # reference scale: temporal delta of the CPU decode (signal the scorer sees)
    a = ref.astype(np.int16)
    td = np.abs(a[1:] - a[:-1]).mean() if len(a) > 1 else 0.0
    res["reference_frame_to_frame_mae_cpu"] = round(float(td), 6)
    return res


def _codec(clip):
    return "hevc" if "ego10k" in clip else "h264"


def sanitize(o, path="$"):
    """JSON-safe copy; numpy scalars are not json-serializable (np.bool_ is
    NOT a subclass of bool in numpy 2). Reports every conversion so a silent
    type confusion in a measurement cannot hide."""
    if isinstance(o, dict):
        return {k: sanitize(v, f"{path}.{k}") for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [sanitize(v, f"{path}[{i}]") for i, v in enumerate(o)]
    if isinstance(o, np.bool_):
        print(f"[sanitize] {path}: np.bool_ -> bool ({bool(o)})", file=sys.stderr)
        return bool(o)
    if isinstance(o, np.integer):
        print(f"[sanitize] {path}: {type(o).__name__} -> int", file=sys.stderr)
        return int(o)
    if isinstance(o, np.floating):
        print(f"[sanitize] {path}: {type(o).__name__} -> float", file=sys.stderr)
        return float(o)
    if isinstance(o, np.ndarray):
        print(f"[sanitize] {path}: ndarray -> list", file=sys.stderr)
        return o.tolist()
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--merge-into", default=None)
    ap.add_argument("--work-res", default="160x90")
    ap.add_argument("--dur", type=float, default=2.0)
    ap.add_argument("--ss", type=float, default=30.0)
    a = ap.parse_args()
    w, h = (int(x) for x in a.work_res.split("x"))

    out = {
        "schema": "jadal.pod_verify.decode_pipeline_parity/1",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "note": "parity at the pipeline working resolution; mirrors "
                "jadal/poc_stream/replay.py and eval/gpu/decode.py NVDEC_CHAIN",
        "work_res": f"{w}x{h}",
        "cases": {},
    }
    out["cases"]["indego_480"] = case(INDEGO_480, w, h, 10, a.ss, a.dur, "480")
    out["cases"]["ego10k_1080"] = case(EGO10K_1080, w, h, 30, a.ss, a.dur, "1080")
    out = sanitize(out)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2)[:4000])

    if a.merge_into and os.path.exists(a.merge_into):
        with open(a.merge_into) as f:
            main_json = json.load(f)
        main_json["pipeline_parity"] = out
        with open(a.merge_into, "w") as f:
            json.dump(main_json, f, indent=2)
        print(f"merged into {a.merge_into}")


if __name__ == "__main__":
    main()
