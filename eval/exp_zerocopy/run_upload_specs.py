#!/usr/bin/env python3
"""Z1.04 — upload-spec ladder (compressed-at-upload assumption).

The deployment may accept an upload-side encoding spec, so the codec /
resolution / quality is a free variable. For each spec we transcode the 30
sample windows' 10-s segments and measure the decode cost (NVDEC with and
without hwdownload, and CPU), then compose the per-window total from the
measured feature / EG2 costs (those stages are resolution-invariant: the
pipeline always works at 160x90 / 512x288).

Specs:
  (a) 1080p30 HEVC (as-is, lossless remux of the 10-s segment)
  (b) 1080p H.264 @CRF23
  (c) 480p H.264 @CRF23
  (d) 360p H.264 @CRF28   (X1: ~40% bytes, ΔAUROC < 0.02)
  (e) 240p H.264 @CRF28

Writes eval/results/exp_zerocopy/upload_specs.json.
Run: bash tools/pod/gpu_run.sh python3 eval/exp_zerocopy/run_upload_specs.py
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
sys.path.insert(0, os.path.join(REPO, "eval", "exp_regimes"))
sys.path.insert(0, os.path.join(REPO, "eval", "exp_rescore"))
sys.path.insert(0, HERE)

import run_e8 as e8  # noqa: E402
import zerocopy as zc  # noqa: E402
from jadal.gpu.features import cpu_window_features  # noqa: E402

OUTDIR = os.path.join(REPO, "eval", "results", "exp_zerocopy")
SCRATCH = os.environ.get("Z_SPEC_SCRATCH", "/tmp/opencode/upload_specs")
W, H = 160, 90
EGW, EGH = 512, 288
WINDOW_S = 10.0
DEVICE = "A40"

SPECS = [
    {"tag": "1080p30_hevc", "mode": "remux", "w": 1920, "h": 1080, "fps": 30,
     "codec": "hevc", "cuvid": "hevc_cuvid", "crf": None},
    {"tag": "1080p_h264_crf23", "mode": "x264", "w": 1920, "h": 1080, "fps": 30,
     "codec": "h264", "cuvid": "h264_cuvid", "crf": 23},
    {"tag": "480p_h264_crf23", "mode": "x264", "w": 854, "h": 480, "fps": 30,
     "codec": "h264", "cuvid": "h264_cuvid", "crf": 23},
    {"tag": "360p_h264_crf28", "mode": "x264", "w": 640, "h": 360, "fps": 30,
     "codec": "h264", "cuvid": "h264_cuvid", "crf": 28},
    {"tag": "240p_h264_crf28", "mode": "x264", "w": 426, "h": 240, "fps": 30,
     "codec": "h264", "cuvid": "h264_cuvid", "crf": 28},
]


def ms(t0):
    return (time.perf_counter() - t0) * 1000.0


def transcode(src, s0, spec, dst, encoder="nvenc"):
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        return
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    common = ["ffmpeg", "-y", "-hide_banner", "-v", "error", "-nostdin"]
    if spec["mode"] == "remux":
        cmd = common + ["-ss", f"{float(s0):.3f}", "-t", f"{WINDOW_S:.3f}",
                        "-i", src, "-an", "-c", "copy",
                        "-avoid_negative_ts", "make_zero", dst]
    elif encoder == "nvenc":
        # NVENC keeps the encode off the (heavily contended) CPU; -cq ~ CRF.
        cmd = common + ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda",
                        "-c:v", "hevc_cuvid",
                        "-ss", f"{float(s0):.3f}", "-t", f"{WINDOW_S:.3f}",
                        "-i", src, "-an",
                        "-vf", f"scale_cuda={spec['w']}:{spec['h']}:format=nv12",
                        "-c:v", "h264_nvenc", "-cq", str(spec["crf"]),
                        "-preset", "p4", "-rc", "vbr", dst]
    else:
        cmd = common + ["-ss", f"{float(s0):.3f}", "-t", f"{WINDOW_S:.3f}",
                        "-i", src, "-an",
                        "-vf", f"scale={spec['w']}:{spec['h']}",
                        "-c:v", "libx264", "-preset", "veryfast",
                        "-crf", str(spec["crf"]), "-r", str(spec["fps"]),
                        "-pix_fmt", "yuv420p", dst]
    p = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                       stdin=subprocess.DEVNULL, timeout=600)
    if p.returncode != 0:
        raise RuntimeError(f"transcode rc={p.returncode}: "
                           f"{p.stderr[-200:].decode('utf-8', 'replace')}")


def _time_cmd(cmd, timeout=300):
    t = time.perf_counter()
    p = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                       stdin=subprocess.DEVNULL, timeout=timeout)
    dt = ms(t)
    if p.returncode != 0:
        raise RuntimeError(p.stderr[-200:].decode("utf-8", "replace"))
    return dt


def _time_min(cmd, reps, timeout=300):
    """Min over ``reps`` runs — robust to pod/GPU contention."""
    best = None
    for _ in range(max(1, reps)):
        dt = _time_cmd(cmd, timeout=timeout)
        best = dt if best is None else min(best, dt)
    return best


def nvdec_cmd(path, spec, w=W, h=H, download=True, null=False):
    vf = f"scale_cuda={w}:{h}:format=nv12"
    if download:
        vf += ",hwdownload,format=nv12,format=rgb24"
    out = ["-f", "null", "-"] if null else ["-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]
    return ["ffmpeg", "-hide_banner", "-v", "error", "-nostdin",
            "-hwaccel", "cuda", "-hwaccel_output_format", "cuda",
            "-c:v", spec["cuvid"], "-ss", "0", "-t", f"{WINDOW_S:.3f}",
            "-i", path, "-vf", vf] + out


def cpu_cmd(path, w=W, h=H):
    return ["ffmpeg", "-hide_banner", "-v", "error", "-nostdin",
            "-ss", "0", "-t", f"{WINDOW_S:.3f}", "-i", path,
            "-vf", f"scale={w}:{h},format=rgb24",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]


def decode_frames(path, spec, w=W, h=H):
    """NVDEC decode to CPU uint8 (T,H,W,3) via scale_cuda+hwdownload."""
    p = subprocess.run(nvdec_cmd(path, spec, w, h, download=True),
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                       stdin=subprocess.DEVNULL, timeout=300)
    if p.returncode != 0:
        raise RuntimeError(p.stderr[-200:].decode("utf-8", "replace"))
    px = w * h * 3
    n = len(p.stdout) // px
    return np.frombuffer(p.stdout[:n * px], dtype=np.uint8).reshape(n, h, w, 3)


def stat(v):
    v = np.asarray(v, float)
    if not len(v):
        return None
    return {"n": int(len(v)), "mean_ms": round(float(v.mean()), 1),
            "p50_ms": round(float(np.percentile(v, 50)), 1),
            "p95_ms": round(float(np.percentile(v, 95)), 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-n", type=int, default=30)
    ap.add_argument("--cpu-n", type=int, default=8, help="windows for the CPU arm")
    ap.add_argument("--reps", type=int, default=3, help="NVDEC min-of-N reps")
    ap.add_argument("--encoder", choices=("nvenc", "x264"), default="nvenc",
                    help="upload transcode encoder (nvenc keeps it off the CPU)")
    ap.add_argument("--out-dir", default=OUTDIR)
    args = ap.parse_args()

    from run_zerocopy import build_sample
    e8.init_worker()
    POD_EG2 = ("/workspace/.cache/huggingface/hub/models--google--embeddinggemma-2"
               "/snapshots/914f7f89142e33e77833254d9c9b90c3cef7303b")
    if os.path.isdir(POD_EG2):
        e8._CFG["views"]["eg2"]["embed_model_dir"] = POD_EG2
    man, sample = build_sample(args.sample_n)
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(SCRATCH, exist_ok=True)

    # --- reference feature / EG2 costs (resolution-invariant; measured once) ---
    # reuse the main run's zerocopy.json baseline stages when present
    ref = None
    zc_path = os.path.join(args.out_dir, "zerocopy.json")
    if os.path.exists(zc_path):
        zj = json.load(open(zc_path))
        def mean_stage(which, s):
            return float(np.mean([r[which]["latency_ms"].get(s, 0.0)
                                  for r in zj["rows"]]))
        ref = {
            "feature_cpu_ms": mean_stage("base", "pixel_features"),
            "eg2_cpu_ms": mean_stage("base", "eg2_embed"),
            "feature_gpu_ms": mean_stage("gpu", "pixel_features"),
            "eg2_gpu_ms": mean_stage("gpu", "eg2_embed"),
            "n": zj["n"], "gpu_mode": zj["gpu_mode"],
        }
    print("reference stages:", ref, flush=True)

    results = {"device": DEVICE, "sample_n": len(sample), "cpu_n": args.cpu_n,
               "encoder": args.encoder, "reps": args.reps,
               "specs": [], "reference_stages": ref,
               "note": ("feature/EG2 costs are resolution-invariant (pipeline works "
                        "at 160x90 / 512x288); decode is the spec-dependent stage")}

    for spec in SPECS:
        tag = spec["tag"]
        rows = []
        cpu_used = 0
        for i, it in enumerate(sample):
            src = it["path"]
            dst = os.path.join(SCRATCH, tag, it["window_id"] + ".mp4")
            t_tc = time.perf_counter()
            try:
                transcode(src, float(it["start_s"]), spec, dst, encoder=args.encoder)
            except Exception as ex:  # noqa: BLE001
                rec = {"window_id": it["window_id"], "transcode_error": str(ex)[:160]}
                rows.append(rec)
                print(f"  {tag} {i+1}/{len(sample)} TRANSCODE ERROR {ex}", flush=True)
                continue
            tc_ms = ms(t_tc)
            seg = dst
            rec = {"window_id": it["window_id"], "transcode_ms": round(tc_ms, 1),
                   "size_bytes": os.path.getsize(seg)}
            try:
                rec["nvdec_hwdownload_ms"] = _time_min(nvdec_cmd(seg, spec, download=True), args.reps)
            except Exception as ex:  # noqa: BLE001
                rec["nvdec_hwdownload_error"] = str(ex)[:120]
            try:
                rec["nvdec_no_download_ms"] = _time_min(nvdec_cmd(seg, spec, download=False, null=True), args.reps)
            except Exception as ex:  # noqa: BLE001
                rec["nvdec_no_download_error"] = str(ex)[:120]
            if i < args.cpu_n:
                try:
                    rec["cpu_ms"] = _time_min(cpu_cmd(seg), 1)
                    cpu_used += 1
                except Exception as ex:  # noqa: BLE001
                    rec["cpu_error"] = str(ex)[:120]
            # decode-frame count / bitrate
            try:
                fr = decode_frames(seg, spec)
                rec["n_frames"] = int(fr.shape[0])
                rec["bytes_per_frame"] = round(rec["size_bytes"] / max(fr.shape[0], 1), 1)
            except Exception as ex:  # noqa: BLE001
                rec["frames_error"] = str(ex)[:120]
            rec["bitrate_kbps"] = round(rec["size_bytes"] * 8 / WINDOW_S / 1000.0, 1)
            rows.append(rec)
            print(f"  {tag} {i+1}/{len(sample)} dl={rec.get('nvdec_hwdownload_ms'):.0f} "
                  f"nodl={rec.get('nvdec_no_download_ms'):.0f} "
                  f"cpu={rec.get('cpu_ms')} size={rec['size_bytes']}", flush=True)

        dl = stat([r["nvdec_hwdownload_ms"] for r in rows if "nvdec_hwdownload_ms" in r])
        nd = stat([r["nvdec_no_download_ms"] for r in rows if "nvdec_no_download_ms" in r])
        cp = stat([r["cpu_ms"] for r in rows if "cpu_ms" in r])
        size = stat([r["size_bytes"] for r in rows])
        bpf = stat([r["bytes_per_frame"] for r in rows if "bytes_per_frame" in r])
        entry = {
            "tag": tag, "w": spec["w"], "h": spec["h"], "fps": spec["fps"],
            "codec": spec["codec"], "crf": spec["crf"],
            "n": len(rows),
            "size_bytes": size, "bytes_per_frame": bpf,
            "bitrate_kbps": stat([r["bitrate_kbps"] for r in rows]),
            "nvdec_hwdownload": dl, "nvdec_no_download": nd, "cpu": cp,
            "n_cpu": cpu_used,
            "cpu_over_nvdec_p50": round(cp["p50_ms"] / max(dl["p50_ms"], 1e-9), 3) if cp and dl else None,
            "hwdownload_cost_p50_ms": round(dl["p50_ms"] - nd["p50_ms"], 1) if dl and nd else None,
            "rows": rows,
        }
        if ref:
            dec = dl["mean_ms"] if dl else None
            dec_nodl = nd["mean_ms"] if nd else None
            if dec is not None:
                entry["total_base_ms"] = round(dec + ref["feature_cpu_ms"] + ref["eg2_cpu_ms"], 1)
                entry["total_gpu_ms"] = round(dec + ref["feature_gpu_ms"] + ref["eg2_gpu_ms"], 1)
                entry["footage_h_per_wall_h_base"] = round(10000.0 / entry["total_base_ms"], 3)
                entry["footage_h_per_wall_h_gpu"] = round(10000.0 / entry["total_gpu_ms"], 3)
            if dec_nodl is not None:
                entry["total_zerocopy_bound_ms"] = round(
                    dec_nodl + ref["feature_gpu_ms"] + ref["eg2_gpu_ms"], 1)
                entry["footage_h_per_wall_h_zerocopy_bound"] = round(
                    10000.0 / entry["total_zerocopy_bound_ms"], 3)
        results["specs"].append(entry)
        json.dump(results, open(os.path.join(args.out_dir, "upload_specs.json"), "w"),
                  indent=1)
        print(f"spec {tag}: dl_p50={dl['p50_ms'] if dl else None} "
              f"cpu_p50={cp['p50_ms'] if cp else None} "
              f"bpf={bpf['mean_ms'] if bpf else None} "
              f"total_base={entry.get('total_base_ms')} "
              f"fh/wh_base={entry.get('footage_h_per_wall_h_base')}", flush=True)
    print("wrote upload_specs.json", flush=True)


if __name__ == "__main__":
    main()
