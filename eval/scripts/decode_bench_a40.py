#!/usr/bin/env python3
"""LB1-decode: verify NVDEC-vs-CPU decode claims on this pod's A40.

Two questions:
  1. CORRECTNESS: does NVDEC decode the same pixels as the CPU decoder?
  2. THROUGHPUT: what are the decode-only fps / RTF, and does NVDEC saturate
     flat across 1-4 concurrent sessions (the FACTS.md claim)?

Decode-only throughout: raw frames are only written for the (short) parity
spans; all throughput runs mux to `-f null` so no disk I/O is measured.

Usage (must be run under the pod GPU slot lock):
  OMP_NUM_THREADS=2 flock -w 3600 /workspace/state/gpu.slot2.lock \
      python3 eval/scripts/decode_bench_a40.py --out <json> [--skip-cpu-long]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np

INDEGO_480 = (
    "/workspace/jadal_pod_bundle/data/indego/1_Assembly/mp4_480/"
    "User_15_410_0411_2_480.mp4"
)
EGO10K_1080 = (
    "/workspace/jadal_pod_bundle/data/ego10k_scale/clips/factory_051/worker_005/"
    "factory051_worker005_00167.mp4"
)

BENCH_RE = re.compile(
    r"bench: utime=(?P<utime>[0-9.]+)s stime=(?P<stime>[0-9.]+)s "
    r"rtime=(?P<rtime>[0-9.]+)s"
)


def run(cmd, timeout=1800):
    t0 = time.time()
    p = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout
    )
    return {
        "cmd": cmd,
        "rc": p.returncode,
        "wall_s": time.time() - t0,
        "stdout": p.stdout.decode("utf-8", "replace"),
        "stderr": p.stderr.decode("utf-8", "replace"),
    }


def probe(path):
    p = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries",
            "stream=codec_name,width,height,r_frame_rate,avg_frame_rate,"
            "nb_frames,duration,pix_fmt,bit_rate",
            "-of", "json", path,
        ],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    st = json.loads(p.stdout)["streams"][0]
    num, den = st["r_frame_rate"].split("/")
    st["fps"] = float(num) / float(den)
    st["frames"] = int(st.get("nb_frames") or 0)
    st["path"] = path
    return st


# ---------------------------------------------------------------- parity ----

def decode_raw(clip, out_path, frames, input_opts, ss=None, extra=None):
    """Decode `frames` frames to raw yuv420p (system memory).

    `input_opts` go BEFORE -i: decoders (-c:v hevc_cuvid) and -hwaccel are
    input options in ffmpeg, while an output-side -c:v would select an
    *encoder* (this is exactly the trap that makes a naive NVDEC benchmark
    silently measure x265 encoding instead of NVDEC decoding).
    """
    cmd = ["ffmpeg", "-hide_banner", "-nostdin", "-v", "error", "-y"]
    if ss is not None:
        cmd += ["-ss", str(ss)]
    cmd += input_opts
    cmd += ["-i", clip]
    cmd += ["-an", "-sn", "-dn", "-frames:v", str(frames),
            "-pix_fmt", "yuv420p", "-f", "rawvideo", out_path]
    if extra:
        cmd += extra
    return run(cmd, timeout=900)


def compare(a_bytes, b_bytes, w, h):
    """Per-frame MAE / max-diff / PSNR over all three yuv420p planes."""
    fa = w * h * 3 // 2
    fb = w * h * 3 // 2
    n = min(len(a_bytes) // fa, len(b_bytes) // fb)
    A = np.frombuffer(a_bytes[: n * fa], dtype=np.uint8).reshape(n, fa)
    B = np.frombuffer(b_bytes[: n * fb], dtype=np.uint8).reshape(n, fb)
    d = np.abs(A.astype(np.int16) - B.astype(np.int16))
    mae = float(d.mean())
    mx = int(d.max())
    exact = float((d.max(axis=1) == 0).mean())
    mse = float((d.astype(np.float64) ** 2).mean())
    psnr = float("inf") if mse == 0 else 10.0 * np.log10(255.0 ** 2 / mse)
    # luma-only (the plane the pipeline's features actually read most)
    ysz = w * h
    dy = d[:, :ysz]
    return {
        "frames_compared": int(n),
        "mae_all_planes": round(mae, 6),
        "max_abs_diff": mx,
        "frac_frames_bit_exact": round(exact, 4),
        "psnr_db": (None if psnr == float("inf") else round(psnr, 3)),
        "psnr_infinite": psnr == float("inf"),
        "mae_luma": round(float(dy.mean()), 6),
        "max_abs_diff_luma": int(dy.max()),
        "pct_pixels_diff_gt2": round(
            float((d > 2).mean()) * 100.0, 6
        ),
    }


def temporal_scale(clip, frames, decoder, w, h, tmpdir, ss=None):
    """Reference scale: mean |frame_t - frame_{t+1}| for the same decode.
    If decoder-vs-decoder MAE << this, the perturbation is far below the
    frame-to-frame signal any downstream scorer sees."""
    out = os.path.join(tmpdir, "temporal.raw")
    r = decode_raw(clip, out, frames, decoder, ss=ss)
    if r["rc"] != 0:
        return None
    raw = open(out, "rb").read()
    fs = w * h * 3 // 2
    n = len(raw) // fs
    A = np.frombuffer(raw[: n * fs], dtype=np.uint8).reshape(n, fs).astype(np.int16)
    d = np.abs(A[1:] - A[:-1])
    return round(float(d.mean()), 6) if d.size else None


def parity_case(clip, w, h, frames, tag, tmpdir, ss=None):
    is480 = "indego" in clip
    cpu_sw = ["-threads", "1"]                       # default software decoder
    nv_cuvid = ["-c:v", "h264_cuvid" if is480 else "hevc_cuvid"]
    nv_hwaccel = ["-hwaccel", "cuda", "-threads", "1"]

    res = {"clip": clip, "width": w, "height": h, "frames_requested": frames,
           "seek_ss": ss}
    paths = {}
    for name, dec in (("cpu", cpu_sw), ("nvdec_cuvid", nv_cuvid),
                      ("nvdec_hwaccel", nv_hwaccel)):
        out = os.path.join(tmpdir, f"{tag}_{name}.raw")
        r = decode_raw(clip, out, frames, dec, ss=ss)
        got = os.path.getsize(out) // (w * h * 3 // 2) if os.path.exists(out) else 0
        res[f"{name}_rc"] = r["rc"]
        res[f"{name}_frames"] = got
        res[f"{name}_wall_s"] = round(r["wall_s"], 3)
        res[f"{name}_cmd"] = " ".join(r["cmd"])
        if r["rc"] != 0:
            res[f"{name}_stderr"] = r["stderr"][-600:]
        paths[name] = out

    cpu = res.get("cpu_frames", 0)
    for name in ("nvdec_cuvid", "nvdec_hwaccel"):
        n = res.get(f"{name}_frames", 0)
        ok = (
            n == cpu == frames
            and res.get(f"{name}_rc") == 0
            and os.path.exists(paths[name])
        )
        res[f"{name}_frame_count_match"] = bool(n == cpu == frames)
        if not ok:
            res[f"{name}_comparison"] = "NOT COMPARABLE (missing/short decode)"
            continue
        a = open(paths["cpu"], "rb").read()
        b = open(paths[name], "rb").read()
        if name == "nvdec_hwaccel":
            # hwaccel path inserts an explicit download; still yuv420p
            pass
        res[f"{name}_comparison"] = compare(a, b, w, h)
    res["reference_frame_to_frame_mae_cpu"] = temporal_scale(
        clip, frames, cpu_sw, w, h, tmpdir, ss=ss
    )
    for p in paths.values():
        if os.path.exists(p):
            os.remove(p)
    return res


# ------------------------------------------------------------ throughput ----

def decode_fps(clip, input_opts, extra=None, timeout=1200):
    """Decode-only fps: mux to null (wrapped_avframe, no pixel conversion),
    read rtime from ffmpeg -benchmark. `input_opts` go before -i."""
    # NOTE: the `bench:` line is emitted at AV_LOG_INFO, so -v error hides it.
    cmd = ["ffmpeg", "-hide_banner", "-nostdin", "-v", "info", "-nostats",
           "-benchmark"]
    cmd += input_opts
    cmd += ["-i", clip]
    cmd += ["-an", "-sn", "-dn", "-f", "null", "-"]
    if extra:
        cmd += extra
    r = run(cmd, timeout=timeout)
    m = BENCH_RE.search(r["stderr"])
    if r["rc"] != 0 or not m:
        return {"ok": False, "rc": r["rc"], "cmd": " ".join(cmd),
                "stderr_tail": r["stderr"][-600:], "wall_s": round(r["wall_s"], 2)}
    rtime = float(m.group("rtime"))
    return {
        "ok": True, "cmd": " ".join(cmd), "rtime_s": rtime,
        "wall_s": round(r["wall_s"], 2),
        "utime_s": float(m.group("utime")), "stime_s": float(m.group("stime")),
        "startup_overhead_s": round(r["wall_s"] - rtime, 3),
    }


def with_metrics(entry, clip_info, n_sessions=1):
    if not entry.get("ok"):
        return entry
    fps_total = clip_info["frames"] * n_sessions / entry["rtime_s"]
    entry["frames_total"] = clip_info["frames"] * n_sessions
    entry["fps"] = round(fps_total, 1)
    entry["fps_per_session"] = round(fps_total / n_sessions, 1)
    entry["fps_from_external_wall"] = round(
        clip_info["frames"] * n_sessions / entry["wall_s"], 1)
    entry["rtf"] = round(fps_total / clip_info["fps"], 2)
    entry["rtf_per_session"] = round(fps_total / n_sessions / clip_info["fps"], 2)
    entry["n_sessions"] = n_sessions
    return entry


def gpu_mem():
    p = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used,utilization.gpu",
         "--format=csv,noheader,nounits"],
        stdout=subprocess.PIPE,
    )
    return p.stdout.decode().strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--tmp", default="/tmp/lb1_decode")
    ap.add_argument("--frames-480", type=int, default=20)
    ap.add_argument("--frames-1080", type=int, default=60)
    ap.add_argument("--parity-seconds", type=float, default=2.0)
    ap.add_argument("--skip-cpu-long", action="store_true")
    a = ap.parse_args()

    for p in (INDEGO_480, EGO10K_1080):
        if not os.path.exists(p):
            sys.exit(f"missing clip: {p}")
    os.makedirs(a.tmp, exist_ok=True)

    c480 = probe(INDEGO_480)
    c1080 = probe(EGO10K_1080)

    ffver = subprocess.run(["ffmpeg", "-version"], stdout=subprocess.PIPE)
    ffver = ffver.stdout.decode().splitlines()[0]

    def _ckpt(obj):
        """Write the report after every section so a late failure does not
        throw away completed measurements."""
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w") as f:
            json.dump(obj, f, indent=2)

    report = {
        "schema": "jadal.pod_verify.decode_verification/1",
        "claim_id": "LB1-decode",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "device": {
            "gpu": subprocess.run(
                ["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                 "--format=csv,noheader"], stdout=subprocess.PIPE
            ).stdout.decode().strip(),
            "nproc": os.cpu_count(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "note": "pod is shared with another swarm; absolute fps carry "
                    "contention noise (host CPU and GPU both shared)",
        },
        "ffmpeg": {
            "version_line": ffver,
            "hwaccels": subprocess.run(["ffmpeg", "-hide_banner", "-hwaccels"],
                                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            .stdout.decode().strip().splitlines(),
        },
        "clips": {"indego_480": c480, "ego10k_1080": c1080},
        "parity": {},
        "throughput": {},
    }

    # ---- correctness parity ------------------------------------------------
    report["parity"]["indego_480_head_2s"] = parity_case(
        INDEGO_480, c480["width"], c480["height"], a.frames_480, "p480", a.tmp
    )
    report["parity"]["ego10k_1080_head_2s"] = parity_case(
        EGO10K_1080, c1080["width"], c1080["height"], a.frames_1080, "p1080", a.tmp
    )
    # mid-stream span (checks it is not just the first GOP that matches)
    report["parity"]["ego10k_1080_midstream_2s"] = parity_case(
        EGO10K_1080, c1080["width"], c1080["height"], a.frames_1080,
        "p1080mid", a.tmp, ss=30.0
    )
    report["parity"]["indego_480_midstream_2s"] = parity_case(
        INDEGO_480, c480["width"], c480["height"], a.frames_480,
        "p480mid", a.tmp, ss=30.0
    )
    _ckpt(report)
    print("[ckpt] parity done", flush=True)

    # ---- throughput --------------------------------------------------------
    tp = report["throughput"]
    mem_before = gpu_mem()

    def _tp(name, entry):
        tp[name] = entry
        if entry.get("ok"):
            print(f"  {name}: {entry['fps']} fps  rtf={entry['rtf']}  "
                  f"rtime={entry['rtime_s']}s", flush=True)
        else:
            print(f"  {name}: FAILED rc={entry.get('rc')} "
                  f"{entry.get('stderr_tail','')[:200]}", flush=True)
        _ckpt(report)
        return entry

    # 480p
    _tp("indego_480_cpu_1thread", with_metrics(
        decode_fps(INDEGO_480, ["-threads", "1"]), c480))
    _tp("indego_480_cpu_process_auto", with_metrics(
        decode_fps(INDEGO_480, ["-threads", "0"]), c480))
    _tp("indego_480_nvdec_cuvid", with_metrics(
        decode_fps(INDEGO_480, ["-c:v", "h264_cuvid"]), c480))

    # 1080p
    _tp("ego10k_1080_cpu_1thread", with_metrics(
        decode_fps(EGO10K_1080, ["-threads", "1"]), c1080))
    if not a.skip_cpu_long:
        _tp("ego10k_1080_cpu_process_auto", with_metrics(
            decode_fps(EGO10K_1080, ["-threads", "0"]), c1080))
    _tp("ego10k_1080_nvdec_cuvid", with_metrics(
        decode_fps(EGO10K_1080, ["-c:v", "hevc_cuvid"]), c1080))
    _tp("ego10k_1080_nvdec_hwaccel", with_metrics(
        decode_fps(EGO10K_1080, ["-hwaccel", "cuda", "-threads", "1"]), c1080))

    # ---- NVDEC concurrency (the "flat across 1-4 sessions" claim) ----------
    conc = {}
    report["throughput"]["nvdec_1080p_concurrency"] = conc
    for n in (1, 2, 4):
        mem0 = gpu_mem()
        procs = []
        t0 = time.time()
        for _ in range(n):
            cmd = ["ffmpeg", "-hide_banner", "-nostdin", "-v", "error",
                   "-benchmark", "-c:v", "hevc_cuvid", "-i", EGO10K_1080,
                   "-an", "-sn", "-dn", "-f", "null", "-"]
            procs.append((subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                           stderr=subprocess.PIPE), cmd))
        outs = [p.communicate(timeout=1200) for p, _ in procs]
        wall = time.time() - t0
        rtimes, rcs = [], []
        for (p, _), (_, err) in zip(procs, outs):
            m = BENCH_RE.search(err.decode("utf-8", "replace"))
            rtimes.append(float(m.group("rtime")) if m else None)
            rcs.append(p.returncode)
        entry = {
            "n_sessions": n,
            "rcs": rcs,
            "rtimes_s": rtimes,
            "concurrent_wall_s": round(wall, 2),
            "frames_total": c1080["frames"] * n,
            "aggregate_fps": round(c1080["frames"] * n / wall, 1),
            "fps_per_session": round(c1080["frames"] * n / wall / n, 1),
            "aggregate_rtf": round(c1080["frames"] * n / wall / c1080["fps"], 2),
            "cmd": " ".join(procs[0][1]),
            "gpu_mem_before": mem0,
            "gpu_mem_after": gpu_mem(),
        }
        conc[f"nvdec_1080p_{n}sess"] = entry
        _ckpt(report)
        print(f"  concurrency {n}: aggregate {entry['aggregate_fps']} fps "
              f"({entry['fps_per_session']}/sess), mem {mem0} -> "
              f"{entry['gpu_mem_after']}", flush=True)

    def _sess1(e):
        return {"ok": True, "cmd": e["cmd"], "rtime_s": e["rtimes_s"][0],
                "frames_total": c1080["frames"], "fps": e["aggregate_fps"],
                "fps_per_session": e["aggregate_fps"], "rtf": e["aggregate_rtf"],
                "rtf_per_session": e["aggregate_rtf"], "n_sessions": 1,
                "note": "same run as nvdec_1080p_1sess"}
    conc["nvdec_1080p_1sess"] = _sess1(conc["nvdec_1080p_1sess"])

    _ok = [e["aggregate_fps"] for e in conc.values() if "aggregate_fps" in e]
    sc, mx = (min(_ok), max(_ok)) if _ok else (None, None)
    conc["scaling_summary"] = {
        "aggregate_fps_by_sessions": {
            str(k): v.get("aggregate_fps") for k, v in conc.items()
            if isinstance(v, dict) and "aggregate_fps" in v and k != "nvdec_1080p_1sess"
        },
        "flat_ratio_max_over_min": (round(mx / sc, 3) if _ok else None),
        "claim_evaluated": "NVDEC saturates at one engine (flat across 1-4 "
                           "concurrent sessions)",
    }
    report["throughput"]["nvdec_1080p_concurrency"] = conc
    # ---- fixed per-process cost (matters for short windows / early exit) ----
    startup = {}
    for name, opts in (
        ("cpu_1080p_0p1s", ["-threads", "1"]),
        ("nvdec_cuvid_1080p_0p1s", ["-c:v", "hevc_cuvid"]),
        ("nvdec_hwaccel_1080p_0p1s", ["-hwaccel", "cuda"]),
        ("cpu_480p_0p1s", ["-threads", "1"]),
        ("nvdec_cuvid_480p_0p1s", ["-c:v", "h264_cuvid"]),
    ):
        clip = EGO10K_1080 if "1080p" in name else INDEGO_480
        cmd = (["ffmpeg", "-hide_banner", "-nostdin", "-v", "error"]
               + opts + ["-i", clip, "-t", "0.1", "-f", "null", "-"])
        e = run(cmd, timeout=300)
        startup[name] = {"rc": e["rc"], "wall_s": round(e["wall_s"], 3),
                         "cmd": " ".join(cmd)}
    report["throughput"]["process_startup_cost"] = startup
    print("  startup:", {k: v["wall_s"] for k, v in startup.items()}, flush=True)
    _ckpt(report)

    report["throughput"]["gpu_mem_before_all"] = mem_before
    report["throughput"]["gpu_mem_after_all"] = gpu_mem()

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
