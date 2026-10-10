#!/usr/bin/env python3
"""LB1-decode, part 4: follow-ups the first pass raised.

  A. CPU software decoder thread scaling (the 1-thread vs auto run was nearly
     flat: 80.1 vs 81.8 fps -- is that the decoder, or the pod?).
  B. `-hwaccel cuda` WITHOUT `-threads 1` (the first pass handicapped it).
  C. NVDEC scaling beyond 4 sessions (does the aggregate ever plateau?).
  D. End-to-end cost of a real pipeline window: `-ss S -t T` + scale to the
     working resolution, process wall time INCLUDING ffmpeg/CUDA startup.
     This is the number the early-exit design actually pays, because the
     current pipeline spawns one ffmpeg per chunk.

--merge-into <part1.json> merges the result under "extensions".
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from decode_bench_a40 import (  # noqa: E402
    BENCH_RE, EGO10K_1080, INDEGO_480, gpu_mem, probe, run,
)


def bench(cmd, timeout=900):
    r = run(cmd, timeout=timeout)
    m = BENCH_RE.search(r["stderr"])
    return r, (float(m.group("rtime")) if m else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--merge-into", default=None)
    a = ap.parse_args()

    c1080 = probe(EGO10K_1080)
    c480 = probe(INDEGO_480)
    out = {"schema": "jadal.pod_verify.decode_extensions/1",
           "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "gpu_state_at_start": gpu_mem()}

    # --- A. CPU thread scaling (30 s span, keeps each run ~10 s at 1 thread) --
    A = {}
    for th in (1, 2, 4, 8, 16, 0):
        cmd = (["ffmpeg", "-hide_banner", "-nostdin", "-v", "info", "-nostats",
                "-benchmark", "-threads", str(th), "-i", EGO10K_1080,
                "-t", "30", "-an", "-sn", "-dn", "-f", "null", "-"])
        r, rt = bench(cmd)
        n = 30 * 30
        A[f"threads_{th}"] = {
            "threads": th, "rc": r["rc"], "rtime_s": rt,
            "wall_s": round(r["wall_s"], 3),
            "fps": (round(n / rt, 1) if rt else None),
            "rtf": (round(n / rt / 30.0, 2) if rt else None),
            "cmd": " ".join(cmd),
        }
        print(f"A threads={th}: {A[f'threads_{th}']['fps']} fps", flush=True)
    out["cpu_thread_scaling_1080p_30s_span"] = A

    # --- B. fair -hwaccel cuda (no -threads 1) ------------------------------
    B = {}
    for name, opts in (("nvdec_cuvid", ["-c:v", "hevc_cuvid"]),
                       ("nvdec_hwaccel_auto", ["-hwaccel", "cuda"]),
                       ("nvdec_hwaccel_t1", ["-hwaccel", "cuda", "-threads", "1"])):
        cmd = (["ffmpeg", "-hide_banner", "-nostdin", "-v", "info", "-nostats",
                "-benchmark"] + opts + ["-i", EGO10K_1080,
                "-an", "-sn", "-dn", "-f", "null", "-"])
        r, rt = bench(cmd)
        B[name] = {"rc": r["rc"], "rtime_s": rt,
                   "wall_s": round(r["wall_s"], 3),
                   "fps": (round(c1080["frames"] / rt, 1) if rt else None),
                   "rtf": (round(c1080["frames"] / rt / 30.0, 2) if rt else None),
                   "cmd": " ".join(cmd)}
        print(f"B {name}: {B[name]['fps']} fps", flush=True)
    out["nvdec_paths_1080p"] = B

    # --- C. concurrency 1/2/4/8, wall-clock aggregate ------------------------
    C = {}
    for n in (1, 2, 4, 8):
        mem0 = gpu_mem()
        procs = []
        t0 = time.time()
        for _ in range(n):
            cmd = ["ffmpeg", "-hide_banner", "-nostdin", "-v", "info",
                   "-nostats", "-benchmark", "-c:v", "hevc_cuvid",
                   "-i", EGO10K_1080, "-an", "-sn", "-dn", "-f", "null", "-"]
            procs.append(subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                                          stderr=subprocess.PIPE))
        rts = []
        for p in procs:
            _, err = p.communicate(timeout=1200)
            m = BENCH_RE.search(err.decode("utf-8", "replace"))
            rts.append(float(m.group("rtime")) if m else None)
        wall = time.time() - t0
        C[f"{n}sess"] = {
            "n_sessions": n, "concurrent_wall_s": round(wall, 2),
            "session_rtimes_s": rts,
            "aggregate_fps": round(c1080["frames"] * n / wall, 1),
            "fps_per_session": round(c1080["frames"] * n / wall / n, 1),
            "aggregate_rtf": round(c1080["frames"] * n / wall / 30.0, 2),
            "gpu_mem_before": mem0, "gpu_mem_after": gpu_mem(),
            "cmd": " ".join(cmd),
        }
        print(f"C {n} sess: aggregate {C[f'{n}sess']['aggregate_fps']} fps "
              f"({C[f'{n}sess']['fps_per_session']}/sess)", flush=True)
    out["nvdec_concurrency_1080p"] = C

    # --- D. real pipeline window: -ss .. -t .. + scale, wall incl. startup ---
    D = {}
    for tag, clip, dur in (("1080p", EGO10K_1080, 2.0),
                           ("1080p", EGO10K_1080, 10.0),
                           ("480p", INDEGO_480, 2.0)):
        for name, pre in (("cpu", []),
                          ("nvdec_cuvid", ["-hwaccel", "cuda", "-c:v",
                                           "hevc_cuvid" if tag == "1080p"
                                           else "h264_cuvid"]),
                          ("nvdec_scale_cuda",
                           ["-hwaccel", "cuda", "-hwaccel_output_format",
                            "cuda"])):
            key = f"{tag}_window{int(dur)}s_{name}"
            vf = (f"scale=160:90,format=rgb24" if name != "nvdec_scale_cuda"
                  else "scale_cuda=160:90:format=nv12,hwdownload,format=nv12")
            pix = "rgb24" if name != "nvdec_scale_cuda" else "nv12"
            cmd = (["ffmpeg", "-hide_banner", "-nostdin", "-v", "error"] + pre
                   + ["-ss", "30", "-t", str(dur), "-i", clip, "-vf", vf,
                      "-f", "rawvideo", "-pix_fmt", pix, "-vsync", "0",
                      "-y", "/dev/null"])
            walls, rcs = [], []
            for _ in range(3):
                r = run(cmd, timeout=600)
                walls.append(r["wall_s"])
                rcs.append(r["rc"])
            D[key] = {
                "wall_s_median": round(statistics.median(walls), 3),
                "wall_s_runs": [round(w, 3) for w in walls],
                "returncodes": rcs,
                "frames_in_window": int(round(dur * (30 if tag == "1080p" else 10))),
                "effective_fps": round(
                    dur * (30 if tag == "1080p" else 10)
                    / statistics.median(walls), 1),
                "cmd": " ".join(cmd),
            }
            print(f"D {key}: {D[key]['wall_s_median']}s "
                  f"({D[key]['effective_fps']} fps eff)", flush=True)
    out["pipeline_window_wall_time"] = D
    out["gpu_state_at_end"] = gpu_mem()

    with open(a.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {a.out}")

    if a.merge_into and os.path.exists(a.merge_into):
        with open(a.merge_into) as f:
            mj = json.load(f)
        mj["extensions"] = out
        with open(a.merge_into, "w") as f:
            json.dump(mj, f, indent=2)
        print(f"merged into {a.merge_into}")


if __name__ == "__main__":
    main()
