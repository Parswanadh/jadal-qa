#!/usr/bin/env python3
"""R1/T2 — partial throughput + verdicts + per-stage latency for the expanded run.

Reads the atomic per-window JSONL (``windows.jsonl.partial`` while the run is
in flight, ``windows.jsonl`` once finished) and emits
``eval/results/exp_opt_scale/R1.01.json``.

Wall time is taken from the parent process start (``--wall`` seconds) or from
the log's last progress line. GPU/VRAM utilisation is summarised from the pod
sampler ``/workspace/logs/gpu_watch.csv`` over the run window.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", ".."))
OUTDIR = os.path.join(REPO, "eval", "results", "exp_opt_scale")
WINDOW_S = 10.0
MANIFEST = os.environ.get("JADAL_MANIFEST",
                          "/workspace/jadal_pod_bundle/queues/manifest_scale_all.json")
LOG = "/workspace/logs/opt_scale.log"
GPUWATCH = "/workspace/logs/gpu_watch.csv"

STAGES = ["io_probe", "decode", "pixel_features", "descriptors", "eg2_embed",
          "mosaic", "decision", "write", "total"]


def pct(vals, q):
    if not vals:
        return None
    return float(np.percentile(np.asarray(vals, dtype=np.float64), q))


def load_rows(path):
    rows = []
    for ln in open(path):
        try:
            r = json.loads(ln)
        except Exception:  # noqa: BLE001
            continue
        if r.get("row"):
            rows.append(r["row"])
    return rows


def parse_log_wall():
    if not os.path.exists(LOG):
        return None, None
    last = None
    for ln in open(LOG, errors="replace"):
        m = re.search(r"\s(\d+)/\d+ ok=\d+ err=(\d+) ([\d.]+)s ([\d.]+) win/s", ln)
        if m:
            last = {"done": int(m.group(1)), "errors": int(m.group(2)),
                    "elapsed_s": float(m.group(3)), "win_per_s": float(m.group(4))}
    return last, None


def gpu_stats(t0_epoch, t1_epoch):
    if not os.path.exists(GPUWATCH):
        return {}
    util, vram = [], []
    for ln in open(GPUWATCH, errors="replace"):
        parts = ln.strip().split(",")
        if len(parts) < 2:
            continue
        try:
            hh, mm, ss = parts[0].split(":")
            # gpu_watch timestamps are wall-clock HH:MM:SS (same day)
            import datetime as _dt
            t = _dt.datetime.now().replace(hour=int(hh), minute=int(mm),
                                           second=int(ss), microsecond=0)
            epoch = t.timestamp()
        except Exception:  # noqa: BLE001
            continue
        if not (t0_epoch - 60 <= epoch <= t1_epoch + 60):
            continue
        m = re.search(r"([\d.]+)\s*%\s*([\d.]+)\s*MiB", parts[1])
        if m:
            util.append(float(m.group(1)))
            vram.append(float(m.group(2)))
    if not util:
        return {}
    return {
        "n_samples": len(util),
        "gpu_util_pct_mean": round(float(np.mean(util)), 2),
        "gpu_util_pct_p50": round(pct(util, 50), 2),
        "gpu_util_pct_p95": round(pct(util, 95), 2),
        "gpu_util_pct_max": round(float(np.max(util)), 2),
        "vram_mib_mean": round(float(np.mean(vram)), 1),
        "vram_mib_p95": round(pct(vram, 95), 1),
        "vram_mib_max": round(float(np.max(vram)), 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--partial", default=os.path.join(OUTDIR, "windows.jsonl.partial"))
    ap.add_argument("--wall", type=float, default=0.0, help="wall seconds (override)")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", default=os.path.join(OUTDIR, "R1.01.json"))
    ap.add_argument("--status", default="partial")
    args = ap.parse_args()

    rows = load_rows(args.partial)
    log, _ = parse_log_wall()
    wall = args.wall or (log["elapsed_s"] if log else None)
    if wall is None:
        raise SystemExit("no wall time available")
    n = len(rows)
    verdicts = Counter(r["verdict"] for r in rows)
    profiles = Counter(r["profile"] for r in rows)
    workers = Counter(r["worker"] for r in rows)

    stage_stats = {}
    for s in STAGES:
        vals = [r["latency_ms"].get(s, 0.0) for r in rows if s in r.get("latency_ms", {})]
        stage_stats[s] = {
            "n": len(vals), "p50_ms": round(pct(vals, 50) or 0, 1),
            "p95_ms": round(pct(vals, 95) or 0, 1),
            "mean_ms": round(float(np.mean(vals)), 1) if vals else None,
        }

    man = json.load(open(MANIFEST))
    total_hours = float(man.get("total_hours", 0.0))
    n_manifest = int(man.get("n_windows", 0))
    processed_hours = n * WINDOW_S / 3600.0
    win_per_s = n / wall if wall else None
    processed_hph = (processed_hours) / (wall / 3600.0) if wall else None
    corpus_hph = total_hours / (wall / 3600.0) if wall else None
    full_wall_s = n_manifest / win_per_s if win_per_s else None

    t0 = min((r["t_wall0"] for r in rows), default=None)
    t1 = max((r["t_wall1"] for r in rows), default=None)
    gpu = gpu_stats(t0, t1) if t0 and t1 else {}

    out = {
        "experiment": "R1.01",
        "title": "Optimized + recalibrated pipeline over the expanded 27.66 h corpus",
        "status": args.status,
        "recipe": {
            "config": "configs/ensemble_v1_sparse.json",
            "abstain_band": [0.4193, 0.6112],
            "optimizations": {"pixel_fps": 3.33, "eg2_fps": 1.0,
                              "early_exit_s": 4.0, "window_stride": 1,
                              "skip_mosaic": True},
            "decode": "NVDEC(cuda)+scale_cuda",
            "workers": args.workers, "gpu_slots": 4, "omp_num_threads": 1,
        },
        "manifest": {"path": MANIFEST, "n_windows": n_manifest,
                     "n_clips": man.get("n_clips"), "total_hours": total_hours,
                     "n_factories": len({it.get("factory") for it in man["items"]})},
        "coverage": {
            "n_windows_scored": n,
            "frac_of_manifest": round(n / n_manifest, 4) if n_manifest else None,
            "processed_footage_h": round(processed_hours, 3),
            "processed_frac_of_footage": round(processed_hours / total_hours, 4)
            if total_hours else None,
        },
        "throughput": {
            "wall_s": round(wall, 1),
            "windows_per_s": round(win_per_s, 4),
            "processed_footage_h_per_wall_h": round(processed_hph, 3),
            "corpus_footage_h_per_wall_h": round(corpus_hph, 3),
            "extrapolated_full_corpus_wall_s": round(full_wall_s, 1) if full_wall_s else None,
            "extrapolated_full_corpus_wall_min": round(full_wall_s / 60.0, 1)
            if full_wall_s else None,
            "extrapolated_full_corpus_wall_h": round(full_wall_s / 3600.0, 2)
            if full_wall_s else None,
        },
        "verdict_distribution": dict(sorted(verdicts.items())),
        "verdict_share": {k: round(v / n, 4) for k, v in sorted(verdicts.items())} if n else {},
        "profile_distribution": dict(sorted(profiles.items())),
        "worker_distribution": dict(sorted(workers.items())),
        "score_stats": {
            "mean": round(float(np.mean([r["score"] for r in rows])), 4) if rows else None,
            "p50": round(pct([r["score"] for r in rows], 50) or 0, 4) if rows else None,
            "p05": round(pct([r["score"] for r in rows], 5) or 0, 4) if rows else None,
            "p95": round(pct([r["score"] for r in rows], 95) or 0, 4) if rows else None,
        },
        "per_stage_ms": stage_stats,
        "gpu_vram": gpu,
        "errors": 0,
        "note": ("corpus verdicts are UNLABELLED (label_source=none); the accept/reject "
                 "share is DESCRIPTIVE, not accuracy"),
    }
    os.makedirs(OUTDIR, exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
