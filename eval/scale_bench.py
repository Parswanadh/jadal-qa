"""Funnel scale bench (GitHub issue #26, Phase 4).

Measures, on THIS laptop: segment decode + L0-stage latency on CPU with P
worker processes (P in {1,2,4,8}), funnel route() latency, and a
cross-process determinism check (route() outputs hash identically in every
worker). Throughput is reported in segments/hour.

Scale-out model: segments are processed ONE AT A TIME; scale-out is by more
processes, never by batching, so per-segment results stay bit-identical to
single-segment runs (SPEC determinism rule).

Honest-labelling notes (read before quoting any number):
- jadal.signals is NOT on main yet, so the L0 stage here is a minimal
  documented PROXY (fixed-frame OpenCV stats: blur variance, brightness,
  motion — same family of cheap measurements the real module will compute),
  clearly marked "l0_proxy" in the output. Decode and route() numbers are
  real; the real signals module will replace the proxy.
- jadal.decision is NOT importable in this worktree (verified), so the L1
  embedding stage is SKIPPED. The script checks nvidia-smi first and only
  runs L1 alone when the module exists AND the GPU is otherwise idle.

Usage:
    PYTHONNOUSERSITE=1 python3 eval/scale_bench.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from jadal import funnel  # noqa: E402

RESULTS_PATH = ROOT / "eval" / "results" / "funnel_scale.json"
SEGMENTS_PATH = ROOT / "eval" / "gold" / "segments.json"

# Deterministic frame sampling: N frames at fixed fractions of the segment.
N_SAMPLE_FRAMES = 10
FRAME_WIDTH = 240


def gpu_busy() -> tuple[bool, str]:
    """True when nvidia-smi shows any non-zero GPU utilisation or processes."""
    try:
        out = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError) as exc:
        return True, f"nvidia-smi unavailable ({exc}); treating GPU as busy"
    if out.returncode != 0:
        return True, "nvidia-smi failed; treating GPU as busy"
    try:
        utils = [float(line.strip()) for line in out.stdout.splitlines()
                 if line.strip()]
    except ValueError:
        return True, "could not parse nvidia-smi; treating GPU as busy"
    if any(u > 0 for u in utils):
        return True, f"GPU utilisation {utils}% — another worker is on it"
    return False, f"GPU idle (utilisation {utils}%)"


def sample_frames(path: str, start_s: float, end_s: float) -> list:
    """Decode N frames at fixed fractions (deterministic, read-only)."""
    import cv2
    import numpy as np
    cap = cv2.VideoCapture(path)
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
        frames = []
        for k in range(N_SAMPLE_FRAMES):
            t = start_s + (k + 0.5) * (end_s - start_s) / N_SAMPLE_FRAMES
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
            ok, frame = cap.read()
            if not ok or frame is None:
                raise RuntimeError(f"cannot read t={t:.1f}s from {path}")
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            small = cv2.resize(gray, (FRAME_WIDTH, FRAME_WIDTH))
            frames.append(np.asarray(small, dtype=np.float32))
        return frames
    finally:
        cap.release()


def l0_proxy(frames: list) -> dict:
    """Minimal documented L0 PROXY (NOT the real jadal.signals module).

    Cheap fixed-frame stats of the same family L0 will compute: Laplacian
    blur variance, brightness mean, frame-difference motion. Deterministic.
    """
    import cv2
    import numpy as np
    stack = np.stack(frames)
    blur = float(np.mean([cv2.Laplacian(
        np.asarray(frame, dtype=np.uint8), cv2.CV_64F).var()
        for frame in frames]))
    brightness = float(np.mean(stack))
    motion = float(np.mean([np.mean(np.abs(stack[i + 1] - stack[i]))
                            for i in range(len(frames) - 1)]))
    return {"stage": "l0_proxy", "blur_var": round(blur, 4),
            "brightness_mean": round(brightness, 4),
            "motion_mean": round(motion, 4)}


def process_one(segment: dict, thresholds: dict) -> dict:
    """Decode -> L0 proxy -> route(); returns timings + output hash."""
    timings: dict[str, float] = {}
    t0 = time.perf_counter()
    frames = sample_frames(segment["path"], segment["start_s"],
                           segment["end_s"])
    timings["decode_s"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    proxy = l0_proxy(frames)
    timings["l0_proxy_s"] = time.perf_counter() - t0
    signals = {
        "decodable": True,
        "duration_s": round(segment["end_s"] - segment["start_s"], 1),
        "blur_var": proxy["blur_var"],
        "brightness_mean": proxy["brightness_mean"],
        "under_exposed_frac": 0.0,
        "over_exposed_frac": 0.0,
        "motion_mean": proxy["motion_mean"],
        "hands_visible_frac": 1.0,
        "dup_of": None,
    }
    decisions = {
        "hands_ok": {"p_yes": 0.99}, "active_work": {"p_yes": 0.99},
        "step": {"choice": "assemble", "probs": {},
                 "answer_conf": 0.99},
    }
    t0 = time.perf_counter()
    out = funnel.route(signals, decisions, thresholds)
    timings["route_s"] = time.perf_counter() - t0
    digest = hashlib.sha256(json.dumps(out, sort_keys=True).encode())
    return {"segment_id": segment["segment_id"], "timings": timings,
            "output_sha256": digest.hexdigest(), "verdict": out["verdict"]}


def bench(segments: list[dict], thresholds: dict, workers: int) -> dict:
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(process_one, segments,
                                [thresholds] * len(segments)))
    wall_s = time.perf_counter() - t0
    agg: dict[str, float] = {}
    for stage in ("decode_s", "l0_proxy_s", "route_s"):
        values = sorted(r["timings"][stage] for r in results)
        agg[stage] = {
            "mean_ms": round(sum(values) / len(values) * 1000, 3),
            "p50_ms": round(values[len(values) // 2] * 1000, 3),
            "max_ms": round(values[-1] * 1000, 3),
        }
    hashes = {r["output_sha256"] for r in results}
    return {
        "workers": workers,
        "n_segments": len(results),
        "wall_s": round(wall_s, 3),
        "segments_per_hour": round(len(results) / wall_s * 3600, 1),
        "per_stage": agg,
        "distinct_output_hashes": len(hashes),
        "verdicts": {v: sum(1 for r in results if r["verdict"] == v)
                     for v in ("accept", "reject", "review")},
        "outputs": {r["segment_id"]: r["output_sha256"] for r in results},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=40,
                        help="segments to process per worker count")
    parser.add_argument("--out", type=Path, default=RESULTS_PATH)
    args = parser.parse_args(argv)

    with open(SEGMENTS_PATH, "r", encoding="utf-8") as fh:
        manifest = json.load(fh)
    segments = manifest["segments"][:args.limit]
    thresholds = funnel.load_thresholds()

    busy, gpu_note = gpu_busy()
    try:
        import jadal.decision  # noqa: F401
        decision_importable = True
    except ImportError:
        decision_importable = False
    l1_note = ("jadal.decision is NOT importable in this worktree: "
               "L1 embedding stage skipped"
               if not decision_importable else
               ("GPU busy, L1 skipped" if busy else "L1 would run alone"))

    print(f"segments={len(segments)} ({manifest['clips'][0]['clip']} + ...), "
          f"frames/segment={N_SAMPLE_FRAMES}")
    print(f"gpu: {gpu_note}; l1: {l1_note}")
    runs = []
    for workers in (1, 2, 4, 8):
        run = bench(segments, thresholds, workers)
        runs.append(run)
        print(f"P={workers}: {run['segments_per_hour']} seg/h "
              f"(wall {run['wall_s']}s, decode {run['per_stage']['decode_s']['mean_ms']}ms, "
              f"l0_proxy {run['per_stage']['l0_proxy_s']['mean_ms']}ms, "
              f"route {run['per_stage']['route_s']['mean_ms']}ms, "
              f"verdicts {run['verdicts']})")
    # Cross-process determinism: every worker count must produce byte-identical
    # per-segment route() outputs (same input -> same hash, any P).
    reference = runs[0]["outputs"]
    deterministic = all(run["outputs"] == reference for run in runs)
    print(f"cross-process determinism: "
          f"{'IDENTICAL' if deterministic else 'MISMATCH'} "
          f"({len(reference)} segments x {len(runs)} worker counts)")
    for run in runs:
        del run["outputs"]
    result = {
        "n_segments": len(segments),
        "frames_per_segment": N_SAMPLE_FRAMES,
        "gpu_note": gpu_note,
        "l1_note": l1_note,
        "l0_note": ("PROXY stage (jadal.signals not on main): fixed-frame "
                    "OpenCV stats; decode + route numbers are real"),
        "scale_model": ("segments processed one at a time; scale-out by "
                        "more processes, never by batching"),
        "runs": runs,
        "cross_process_determinism": deterministic,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
        fh.write("\n")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
