"""Decode-only floor for the spectral study (harness worker).

Measures, single-core (taskset), per 10-s 480p segment:
  - full decode via cv2 (all frames, no processing)
  - reduced-cost decode via ffmpeg scale=160:-1 to null
  - keyframe-only decode via ffmpeg -skip_frame nokey to null
Plus a simple CPU benchmark (numpy) so numbers can be scaled to other machines.
Writes eval/results/spectral/decode_floor.json.

Usage: python -m eval.spectral.decode_floor  (run from worktree root)
"""
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "eval" / "results" / "spectral" / "decode_floor.json"
MANIFEST = Path("/home/parshu/projects/jadal-qa-wt/spectral_data/manifest_gold.json")


def cpu_benchmark(seconds: float = 3.0) -> dict:
    """Simple single-core float benchmark: 512x512 matmul reps in `seconds`."""
    rng = np.random.default_rng(0)
    a = rng.random((512, 512))
    b = rng.random((512, 512))
    t0 = time.perf_counter()
    n = 0
    while time.perf_counter() - t0 < seconds:
        a @ b
        n += 1
    dt = time.perf_counter() - t0
    return {"matmul_512_reps": n, "wall_s": round(dt, 3),
            "reps_per_s": round(n / dt, 3)}


def clip_segment_tmp(src: str, start: float, end: float, tmp: Path) -> Path:
    out = tmp / "probe.mp4"
    dur = end - start
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
         "-ss", str(start), "-i", src, "-t", str(dur),
         "-c", "copy", str(out)],
        check=True)
    return out


def measure_cv2_full(seg: Path) -> dict:
    import cv2
    t0 = time.perf_counter()
    cap = cv2.VideoCapture(str(seg))
    n = 0
    while True:
        ok, _ = cap.read()
        if not ok:
            break
        n += 1
    cap.release()
    dt = time.perf_counter() - t0
    return {"frames": n, "wall_s": round(dt, 3)}


def measure_ffmpeg(seg: Path, pre: list[str], post: list[str]) -> dict:
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error"] + pre + \
        ["-i", str(seg)] + post + ["-f", "null", "-"]
    t0 = time.perf_counter()
    r = subprocess.run(cmd, capture_output=True, text=True)
    dt = time.perf_counter() - t0
    return {"wall_s": round(dt, 3), "rc": r.returncode,
            "stderr_tail": r.stderr[-300:] if r.returncode else ""}


def probe_duration(seg: Path) -> float:
    r = subprocess.run(
        ["ffprobe", "-hide_banner", "-loglevel", "error",
         "-show_entries", "format=duration", "-of", "csv=p=0", str(seg)],
        capture_output=True, text=True)
    try:
        return float(r.stdout.strip())
    except ValueError:
        return 10.0


def main() -> None:
    import tempfile
    man = json.loads(MANIFEST.read_text())
    seen = {}
    for it in man["items"]:
        if it["clip"] not in seen:
            seen[it["clip"]] = it
    probes = list(seen.values())[:7]
    bench = cpu_benchmark()
    rows = []
    with tempfile.TemporaryDirectory(dir=str(ROOT / ".scratch")) as td:
        tmp = Path(td)
        for it in probes:
            seg = clip_segment_tmp(it["path"], it["start_s"], it["end_s"], tmp)
            dur = probe_duration(seg)
            full = measure_cv2_full(seg)
            scaled = measure_ffmpeg(seg, [], ["-vf", "scale=160:-1"])
            keyonly = measure_ffmpeg(
                seg, ["-skip_frame", "nokey"], [])
            rows.append({
                "id": it["id"], "clip": it["clip"],
                "video_s": round(dur, 2),
                "full_decode_cv2": full,
                "scaled160_ffmpeg": scaled,
                "keyframe_only_ffmpeg": keyonly,
                "rtf_full": round(dur / full["wall_s"], 2) if full["wall_s"] else None,
                "rtf_scaled160": round(dur / scaled["wall_s"], 2) if scaled["wall_s"] else None,
                "rtf_keyonly": round(dur / keyonly["wall_s"], 2) if keyonly["wall_s"] else None,
            })
            seg.unlink(missing_ok=True)
    out = {
        "note": "single-core (taskset) decode floor; NOT measured on a Pi; "
                "use cpu_benchmark to scale across machines",
        "single_core": True,
        "cpu_benchmark": bench,
        "machine": {"nproc": 22},
        "rows": rows,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    sys.exit(main())
