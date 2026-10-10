"""Edge benchmark (B9): L0 signals + funnel.route over real IndEgo segments.

Runs ``jadal.signals.compute_signals`` + ``jadal.funnel.route`` (placeholder
thresholds are fine for a throughput benchmark; the verdicts here measure
speed, not quality — said explicitly in the report) over N real IndEgo
segments (480x480, 10 fps, read-only source clips; 10 s each, deterministic
selection: clips sorted by path, per-clip windows at fixed fractional
offsets) under pinned CPUs.

Method: the orchestrator extracts the segments once, then re-execs itself as
a worker under ``taskset -c 0`` (1 core) and ``taskset -c 0-3`` (4 cores),
with ``cv2.setNumThreads`` and the OMP/OPENBLAS/MKL thread env vars set to
the core count. Each worker discards 1 warm-up pass, then does 3 timed
repeats; per-segment wall time (median/p95), segments/hour, video-hours/day
(= segments/hour * seg_seconds * 24 / 3600), and peak RSS are reported.

Writes ``eval/results/edge_bench_<label>.json`` including platform, machine,
CPU model, python/opencv/numpy versions, core list, git sha, and the proxy
disclaimer: every number here is "x86 laptop cores as a proxy, NOT measured
on a Raspberry Pi" unless run with ``--device-label`` on real hardware. We
never extrapolate Pi speed from laptop numbers with an invented factor.

Determinism check: ``route()`` outputs are hashed per segment; the hashes
must be identical across the 3 repeats AND across the two core counts.

When ``jadal.signals`` (branch origin/feat/signals, PR #12) or the IndEgo
clips are absent, the run is SKIPPED with an explanation (exit code 2).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
RESULTS_DIR = ROOT / "eval" / "results"
SEG_DIR = ROOT / "data_runtime" / "edge_tmp" / "bench_segs"

DISCLAIMER = (
    "x86 laptop cores as a proxy, NOT measured on a Raspberry Pi. "
    "Never claim 'runs on a Pi' without a run with --device-label on real hardware."
)

# Placeholder thresholds: fine for throughput; verdict quality is NOT claimed.
PLACEHOLDER_THRESHOLDS = {
    "accept_conf": 0.99,
    "min_hands_frac": 0.5,
    "max_dup_hamming": 10,
}
PLACEHOLDER_NOTE = (
    "placeholder thresholds: throughput benchmark only; "
    "verdicts/accept-rates here carry no quality claim.")

REQUIRED_RESULT_KEYS = frozenset({
    "device_label", "disclaimer", "placeholder_thresholds_note", "platform",
    "machine", "cpu_model", "python", "opencv", "numpy", "git_sha",
    "seg_seconds", "n_segments", "repeats", "core_configs", "results",
    "determinism",
})


def hash_route_output(obj) -> str:
    """Stable sha256 of a route()/signals output (canonical JSON). Pure."""
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":"),
                   default=str).encode("utf-8")).hexdigest()


def validate_bench_result(obj: dict) -> list[str]:
    """Schema violations (empty = valid). Hermetic/pure."""
    errors: list[str] = []
    if not isinstance(obj, dict):
        return ["result must be a JSON object"]
    for key in sorted(REQUIRED_RESULT_KEYS):
        if key not in obj:
            errors.append(f"missing key: {key}")
    if DISCLAIMER not in str(obj.get("disclaimer", "")):
        errors.append("disclaimer must contain the proxy-hardware label")
    det = obj.get("determinism", {})
    if isinstance(det, dict):
        for key in ("identical_across_repeats", "identical_across_cores"):
            if key not in det:
                errors.append(f"determinism missing key: {key}")
    else:
        errors.append("determinism must be a dict")
    return errors


def git_sha() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, cwd=ROOT,
                             timeout=15)
        return out.stdout.strip() or None
    except Exception:
        return None


def cpu_model() -> str | None:
    try:
        with open("/proc/cpuinfo") as fh:
            for line in fh:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or None


def clip_durations(clips: list[Path]) -> dict[str, float]:
    import cv2
    durs: dict[str, float] = {}
    for c in clips:
        cap = cv2.VideoCapture(str(c))
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 10.0
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        durs[str(c)] = n / fps if fps > 0 else 0.0
    return durs


def select_segments(clip_dir: Path, n: int, seg_seconds: float) -> list[dict]:
    """Deterministic segment selection: clips sorted, fixed fractional offsets."""
    clips = sorted(clip_dir.rglob("*_480.mp4"))
    clips = [c for c in clips if c.is_file()]
    if not clips:
        raise SystemExit(f"SKIP: no IndEgo clips under {clip_dir}")
    durs = clip_durations(clips)
    usable = [(c, d) for c, d in
              ((c, durs[str(c)]) for c in clips) if d >= seg_seconds + 1]
    if not usable:
        raise SystemExit(f"SKIP: no clip >= {seg_seconds + 1}s under {clip_dir}")
    per = max(1, -(-n // len(usable)))  # ceil
    segs: list[dict] = []
    for clip, dur in usable:
        span = dur - seg_seconds
        for i in range(per):
            frac = (i / (per - 1)) if per > 1 else 0.0
            start = round(span * frac, 3)
            segs.append({"clip": str(clip), "start_s": start,
                         "seg_id": f"{clip.stem}-{start:08.3f}"})
            if len(segs) >= n:
                return segs
    return segs


def extract_segments(segs: list[dict], seg_seconds: float) -> list[Path]:
    """Extract 10 s windows with cv2 (deterministic: same file reused by all repeats)."""
    import cv2
    SEG_DIR.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for s in segs:
        out = SEG_DIR / f"{s['seg_id']}.mp4"
        if out.is_file():
            paths.append(out)
            continue
        cap = cv2.VideoCapture(s["clip"])
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 10.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.set(cv2.CAP_PROP_POS_MSEC, s["start_s"] * 1000.0)
        need = int(round(seg_seconds * fps))
        writer = cv2.VideoWriter(str(out),
                                 cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        got = 0
        while got < need:
            ok, frame = cap.read()
            if not ok:
                break
            writer.write(frame)
            got += 1
        cap.release()
        writer.release()
        if got == 0:
            raise SystemExit(f"SKIP: could not extract {s['seg_id']}")
        paths.append(out)
    return paths


def run_worker(seg_paths: list[str], seg_seconds: float, n_cores: int,
               repeats: int) -> dict:
    """Worker: pinned process. Sets thread counts, times signals+route."""
    import resource
    import cv2
    cv2.setNumThreads(n_cores)
    try:
        from jadal import signals as S
    except ImportError as e:
        raise SystemExit(
            "SKIP: jadal.signals absent (branch origin/feat/signals, PR #12, "
            f"not yet on main): {e}")
    try:
        from jadal import funnel as F
    except ImportError as e:
        raise SystemExit(f"SKIP: jadal.funnel absent: {e}")
    decisions = {"hands_ok": {"p_yes": 0.9}, "active_work": {"p_yes": 0.9},
                 "step": {"choice": None, "probs": {}, "answer_conf": 0.9},
                 "model_versions": {"edge_bench": "placeholder-L1"}}
    # 1 warm-up pass, discarded.
    for p in seg_paths:
        F.route(S.compute_signals(p), decisions, PLACEHOLDER_THRESHOLDS)
    rep_times: list[list[float]] = []
    rep_hashes: list[list[str]] = []
    for _ in range(repeats):
        times: list[float] = []
        hashes: list[str] = []
        for p in seg_paths:
            t0 = time.perf_counter()
            sig = S.compute_signals(p)
            out = F.route(sig, decisions, PLACEHOLDER_THRESHOLDS)
            times.append(time.perf_counter() - t0)
            hashes.append(hash_route_output({"signals": sig, "route": out}))
        rep_times.append(times)
        rep_hashes.append(hashes)
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    try:
        backend = S.backend_info()
    except Exception as e:  # pragma: no cover - defensive
        backend = {"error": str(e)}
    return {"per_segment_s": rep_times, "hashes": rep_hashes,
            "peak_rss_MB": round(peak, 1), "backend": backend,
            "cv2_threads": cv2.getNumThreads()}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Edge throughput benchmark (B9).")
    ap.add_argument("--device-label", default="x86-laptop-proxy")
    ap.add_argument("--cores", default="0,0-3",
                    help="Comma-separated taskset core lists, e.g. '0,0-3'.")
    ap.add_argument("--segments", type=int, default=30)
    ap.add_argument("--seg-seconds", type=float, default=10.0)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--clip-dir",
                    default="/home/parshu/projects/awing/data/indego")
    ap.add_argument("--worker", action="store_true",
                    help=argparse.SUPPRESS)  # internal: pinned worker mode
    ap.add_argument("--worker-cores", type=int, default=1)
    ap.add_argument("--worker-ncores-label", default="")
    ap.add_argument("--worker-segs", default="")
    args = ap.parse_args(argv)

    if args.worker:
        paths = args.worker_segs.split(os.pathsep)
        try:
            res = run_worker(paths, args.seg_seconds, args.worker_cores,
                             args.repeats)
        except SystemExit as e:
            print(f"WORKER_SKIP: {e}", file=sys.stderr)
            return 2
        print(json.dumps(res))
        return 0

    # --- orchestrator (unpinned) ---
    try:
        import jadal.signals  # noqa: F401  (presence check only)
    except ImportError:
        print("SKIP: jadal.signals absent (branch origin/feat/signals, PR #12, "
              "not yet on main): benchmark needs the real L0 module.",
              file=sys.stderr)
        return 2
    clip_dir = Path(args.clip_dir)
    if not clip_dir.is_dir():
        print(f"SKIP: clip dir absent: {clip_dir}", file=sys.stderr)
        return 2
    segs = select_segments(clip_dir, args.segments, args.seg_seconds)
    seg_paths = [str(p) for p in extract_segments(segs, args.seg_seconds)]
    print(f"segments: {len(seg_paths)} x {args.seg_seconds}s "
          f"(deterministic selection, cached under {SEG_DIR})", flush=True)

    core_cfgs = [c.strip() for c in args.cores.split(",") if c.strip()]
    ncpu = os.cpu_count() or 1
    worker_results: dict[str, dict] = {}
    for cfg in core_cfgs:
        cores = [c for c in cfg.replace("-", " ").split()]
        n_cores = (int(cores[-1]) - int(cores[0]) + 1) if len(cores) == 2 \
            else len(cores)
        n_cores = max(1, min(n_cores, ncpu))
        env = dict(os.environ)
        env["OMP_NUM_THREADS"] = env["OPENBLAS_NUM_THREADS"] = \
            env["MKL_NUM_THREADS"] = str(n_cores)
        cmd = ["taskset", "-c", cfg, sys.executable, __file__, "--worker",
               "--worker-cores", str(n_cores),
               "--worker-ncores-label", cfg,
               "--worker-segs", os.pathsep.join(seg_paths),
               "--seg-seconds", str(args.seg_seconds),
               "--repeats", str(args.repeats)]
        print(f"worker cores={cfg} (threads={n_cores}) ...", flush=True)
        proc = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT,
                              env=env, timeout=3600)
        if proc.returncode == 2:
            print(f"SKIP worker {cfg}: {proc.stderr.strip()[-500:]}",
                  file=sys.stderr)
            return 2
        if proc.returncode != 0:
            print(f"worker {cfg} failed: {proc.stderr.strip()[-2000:]}",
                  file=sys.stderr)
            return 1
        worker_results[cfg] = json.loads(proc.stdout.strip().splitlines()[-1])
        med = statistics.median(worker_results[cfg]["per_segment_s"][0])
        print(f"worker cores={cfg}: repeat0 median/seg = {med:.3f}s",
              flush=True)

    import cv2
    import numpy
    results: dict[str, dict] = {}
    for cfg, w in worker_results.items():
        flat = [t for rep in w["per_segment_s"] for t in rep]
        med = statistics.median(flat)
        p95 = float(numpy.quantile(flat, 0.95))
        sph = 3600.0 / med if med > 0 else 0.0
        vhd = sph * args.seg_seconds * 24.0 / 3600.0
        results[cfg] = {
            "n_cores_label": cfg,
            "per_segment_wall_s": {
                "median": round(med, 4),
                "p95": round(float(p95), 4),
                "n": len(flat),
            },
            "segments_per_hour": round(sph, 1),
            "video_hours_per_day": round(vhd, 2),
            "peak_rss_MB": w["peak_rss_MB"],
            "backend": w["backend"],
        }
    first_hashes = {cfg: w["hashes"][0] for cfg, w in worker_results.items()}
    across_repeats = all(
        all(h == w["hashes"][0] for h in w["hashes"][1:])
        for w in worker_results.values())
    across_cores = len({tuple(h) for h in first_hashes.values()}) == 1
    bench = {
        "device_label": args.device_label,
        "disclaimer": DISCLAIMER,
        "placeholder_thresholds_note": PLACEHOLDER_NOTE,
        "placeholder_thresholds": PLACEHOLDER_THRESHOLDS,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_model": cpu_model(),
        "python": platform.python_version(),
        "opencv": cv2.__version__,
        "numpy": numpy.__version__,
        "git_sha": git_sha(),
        "seg_seconds": args.seg_seconds,
        "n_segments": len(seg_paths),
        "segment_ids": [s["seg_id"] for s in segs],
        "repeats": args.repeats,
        "warmup_discarded": 1,
        "core_configs": core_cfgs,
        "results": results,
        "determinism": {
            "identical_across_repeats": across_repeats,
            "identical_across_cores": across_cores,
            "hash_algo": "sha256(canonical-json(signals+route))",
        },
    }
    errors = validate_bench_result(bench)
    out_path = RESULTS_DIR / f"edge_bench_{args.device_label}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(bench, indent=2, sort_keys=True) + "\n")
    print(json.dumps(results, indent=2))
    print(f"determinism: repeats={across_repeats} cores={across_cores}")
    print(f"wrote: {out_path}")
    if errors:  # pragma: no cover - defensive
        print(f"SCHEMA ERRORS: {errors}", file=sys.stderr)
        return 1
    if not (across_repeats and across_cores):
        print("FAIL: route() outputs differ across repeats/cores",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
