"""Edge-tier import audit (B9): prove the edge path is torch-free and light.

EDGE TIER = jadal.segmenter, jadal.signals, jadal.funnel, jadal.certify,
jadal.receipts (+ jadal.config). The CENTRAL tier (decision/EmbeddingGemma 2,
hub, service) is NOT required on the edge and is never imported here.

What this script does (all imports happen in CLEAN subprocesses so this
process's own imports can never pollute the audit):
  1. Import the edge modules in a fresh interpreter, then inspect
     ``sys.modules``. FAIL (exit 1) if any forbidden heavy module was pulled
     in: torch, torchvision, sentence_transformers, transformers, sklearn,
     pandas, matplotlib, fastapi/uvicorn, or any CUDA library.
  2. Report which heavy third-party modules the edge path DOES pull in
     (numpy, cv2, scipy, mediapipe?), the import time, and the peak RSS after
     processing one real 10 s segment.
  3. Record the active hands/faces backend names
     (``jadal.signals.backend_info()``). mediapipe on Pi ARM64 is UNVERIFIED,
     so the edge path is also run with mediapipe BLOCKED
     (``sys.modules['mediapipe'] = None`` in a subprocess) to prove the
     OpenCV colour-proxy fallback works.
  4. Write ``eval/results/edge_check_<label>.json`` (schema validated by
     :func:`validate_result`).

When ``jadal.signals`` / ``jadal.segmenter`` are absent (they live on branch
``origin/feat/signals``, PR #12, not yet on main) the signals-dependent steps
are SKIPPED with an explanation; the import audit still runs over the modules
that ARE present and the exit code still reflects the forbidden-import check.

Every number in the output JSON carries the proxy disclaimer: the benchmark
here runs on THIS x86 laptop, which is NOT a Raspberry Pi. Never claim
"runs on a Pi" without a run with ``--device-label`` on real hardware.

Exit codes: 0 = pass (no forbidden import at import time OR at call time;
skips, if any, are explained in the JSON), 1 = FAIL (forbidden import found),
2 = skipped/not-measured (`--require-probes` with probes skipped, or refusal
to overwrite a measured result with a probe-skipped one without `--force`).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "eval" / "results"

DISCLAIMER = (
    "x86 laptop cores as a proxy, NOT measured on a Raspberry Pi. "
    "Never claim 'runs on a Pi' without a run with --device-label on real hardware."
)

# Exact module names (or top-level package prefixes) that must never appear in
# sys.modules after importing the edge tier.
FORBIDDEN_EXACT = frozenset({
    "torch",
    "torchvision",
    "sentence_transformers",
    "transformers",
    "sklearn",
    "pandas",
    "matplotlib",
    "fastapi",
    "uvicorn",
    "onnxruntime",
    "tensorflow",
})
# Prefixes catching CUDA libraries (nvidia wheels, cupy, triton, ...).
FORBIDDEN_PREFIXES = ("cuda", "cublas", "cudnn", "cufft", "curand",
                      "cusparse", "nvidia", "cupy", "triton", "tensorrt")

# Heavy third-party modules we explicitly report (present or absent).
HEAVY_REPORT = ("numpy", "cv2", "scipy", "mediapipe", "PIL", "imagehash")

EDGE_MODULES = ("jadal.segmenter", "jadal.signals", "jadal.funnel",
                "jadal.certify", "jadal.receipts", "jadal.config")

CLIP_SEARCH = (
    "1_Assembly/mp4_480/User_15_410_0411_2_480.mp4",
    "1_Assembly/mp4_480/User_1_412_2010_3_480.mp4",
    "3_Logistics_Organization/mp4_480/User_1_2110_Organisation_480.mp4",
)

REQUIRED_RESULT_KEYS = frozenset({
    "device_label", "disclaimer", "platform", "python", "git_sha",
    "forbidden_found", "heavy_present", "import_time_s",
    "edge_modules_imported", "edge_modules_missing",
})


def check_forbidden(modules) -> list[str]:
    """Return the sorted forbidden entries found in a sys.modules-like mapping.

    Hermetic/pure: takes any iterable of module names (or a dict, in which
    case its keys are used). Never imports anything.
    """
    found: set[str] = set()
    names = modules.keys() if isinstance(modules, dict) else modules
    for name in names:
        top = str(name).split(".")[0]
        if top in FORBIDDEN_EXACT:
            found.add(top)
        else:
            low = top.lower()
            for prefix in FORBIDDEN_PREFIXES:
                if low == prefix or low.startswith(prefix):
                    found.add(top)
                    break
    return sorted(found)


def validate_result(obj: dict) -> list[str]:
    """Return a list of schema violations (empty = valid). Hermetic/pure."""
    errors: list[str] = []
    if not isinstance(obj, dict):
        return ["result must be a JSON object"]
    for key in sorted(REQUIRED_RESULT_KEYS):
        if key not in obj:
            errors.append(f"missing key: {key}")
    disc = obj.get("disclaimer", "")
    if DISCLAIMER not in str(disc):
        errors.append("disclaimer must contain the proxy-hardware label")
    if not isinstance(obj.get("forbidden_found", None), list):
        errors.append("forbidden_found must be a list")
    if "runtime_forbidden_found" in obj and not isinstance(
            obj["runtime_forbidden_found"], list):
        errors.append("runtime_forbidden_found must be a list")
    if not isinstance(obj.get("heavy_present", None), dict):
        errors.append("heavy_present must be a dict")
    return errors


def git_sha() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, cwd=ROOT,
                             timeout=15)
        return out.stdout.strip() or None
    except Exception:
        return None


_AUDIT_CHILD = r"""
import json, sys, time
t0 = time.time()
imported, missing = [], []
for mod in {modules!r}:
    try:
        __import__(mod)
        imported.append(mod)
    except ImportError as e:
        missing.append({{"module": mod, "error": str(e)}})
import_time = time.time() - t0
heavy = {{m: (m in sys.modules) for m in {heavy!r}}}
versions = {{}}
for m in ("numpy", "cv2", "scipy", "mediapipe", "PIL", "imagehash"):
    try:
        mod = sys.modules.get(m)
        versions[m] = getattr(mod, "__version__", None) if mod is not None else None
    except Exception:
        versions[m] = None
backend = None
if "jadal.signals" in imported:
    try:
        from jadal import signals as _s
        backend = _s.backend_info()
    except Exception as e:
        backend = {{"error": str(e)}}
print(json.dumps({{"imported": imported, "missing": missing,
                   "modules": sorted(sys.modules.keys()),
                   "heavy_present": heavy, "heavy_versions": versions,
                   "import_time_s": round(import_time, 3), "backend_info": backend}}))
"""

_PROCESS_CHILD = r"""
import json, sys, resource, time
{BLOCK}
import cv2
from pathlib import Path
SEG = {seg!r}
OUT = Path({out!r})
OUT.parent.mkdir(parents=True, exist_ok=True)
cap = cv2.VideoCapture(SEG)
fps = float(cap.get(cv2.CAP_PROP_FPS)) or 10.0
need = int(round(float({seconds}) * fps))
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
writer = cv2.VideoWriter(str(OUT), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
n = 0
ok, frame = cap.read()
while ok and n < need:
    writer.write(frame)
    n += 1
    ok, frame = cap.read()
cap.release(); writer.release()
t0 = time.time()
from jadal import signals as S
sig = S.compute_signals(str(OUT))
dt = time.time() - t0
peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
print(json.dumps({{"segment": str(OUT), "frames_written": n, "fps": fps,
                   "compute_signals_s": round(dt, 3), "peak_rss_MB": round(peak, 1),
                   "signals": sig, "backend_info": S.backend_info(),
                   "modules": sorted(sys.modules.keys())}}))
"""


def _run_child(code: str, timeout: int = 600) -> dict:
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True,
                          text=True, cwd=ROOT, timeout=timeout)
    if proc.returncode != 0:
        raise ChildError(proc.stderr.strip()[-2000:] or f"exit {proc.returncode}")
    return json.loads(proc.stdout.strip().splitlines()[-1])


class ChildError(Exception):
    pass


def find_clip(clip_dir: Path) -> Path | None:
    for rel in CLIP_SEARCH:
        p = clip_dir / rel
        if p.is_file():
            return p
    return None


def should_overwrite(old: dict | None, new_has_probes: bool,
                     force: bool) -> bool:
    """Pure overwrite rule (Minor 5): never clobber measured probes with a
    probe-skipped result unless --force. Hermetic/pure."""
    if force or new_has_probes:
        return True
    if not isinstance(old, dict):
        return True
    return not bool(old.get("process_probe"))


def git_branch() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"],
                             capture_output=True, text=True, cwd=ROOT,
                             timeout=15)
        return out.stdout.strip() or None
    except Exception:
        return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Edge-tier import audit (B9).")
    ap.add_argument("--device-label", default="x86-laptop-proxy")
    ap.add_argument("--clip-dir",
                    default="/home/parshu/projects/awing/data/indego",
                    help="Read-only IndEgo clip directory.")
    ap.add_argument("--seg-seconds", type=float, default=10.0)
    ap.add_argument("--out", default=None,
                    help="Result JSON path (default eval/results/edge_check_<label>.json).")
    ap.add_argument("--require-probes", action="store_true",
                    help="Exit 2 when any probe is skipped (for the B9 gate: "
                         "a PASS must have measured, not just audited).")
    ap.add_argument("--force", action="store_true",
                    help="Allow overwriting a result that contains measurements "
                         "with one where the probes were skipped.")
    args = ap.parse_args(argv)

    audit = _run_child(
        _AUDIT_CHILD.format(modules=list(EDGE_MODULES), heavy=list(HEAVY_REPORT)))
    forbidden = check_forbidden(audit["modules"])
    signals_present = "jadal.signals" in audit["imported"]

    result: dict = {
        "device_label": args.device_label,
        "disclaimer": DISCLAIMER,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "git_sha": git_sha(),
        "git_branch": git_branch(),
        "edge_modules_imported": audit["imported"],
        "edge_modules_missing": audit["missing"],
        "forbidden_found": forbidden,
        "runtime_forbidden_found": [],
        "heavy_present": audit["heavy_present"],
        "heavy_versions": audit["heavy_versions"],
        "import_time_s": audit["import_time_s"],
        "signals_backend": audit["backend_info"],
        "process_probe": None,
        "mediapipe_blocked_probe": None,
        "skipped": [],
    }

    if not signals_present:
        result["skipped"].append(
            "jadal.signals/jadal.segmenter absent (branch origin/feat/signals, "
            "PR #12, not yet on main): segment processing and backend probes "
            "skipped; import audit covers the modules present on this branch.")
    else:
        clip = find_clip(Path(args.clip_dir))
        if clip is None:
            result["skipped"].append(
                f"no IndEgo clip found under {args.clip_dir}: segment "
                "processing probe skipped.")
        else:
            seg_path = (ROOT / "data_runtime" / "edge_tmp" / "edge_check_seg.mp4")
            probe = _run_child(_PROCESS_CHILD.format(
                BLOCK="", seg=str(clip), out=str(seg_path),
                seconds=args.seg_seconds))
            result["process_probe"] = probe
            # Important 1 fix: the forbidden gate also covers modules pulled
            # in LAZILY at call time (compute_signals runs in this child).
            result["runtime_forbidden_found"] = check_forbidden(
                probe.get("modules", []))
            try:
                blocked = _run_child(_PROCESS_CHILD.format(
                    BLOCK="sys.modules['mediapipe'] = None",
                    seg=str(clip), out=str(seg_path),
                    seconds=args.seg_seconds))
                result["mediapipe_blocked_probe"] = {
                    **blocked, "mediapipe_blocked": True,
                    "fallback_works": (
                        blocked["backend_info"].get("faces") == "proxy"
                        and blocked["backend_info"].get("hands") == "proxy"),
                }
                result["runtime_forbidden_found"] = sorted(set(
                    result["runtime_forbidden_found"])
                    | set(check_forbidden(blocked.get("modules", []))))
            except ChildError as e:
                result["mediapipe_blocked_probe"] = {
                    "mediapipe_blocked": True, "fallback_works": False,
                    "error": str(e),
                }
                result["skipped"].append(
                    "mediapipe-blocked probe failed; see mediapipe_blocked_probe.error")

    errors = validate_result(result)
    out_path = Path(args.out) if args.out else (
        RESULTS_DIR / f"edge_check_{args.device_label}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"device_label: {args.device_label}")
    print(f"disclaimer: {DISCLAIMER}")
    print(f"imported: {audit['imported']}")
    print(f"missing: {[m['module'] for m in audit['missing']]}")
    print(f"import_time_s: {audit['import_time_s']}")
    print(f"heavy_present: {audit['heavy_present']}")
    print(f"forbidden_found: {forbidden or 'none'}")
    print(f"runtime_forbidden_found: "
          f"{result['runtime_forbidden_found'] or 'none'}")
    if result["process_probe"]:
        p = result["process_probe"]
        print(f"probe: compute_signals_s={p['compute_signals_s']} "
              f"peak_rss_MB={p['peak_rss_MB']} backend={p['backend_info']}")
    if result["mediapipe_blocked_probe"]:
        b = result["mediapipe_blocked_probe"]
        print(f"mediapipe-blocked: fallback_works={b.get('fallback_works')} "
              f"backend={b.get('backend_info')}")
    for s in result["skipped"]:
        print(f"SKIPPED: {s}")
    if errors:
        print(f"SCHEMA ERRORS: {errors}", file=sys.stderr)
        return 1
    if forbidden or result["runtime_forbidden_found"]:
        print(f"FAIL: forbidden modules imported: import-time={forbidden} "
              f"runtime={result['runtime_forbidden_found']}", file=sys.stderr)
        return 1
    if not args.force:
        try:
            old = json.loads(out_path.read_text()) if out_path.is_file() else None
        except Exception:
            old = None
        if not should_overwrite(old, result["process_probe"] is not None,
                                args.force):
            print(f"NOT OVERWRITTEN: {out_path} already contains measured "
                  f"probes; re-run with --force to replace with this "
                  f"probe-skipped result.", file=sys.stderr)
            return 2
    out_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"wrote: {out_path}")
    if args.require_probes and result["skipped"]:
        print("SKIP: --require-probes set and probes were skipped.",
              file=sys.stderr)
        return 2
    print("PASS: no forbidden import on the edge path.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
