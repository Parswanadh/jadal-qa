#!/usr/bin/env python3
"""Known-defect demo clip generator (GitHub issue #38).

Reads the real IndEgo clips read-only, cuts one ~20 s window from each of two
sources (ECU assembly with hands in view, logistics), and deterministically
emits a small LABELLED set of demo variants plus a manifest.

Outputs go ONLY under ``data_runtime/demo_clips/`` (git-ignored; the default
is anchored at the repo root so it works from any CWD). ``--out`` accepts
another directory only with ``--allow-outside-data-runtime`` (debug-only;
temp dirs are always refused per SPEC).
Source clips are never copied into the repo. IndEgo inputs default to
``/home/parshu/projects/awing/data/indego`` (override: ``JADAL_INDEGO_ROOT``).

Variants per source window (9 files):
  good               - re-encoded copy of the window (no defect)
  blurred            - strong cv2.GaussianBlur
  too_dark           - pixel values scaled down
  overexposed        - gain + offset, clipped
  frozen             - first frame repeated (static / duplicate-like)
  wrong_fps          - same frames written at 15 fps (container fps defect)
  truncated          - good file cut to its first half (damaged container)
  garbage            - seeded random bytes with a .mp4 name (undecodable)
  reupload_duplicate - byte-identical copy of good under a new name

Labels (true by construction — we injected the defect — and MUST NOT be
presented as human-labelled data):
  good               -> defect "none",      label "SYNTHETIC-GOOD"
  reupload_duplicate -> defect "duplicate", label "SYNTHETIC-DEFECT"
  everything else    -> its defect code,    label "SYNTHETIC-DEFECT"

Determinism: fixed seeds, fixed encoder settings (cv2.VideoWriter, fourcc
``mp4v``, 10 fps — 15 fps for the wrong_fps variant — source frame size).
The manifest stores no timestamps, so same inputs -> byte-identical clips
AND byte-identical manifest.json on the same machine + OpenCV build
(verified: two back to back runs hash equal). Cross-machine /
cross-OpenCV-build byte stability is expected for garbage / truncated /
reupload_duplicate (pure byte ops) but NOT guaranteed for the re-encoded
mp4s (encoder version differences); the defect itself is deterministic
everywhere.

What is NOT done: no hands-in-view detector (windows are hand-picked from
the IndEgo VIA annotations, see DEFAULT_SOURCES), no face blurring, no
service wiring.

Usage:
  python scripts/make_demo_clips.py [--out DIR] [--seed INT] [--list]
  python scripts/make_demo_clips.py --out /elsewhere --allow-outside-data-runtime

Exit codes: 0 ok; 2 IndEgo sources missing (nothing written);
3 --out outside data_runtime/ without --allow-outside-data-runtime.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from datetime import datetime, timezone

import cv2
import numpy as np

FPS = 10.0
WRONG_FPS = 15.0
FOURCC = "mp4v"
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RUNTIME = os.path.join(REPO_ROOT, "data_runtime")
DEFAULT_OUT = os.path.join(DATA_RUNTIME, "demo_clips")
# Overridable for tests via JADAL_INDEGO_ROOT (production default below).
INDEGO_ROOT = os.environ.get("JADAL_INDEGO_ROOT", "/home/parshu/projects/awing/data/indego")

# Hand-picked windows (from the IndEgo VIA annotations shipped with the data):
#  - ECU clip: step 02_attach_ram_chip spans 42.7-140.8 s -> [60, 80) s.
#  - Logistics organise clip: step 01_organize spans 1.6-232.8 s -> [30, 50) s.
DEFAULT_SOURCES = [
    {
        "key": "ecu",
        "relpath": "1_Assembly/mp4_480/User_15_410_0411_2_480.mp4",
        "start_s": 60.0,
        "duration_s": 20.0,
        "note": "ECU assembly, 02_attach_ram_chip window (hands in view)",
    },
    {
        "key": "logistics",
        "relpath": "3_Logistics_Organization/mp4_480/User_1_2110_Organisation_480.mp4",
        "start_s": 30.0,
        "duration_s": 20.0,
        "note": "logistics organise, 01_organize window",
    },
]

# Variant table: (file suffix, defect code, label, parameters).
# Parameters are recorded verbatim in the manifest.
VARIANTS: list[tuple[str, str, str, dict]] = [
    ("good", "none", "SYNTHETIC-GOOD", {}),
    ("blurred", "blur", "SYNTHETIC-DEFECT", {"op": "cv2.GaussianBlur", "ksize": [31, 31], "sigmaX": 0}),
    ("too_dark", "too_dark", "SYNTHETIC-DEFECT", {"op": "scale", "scale": 0.25}),
    ("overexposed", "overexposed", "SYNTHETIC-DEFECT", {"op": "gain_offset_clip", "gain": 1.7, "offset": 30.0}),
    ("frozen", "frozen", "SYNTHETIC-DEFECT", {"op": "repeat_first_frame"}),
    ("wrong_fps", "wrong_fps", "SYNTHETIC-DEFECT", {"op": "write_same_frames_at_fps", "fps": WRONG_FPS}),
    ("truncated", "truncated", "SYNTHETIC-DEFECT", {"op": "cut_bytes", "fraction": 0.5}),
    ("garbage", "garbage", "SYNTHETIC-DEFECT", {"op": "seeded_random_bytes", "nbytes": 65536}),
    ("reupload_duplicate", "duplicate", "SYNTHETIC-DEFECT", {"op": "byte_identical_copy_of_good"}),
]

PIXEL_VARIANTS = ("good", "blurred", "too_dark", "overexposed", "frozen")
DECODABLE_VARIANTS = PIXEL_VARIANTS + ("wrong_fps", "reupload_duplicate")
UNDECODABLE_VARIANTS = ("truncated", "garbage")


def _is_inside(path: str, root: str) -> bool:
    return os.path.commonpath([os.path.abspath(path), os.path.abspath(root)]) == os.path.abspath(root)


def _is_temp_dir(path: str) -> bool:
    roots = ["/tmp"]
    tmpdir = os.environ.get("TMPDIR")
    if tmpdir:
        roots.append(tmpdir)
    return any(_is_inside(path, root) for root in roots)


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def is_decodable(path: str) -> bool:
    """True if OpenCV can open the file and read at least one frame."""
    cap = cv2.VideoCapture(path)
    try:
        if not cap.isOpened():
            return False
        ok, frame = cap.read()
        return bool(ok) and frame is not None
    finally:
        cap.release()


def read_window(
    path: str, start_s: float, duration_s: float
) -> tuple[list, int, tuple[int, int], float]:
    """Read a window, resampled to FPS. Returns (frames, start_frame, (w, h), src_fps)."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open source clip: {path}")
    try:
        src_fps = cap.get(cv2.CAP_PROP_FPS) or FPS
        start_frame = int(round(start_s * src_fps))
        want = int(round(duration_s * src_fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        raw: list = []
        for _ in range(want):
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            raw.append(frame)
        if not raw:
            raise RuntimeError(f"no frames read from {path} at {start_s}s")
        if len(raw) < want * 0.9:
            raise RuntimeError(
                f"short read from {path} at {start_s}s: got {len(raw)}/{want} frames"
            )
        target = int(round(duration_s * FPS))
        if abs(src_fps - FPS) > 1e-6 and len(raw) != target:
            # Even resample to the nominal frame rate (deterministic
            # nearest-neighbour; repeats frames when upsampling).
            idx = np.linspace(0, len(raw) - 1, target).round().astype(int)
            frames = [raw[i] for i in idx]
        else:
            frames = raw
        h, w = frames[0].shape[:2]
        return frames, start_frame, (w, h), float(src_fps)
    finally:
        cap.release()


def write_frames(frames: list, out_path: str, fps: float = FPS) -> None:
    h, w = frames[0].shape[:2]
    vw = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*FOURCC), fps, (w, h))
    if not vw.isOpened():
        raise RuntimeError(f"VideoWriter failed to open: {out_path}")
    try:
        for frame in frames:
            vw.write(frame)
    finally:
        vw.release()


def apply_variant(frames: list, variant: str) -> list:
    if variant in ("good", "wrong_fps"):
        return [f.copy() for f in frames]
    if variant == "blurred":
        return [cv2.GaussianBlur(f, (31, 31), 0) for f in frames]
    if variant == "too_dark":
        return [(f.astype(np.float32) * 0.25).clip(0, 255).astype(np.uint8) for f in frames]
    if variant == "overexposed":
        return [(f.astype(np.float32) * 1.7 + 30.0).clip(0, 255).astype(np.uint8) for f in frames]
    if variant == "frozen":
        return [frames[0].copy() for _ in frames]
    raise ValueError(f"unknown pixel variant: {variant}")


def build_source_set(
    source_abs: str,
    source_rel: str,
    key: str,
    start_s: float,
    duration_s: float,
    out_dir: str,
    seed: int,
    source_index: int,
) -> tuple[list[dict], tuple[int, int]]:
    """Build all 9 variants for one source window.

    Returns (manifest entries, (w, h)). Good is encoded once; truncated and
    the duplicate derive from those exact bytes. Postcondition: the
    decodable set decodes, truncated/garbage do not.

    Note: output containment under data_runtime/ is a CLI (main()) property;
    this API writes wherever out_dir points (tests use temp dirs).
    """
    frames, start_frame, size, src_fps = read_window(source_abs, start_s, duration_s)
    os.makedirs(out_dir, exist_ok=True)
    entries: list[dict] = []
    good_name = f"{key}_good.mp4"
    good_path = os.path.join(out_dir, good_name)
    write_frames(apply_variant(frames, "good"), good_path)
    with open(good_path, "rb") as f:
        good_bytes = f.read()
    good_sha = hashlib.sha256(good_bytes).hexdigest()

    garbage_rng = np.random.default_rng(seed + 1000 + source_index)
    garbage_bytes = garbage_rng.integers(0, 256, size=65536, dtype=np.uint8).tobytes()

    for variant, defect, label, base_params in VARIANTS:
        name = f"{key}_{variant}.mp4"
        path = os.path.join(out_dir, name)
        out_fps = FPS
        if variant == "good":
            pass  # already written above; entry reuses these exact bytes
        elif variant in ("blurred", "too_dark", "overexposed", "frozen"):
            write_frames(apply_variant(frames, variant), path)
        elif variant == "wrong_fps":
            out_fps = WRONG_FPS
            write_frames(apply_variant(frames, variant), path, fps=WRONG_FPS)
        elif variant == "truncated":
            with open(path, "wb") as f:
                f.write(good_bytes[: len(good_bytes) // 2])
        elif variant == "garbage":
            with open(path, "wb") as f:
                f.write(garbage_bytes)
        elif variant == "reupload_duplicate":
            shutil.copyfile(good_path, path)
        else:  # pragma: no cover - guarded by VARIANTS table
            raise ValueError(variant)
        params = dict(base_params)
        if variant == "reupload_duplicate":
            params.update({"copy_of": good_name, "good_sha256": good_sha})
        entries.append(
            {
                "file": name,
                "defect": defect,
                "variant": variant,
                "parameters": params,
                "fps": out_fps,
                "actual_duration_s": round(len(frames) / out_fps, 3),
                "source_clip": source_rel,
                "window": {
                    "start_s": start_s,
                    "duration_s": duration_s,
                    "start_frame": start_frame,
                    "n_frames": len(frames),
                    "source_fps": src_fps,
                    "frame_size": [size[0], size[1]],
                },
                "sha256": sha256_file(path),
                "label": label,
            }
        )
    # Postconditions: decode behaviour + duplicate identity (M3).
    by_name = {e["file"]: e for e in entries}
    assert by_name[f"{key}_reupload_duplicate.mp4"]["sha256"] == good_sha, "duplicate diverged"
    for variant in DECODABLE_VARIANTS:
        assert is_decodable(os.path.join(out_dir, f"{key}_{variant}.mp4")), f"{key}_{variant} unreadable"
    for variant in UNDECODABLE_VARIANTS:
        assert not is_decodable(os.path.join(out_dir, f"{key}_{variant}.mp4")), f"{key}_{variant} decodes"
    return entries, size


def make_manifest(seed: int, entries: list[dict], width: int, height: int) -> dict:
    """Manifest dict. No timestamps: same inputs -> identical bytes."""
    return {
        "generator": "scripts/make_demo_clips.py",
        "seed": seed,
        "encoder": {"backend": "cv2.VideoWriter", "fourcc": FOURCC, "fps": FPS, "width": width, "height": height},
        "label_note": "SYNTHETIC-* labels are true by construction (defect injected by this script); MUST NOT be presented as human-labelled data.",
        "sources": DEFAULT_SOURCES,
        "files": entries,
    }


def write_manifest(out_dir: str, seed: int, entries: list[dict], width: int, height: int) -> str:
    manifest_path = os.path.join(out_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(make_manifest(seed, entries, width, height), f, indent=2, sort_keys=True)
        f.write("\n")
    return manifest_path


def plan_list() -> list[str]:
    names: list[str] = []
    for src in DEFAULT_SOURCES:
        for variant, _defect, _label, _params in VARIANTS:
            names.append(f"{src['key']}_{variant}.mp4")
    return names


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build labelled demo clips with known defects.")
    ap.add_argument("--out", default=DEFAULT_OUT, help="output dir (default: %(default)s)")
    ap.add_argument("--seed", type=int, default=42, help="fixed RNG seed (default: %(default)s)")
    ap.add_argument("--list", action="store_true", help="print planned outputs and exit")
    ap.add_argument(
        "--allow-outside-data-runtime",
        action="store_true",
        help="debug-only: permit --out outside data_runtime/ (never under /tmp)",
    )
    args = ap.parse_args(argv)

    if args.list:
        for name in plan_list():
            print(name)
        return 0

    out_abs = os.path.abspath(args.out)
    data_abs = os.path.abspath(DATA_RUNTIME)
    if _is_temp_dir(out_abs):
        print(
            f"make_demo_clips: refusing --out under a temp dir ({out_abs}); "
            "SPEC forbids writing results to /tmp.",
            file=sys.stderr,
        )
        return 3
    if not _is_inside(out_abs, data_abs) and not args.allow_outside_data_runtime:
        print(
            f"make_demo_clips: refusing --out outside data_runtime/ ({out_abs}). "
            "Pass --allow-outside-data-runtime to override (debug-only; "
            "never under /tmp).",
            file=sys.stderr,
        )
        return 3

    missing = [
        src for src in DEFAULT_SOURCES if not os.path.isfile(os.path.join(INDEGO_ROOT, src["relpath"]))
    ]
    if missing:
        print(
            "make_demo_clips: IndEgo source clips not found (inputs are read-only and "
            f"expected under {INDEGO_ROOT}; override with JADAL_INDEGO_ROOT):",
            file=sys.stderr,
        )
        for src in missing:
            print(f"  - {os.path.join(INDEGO_ROOT, src['relpath'])}", file=sys.stderr)
        print("Nothing was written. Mount or download the IndEgo clips and retry.", file=sys.stderr)
        return 2

    os.makedirs(args.out, exist_ok=True)
    all_entries: list[dict] = []
    sizes: set[tuple[int, int]] = set()
    for i, src in enumerate(DEFAULT_SOURCES):
        src_entries, size = build_source_set(
            os.path.join(INDEGO_ROOT, src["relpath"]),
            src["relpath"],
            src["key"],
            src["start_s"],
            src["duration_s"],
            args.out,
            args.seed,
            i,
        )
        sizes.add(size)
        all_entries.extend(src_entries)
    if len(sizes) > 1:
        raise RuntimeError(f"mixed source frame sizes: {sorted(sizes)}")
    (width, height) = sorted(sizes)[0]
    write_manifest(args.out, args.seed, all_entries, width, height)
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    print(f"wrote {len(all_entries)} clips + manifest.json to {args.out} at {now}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
