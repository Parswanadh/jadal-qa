#!/usr/bin/env python3
"""Known-defect demo clip generator (GitHub issue #38).

Reads the real IndEgo clips read-only, cuts one ~20 s window from each of two
sources (ECU assembly with hands in view, logistics), and deterministically
emits a small LABELLED set of demo variants plus a manifest.

Outputs go ONLY under ``data_runtime/demo_clips/`` (git-ignored; override with
``--out``). Source clips are never copied into the repo.

Variants per source window (8 files):
  good               - re-encoded copy of the window (no defect)
  blurred            - strong cv2.GaussianBlur
  too_dark           - pixel values scaled down
  overexposed        - gain + offset, clipped
  frozen             - first frame repeated (static / duplicate-like)
  truncated          - good file cut to its first half (damaged container)
  garbage            - seeded random bytes with a .mp4 name (undecodable)
  reupload_duplicate - byte-identical copy of good under a new name

Every manifest entry carries ``label: "SYNTHETIC-DEFECT"``. These labels are
true by construction (we injected the defect) and MUST NOT be presented as
human-labelled data.

Determinism: fixed seeds, fixed encoder settings (cv2.VideoWriter, fourcc
``mp4v``, 10 fps, source frame size). Same inputs -> byte-identical outputs
for ALL variants on the same machine + OpenCV build (verified: two back to
back re-encodes hash equal). Cross-machine / cross-OpenCV-build byte
stability is expected for garbage / truncated / reupload_duplicate (pure
byte ops) but NOT guaranteed for the re-encoded mp4s (encoder version
differences); the defect itself is deterministic everywhere.

What is NOT done: no wrong-FPS variant (out of scope for this cut), no
hands-in-view detector (windows are hand-picked from the IndEgo VIA
annotations, see DEFAULT_SOURCES), no face blurring, no service wiring.

Usage:
  python scripts/make_demo_clips.py [--out DIR] [--seed INT] [--list]
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
FOURCC = "mp4v"
DEFAULT_OUT = os.path.join("data_runtime", "demo_clips")
INDEGO_ROOT = "/home/parshu/projects/awing/data/indego"

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

# Variant name -> (defect code, parameters). Parameters are recorded verbatim
# in the manifest.
VARIANTS: list[tuple[str, str, dict]] = [
    ("good", "none", {}),
    ("blurred", "blur", {"op": "cv2.GaussianBlur", "ksize": [31, 31], "sigmaX": 0}),
    ("too_dark", "too_dark", {"op": "scale", "scale": 0.25}),
    ("overexposed", "overexposed", {"op": "gain_offset_clip", "gain": 1.7, "offset": 30.0}),
    ("frozen", "frozen", {"op": "repeat_first_frame"}),
    ("truncated", "truncated", {"op": "cut_bytes", "fraction": 0.5}),
    ("garbage", "garbage", {"op": "seeded_random_bytes", "nbytes": 65536}),
    ("reupload_duplicate", "none", {"op": "byte_identical_copy_of_good"}),
]


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


def read_window(path: str, start_s: float, duration_s: float) -> tuple[list, int, tuple[int, int]]:
    """Read a window of frames. Returns (frames, start_frame, (w, h))."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open source clip: {path}")
    try:
        src_fps = cap.get(cv2.CAP_PROP_FPS) or FPS
        start_frame = int(round(start_s * src_fps))
        want = int(round(duration_s * src_fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        frames: list = []
        for _ in range(want):
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            frames.append(frame)
        if not frames:
            raise RuntimeError(f"no frames read from {path} at {start_s}s")
        h, w = frames[0].shape[:2]
        return frames, start_frame, (w, h)
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
    if variant == "good":
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
) -> list[dict]:
    """Build all 8 variants for one source window. Returns manifest entries."""
    frames, start_frame, _size = read_window(source_abs, start_s, duration_s)
    entries: list[dict] = []
    good_name = f"{key}_good.mp4"
    good_path = os.path.join(out_dir, good_name)
    write_frames(apply_variant(frames, "good"), good_path)
    good_sha = sha256_file(good_path)
    good_bytes = open(good_path, "rb").read()

    garbage_rng = np.random.default_rng(seed + 1000 + source_index)
    garbage_bytes = garbage_rng.integers(0, 256, size=65536, dtype=np.uint8).tobytes()

    for variant, defect, params in VARIANTS:
        name = f"{key}_{variant}.mp4"
        path = os.path.join(out_dir, name)
        if variant in ("good", "blurred", "too_dark", "overexposed", "frozen"):
            write_frames(apply_variant(frames, variant), path)
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
        entries.append(
            {
                "file": name,
                "defect": defect,
                "variant": variant,
                "parameters": params,
                "source_clip": source_rel,
                "window": {
                    "start_s": start_s,
                    "duration_s": duration_s,
                    "start_frame": start_frame,
                    "n_frames": len(frames),
                },
                "sha256": sha256_file(path),
                "label": "SYNTHETIC-DEFECT",
            }
        )
    # Sanity: the duplicate really is a duplicate.
    by_name = {e["file"]: e for e in entries}
    assert by_name[f"{key}_reupload_duplicate.mp4"]["sha256"] == good_sha, "duplicate diverged"
    return entries


def plan_list() -> list[str]:
    names: list[str] = []
    for src in DEFAULT_SOURCES:
        for variant, _defect, _params in VARIANTS:
            names.append(f"{src['key']}_{variant}.mp4")
    return names


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build labelled demo clips with known defects.")
    ap.add_argument("--out", default=DEFAULT_OUT, help="output dir (default: %(default)s)")
    ap.add_argument("--seed", type=int, default=42, help="fixed RNG seed (default: %(default)s)")
    ap.add_argument("--list", action="store_true", help="print planned outputs and exit")
    args = ap.parse_args(argv)

    if args.list:
        for name in plan_list():
            print(name)
        return 0

    missing = [
        src for src in DEFAULT_SOURCES if not os.path.isfile(os.path.join(INDEGO_ROOT, src["relpath"]))
    ]
    if missing:
        print(
            "make_demo_clips: IndEgo source clips not found (inputs are read-only and "
            "expected under /home/parshu/projects/awing/data/indego):",
            file=sys.stderr,
        )
        for src in missing:
            print(f"  - {os.path.join(INDEGO_ROOT, src['relpath'])}", file=sys.stderr)
        print("Nothing was written. Mount or download the IndEgo clips and retry.", file=sys.stderr)
        return 2

    os.makedirs(args.out, exist_ok=True)
    all_entries: list[dict] = []
    for i, src in enumerate(DEFAULT_SOURCES):
        all_entries.extend(
            build_source_set(
                os.path.join(INDEGO_ROOT, src["relpath"]),
                src["relpath"],
                src["key"],
                src["start_s"],
                src["duration_s"],
                args.out,
                args.seed,
                i,
            )
        )
    manifest = {
        "generator": "scripts/make_demo_clips.py",
        "seed": args.seed,
        "encoder": {"backend": "cv2.VideoWriter", "fourcc": FOURCC, "fps": FPS, "size": "source (480x480 for IndEgo)"},
        "label_note": "SYNTHETIC-DEFECT labels are true by construction (defect injected by this script); MUST NOT be presented as human-labelled data.",
        "sources": DEFAULT_SOURCES,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "files": all_entries,
    }
    manifest_path = os.path.join(args.out, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"wrote {len(all_entries)} clips + manifest.json to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
