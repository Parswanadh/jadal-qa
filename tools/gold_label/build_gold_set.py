"""Build the deterministic gold-label segment manifest (issue #26, Phase 3).

Selects ~150 ten-second segments from the real IndEgo clips listed in
/home/parshu/projects/awing/data/indego/CATALOG.md (480x480, 10 fps
head-cam). Clips are READ-ONLY: never copied into the repo, only
referenced by path. Selection is fully deterministic (no RNG):
per-clip counts by largest remainder on duration share, start times evenly
spaced across each clip. Writes eval/gold/segments.json.

Usage:
    PYTHONNOUSERSITE=1 /home/parshu/envs/jadal/bin/python \\
        tools/gold_label/build_gold_set.py [--n 150] [--seg-seconds 10]
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
OUT_PATH = ROOT / "eval" / "gold" / "segments.json"
INDEGO = Path("/home/parshu/projects/awing/data/indego")

# (clip_id, relative path under INDEGO) — from CATALOG.md, assembly + logistics.
CLIPS = [
    ("ecu_full", "1_Assembly/mp4_480/User_15_410_0411_2_480.mp4"),
    ("ecu_late", "1_Assembly/mp4_480/User_1_412_2010_3_480.mp4"),
    ("desk", "1_Assembly/03_Mechanical_Desk/mp4_480/"
             "User_1_User_15_411_2910_2_1_480.mp4"),
    ("pallet2", "3_Logistics_Organization/mp4_480/"
                "User_1_H_pal_transport2_Nar_480.mp4"),
    ("pallet", "3_Logistics_Organization/mp4_480/"
               "User_1_H_pal_transport_Nar_480.mp4"),
    ("organise", "3_Logistics_Organization/mp4_480/"
                 "User_1_2110_Organisation_480.mp4"),
    ("logistics4", "3_Logistics_Organization/mp4_480/"
                   "User_1_User_16_451_3101_Logistics_4_480.mp4"),
]


def probe_duration(path: Path) -> float:
    """Clip duration in seconds via OpenCV (read-only)."""
    import cv2
    cap = cv2.VideoCapture(str(path))
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        if frames <= 0:
            raise RuntimeError(f"cannot read frame count: {path}")
        return frames / fps
    finally:
        cap.release()


def apportion(durations: list[float], total: int) -> list[int]:
    """Largest-remainder split of `total` over durations (deterministic)."""
    whole = sum(durations)
    exact = [total * d / whole for d in durations]
    counts = [int(e) for e in exact]
    remainder = total - sum(counts)
    order = sorted(range(len(durations)),
                   key=lambda i: (exact[i] - counts[i], -i), reverse=True)
    for i in order[:remainder]:
        counts[i] += 1
    return counts


def build(n: int = 150, seg_seconds: float = 10.0) -> dict:
    durations = []
    for _, rel in CLIPS:
        path = INDEGO / rel
        if not path.is_file():
            raise FileNotFoundError(f"clip missing (read-only check): {path}")
        durations.append(probe_duration(path))
    counts = apportion(durations, n)
    segments = []
    idx = 0
    for (clip_id, rel), duration, count in zip(CLIPS, durations, counts):
        span = duration - seg_seconds
        for j in range(count):
            start = span / 2.0 if count == 1 else span * j / (count - 1)
            start = round(start, 1)
            segments.append({
                "segment_id": f"g{idx:03d}",
                "clip": clip_id,
                "path": str(INDEGO / rel),
                "start_s": start,
                "end_s": round(start + seg_seconds, 1),
            })
            idx += 1
    return {
        "n": len(segments),
        "seg_seconds": seg_seconds,
        "clips": [{"clip": c, "path": str(INDEGO / r), "duration_s": round(d, 1),
                   "n_segments": k}
                  for (c, r), d, k in zip(CLIPS, durations, counts)],
        "built_utc": datetime.now(timezone.utc).isoformat(),
        "segments": segments,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=150)
    parser.add_argument("--seg-seconds", type=float, default=10.0)
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    args = parser.parse_args(argv)
    manifest = build(args.n, args.seg_seconds)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
        fh.write("\n")
    for clip in manifest["clips"]:
        print(f"{clip['clip']}: duration={clip['duration_s']}s "
              f"n={clip['n_segments']}")
    print(f"wrote {args.out} ({manifest['n']} segments)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
