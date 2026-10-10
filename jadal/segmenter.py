"""Deterministic clip -> fixed-length segment splitter (OpenCV decode / encode).

Splits a video into consecutive, fixed-duration segments by re-encoding the
frames in each window with OpenCV. Segment files are written to `out_dir` with
deterministic names and their sha256 is recorded in each Segment dict
(docs/SPEC.md). The source clip is only read.

WHAT IS NOT DONE:
- Segments are cut at exact frame indices by re-encoding (mp4v codec), which is
  byte-deterministic on this OpenCV build (verified in tests). GOP/keyframe-exact
  cutting or ffmpeg stream-copy is not attempted, so the segment bytes differ
  from the source bytes (only the sha256 of the written segment is stored).
- No audio is carried into segments (video only).
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional

import cv2

from .config import DEFAULT, Config

_SEG_DECIMALS = 4


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def split_clip(
    path: str,
    seg_seconds: float,
    out_dir: Optional[str] = None,
    worker_id: str = "",
    site_id: str = "",
    source_clip: Optional[str] = None,
    config: Optional[Config] = None,
) -> list[dict]:
    """Split `path` into fixed `seg_seconds` segments; return a list of Segment dicts.

    Segments are written to `out_dir` (defaults to config.segments_dir) as
    ``{stem}_seg{NNNN}.mp4``. Each Segment carries its own sha256. Returns an
    empty list when the clip is undecodable.
    """
    cfg = config if config is not None else DEFAULT
    if out_dir is None:
        out_dir = cfg.segments_dir
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    src = Path(path)
    stem = "".join(c if (c.isalnum() or c in "-_") else "_" for c in src.stem)
    source = str(source_clip) if source_clip is not None else str(path)

    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        cap.release()
        return []
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if fps <= 0:
            fps = 30.0
        seg_frames = max(1, int(round(seg_seconds * fps)))

        ok, first = cap.read()
        if not ok:
            return []
        h, w = first.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        writers: dict[int, cv2.VideoWriter] = {}
        counts: dict[int, int] = {}
        seg_for: dict[int, int] = {}

        def _writer(seg: int) -> cv2.VideoWriter:
            if seg not in writers:
                vw = cv2.VideoWriter(str(out_path / f"{stem}_seg{seg:04d}.mp4"), fourcc, fps, (w, h))
                writers[seg] = vw
            return writers[seg]

        i = 0
        frame = first
        while True:
            seg = i // seg_frames
            _writer(seg).write(frame)
            counts[seg] = counts.get(seg, 0) + 1
            seg_for[i] = seg
            ok, frame = cap.read()
            i += 1
            if not ok:
                break
        for vw in writers.values():
            vw.release()

        segments: list[dict] = []
        for seg in sorted(counts):
            n_frames = counts[seg]
            first_frame = seg * seg_frames
            seg_file = out_path / f"{stem}_seg{seg:04d}.mp4"
            if not seg_file.exists():
                continue
            start_s = round(first_frame / fps, _SEG_DECIMALS)
            end_s = round((first_frame + n_frames) / fps, _SEG_DECIMALS)
            segments.append(
                {
                    "segment_id": f"{stem}_seg{seg:04d}",
                    "source_clip": source,
                    "worker_id": worker_id,
                    "site_id": site_id,
                    "start_s": start_s,
                    "end_s": end_s,
                    "path": str(seg_file),
                    "sha256": _sha256_file(seg_file),
                }
            )
        return segments
    finally:
        cap.release()
