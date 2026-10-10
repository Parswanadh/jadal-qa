"""L0 signal extraction for a single segment (OpenCV decode + deterministic metrics).

Computes the exact L0 signals defined in docs/SPEC.md for one video segment:
decode with OpenCV (no torchcodec), fixed-rate frame sampling, blur (variance of
Laplacian), brightness/exposure, motion (Farneback optical flow on a fixed grid),
hands_visible_frac / hand_count_mode, faces_found / faces_blurred, a per-segment
perceptual hash (pHash) and near-duplicate lookup. All floats are rounded to
``config.float_decimals`` (4) before they are returned. Blurred output is written
to a SEPARATE file under ``config.blurred_dir`` with its own sha256; the original
segment is only ever read, never modified.

WHAT IS NOT DONE:
- mediapipe Hands / FaceDetector are NOT wired as the default. This environment
  (mediapipe 1.1.0) ships only the ``tasks`` API and no offline ``.task`` model
  files, and downloading models is disallowed (no network). If valid model paths
  are supplied via env vars ``JADAL_HAND_LANDMARKER_TASK`` and
  ``JADAL_FACE_DETECTOR_TASK`` the mediapipe adapters are used; otherwise a fully
  offline, deterministic OpenCV colour (YCrCb skin-region) detector is used as a
  PROXY. The colour proxy cannot read hand pose or verify faces; it partitions
  skin blobs by size/aspect into "face-like" and "hand-like" regions. Faces/hands
  counts are therefore approximate and are NOT a substitute for mediapipe.
- Only face-classified regions are blurred in the blurred output video (not hands).
- Blurring/encoding uses the mp4v codec (verified byte-deterministic on this
  OpenCV build in tests); no attempt at GOP-exact cutting or stream-copy.
"""
from __future__ import annotations

import hashlib
import os
from collections import Counter
from pathlib import Path
from typing import Iterable, Optional

import cv2
import imagehash
import numpy as np
from PIL import Image

from .config import DEFAULT, Config

# --- fixed, deterministic parameters -------------------------------------------------
SAMPLE_FRAMES = 16          # frames sampled per segment for the aggregated metrics
MOTION_GRID = 8             # Farneback flow is averaged on an (MOTION_GRID x MOTION_GRID) point grid
DARK_PIX_MAX = 20           # a gray pixel below this is "under exposed"
CLIP_PIX_MIN = 245          # a gray pixel at/above this is "clipped / over exposed"
# YCrCb skin range (Cr, Cb) -- classic Chai & Ngan bounds; fully offline.
SKIN_YCRCB_LO = (0, 133, 77)
SKIN_YCRCB_HI = (255, 177, 127)
MIN_SKIN_AREA_FRAC = 0.0015  # noise floor, a fraction of frame area
MIN_SKIN_AREA_PX = 120
FACE_AREA_FRAC = 0.02        # "face-like" blobs are relatively large
FACE_AREA_MIN_PX = 600
FACE_ASPECT_LO, FACE_ASPECT_HI = 0.6, 1.6
FACE_FILL_MIN = 0.5          # blob_area / bbox_area for a filled/roundish region
DEFAULT_PHASH_DUP_DISTANCE = 10  # Hamming distance (of 64) treated as a duplicate


# --- small deterministic helpers ------------------------------------------------------
def _round(x: float, config: Optional[Config]) -> float:
    d = config.float_decimals if config is not None else DEFAULT.float_decimals
    return round(float(x), d)


def _hamming(a: str, b: str) -> int:
    """Hamming distance between two hex bit-strings of equal width."""
    try:
        return (int(a, 16) ^ int(b, 16)).bit_count()
    except (ValueError, TypeError):
        n = min(len(a), len(b))
        return sum(a[i] != b[i] for i in range(n))


def _skin_mask(bgr: np.ndarray) -> np.ndarray:
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    mask = cv2.inRange(ycrcb, tuple(int(v) for v in SKIN_YCRCB_LO), tuple(int(v) for v in SKIN_YCRCB_HI))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    return mask


def _skin_blobs(bgr: np.ndarray, min_area: int) -> list[dict]:
    """Return connected skin regions (list of dicts) with area >= min_area."""
    h, w = bgr.shape[:2]
    mask = _skin_mask(bgr)
    n, _lab, stats, _cent = cv2.connectedComponentsWithStats(mask, 8)
    blobs = []
    for i in range(1, n):
        x, y, bw, bh, area = (int(v) for v in stats[i])
        if area < min_area:
            continue
        aspect = bw / bh if bh else 0.0
        fill = area / (bw * bh) if (bw and bh) else 0.0
        is_face = (
            FACE_ASPECT_LO <= aspect <= FACE_ASPECT_HI
            and fill >= FACE_FILL_MIN
            and area >= max(FACE_AREA_MIN_PX, int(FACE_AREA_FRAC * w * h))
        )
        blobs.append({"x": x, "y": y, "w": bw, "h": bh, "area": area, "face": is_face})
    return blobs


def _area_thresholds(bgr: np.ndarray) -> int:
    h, w = bgr.shape[:2]
    return max(MIN_SKIN_AREA_PX, int(MIN_SKIN_AREA_FRAC * w * h))


# --- frame IO -------------------------------------------------------------------------
def _open(path: str) -> Optional[cv2.VideoCapture]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None
    return cap


def _sample_indices(n_total: int, k: int) -> list[int]:
    """Deterministic, strictly increasing set of <=k even sample indices in [0, n_total-1]."""
    if n_total <= 0:
        return []
    if n_total <= k:
        return list(range(n_total))
    idx = {int(round(i * (n_total - 1) / (k - 1))) for i in range(k)}
    return sorted(idx)


def _read_sampled_frames(path: str, k: int) -> tuple[list[np.ndarray], list[int]]:
    """Read up to `k` evenly spaced frames in a single sequential pass."""
    cap = _open(path)
    if cap is None:
        return [], []
    try:
        n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        want = _sample_indices(n_total, k) if n_total > 0 else []
        if not want:  # unknown length: fall back to first k frames
            want = list(range(k))
        want_set = set(want)
        max_i = max(want)
        frames: dict[int, np.ndarray] = {}
        i = 0
        while i <= max_i:
            ok, frame = cap.read()
            if not ok:
                break
            if i in want_set:
                frames[i] = frame
            i += 1
        got_idx = sorted(frames)
        return [frames[j] for j in got_idx], got_idx
    finally:
        cap.release()


# --- metric primitives ----------------------------------------------------------------
def _laplacian_var(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _brightness(gray: np.ndarray) -> float:
    return float(gray.mean())


def _exposure_fracs(gray: np.ndarray) -> tuple[float, float]:
    flat = gray.ravel()
    return float((flat < DARK_PIX_MAX).mean()), float((flat >= CLIP_PIX_MIN).mean())


def _farneback_motion(gray_frames: list[np.ndarray], grid: int = MOTION_GRID) -> float:
    """Mean optical-flow magnitude between consecutive frames, on a fixed grid of points."""
    if len(gray_frames) < 2:
        return 0.0
    h, w = gray_frames[0].shape[:2]
    ys = np.linspace(0, h - 1, grid).round().astype(int)
    xs = np.linspace(0, w - 1, grid).round().astype(int)
    mags = []
    prev = gray_frames[0]
    for cur in gray_frames[1:]:
        flow = cv2.calcOpticalFlowFarneback(prev, cur, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
        mags.append(mag[ys[:, None], xs[None, :]])
        prev = cur
    stack = np.stack(mags)
    return float(stack.mean()) if stack.size else 0.0


def _phash_of_frame(bgr: np.ndarray) -> str:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return str(imagehash.phash(Image.fromarray(rgb)))


def _mode_smallest_on_tie(counts: Iterable[int]) -> int:
    c = Counter(int(x) for x in counts)
    if not c:
        return 0
    best_freq = max(c.values())
    return min(v for v, f in c.items() if f == best_freq)


# --- mediapipe adapters (only used if a .task model path is provided; NOT DONE offline) -
def _mp_available(model_env: str) -> bool:
    return bool(os.environ.get(model_env))


def _mp_face_boxes(bgr: np.ndarray) -> Optional[list[tuple[int, int, int, int]]]:
    model = os.environ.get("JADAL_FACE_DETECTOR_TASK")
    if not model:
        return None
    try:
        import mediapipe as mp
        from mediapipe.tasks.python import core as mp_core
        from mediapipe.tasks.python import vision as mp_vision
        h, w = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        det = mp_vision.FaceDetector.create_from_options(
            mp_vision.FaceDetectorOptions(
                base_options=mp_core.BaseOptions(model_asset_path=model),
                min_detection_confidence=0.5,
            )
        )
        res = det.detect(img)
        out = []
        for d in (res.detections or []):
            b = d.bounding_box
            out.append((int(b.origin_x), int(b.origin_y), int(b.width), int(b.height)))
        return out
    except Exception:
        return None


def _detect_faces(bgr: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Face-like boxes for a frame (mediapipe if available, else the colour proxy)."""
    boxes = _mp_face_boxes(bgr)
    if boxes is not None:
        return boxes
    min_area = _area_thresholds(bgr)
    return [(b["x"], b["y"], b["w"], b["h"]) for b in _skin_blobs(bgr, min_area) if b["face"]]


# --- public: blur_faces ----------------------------------------------------------------
def blur_faces(frame: np.ndarray) -> tuple[np.ndarray, int, int]:
    """Detect faces in one BGR frame, blur them, return (frame, n_found, n_blurred)."""
    if frame is None:
        return frame, 0, 0
    out = frame.copy()
    boxes = _detect_faces(out)
    n_found = len(boxes)
    n_blurred = 0
    h, w = out.shape[:2]
    for (x, y, bw, bh) in boxes:
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(w, x + bw), min(h, y + bh)
        if x1 <= x0 or y1 <= y0:
            continue
        roi = out[y0:y1, x0:x1]
        k = max(15, (max(bw, bh) // 2) | 1)
        out[y0:y1, x0:x1] = cv2.GaussianBlur(roi, (k, k), 0)
        n_blurred += 1
    return out, n_found, n_blurred


# --- public: phash + duplicate lookup --------------------------------------------------
def phash_segment(path: str) -> str:
    """64-bit perceptual hash (hex) of a segment's middle sampled frame; "" if undecodable."""
    frames, idx = _read_sampled_frames(path, SAMPLE_FRAMES)
    if not frames:
        return ""
    return _phash_of_frame(frames[len(frames) // 2])


def find_duplicate(phash: str, index: dict, threshold: int = DEFAULT_PHASH_DUP_DISTANCE) -> Optional[str]:
    """Return the segment_id whose stored phash is within `threshold` hamming of `phash`.

    ``index`` maps phash_hex -> segment_id. Ties break on sorted phash (first wins).
    Returns None when no near-duplicate exists.
    """
    if not phash or not index:
        return None
    best_id = None
    best_d = None
    for p in sorted(index.keys()):
        d = _hamming(phash, p)
        if best_d is None or d < best_d:
            best_d, best_id = d, index[p]
    return best_id if (best_d is not None and best_d <= threshold) else None


# --- public: blurred output video ------------------------------------------------------
def blur_and_hash_segment(path: str, config: Optional[Config] = None) -> dict:
    """Stream the segment, blur detected faces in every frame, write a SEPARATE video.

    Returns {"path", "sha256", "faces_found", "faces_blurred"}. The original at
    `path` is only read. Output lands under config.blurred_dir (deterministic name).
    """
    cfg = config if config is not None else DEFAULT
    out_dir = Path(cfg.blurred_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(path).stem
    out_path = out_dir / f"{stem}_blurred.mp4"
    cap = _open(path)
    if cap is None:
        return {"path": str(out_path), "sha256": "", "faces_found": 0, "faces_blurred": 0}
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
        ok, first = cap.read()
        if not ok:
            return {"path": str(out_path), "sha256": "", "faces_found": 0, "faces_blurred": 0}
        h, w = first.shape[:2]
        writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        if not writer.isOpened():
            return {"path": str(out_path), "sha256": "", "faces_found": 0, "faces_blurred": 0}
        total_found = total_blurred = 0
        ok, frame = True, first
        while ok:
            blurred, found, nblur = blur_faces(frame)
            total_found += found
            total_blurred += nblur
            writer.write(blurred)
            ok, frame = cap.read()
        writer.release()
    finally:
        cap.release()
    return {
        "path": str(out_path),
        "sha256": hashlib.sha256(out_path.read_bytes()).hexdigest(),
        "faces_found": total_found,
        "faces_blurred": total_blurred,
    }


# --- public: compute_signals -----------------------------------------------------------
def compute_signals(path: str, config: Optional[Config] = None, index: Optional[dict] = None) -> dict:
    """Exact L0 signals for one segment (see docs/SPEC.md). Deterministic.

    Aggregated metrics use <SAMPLE_FRAMES> evenly-spaced frames; signals are
    computed on the original frames only. If `index` (phash -> segment_id) is
    provided, `dup_of` is filled via near-duplicate lookup, else it is None.
    """
    cfg = config if config is not None else DEFAULT
    frames, _idx = _read_sampled_frames(path, SAMPLE_FRAMES)
    decodable = len(frames) > 0
    if not decodable:
        return {
            "decodable": False, "blur_var": _round(0.0, cfg), "brightness_mean": _round(0.0, cfg),
            "under_exposed_frac": _round(0.0, cfg), "over_exposed_frac": _round(0.0, cfg),
            "motion_mean": _round(0.0, cfg), "hands_visible_frac": _round(0.0, cfg),
            "hand_count_mode": 0, "faces_found": 0, "faces_blurred": 0,
            "phash": "", "dup_of": None,
        }

    grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
    blur_var = float(np.mean([_laplacian_var(g) for g in grays]))
    brightness = _brightness(grays[0])
    under = float(np.mean([_exposure_fracs(g)[0] for g in grays]))
    over = float(np.mean([_exposure_fracs(g)[1] for g in grays]))
    motion = _farneback_motion(grays, MOTION_GRID)

    faces_found = 0
    hand_counts = []
    for f in frames:
        min_area = _area_thresholds(f)
        blobs = _skin_blobs(f, min_area)
        faces_found += sum(1 for b in blobs if b["face"])
        n_hands = min(sum(1 for b in blobs if not b["face"]), 2)
        hand_counts.append(n_hands)
    hands_visible_frac = float(np.mean([c >= 1 for c in hand_counts])) if hand_counts else 0.0
    hand_count_mode = _mode_smallest_on_tie(hand_counts)

    phash = _phash_of_frame(frames[len(frames) // 2])
    dup_of = find_duplicate(phash, index) if index else None

    return {
        "decodable": True,
        "blur_var": _round(blur_var, cfg),
        "brightness_mean": _round(brightness, cfg),
        "under_exposed_frac": _round(under, cfg),
        "over_exposed_frac": _round(over, cfg),
        "motion_mean": _round(motion, cfg),
        "hands_visible_frac": _round(hands_visible_frac, cfg),
        "hand_count_mode": int(hand_count_mode),
        "faces_found": int(faces_found),
        "faces_blurred": int(faces_found),
        "phash": phash,
        "dup_of": dup_of,
    }
