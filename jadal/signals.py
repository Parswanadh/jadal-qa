"""L0 signal extraction for a single segment (OpenCV decode + deterministic metrics).

Computes the exact L0 signals defined in docs/SPEC.md for one video segment:
decode with OpenCV (no torchcodec), fixed-rate frame sampling, blur (variance of
Laplacian), brightness/exposure, motion (Farneback optical flow on a fixed grid),
hands_visible_frac / hand_count_mode (mediapipe HandLandmarker when its model
file is available, else a colour proxy), faces_found / faces_blurred (mediapipe
FaceDetector when its model file is available, else a colour proxy), a
per-segment perceptual hash (pHash) and near-duplicate lookup. All floats are
rounded to ``config.float_decimals`` (4) before they are returned. Blurred output
is written to a SEPARATE file under ``config.blurred_dir`` with its own sha256;
the original segment is only ever read, never modified.

Model files resolve from ``config.face_detector_task`` /
``config.hand_landmarker_task`` first; when the configured path is absent the
absolute directory in env var ``JADAL_MODELS_DIR`` is also searched
(``<JADAL_MODELS_DIR>/blaze_face_short_range.tflite``,
``<JADAL_MODELS_DIR>/hand_landmarker.task``); when neither exists the offline
OpenCV colour proxy is used. The active backend is inspectable via
``backend_info()`` (never added to the SPEC Signals dict). Detectors are created
once and cached at module level (CPU only, IMAGE running mode).

Definitions: ``faces_found`` is the total faces detected across the sampled
frames; ``faces_blurred`` is the total faces actually blurred across the same
sampled frames (via ``blur_faces``). ``blur_and_hash_segment`` instead reports
all-frame totals for the written artifact, which therefore differ from the
sampled-frame signals by construction.

WHAT IS NOT DONE:
- The written blurred file is a best-effort privacy aid, NOT a guarantee: the
  mediapipe face model can miss faces (profile views, occlusions, small faces)
  and the colour fallback only blurs skin-coloured blobs. Do not expose the
  blurred output as a privacy guarantee (e.g. on a dashboard).
- Only face-classified regions are blurred in the blurred output video (not hands).
- Blurring/encoding uses the mp4v codec (verified byte-deterministic on this
  OpenCV build in tests); no attempt at GOP-exact cutting or stream-copy.
- Mediapipe confidence thresholds are fixed module constants (config carries no
  L0 mediapipe threshold fields); hand counts are capped at 2 (num_hands=2).
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
# Mediapipe model filenames (resolved from config paths, else $JADAL_MODELS_DIR).
FACE_MODEL_NAME = "blaze_face_short_range.tflite"
HAND_MODEL_NAME = "hand_landmarker.task"
# Fixed mediapipe thresholds (config carries no L0 mediapipe threshold fields).
MP_FACE_MIN_DETECTION_CONFIDENCE = 0.5
MP_HAND_NUM_HANDS = 2
MP_HAND_MIN_DETECTION_CONFIDENCE = 0.5
MP_HAND_MIN_PRESENCE_CONFIDENCE = 0.5


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
    if k <= 1:
        return [0]
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


# --- detection backends: mediapipe (models) with colour-proxy fallback -------------------
def _resolve_model(config_path, filename: str) -> Optional[Path]:
    """Resolve a model file: config path first, then $JADAL_MODELS_DIR/<filename>."""
    p = Path(config_path)
    if p.is_file():
        return p
    d = os.environ.get("JADAL_MODELS_DIR")
    if d:
        q = Path(d) / filename
        if q.is_file():
            return q
    return None


def _face_model_path(config: Optional[Config]) -> Optional[Path]:
    cfg = config if config is not None else DEFAULT
    return _resolve_model(cfg.face_detector_task, FACE_MODEL_NAME)


def _hand_model_path(config: Optional[Config]) -> Optional[Path]:
    cfg = config if config is not None else DEFAULT
    return _resolve_model(cfg.hand_landmarker_task, HAND_MODEL_NAME)


_FACE_DETECTORS: dict[str, object] = {}
_HAND_LANDMARKERS: dict[str, object] = {}


def _get_face_detector(model_path: Path):
    """Create (once per path, cached) the mediapipe FaceDetector. Errors propagate."""
    import mediapipe as mp  # noqa: F401  (imported lazily; keeps module import light)
    from mediapipe.tasks.python import vision as mp_vision
    from mediapipe.tasks.python.core.base_options import BaseOptions

    key = str(model_path)
    det = _FACE_DETECTORS.get(key)
    if det is None:
        det = mp_vision.FaceDetector.create_from_options(
            mp_vision.FaceDetectorOptions(
                base_options=BaseOptions(model_asset_path=key),
                min_detection_confidence=MP_FACE_MIN_DETECTION_CONFIDENCE,
            )
        )
        _FACE_DETECTORS[key] = det
    return det


def _get_hand_landmarker(model_path: Path):
    """Create (once per path, cached) the mediapipe HandLandmarker. Errors propagate."""
    from mediapipe.tasks.python import vision as mp_vision
    from mediapipe.tasks.python.core.base_options import BaseOptions

    key = str(model_path)
    lm = _HAND_LANDMARKERS.get(key)
    if lm is None:
        lm = mp_vision.HandLandmarker.create_from_options(
            mp_vision.HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=key),
                num_hands=MP_HAND_NUM_HANDS,
                min_hand_detection_confidence=MP_HAND_MIN_DETECTION_CONFIDENCE,
                min_hand_presence_confidence=MP_HAND_MIN_PRESENCE_CONFIDENCE,
            )
        )
        _HAND_LANDMARKERS[key] = lm
    return lm


def _mp_image(bgr: np.ndarray):
    import mediapipe as mp

    rgb = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)


def _mp_face_boxes(bgr: np.ndarray, model_path: Optional[Path]) -> Optional[list[tuple[int, int, int, int]]]:
    """mediapipe face boxes, or None when no model file is available (proxy fallback).

    A configured-but-broken model raises instead of silently falling back.
    """
    if model_path is None:
        return None
    det = _get_face_detector(model_path)
    res = det.detect(_mp_image(bgr))
    out = []
    for d in (res.detections or []):
        b = d.bounding_box
        out.append((int(b.origin_x), int(b.origin_y), int(b.width), int(b.height)))
    return out


def _mp_hand_count(bgr: np.ndarray, model_path: Optional[Path]) -> Optional[int]:
    """mediapipe hand count (capped at num_hands), or None when no model is available.

    A configured-but-broken model raises instead of silently falling back.
    """
    if model_path is None:
        return None
    lm = _get_hand_landmarker(model_path)
    res = lm.detect(_mp_image(bgr))
    return min(len(res.hand_landmarks or []), MP_HAND_NUM_HANDS)


def _proxy_face_boxes(bgr: np.ndarray) -> list[tuple[int, int, int, int]]:
    min_area = _area_thresholds(bgr)
    return [(b["x"], b["y"], b["w"], b["h"]) for b in _skin_blobs(bgr, min_area) if b["face"]]


def _proxy_hand_count(bgr: np.ndarray) -> int:
    min_area = _area_thresholds(bgr)
    return min(sum(1 for b in _skin_blobs(bgr, min_area) if not b["face"]), 2)


def _detect_faces(bgr: np.ndarray, config: Optional[Config] = None) -> list[tuple[int, int, int, int]]:
    """Face-like boxes for a frame (mediapipe when its model resolves, else colour proxy)."""
    boxes = _mp_face_boxes(bgr, _face_model_path(config))
    return boxes if boxes is not None else _proxy_face_boxes(bgr)


def _count_hands(bgr: np.ndarray, config: Optional[Config] = None) -> int:
    """Hand count for a frame (mediapipe when its model resolves, else colour proxy)."""
    n = _mp_hand_count(bgr, _hand_model_path(config))
    return n if n is not None else _proxy_hand_count(bgr)


def _sha256_path(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def backend_info(config: Optional[Config] = None) -> dict:
    """Inspectable active detection backend (never added to the SPEC Signals dict)."""
    fm = _face_model_path(config)
    hm = _hand_model_path(config)
    return {
        "faces": "mediapipe" if fm is not None else "proxy",
        "hands": "mediapipe" if hm is not None else "proxy",
        "face_model": str(fm) if fm is not None else None,
        "hand_model": str(hm) if hm is not None else None,
        "face_model_sha256": _sha256_path(fm) if fm is not None else None,
        "hand_model_sha256": _sha256_path(hm) if hm is not None else None,
    }


# --- public: blur_faces ----------------------------------------------------------------
def blur_faces(frame: np.ndarray, config: Optional[Config] = None) -> tuple[np.ndarray, int, int]:
    """Detect faces in one BGR frame, blur them, return (frame, n_found, n_blurred)."""
    if frame is None:
        return frame, 0, 0
    out = frame.copy()
    boxes = _detect_faces(out, config)
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
    Returns None when no near-duplicate exists. The query's own entry is NOT
    excluded: if the pipeline inserts a segment's phash before lookup, the segment
    matches itself. Two segments sharing one phash overwrite each other in the
    index; key the index by segment_id upstream if that matters.
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

    Returns {"path", "sha256", "faces_found", "faces_blurred"} counted across ALL
    frames (unlike compute_signals, which uses sampled frames). The original at
    `path` is only read. Output lands under config.blurred_dir (deterministic name).
    Best-effort privacy aid, NOT a guarantee (see module docstring).
    """
    cfg = config if config is not None else DEFAULT
    out_dir = Path(cfg.blurred_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = "".join(c if (c.isalnum() or c in "-_") else "_" for c in Path(path).stem)
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
            blurred, found, nblur = blur_faces(frame, cfg)
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
    computed on the original frames only. ``faces_found`` is the total faces
    detected across the sampled frames and ``faces_blurred`` the total actually
    blurred there (via ``blur_faces``). If `index` (phash -> segment_id) is
    provided, `dup_of` is filled via near-duplicate lookup, else it is None.
    Hands use the mediapipe HandLandmarker when its model resolves, else the
    colour proxy (see ``backend_info``).
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
    brightness = float(np.mean([_brightness(g) for g in grays]))
    under = float(np.mean([_exposure_fracs(g)[0] for g in grays]))
    over = float(np.mean([_exposure_fracs(g)[1] for g in grays]))
    motion = _farneback_motion(grays, MOTION_GRID)

    faces_found = faces_blurred = 0
    hand_counts = []
    for f in frames:
        _blurred, found, nblur = blur_faces(f, cfg)
        faces_found += found
        faces_blurred += nblur
        hand_counts.append(_count_hands(f, cfg))
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
        "faces_blurred": int(faces_blurred),
        "phash": phash,
        "dup_of": dup_of,
    }
