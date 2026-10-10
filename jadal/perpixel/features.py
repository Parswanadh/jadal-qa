"""Unified whole-video feature row: arm B T1 (40) + GPU CPU-ref (26) + series desc (55).

Total F=121. extract_row(video_path, start_s, end_s) -> (row float64, names, series float32, meta)

extract_row(video_path, start_s, end_s) -> (row float64, names, series float32, meta)
All deterministic (fixed seeds n/a — no randomness; float64 reductions).

NOTE (fix round 2 / I2): G_dup_hash_mean here is the CPU DCT-hash
popcount-share. The GPU kernel column of the same name
(jadal/gpu/features.py, read-only / GPU worker's file) is a parity-noise
proxy, NOT the same quantity — any CPU/GPU parity claim for that column is
false until the GPU worker implements the real hash or renames it (referred,
not touched: read-only ownership). The frozen scorer/digest use the CPU row
only; no parity is claimed.
"""

from __future__ import annotations

import numpy as np

from jadal.gpu import features as gpu
from jadal.spectral import features as armB

from jadal.perpixel import series as S


def extract_row(video_path: str, start_s: float, end_s: float,
                return_frames: bool = False):
    frames90, fps90 = S.decode_160x90(video_path, start_s, end_s)
    Smat = S.per_frame_series(frames90)          # (T,8) float64
    desc = S.describe_series(Smat, fps90)        # ~60 scalars
    # GPU CPU-reference features on the same 160x90 stack (documented below)
    luma = (0.299 * frames90[:, :, :, 0] + 0.587 * frames90[:, :, :, 1]
            + 0.114 * frames90[:, :, :, 2])
    gfeat = gpu.cpu_window_features(luma, frames90, fps=fps90)
    # arm B T1 40 scalars (own 160x160 decode + ffprobe; read-only reuse)
    bfeat = armB.extract(video_path, start_s, end_s, tier="T1")
    names = ([f"B_{k}" for k in armB.ALL_FEATURES]
             + [f"G_{k}" for k in gpu.FEATURE_NAMES]
             + [f"P_{k}" for k in sorted(desc.keys())])
    row = np.array([bfeat[k] for k in armB.ALL_FEATURES]
                   + [gfeat[k] for k in gpu.FEATURE_NAMES]
                   + [desc[k] for k in sorted(desc.keys())], dtype=np.float64)
    meta = {"nframes90": int(frames90.shape[0]), "fps90": float(fps90)}
    if return_frames:
        return row, names, Smat.astype(np.float32), meta, frames90, fps90
    return row, names, Smat.astype(np.float32), meta


FEATURE_NAMES: list[str] | None = None
