"""Incremental/windowed spectral features for live streaming.

SpectralStream is fed 2-second chunks (BGR uint8 frames at any resolution,
in order) and returns the running feature vector after each chunk, using
only past chunks. Memory is O(window): a ring buffer of small gray frames
(default last 100 frames = 10 s at 10 fps) plus scalar accumulators.
Matches batch extract() on the same frames up to the window boundary.
"""

from __future__ import annotations

import cv2
import numpy as np

from jadal.spectral.features import ALL_FEATURES, TIERS, FrameAccumulator


class SpectralStream:
    def __init__(self, tier: str = "T1", chunk_s: float = 2.0,
                 window_frames: int = 100, fps_hint: float = 10.0):
        self.size = TIERS[tier]["size"]
        self.tier = tier
        self.chunk_s = chunk_s
        self.fps_hint = fps_hint
        self.acc = FrameAccumulator(self.size, window=window_frames)
        self.n_chunks = 0

    def feed(self, bgr_frames) -> dict:
        """Feed one chunk (iterable of BGR uint8 arrays); return running
        feature dict (same keys as extract(), minus compressed-domain
        features which need an ffmpeg pass — set to 0.0 with key present
        except mv_per_frame which is absent until finalize)."""
        for f in bgr_frames:
            small = cv2.resize(f, (self.size, self.size),
                               interpolation=cv2.INTER_AREA)
            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB).astype(np.float64) / 255.0
            self.acc.update(rgb)
        self.n_chunks += 1
        feats = self.acc.running_features(fps_hint=self.fps_hint)
        for k in ("mv_per_frame", "i_size", "p_size_mean", "pb_size_ratio",
                  "frame_size_std", "p_count_frac"):
            feats.setdefault(k, 0.0)
        return {k: float(feats.get(k, 0.0)) for k in ALL_FEATURES}

    @property
    def n_frames(self) -> int:
        return self.acc.n
