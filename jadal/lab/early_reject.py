"""Jadal lab: streaming early-reject gate for motion/exposure collapse.

Breakthrough-lab direction 3 prototype, narrowed to its honest claim:
one-sided sequential tests on a noise-normalised frame-difference series
catch *physical* collapse (dark / frozen / lens-covered) within ~1 s, but
are near-blind to semantic bad reasons (HANDS_NOT_VISIBLE, NO_ACTIVE_WORK).

Evidence: eval/lab/seqchange/REPORT.md — LOCO over 150 gold windows
(detection 0.24 / false-alarm 0.45 pooled, CIs wide); synthetic
black/frozen/darken caught 6/6 at <= 1.4 s; real TOO_DARK 6/6 (CUSUM).

Scope contract (read before changing defaults):
- This gate REJECTS only on collapse evidence. It never ACCEPTS a window;
  no-fire means "undecided", never "good".
- DEFAULT_MU0 / DEFAULT_SIG0 are fitted on IndEgo 480p gold-train windows
  (median MAD 10.20 DN, robust scale 7.54 DN). They are NOT certified and
  MUST be recalibrated per deployment (see honest_bounds.required_n).
- Nominal error rates (SPRT alpha/beta, CUSUM ARL0) assume i.i.d.
  sub-Gaussian increments. Video frames violate this; observed false-alarm
  was 3-9x nominal. Treat nominal rates as procedure properties, not
  deployment promises.

Deterministic: fixed arithmetic, no RNG, no wall-clock. Streaming: O(1)
state, update() per frame/chunk with past data only. Hashable: verdict()
carries sha256 over canonical JSON of (inputs hash, params, outcome).
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass

# Fitted on IndEgo gold-train good windows (64x64 luma, 5 fps MAD series).
# Deployment MUST recalibrate; see module docstring.
DEFAULT_MU0 = 10.20
DEFAULT_SIG0 = 7.54

# One-sided CUSUM tuned to a 1-sigma downward shift (Page 1954; Siegmund ARL).
DEFAULT_CUSUM_K = 0.5
DEFAULT_CUSUM_H = 5.0

# Gaussian SPRT for mean shift delta at level (alpha, beta) (Wald 1947).
DEFAULT_SPRT_DELTA = 1.0
DEFAULT_SPRT_ALPHA = 0.05
DEFAULT_SPRT_BETA = 0.10

# Dark-frame fast path: fraction of pixels with luma < 16.
DEFAULT_DARK_FRAC = 0.85
DEFAULT_DARK_MIN_FRAMES = 5


def _canon(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def evidence_hash(payload: dict) -> str:
    return hashlib.sha256(_canon(payload).encode()).hexdigest()


@dataclass
class CollapseGate:
    """One-sided CUSUM + dark-fraction fast path over a MAD series.

    Feed per-frame ``mad`` (mean |Y_t - Y_{t-1}|, DN) and ``dark_frac``
    via update(); check ``fired``. First-firing frame index -> seconds
    via fps. Call reset() per window/chunk-stream.
    """

    mu0: float = DEFAULT_MU0
    sig0: float = DEFAULT_SIG0
    cusum_k: float = DEFAULT_CUSUM_K
    cusum_h: float = DEFAULT_CUSUM_H
    dark_frac: float = DEFAULT_DARK_FRAC
    dark_min_frames: int = DEFAULT_DARK_MIN_FRAMES
    fps: float = 5.0

    def __post_init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.s = 0.0
        self.n = 0
        self.dark_run = 0
        self.fired = False
        self.fire_frame: int | None = None
        self.fire_cause: str | None = None

    def _z(self, mad: float) -> float:
        return (self.mu0 - mad) / self.sig0

    def update(self, mad: float, dark_frac: float = 0.0) -> bool:
        """Feed one frame. Returns True on the firing frame (sticky after)."""
        if self.fired:
            return True
        z = self._z(float(mad))
        self.s = max(0.0, self.s + z - self.cusum_k)
        if float(dark_frac) >= self.dark_frac:
            self.dark_run += 1
        else:
            self.dark_run = 0
        if self.s > self.cusum_h:
            self.fired = True
            self.fire_cause = "cusum"
            self.fire_frame = self.n
        elif self.dark_run >= self.dark_min_frames:
            self.fired = True
            self.fire_cause = "dark"
            self.fire_frame = self.n
        self.n += 1
        return self.fired

    @property
    def fire_s(self) -> float | None:
        if self.fire_frame is None:
            return None
        return self.fire_frame / self.fps

    def nominal_arl0_frames(self) -> float:
        """Siegmund approximation for CUSUM ARL0 (nominal, i.i.d. assumed)."""
        k, h = self.cusum_k, self.cusum_h
        return (math.exp(2 * k * h) - 2 * k * h - 1) / (2 * k * k)

    def verdict(self, window_id: str) -> dict:
        payload = {
            "window": window_id,
            "fired": self.fired,
            "cause": self.fire_cause,
            "fire_frame": self.fire_frame,
            "fire_s": self.fire_s,
            "frames_seen": self.n,
            "params": {
                "mu0": self.mu0,
                "sig0": self.sig0,
                "k": self.cusum_k,
                "h": self.cusum_h,
                "dark_frac": self.dark_frac,
                "dark_min_frames": self.dark_min_frames,
            },
        }
        payload["hash"] = evidence_hash(payload)
        return payload


@dataclass
class SprtGate:
    """One-sided Gaussian SPRT for a downward mean shift (Wald boundaries).

    Same streaming contract as CollapseGate. H0: z ~ N(0,1) (healthy
    motion); H1: z ~ N(delta,1) (collapse). Fires when cumulative LLR >= B.
    Lower-boundary crossing (accept) is recorded but callers should treat
    no-fire as undecided per this module's scope contract.
    """

    delta: float = DEFAULT_SPRT_DELTA
    alpha: float = DEFAULT_SPRT_ALPHA
    beta: float = DEFAULT_SPRT_BETA
    mu0: float = DEFAULT_MU0
    sig0: float = DEFAULT_SIG0
    fps: float = 5.0

    def __post_init__(self) -> None:
        self.a = math.log(self.beta / (1 - self.alpha))
        self.b = math.log((1 - self.beta) / self.alpha)
        self.reset()

    def reset(self) -> None:
        self.llr = 0.0
        self.n = 0
        self.fired = False
        self.fire_frame: int | None = None

    def update(self, mad: float) -> bool:
        if self.fired:
            return True
        z = (self.mu0 - float(mad)) / self.sig0
        self.llr += self.delta * z - self.delta * self.delta / 2.0
        if self.llr >= self.b:
            self.fired = True
            self.fire_frame = self.n
        self.n += 1
        return self.fired
