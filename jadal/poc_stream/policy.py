"""Adopted PoC streaming policy (claim LB2) -- single source of truth.

Everything the PoC scorer path needs to obey the adopted settings lives in
``ADOPTED`` below.  Nothing here trains or reads a GPU: the module is pure
CPU/numpy and deterministic, and it only *loads* frozen model constants that
were fitted earlier (``caches/ensemble/ensemble_v1.json``).

Adopted settings and their evidence (all on the 150 human-labelled IndEgo
windows, leave-one-clip-out unless stated):

- every-2nd window sampling grid                      E3.01 / E3.05 (adopted_policy="every-2")
- 4 s early exit inside a window (2 fps EG2 prefix)   E3.03  (8 frames @ 0.5 s -> AUROC 0.860 vs 0.848 full)
- EmbeddingGemma-2 2 fps view                         E1.02  (2 fps AUROC 0.848 [0.73, 0.91])
- 26-feature per-pixel logistic view                  E1.01  (10 fps AUROC 0.809 [0.69, 0.89])
- equal-weight probability fusion, thresholds 0.5/0.4  caches/ensemble/ensemble_v1.json (frozen, pre-registered)

Cost (E3.04, per 4.404 h Egocentric-10K corpus; arithmetic, decode-bound upper bound):
every-2 alone -> x1.94; every-2 + 4 s exit -> x4.86 (~0.207 of footage, ~0.91 h).

DESCRIPTIVE ONLY.  The policy is a *working policy on IndEgo* -- the E3.05
selection itself says "not validated" (no held-out clips, no OOD validation).
Nothing in this module supports an out-of-domain or accuracy claim.
"""

from __future__ import annotations

import json
import math
import os

import numpy as np

# --------------------------------------------------------------------------
# Adopted constants -- the one place to change them.
# --------------------------------------------------------------------------
ADOPTED = {
    # --- sampling grid (E3.01/E3.05) ---
    "sample_every_n_windows": 2,
    "sample_offset": 0,  # eval/exp_temporal/run_e3.py: pol_every_k = range(0, n, k)
    # --- early exit inside a window (E3.03) ---
    "early_exit_s": 4.0,
    "eg2_frame_step_s": 0.5,  # E1.02 / E3.03
    "pixel_prefix_step_s": 2.0,  # features.json: prefix[] at 2,4,6,8,10 s
    # --- views (E1.01 / E1.02) ---
    "pixel_view": "pixel26",
    "eg2_view": "eg2_2fps",
    "views": ("pixel26", "eg2_2fps"),
    # --- fusion / verdicts (frozen ensemble_v1) ---
    "fusion": "equal_weight_probability_average",
    "accept_prob": 0.5,  # usable_prob > 0.5 -> accept
    "reject_prob": 0.4,  # usable_prob < 0.4 -> reject, else uncertain
    # --- clip-level decision (E3.01/E3.05) ---
    "clip_bad_fraction_threshold": 0.30,
    # --- cost ratios (E3.04, reported, not re-derived) ---
    "cost": {
        "every_2_speedup": 1.94,
        "every_2_4s_speedup": 4.86,
        "every_2_4s_total_frac": 0.2059,
        "every_2_4s_proc_hours_per_4_404h": 0.907,
        "window_seconds": 10.0,
    },
    # --- evidence status ---
    "descriptive_only": True,
    "status": "working policy on IndEgo, not validated",
    "provenance": {
        "sampling_grid": "E3.01/E3.05",
        "early_exit": "E3.03",
        "pixel_view": "E1.01",
        "eg2_view": "E1.02",
        "fusion_and_thresholds": "configs/frozen/ensemble_v1.prereg.json (sha256 4da1068a…)",
        "cost": "E3.04",
    },
}

DEFAULT_BUNDLE = os.environ.get("JADAL_BUNDLE", "/workspace/jadal_pod_bundle")

# --- pre-registered frozen model (fail closed) -----------------------------
# The copies of ensemble_v1.json that lived on disk were path-remapped AFTER the
# freeze (embed_command / pixel_command / embed_model_dir) and no longer hash to
# the recorded value.  The byte-exact recovered file is
# configs/frozen/ensemble_v1.prereg.json (git blob 0ede3b83... @ b21e3af),
# recovered by LB4 (PR #97).  We verify its sha256 on every load and refuse to
# fall back to the remapped configs/ensemble_v1.json.
FROZEN_PREREG_SHA256 = "4da1068a78d5d15021f4e167d6aa24fd2e0bb08aa88f83c83c2f167cc086fe31"
FROZEN_PREREG_REL = os.path.join("configs", "frozen", "ensemble_v1.prereg.json")
LEGACY_REMAPPED_REL = os.path.join("configs", "ensemble_v1.json")  # NOT pre-registered
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# label/threshold constants of the frozen artifact (read back and asserted)
FROZEN_EXPECTED = {
    "name": "ensemble_v1",
    "threshold": 0.5,
    "fusion_kind": "equal_weight_probability_average",
}


def sampled_indices(n_windows: int, every: int | None = None,
                    offset: int | None = None) -> list[int]:
    """Deterministic every-k-th window grid: range(offset, n, every)."""
    every = ADOPTED["sample_every_n_windows"] if every is None else int(every)
    offset = ADOPTED["sample_offset"] if offset is None else int(offset)
    if every < 1:
        raise ValueError("every must be >= 1")
    if not 0 <= offset < every:
        raise ValueError("offset must satisfy 0 <= offset < every")
    return list(range(offset, int(n_windows), every))


def eg2_prefix_frames(exit_s: float | None = None) -> int:
    """Number of 2 fps EG2 frames observed by ``exit_s`` seconds.

    E3.03 curve: 4 frames -> 2.0 s, 8 -> 4.0 s, 12 -> 6.0 s, 16 -> 8.0 s.
    Frame times are 0, 0.5, ..., so m frames cover (m-1)*0.5 s.
    """
    exit_s = ADOPTED["early_exit_s"] if exit_s is None else float(exit_s)
    return int(math.ceil(exit_s / ADOPTED["eg2_frame_step_s"]))


def pixel_prefix_index(exit_s: float | None = None) -> int:
    """Index into features.json ``prefix`` list for the observed seconds.

    ``prefix`` holds 20,40,60,80,100 decoded frames (2,4,6,8,10 s at 10 fps);
    index 4 is the full window.
    """
    exit_s = ADOPTED["early_exit_s"] if exit_s is None else float(exit_s)
    if exit_s >= 10.0:
        return 4
    return max(0, int(exit_s / ADOPTED["pixel_prefix_step_s"]) - 1)


def eg2_prefix_mean(F: "np.ndarray", counts=None, exit_s: float | None = None) -> "np.ndarray":
    """L2-normalised mean of the first ``eg2_prefix_frames`` valid frames.

    ``F`` is (n_windows, max_frames, dim) from caches/frames10k/segments.npz;
    rows past ``counts[i]`` are NaN padding and are never used.
    """
    F = np.asarray(F, dtype=np.float32)
    m = eg2_prefix_frames(exit_s)
    n = F.shape[0]
    out = np.empty((n, F.shape[2]), dtype=np.float32)
    for i in range(n):
        c = int(counts[i]) if counts is not None else F.shape[1]
        k = max(1, min(m, c))
        v = F[i, :k, :].mean(axis=0)
        nrm = float(np.linalg.norm(v))
        out[i] = v / nrm if nrm > 0 else v
    return out


def sigmoid(z):
    z = np.asarray(z, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-z))


def predict_head(X, head) -> "np.ndarray":
    """Apply a frozen linear head: sigmoid(((X-mean)/scale) @ coef + intercept)."""
    X = np.asarray(X, dtype=np.float64)
    mean = np.asarray(head["mean"], dtype=np.float64)
    scale = np.asarray(head["scale"], dtype=np.float64)
    coef = np.asarray(head["coef"], dtype=np.float64)
    return sigmoid(((X - mean) / scale) @ coef + float(head["intercept"]))


def fuse_probabilities(p_pixel, p_eg2):
    """Equal-weight probability average (frozen ensemble fusion)."""
    return 0.5 * np.asarray(p_pixel, dtype=np.float64) + 0.5 * np.asarray(p_eg2, dtype=np.float64)


def window_verdict(usable_prob: float) -> str:
    """accept / reject / uncertain per the frozen ensemble verdict_rule."""
    p = float(usable_prob)
    if p > ADOPTED["accept_prob"]:
        return "accept"
    if p < ADOPTED["reject_prob"]:
        return "reject"
    return "uncertain"


def clip_verdict(sampled_bad: list[bool], threshold: float | None = None) -> str:
    """Clip verdict from the sampled windows' bad fraction (E3.01/E3.05)."""
    threshold = ADOPTED["clip_bad_fraction_threshold"] if threshold is None else float(threshold)
    if not sampled_bad:
        return "uncertain"
    frac = float(np.mean([bool(b) for b in sampled_bad]))
    return "reject" if frac >= threshold else "accept"


def sampled_seconds_per_window(exit_s: float | None = None) -> float:
    """Footage seconds actually observed per sampled window under early exit."""
    return ADOPTED["early_exit_s"] if exit_s is None else float(exit_s)


def cost_ratio(key: str = "every_2_4s_speedup") -> float:
    """Adopted cost ratio from E3.04 (arithmetic, not re-measured here)."""
    return float(ADOPTED["cost"][key])


def sha256_file(path: str) -> str:
    """Hex sha256 of a file (used to pin the pre-registered model)."""
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_frozen_prereg(bundle: str | None = None, path: str | None = None) -> str:
    """Locate the byte-exact pre-registered config; never fall back to the remapped copy.

    Order: explicit ``path`` -> repo ``configs/frozen/ensemble_v1.prereg.json``
    -> bundle ``configs/frozen/ensemble_v1.prereg.json``.  If none exists we
    raise instead of silently loading the path-remapped ``configs/ensemble_v1.json``.
    """
    if path is not None:
        if not os.path.exists(path):
            raise FileNotFoundError(f"frozen ensemble not found: {path}")
        return path
    bundle = DEFAULT_BUNDLE if bundle is None else bundle
    for cand in (os.path.join(_REPO_ROOT, FROZEN_PREREG_REL),
                 os.path.join(bundle, FROZEN_PREREG_REL)):
        if os.path.exists(cand):
            return cand
    raise FileNotFoundError(
        "pre-registered frozen ensemble not found in "
        f"{os.path.join(_REPO_ROOT, FROZEN_PREREG_REL)!r} or "
        f"{os.path.join(bundle, FROZEN_PREREG_REL)!r}; refusing to fall back to "
        f"the path-remapped {LEGACY_REMAPPED_REL!r} (wrong failure direction for a "
        "frozen model). Recover it from b/ood-ready (PR #97) or pass path=.")


def load_frozen_ensemble(bundle: str | None = None, path: str | None = None,
                         expected_sha256: str | None = FROZEN_PREREG_SHA256) -> dict:
    """Load the pre-registered frozen ensemble constants, verifying sha256 first.

    Fails closed: a hash mismatch or a missing file raises.  Pass
    ``expected_sha256=None`` only for an explicit non-frozen override.
    """
    resolved = resolve_frozen_prereg(bundle, path)
    if expected_sha256 is not None:
        digest = sha256_file(resolved)
        if digest != expected_sha256:
            raise ValueError(
                f"frozen ensemble sha256 mismatch: {digest} != {expected_sha256} "
                f"({resolved}); refusing to use a file that is not the pre-registered model")
    with open(resolved, "r", encoding="utf-8") as fh:
        ens = json.load(fh)
    if ens.get("name") != FROZEN_EXPECTED["name"]:
        raise ValueError(f"frozen ensemble mismatch: name={ens.get('name')!r}")
    if float(ens.get("threshold")) != FROZEN_EXPECTED["threshold"]:
        raise ValueError(f"frozen ensemble mismatch: threshold={ens.get('threshold')!r}")
    if ens.get("fusion", {}).get("kind") != FROZEN_EXPECTED["fusion_kind"]:
        raise ValueError(f"frozen ensemble mismatch: fusion={ens.get('fusion')!r}")
    if list(ens.get("fusion", {}).get("weights", [])) != [0.5, 0.5]:
        raise ValueError(f"frozen ensemble mismatch: fusion weights={ens.get('fusion')!r}")
    for view in ("eg2", "pixel"):
        for field in ("mean", "scale", "coef", "intercept"):
            if field not in ens["views"][view]:
                raise ValueError(f"frozen ensemble view {view} missing {field}")
    return ens


def policy_fingerprint() -> dict:
    """The adopted policy, JSON-serialisable, for results provenance."""
    return {
        "sample_every_n_windows": ADOPTED["sample_every_n_windows"],
        "sample_offset": ADOPTED["sample_offset"],
        "early_exit_s": ADOPTED["early_exit_s"],
        "eg2_prefix_frames": eg2_prefix_frames(),
        "pixel_prefix_index": pixel_prefix_index(),
        "views": list(ADOPTED["views"]),
        "fusion": ADOPTED["fusion"],
        "accept_prob": ADOPTED["accept_prob"],
        "reject_prob": ADOPTED["reject_prob"],
        "clip_bad_fraction_threshold": ADOPTED["clip_bad_fraction_threshold"],
        "descriptive_only": ADOPTED["descriptive_only"],
        "status": ADOPTED["status"],
        "frozen_ensemble_sha256": FROZEN_PREREG_SHA256,
        "frozen_ensemble_path": FROZEN_PREREG_REL,
        "cost": dict(ADOPTED["cost"]),
        "provenance": dict(ADOPTED["provenance"]),
    }
