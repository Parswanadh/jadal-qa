"""GRID.09: brightness-calibrated sharpness (blur) statistic for the S0 probe.

The shipped S0 probe measures brightness (p95, mean, p5), spatial variance and
temporal energy, but NO sharpness.  A strongly blurred clip therefore reaches S1
unfiltered.  This module adds the one sharpness statistic both input paths use:

    lapvar_cal(grey) = Laplacian variance of ``grey`` / (mean luma + eps)

Raw Laplacian variance scales with contrast, so a dark-but-sharp frame and a
bright blurred frame are otherwise confusable; dividing by the mean luma removes
the first-order contrast dependence.  The GRID.09 brightness controls test
whether that calibration (not raw contrast) is doing the work.

One definition, two call sites
------------------------------
* ``jadal.mosaic.features._lapvar_cal`` (per tile on the 224 px canvas) delegates
  here;
* ``scripts/poc_run.py::s0_from_luma`` (per frame of the decoded 160x90 working
  luma, with the 2x2-boxed 80x45 value kept as ``lapvar_cal_45``) uses
  :func:`probe_lapvar_cal`.

Resolution matters (measured, not assumed)
------------------------------------------
The first candidate was the 80x45 grey probe the shipped S0 statistics use, and
it is a weak blur detector: a sigma=3 Gaussian blur at 480p survives the
downsample to 160x90 but is largely removed by the extra 2x2 box to 80x45.  On
the GRID.09 calibration split the calibrated 80x45 statistic scored 0.796 AUROC
against 0.981 for the 160x90 statistic (both at their own clean-only thresholds;
see ``eval/results/exp_grid/GRID.09.json``).  The production rule therefore uses
the 160x90 statistic and reports the 80x45 candidate alongside it.

The numeric value is NOT comparable across the two input paths (a 224 px tile
keeps far more high-frequency energy than a 160x90 probe), so the mosaic and the
full-pipeline operating points are calibrated separately, each on clean data
only.  Neither is inherited from the other.

Determinism: pure float64 numpy, no sampling, no randomness.
"""
from __future__ import annotations

import numpy as np

#: epsilon added to the mean luma; keeps a fully black image finite (0/eps = 0)
EPS = 1e-6

#: SHIPPED DEFAULT for the BLUR rule's operating point.
#:
#: 0.0 = DISABLED.  Adding BLUR to ``cascade.S0_RULES`` changes the shipped
#: verdict path (a blurred window that used to reach S1 is rejected at S0), and
#: the stream owner's wrap-up instruction is that a verdict-changing rule ships
#: OFF by default.  With 0.0 the comparison ``lapvar_cal < 0.0`` can never fire,
#: so the shipped five rules decide EXACTLY as before; the probe still reports
#: ``lapvar_cal`` (observability only, no verdict change).
BLUR_LAPVAR_CAL_MIN = 0.0

#: The CALIBRATED operating point this branch measured, frozen 2026-10-18 from
#: eval/results/exp_grid/grid09_calibration.json.  Rule = the largest threshold
#: whose clean-only false-reject rate on the CALIBRATION split is <= 2% (the 2nd
#: percentile of the clean calibration-split distribution, n_calib = 106 windows /
#: 26 groups; even-index IndEgo clips and OOD-160 factories are CALIB, odd are
#: EVAL).  Flip ``BLUR_LAPVAR_CAL_MIN`` to this value to enable the rule; every
#: number in eval/results/exp_grid/GRID.09.json was measured at THIS threshold.
#: The 80x45 candidate would have been 0.04766704819122121.
BLUR_LAPVAR_CAL_MIN_CALIBRATED = 0.021092022956273312

#: the S0 rule tuple appended to ``jadal.poc.cascade.S0_RULES`` (last, so the
#: five shipped rules keep priority: a black clip still reports BLACK, not BLUR)
BLUR_RULE = ("BLUR", "lapvar_cal", "<", BLUR_LAPVAR_CAL_MIN)

#: feature name shown in the evidence block
BLUR_EVIDENCE_NAME = "lapvar_cal"


def laplacian_var(grey: np.ndarray) -> float:
    """4-neighbour Laplacian variance of ``grey`` (luma in [0, 1])."""
    g = np.asarray(grey, dtype=np.float64)
    if g.ndim != 2 or min(g.shape) < 3:
        return float("nan")
    lap = (g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:]
           - 4.0 * g[1:-1, 1:-1])
    return float(lap.var())


def lapvar_cal(grey: np.ndarray) -> float:
    """Laplacian variance of ``grey`` divided by (mean luma + eps).

    Uniform input (black, white or grey) -> 0.0.  A fully black probe is handled
    by the BLACK/TOO_DARK rules, which run BEFORE the blur rule in
    ``jadal.poc.cascade.S0_RULES``.
    """
    g = np.asarray(grey, dtype=np.float64)
    if g.ndim != 2 or min(g.shape) < 3:
        return float("nan")
    return float(laplacian_var(g) / (float(g.mean()) + EPS))


def probe_lapvar_cal(grey_stack: np.ndarray) -> float:
    """The S0-probe statistic: mean per-frame Laplacian variance / (mean luma).

    ``grey_stack`` is the whole (T, H, W) grey probe.  Per frame and then
    averaged (not computed on the temporal mean: averaging frames first
    low-passes the very high-frequency content this statistic measures).  The
    denominator is the probe's global mean luma, so the reading is "sharpness at
    this window's brightness" -- the definition GRID.09 pre-registered.
    """
    g = np.asarray(grey_stack, dtype=np.float64)
    if g.ndim != 3 or g.shape[0] < 1:
        return float("nan")
    vals = [laplacian_var(f) for f in g]
    vals = [v for v in vals if np.isfinite(v)]
    if not vals:
        return float("nan")
    return float(np.mean(vals) / (float(g.mean()) + EPS))


def probe_lapvar_cal_of_mean(grey_stack: np.ndarray) -> float:
    """Ablation: :func:`lapvar_cal` of the TEMPORAL MEAN frame of the probe."""
    g = np.asarray(grey_stack, dtype=np.float64)
    if g.ndim != 3:
        return float("nan")
    return lapvar_cal(g.mean(axis=0))


def probe_laplacian_var(grey_stack: np.ndarray) -> float:
    """Ablation: per-frame mean of the RAW (uncalibrated) Laplacian variance."""
    g = np.asarray(grey_stack, dtype=np.float64)
    vals = [laplacian_var(f) for f in g]
    vals = [v for v in vals if np.isfinite(v)]
    return float(np.mean(vals)) if vals else float("nan")
