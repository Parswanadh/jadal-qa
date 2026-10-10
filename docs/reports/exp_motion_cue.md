# E6 — motion cue + per-clip sign calibration

Branch `feat/exp-motion-cue`. Registry ids **E6.01–E6.04** (`experiments/EXPERIMENTS.md`).
All numbers below come from the results JSONs in `eval/results/exp_motion_cue/`; nothing here
is claimed beyond them.

## 1. What was measured

**The cue under test** = residual motion after translation-only ego-motion compensation:

    resid_mean = mean_t  mean |I_t - warp(I_{t-1}; phaseCorrelate)|

computed at 160x90 (area), every frame of the 10 s window (`eval/exp_motion_cue/motion_cue.py`).
High residual = motion left unexplained by a pure translation. Canonical orientation:
**low residual = usable** (the score reported as `auroc_oriented` is AUROC with that orientation,
so 0.5 = chance).

**Ego-motion magnitude** (the conditioning variable, as specified) = per-window mean of the cached
`series_gold.npz` channel `shift_px` (phase-correlation shift vs the first frame, px). It is *not*
the same quantity as the cue, and the cached `diff_e`/`motion_energy` (uncompensated frame
difference) is a third, different scalar.

Data: **150 IndEgo human-labelled windows, 7 clips, one labeller** (100 usable / 50 bad) for
E6.01/E6.02/E6.04; **60 human-labelled Egocentric-10K windows, 51 good / 9 bad** for E6.03
(exploratory). Device: **A40** (the extraction is CPU-only; the GPU was idle). Cost: 100 frames per
window, 150 windows extracted in 98.6 s wall on 3 workers = **657 ms/window**, 0 MB VRAM; OOD
95.6 s / 60 windows = 1,593 ms/window. Command:

    JADAL_BUNDLE=$JADAL_BUNDLE python3 eval/exp_motion_cue/run_e6.py \
        --experiments E6.01,E6.02,E6.03,E6.04 --workers 3

Prior evidence (bundle `docs/FACTS.md`, 2026-10-10 18:25 IST) reported the motion cue
unoriented AUROC ~0.20 (~0.80 oriented), `pallet` raw 0.27, `pallet2` 0.75. Our recomputation of
the **cached** scalars reproduces the *magnitude* of that number for the ego-motion magnitude
(oriented 0.805, below), while the compensated residual cue is weaker (oriented 0.692). We do not
have the harness's `flow_act` (hand-region) scalar in the caches, so we do not claim to reproduce
its per-clip numbers except where they match (`pallet2`).

## 2. E6.01 — conditioning on ego-motion magnitude (`E6.01.json`, n=150)

Clip-bootstrap 95% CI, 2,000 draws. Spearman ρ; partial ρ controls for ego-motion magnitude.

| cue | oriented AUROC | ρ(cue, label) [CI] | ρ(cue, ego) | partial ρ given ego [CI] |
|---|---|---|---|---|
| `resid_mean` (cue) | 0.692 | −0.314 [−0.50, +0.04] | +0.473 | **−0.102 [−0.347, +0.072]** |
| `resid_med` | 0.684 | −0.300 [−0.48, +0.01] | +0.438 | −0.105 [−0.333, +0.066] |
| `resid_p90` | 0.698 | −0.323 [−0.49, +0.02] | +0.502 | −0.097 [−0.315, +0.063] |
| `resid_std` | 0.694 | −0.316 [−0.44, −0.05] | +0.564 | −0.049 [−0.188, +0.068] |
| ego-motion magnitude itself (`shift_px` mean) | **0.805 [0.675, 0.893]** | — | — | (the conditioning variable) |

Within terciles of ego-motion magnitude (edges 12.5 / 25.4 px):

| stratum | n | good / bad | `resid_mean` oriented AUROC [CI] |
|---|---|---|---|
| low motion (1.5–12.5 px) | 50 | **49 / 1** | 0.306 [0.061, 0.446] |
| mid motion (12.5–25.4 px) | 50 | 30 / 20 | 0.682 [0.461, 0.830] |
| high motion (25.4–46.8 px) | 50 | 21 / 29 | 0.517 [0.347, 0.741] |

**Reading.** (a) The low-motion stratum contains 49 of 50 usable windows and a single bad one:
at low ego-motion the window is almost always usable, i.e. the dominant effect is *the wearer is
not walking*. (b) The partial correlation of the cue with the label, controlling for ego-motion
magnitude, is −0.10 and its CI includes 0 → the residual cue does **not** demonstrably survive the
conditioning. (c) The strongest motion signal available is the ego-motion magnitude itself
(oriented AUROC 0.805 [0.675, 0.893]), and it is the "not walking" quantity. Verdict: **drop**
(the motion/stability cue is not an independent usable-window signal beyond ego-motion magnitude).

## 3. E6.02 — per-clip sign calibration from UNLABELLED windows only (`E6.02.json`, n=150)

Direction is chosen on the 6 training clips (LOCO); the recalibration for the held-out clip uses
**only that clip's own window values** (no labels). Two label-free rules: `-|z|` within clip
(extremes in either direction = bad) and a **majority-prior sign** (orient so the majority of the
clip's windows score usable). Both protocols are reported because they disagree — pooled AUROC is
dominated by between-clip score offsets, while the equal-weight mean per-clip AUROC is what a
per-clip rule can move.

| rule | pooled AUROC [CI] | mean per-clip AUROC [CI] |
|---|---|---|
| raw (global orientation) | **0.692 [0.477, 0.839]** | 0.664 [0.486, 0.809] |
| `-|z|` within clip | 0.606 [0.525, 0.708] | 0.621 [0.536, 0.709] |
| majority-prior sign | 0.610 [0.449, 0.767] | 0.691 [0.535, 0.808] |
| oracle per-clip sign (UPPER BOUND, uses the test labels — not usable) | 0.537 | 0.757 [0.698, 0.817] |

Per-clip per-clip raw → majority-prior: `pallet` 0.36 → **0.64**, `organise` 0.31 → **0.69**,
`desk` 0.73 → **0.27**. So the label-free sign rule repairs the two inverted clips and breaks one
correctly-oriented clip; the pooled AUROC is *lower* than raw, and the mean-per-clip gain (+0.027)
has overlapping CIs. The oracle row shows that a per-clip sign would help *within* clips, but the
sign cannot be recovered without labels at this n. Verdict: **drop** — per-clip sign calibration
from unlabelled windows is not a reliable fix for the `pallet` inversion.

## 4. E6.03 — out of domain on the 60 human-labelled Egocentric-10K windows (`E6.03.json`)

Cue computed on `queues/label_queue_random.json` `path` (480p/30 fps 10 s transcode, stride 3 →
~10 fps effective) — the file the human labelled. The 1080p/30 fps `source_path` originals exist;
they were **not** used (the labelled window is the transcode). n = 60, 51 good / 9 bad →
**exploratory**. Window-bootstrap CI (optimistic: one window per factory).

| cue | AUROC with the IndEgo-frozen direction [CI] | ρ(cue, label) on OOD | post-hoc best orientation (not usable) |
|---|---|---|---|
| `resid_mean` | **0.281 [0.145, 0.434]** | +0.271 | 0.719 |
| `resid_med` | 0.264 [0.136, 0.412] | +0.292 | 0.736 |
| `resid_p90` | 0.383 [0.199, 0.568] | +0.144 | 0.617 |
| `resid_std` | 0.575 [0.349, 0.784] | −0.093 | 0.575 |

The correlation with the label has the **opposite sign** out of domain (+0.27) from in-domain
(−0.31), so the frozen direction is worse than chance: the cue's orientation does not transfer.
Per-source sign calibration is **not applicable** — the 60 windows come from 47+ distinct source
videos, almost all contributing a single window. Verdict: **drop** (out of domain, direction not
transferable; do not present as a quality measure).

## 5. E6.04 — robustness of the conditioning (`E6.04.json`, n=150)

Within-stratum oriented AUROC of `resid_mean` (only strata with ≥ 3 bad windows are interpretable):

| stratification | edges (px) | strata with ≥3 bad | min / max / mean AUROC |
|---|---|---|---|
| quantile terciles | 12.5, 25.4 | 2 | 0.517 / 0.682 / 0.599 |
| quantile quintiles | 9.1, 15.5, 23.1, 31.6 | 4 | 0.503 / 0.756 / 0.629 |
| fixed bins | 5, 15 | 1 | 0.594 (only one interpretable stratum) |
| fixed bins | 1, 5, 20 | 2 | 0.569 / 0.806 / 0.688 |

The within-motion-level separation is weak (≈ 0.50–0.70), depends on where the strata are cut, and
in no scheme does it approach the ego-motion magnitude alone (0.805). Decisive control
(LOCO logistic on the same folds): cue only **0.660**, ego-motion magnitude only **0.798**,
cue + ego **0.793** — adding the residual cue to the ego-motion magnitude does not help
(−0.005). Verdict: **drop** (negative result: the effect is not stable; the cue adds nothing on
top of ego-motion magnitude).

## 6. What this does NOT say

- Not "hand work is detected", not "hand-region motion": no spatial/hand-region channel was
  computed; the cue is a global scalar.
- Nothing about spatial localisation or rhythm (closed lines elsewhere) and nothing about the
  harness's `flow_act`, which is not in the caches.
- One labeller, 7 clips (effective n ≈ 11–43 by clip correlation); the OOD set has n_bad = 9 and
  its window-bootstrap CI is optimistic. No accuracy claim is made on unlabelled video.
- The products's *strongest* motion signal on this data is ego-motion magnitude, i.e. an
  "is the wearer walking / is the camera stable" proxy, not evidence of active work.

## 7. The one sentence the product can honestly use

**On 150 IndEgo windows from 7 clips (one labeller), motion/stability cues predict usable windows,
but the strongest motion signal is essentially ego-motion magnitude (the wearer not walking) and
residual motion after translation-only compensation adds no measurable information on top — part of
the cue may be head stability.**
