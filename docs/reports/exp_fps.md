# E1: frame-rate and frame-selection experiments

Status: descriptive results on the IndEgo gold set (n=150 windows, 50 bad). No
equivalence margin was pre-stated and no paired test was run. Code:
`eval/exp_fps/`. Results: `eval/results/exp_fps/`.

## E1.01 fps sweep (headline)

Two numbers, and they are not the same question:

- **All 26 features** (the headline as originally run): 10 fps AUROC 0.809
  [0.693, 0.890]; 3.33 fps AUROC 0.807 [0.697, 0.873]. This set includes the 8
  timing-dependent features (temp_b0-b3, frozen_frac, flicker_idx,
  motion_energy, rppg_pulse), whose values change with the frame interval. The
  0.809 vs 0.807 comparison is therefore not a clean test of fps.
- **Spatial-only, 18 features** (dt-independent): 10 fps AUROC 0.690
  [0.565, 0.808]; 3.33 fps AUROC 0.678 [0.591, 0.741].

Both are descriptive. The 95% CIs are about 0.2 wide at n=150, so differences
between fps settings are not resolvable at this sample size. The 18-feature
numbers are the fair comparison across fps, and even they still carry the
caveat that three of them are span-dependent (see E1.04).

## E1.04 frame selection at a 20-frame budget

Arms: uniform20 (step 5), motion20 (top inter-frame-diff frames plus frame 0),
random20 (seed 0), keyframe20 (ffprobe I-frame positions).

Scored on the 18 spatial features (`e1_04.json` `rows`):

| arm | AUROC | 95% CI |
|---|---|---|
| uniform20 | 0.679 | [0.550, 0.792] |
| motion20 | 0.578 | [0.499, 0.684] |
| random20 | 0.548 | [0.381, 0.652] |
| keyframe20 | INVALID | not reported |

keyframe20 is INVALID: every one of the 150 windows had fewer than 20 distinct
I-frame positions on the 10 fps grid (median 1, max 5), so the arm would not be
a 20-frame keyframe selection. No keyframe AUROC is reported.

**Span confound.** Three of the 18 "spatial" features are dt-independent but
span-dependent: `global_shift` (first vs middle frame), `zncc_self` (first vs
last frame) and `dup_hash_mean` (middle frame). uniform20, motion20 and random20
pick different endpoints and middle frames, so the arm comparison is confounded
for these three. They are listed in `e1_04.json` `span_dependent_features`.

**Sensitivity without the three span-dependent features** (15 features, same
LOCO and bootstrap, `e1_04.json` `sensitivity_without_span_dependent`):

| arm | AUROC | 95% CI |
|---|---|---|
| uniform20 | 0.649 | [0.511, 0.747] |
| motion20 | 0.568 | [0.475, 0.639] |
| random20 | 0.533 | [0.405, 0.615] |

The ordering is unchanged, but the uniform20 gap shrinks from 0.10 to 0.08
versus motion20 and from 0.13 to 0.12 versus random20. All CIs overlap, so the
selection-arm ranking is not established at this n.

## Other notes

- E1.01 `fps_comparability.comparable_endpoint_spatial` lists zncc_self and
  global_shift as comparable across fps. That is not strictly true: uniform
  subsampling ends at different native frames (99, 98, 99, 95, 90 for
  10, 5, 3.33, 2, 1 fps), so the endpoint span changes with fps.
- E1.02 (EG2 fps): see `e1_02.json`. E1.03 (OOD) is partial and its shard
  outputs are not committed.
- E1.02b (denser EG2 5/10 fps, GPU): see `e1_02b.json` and
  `docs/reports/e1_02b_dense_eg2.md`. Verdict drop: 5 fps 0.8486 and 10 fps
  0.8478 vs 2 fps 0.8480; every paired clip-bootstrap delta CI includes 0 and
  the point estimates sit at the 0.0012 re-embedding noise floor, at 2.5x/5x the
  embedding cost. No gain established (the CIs exclude only gains above ~+0.015).
