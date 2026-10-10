# rhythm STATUS — COMPLETE (all planned work done)

Branch `feat/custom-algo-rhythm`, pushed. **PR #77 open** (do NOT merge):
https://github.com/Parswanadh/jadal-qa/pull/77

## Headline (150 real human windows, LOCO, clip-bootstrap 2000)
| method | AUROC (95% CI) |
|---|---|
| 26 GPU per-pixel features + logistic | 0.810 |
| **4-epoch control (same architecture)** | **0.799 (0.703-0.847)** |
| pooled control, no layout | 0.790 |
| flow_act-like scalar family | 0.764 |
| **rhythm ensemble, 25-epoch headline** | **0.753 (0.651-0.813)** |
| rhythm ensemble, WITH synthetic negatives | 0.351 |

**Outcome: documented negative result.** Bar was >=0.85 / CI lower >=0.72 / no
clip <0.60. Missed on all three.

## Full ablation matrix (identical settings, 4 epochs, K=32, no synthetic)
control 0.799 | no_ego_motion 0.794 | no_skin 0.795 | G=8 0.790 | G=4 0.785 |
single_member 0.783 | no_aug 0.772 | single_no_aug 0.768.
**Every CI overlaps every other — nothing in the architecture is individually
load-bearing.** pallet 0.515-0.530 in EVERY configuration.

## Determinism (measured)
10 repeats x 5 windows -> identical digests/probs/verdicts. Tensors byte-identical
on re-extraction. OMP_NUM_THREADS=1 vs 8 -> **0 differing fields**. Scope: same
machine, pinned stack, CPU path. Content-hashed, NOT signed.

## Three corrections to the project's story
1. **Motion cue is INVERTED**: good windows have LESS residual flow (signed AUROC
   0.196). Translation-only ego-motion breaks when the wearer walks. The baseline
   is closer to a "wearer is not walking" detector than a "work is happening" one.
2. **Spatial layout adds nothing** beyond pooling the same channels
   (sp_cellmax - poolmean_only = -0.0018; my own pooled control 0.790 > my model 0.753).
3. **Synthetic negatives cost 0.40 AUROC on real data** (0.753 -> 0.351).

## Bug found and fixed
`cv2.phaseCorrelate` mutates its inputs in place (OpenCV 5.0.0). First cache was
corrupt and discarded; guarded by 3 regression tests.

## Deliverables
`jadal/rhythm/{features,model,gpu_batch,scorer}.py`,
`eval/rhythm/{extract,train,evaluate,merge_folds,freeze,sign_robust,determinism}.py`,
`eval/results/rhythm/*.json` (14 files), `configs/rhythm_v1.json(.sha256)` +
`rhythm_v1.npz` (1427 B, sha256 0faf3609...), `tests/test_rhythm_features.py`,
`docs/reports/rhythm.md` (467 lines), this file.

## Still not done (no number claimed)
Streaming (2-second chunk) path. GPU-vs-CPU digest comparison. Cost in
windows/hour. Cross-device determinism.
