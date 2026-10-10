# EXP-TEMPORAL (track E3) — interim report (2026-10-10 ~19:55 IST)

Sampling policy for long video: which windows must be processed?

## Data & protocol
- IndEgo human gold: 150 windows / 7 clips; per-clip subsampling simulation on gold-window
  order. "Full coverage" = all gold windows of a clip (gold is a sparse sample, not exhaustive video).
- Scores: EG2/26-feat LOCO OOF `p_ens` (orientation checked: AUROC>0.5). Two estimate modes:
  L (label oracle: isolates sampling error) and M (model: end-to-end).
- Clip verdict threshold: bad-fraction ≥ 0.30. CIs: clip bootstrap (2000). CUSUM tuned with LOCO.
- OOD: 60 labelled Egocentric-10K windows (51 good / 9 bad); labels2 not yet present.
- Regen: `PYTHONNOUSERSITE=1 OMP_NUM_THREADS=2 /home/parshu/envs/jadal/bin/python eval/exp_temporal/run_e3.py --out eval/results/exp_temporal` (cwd = worktree). Results: `eval/results/exp_temporal/E3.0{1..5}*.json`. Tests: `tests/test_exp_temporal_e3.py` (5 pass).

## E3.01 — probe policies (mean over 7 clips; delay in windows, spacing ~15 s)
| policy | frac | errL [95% CI] | verdicts | delay | miss |
|---|---|---|---|---|---|
| every-2 | 0.51 | 0.049 [0.019,0.087] | 7/7 | 0.3 | 0 |
| every-3 | 0.35 | 0.117 [0.069,0.161] | 6/7 | 1.3 | 0 |
| every-4 | 0.29 | 0.124 [0.068,0.182] | 7/7 | 3.1 | 0 |
| every-6/8 | 0.18/0.15 | 0.228/0.247 | 5/7 | 3.5/7.8 | 3/2 |
| skip-30/60/120/240s | 0.84..0.22 | 0.088..0.418 | 6/7..5/7 | 1.9..12.7 | 0 |
| random-1/2 (s7) | 0.49 | 0.083 | 7/7 | **8.9** | 0 |
| cusum-default | 0.29 | 0.124 (=every-4: never fires) | 7/7 | 3.1 | 0 |
| cusum-LOCO | 0.30 | errM 0.214 [0.118,0.319] | — | — | — |
- Random probing matches systematic error but destroys latency (delay 8.9 vs 0.3 windows): use grids.
- Mode M is model-dominated: full-coverage errM=0.167, only 4/7 verdicts right. Sampling adds ≤0.05 on top (every-2).

## E3.02 — OOD position bias: NO effect (MW p=0.42 raw / 0.47 normalised; medians bad 70 s vs good 90 s; logreg coef −0.31). n_bad=9: absence of evidence, not evidence of absence. No skip rule for Ego10K; re-run at n_bad≈24.

## E3.03 — early exit: AUROC 2 s 0.804 → 4 s 0.860 → 6 s 0.869 → 8 s 0.857 → 10.5 s 0.848. Saturates at 4–6 s; 4 s within 0.01 of peak. Don't observe less than 4 s.

## E3.04 — compute (4.404 h corpus): every-2 → 2.27 h (×1.9); every-2 + 4 s exit → ~0.9 h (×4.9); every-3 + 4 s → ×7.2. Upper-bound (decode-bound) arithmetic.

## E3.05 — decisions
- ADOPT (IndEgo-like): every-2nd window + 4 s early exit. Budget option: every-3rd.
- DROP CUSUM-as-sampler; DROP skip-first-M (both regimes); OOD: no certified policy until scorer transfers (E5).
- Sampling is NOT the bottleneck (model err 0.167 at full coverage). Thin evidence: 7 clips; n_bad=9 OOD; gold is sparse; early-exit curve IndEgo-only.
