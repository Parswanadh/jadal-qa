# R1.01 — Optimized + recalibrated pipeline over the expanded 27.66 h corpus

**Why:** L3.01 (PR #148) showed the adopted optimizations (pixel 3.33 fps, EG2 1 fps, 4 s
early exit, skip mosaic) shift the frozen-head score by +0.20 and break the verdict
distribution (54% agreement). N5 (PR #156) refit both heads on sparse features and shipped
`configs/ensemble_v1_sparse.json` + band `[0.4193, 0.6112]`. The expanded corpus
(`manifest_scale_all.json`: 9,757 windows / 391 clips / 85 factories / 27.66 h) now needs
a full run with the recalibrated heads to measure throughput and the descriptive verdict
distribution at the restored operating point.

**Hypothesis:** The optimized + recalibrated pipeline runs over the expanded corpus at
~20 footage-h/wall-h (L3 rate) and produces a verdict distribution consistent with the
frozen operating point (high review share under the wide profile bands), with no OOD
regression.

**Data / label source:** `queues/manifest_scale_all.json` — 9,757 windows / 391 clips /
85 factories / 27.662 h, 1080p30 HEVC, `label_source=none` (descriptive verdicts).

**Data regime:** factory Egocentric-10K, 1080p, 30 fps, HEVC (in-domain-shaped, unlabelled).

**Method:** repo E8.01 pipeline (`eval/exp_rescore/run_e8.py`) + L3 runner
(`eval/exp_opt_scale/run_opt_scale.py`, copied from m/latency-fast), one NVDEC
three-branch pass per window, 6 workers, `JADAL_GPU_SLOTS=4`, `OMP_NUM_THREADS=1`.
Heads: `configs/ensemble_v1_sparse.json`. Operating-point + determinism verified on
labelled subsets (150 gold + 160 OOD).

**Owner:** R1  **Started:** 2026-10-10 19:52

## Results (partial, 2026-10-10 20:31)

**Verdict: maybe (run still in flight).** 830/9,757 windows (8.5%) scored, 0 errors,
wall 1509 s, **0.55 windows/s**, **5.50 processed footage-h/wall-h**; extrapolated
full-corpus wall **296 min (4.9 h)**. Descriptive verdicts: accept 210 (25.3%),
reject 100 (12.0%), review 520 (62.7%). Per-stage p50/p95 ms: decode 7590/18101,
descriptors 1403/4539, eg2_embed 464/1113, pixel_features 215/706, total 10003/23569.
GPU util mean 12.4% / p95 90%; VRAM mean 13.5 GiB / max 16.9 GiB.

Operating point (independent re-application of frozen vs recalibrated heads on the N5
sparse features): gold review share 0.1867, confident BA 0.7632 (frozen heads) →
**0.9073 (recalibrated)**; OOD AUROC 0.6347 (frozen heads) → **0.5231 (recalibrated)**,
inside the frozen factory-grouped CI. Determinism: 10 windows × 2 runs, content /
`features_digest` / score+verdict **10/10 identical**.

Throughput is ~4× below L3's uncontended 19.3 h/h because a peer transcode job saturated
the network FS (decode p50 7.3 s vs L3 2.8 s).

Results file: `eval/results/exp_opt_scale/R1.01.json`
Figure: `docs/experiments/exp_opt_scale/images/opt_scale.png`
Report: `docs/reports/exp_opt_scale.md`
