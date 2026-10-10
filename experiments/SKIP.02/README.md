# SKIP.02 — Stratified vs contiguous probe + dense-where-uncertain

**Why:** Q2: cheap sparse probe first window of each kept section; compare to equal number of contiguous windows and to densifying the section closest to threshold

**Hypothesis:** Stratified beats contiguous at equal budget; densifying the closest-to-threshold section beats uniform dropping at equal budget

**Data / label source:** IndEgo human gold 150 windows/7 clips + cached oof_probs.npz

**Data regime:** 480p/10fps IndEgo

**Method:** stratified first-window probe; contiguous first-m control; CUSUM/SPRT-style densify of nearest-threshold section; equal-budget comparison vs uniform every-k grid

**Owner:** X2  **Started:** 2026-10-10 16:27

## Results
(pending — `exp.py finish` fills the registry; put figures/JSON in this folder)

**Verdict (2026-10-10 16:27): drop.** strat-probe(1/section) err 0.142 @ 33.1% beats equal-budget contiguous probe 0.211 @ 33.1% and lifts agreement 0.57->0.86, but is not better than uniform-every-3 (0.117 @ 34.9%). dense-where-uncertain x1 0.100 @ 45.9% loses to uniform at equal budget (~0.072); x2 0.057 @ 62.5% also loses. Reproduces E3 CUSUM rejection.

Results file: eval/results/exp_skip/SKIP.02.json

Decision: Drop dense-where-uncertain and stratified-as-primary; uniform every-k grid stays.

Cost: CPU-only, cached scores + labels, no GPU
