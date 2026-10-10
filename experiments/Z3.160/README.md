# Z3.160 — Z3 domain-agnostic features re-run on 160 OOD labels

**Why:** PR #127 used the old 60 OOD labels; re-run the candidate transfer table and z-score combos on the merged 160 labels with grouped CIs.

**Hypothesis:** Per-domain z-score stays the best transferable combo, with a smaller but still significant gain over frozen.

**Data / label source:** IndEgo 150 + labels/ood160_labels.json 160 (138 good/22 bad, 47 factories); z3_160.npz + ood160_pixel.npz

**Data regime:** evaluation only; 10 fps-effective

**Method:** signed per-feature AUROC + transfer class; per-domain z-score combos; factory-grouped CI

**Owner:** M5  **Started:** 2026-10-10 19:25

## Results
(pending — `exp.py finish` fills the registry; put figures/JSON in this folder)

**Verdict (2026-10-10 19:25): adopt.** 6 transferable features unchanged; per-domain z-score OOD-160 0.759 [0.597,0.886] and stable-8+6 0.781 [0.634,0.895] vs frozen 0.566; paired grouped gain excludes 0. Headline magnitude lower than the optimistic n=60 0.85.

Results file: eval/results/exp_features160/Z3.01_160.json

Decision: n/a

Cost: n/a
