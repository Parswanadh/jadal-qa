# E2.04 — Mosaic without timestamps

**Why:** Do the burned-in timestamps add signal beyond the pixels?

**Hypothesis:** Removing timestamps does not change AUROC much if content dominates.

**Data / label source:** same windows/folds; label_source human

**Data regime:** same as E2.01

**Method:** mosaic N=9 tile 224 timestamps=False, same eval

**Owner:** experimenter  **Started:** 2026-10-10 15:20

## Results
(pending — `exp.py finish` fills the registry; put figures/JSON in this folder)

**Verdict (2026-10-10 16:07): drop.** no-timestamp mosaic IndEgo 0.810 [0.68,0.88] vs with-timestamps 0.785: delta +0.025 inside CI; OOD 0.444. Burned-in timestamps do not add signal beyond pixels.

Results file: eval/results/exp_mosaic/E2.04.json

Decision: Drop the timestamp-helps hypothesis; keep timestamps only for the human-evidence view.

Cost: 1 embed pass/window; A40
