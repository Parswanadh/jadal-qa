# E6.01 — Motion cue conditioning on ego-motion magnitude (terciles + Spearman partial)

**Why:** The residual-motion cue is documented inverted (signed AUROC ~0.20). If it is only 'the wearer is walking', it is an ego-motion magnitude proxy and cannot separate usable windows within a motion level.

**Hypothesis:** The residual-motion cue survives within ego-motion terciles and its Spearman partial correlation with the label controlling for shift_px stays substantially negative.

**Data / label source:** IndEgo human gold, 150 windows / 7 clips (one labeller); cue recomputed from video, ego-motion from caches/perpixel/series_gold.npz shift_px

**Data regime:** 480p/10fps IndEgo mp4_480, ~100 frames per 10 s window

**Method:** per-window mean residual motion after translation-only phase-correlation compensation at 160x90; ego-motion = mean shift_px; within-tercile AUROC + Spearman partial correlation; clip-bootstrap CI (2000)

**Owner:** experimenter  **Started:** 2026-10-10 15:21

## Results
(pending — `exp.py finish` fills the registry; put figures/JSON in this folder)

**Verdict (2026-10-10 15:48): drop.** n=150 IndEgo human windows (100 good/50 bad, 7 clips, one labeller). Strongest motion scalar = ego-motion magnitude (series_gold shift_px mean), oriented AUROC 0.805 [0.675,0.893] (unoriented 0.195). Residual-motion cue (translation-compensated, 160x90) oriented AUROC 0.692; Spearman partial correlation with the label controlling for ego-motion = -0.102 [-0.347,+0.072] (CI includes 0). Low-motion tercile is 49 good/1 bad, so the cue is largely 'the wearer is not walking'.

Results file: eval/results/exp_motion_cue/E6.01.json

Decision: Do not use the residual-motion cue as an independent usable-window signal; use ego-motion magnitude as a regime/stability descriptor in profiles, not as evidence of active work.

Cost: CPU-only: 100 frames/window, 150 windows in 98.6 s (3 workers) = 657 ms/window, 0 MB VRAM; bootstraps 2000 draws; device A40 (GPU idle).
