# Experiment log

Every experiment: why it was run, on what data, what came out, and the decision. Generated from `experiments/registry.jsonl` (append-only). Negative results are kept.

| id | track | status | verdict | title | data / regime |
|---|---|---|---|---|---|
| E6.01 | E6 | done | drop | Motion cue conditioning on ego-motion magnitude (terciles + Spearman partial) | IndEgo human gold, 150 windows / 7 clips (one labeller); cue / 480p/10fps IndEgo mp4_480, ~100 frames per 10 s window |
| E6.02 | E6 | done | drop | Per-clip sign calibration from unlabelled windows only (LOCO) | IndEgo human gold, 150 windows / 7 clips (one labeller) for  / 480p/10fps IndEgo mp4_480 |
| E6.03 | E6 | done | drop | Motion cue out-of-domain on 60 human-labelled Egocentric-10K windows | Egocentric-10K queues/label_queue_random.json 480p/30fps tra / 480p/30fps transcode of 1080p/30fps HEVC factory clips, stride 3 -> ~10 fps effective |
| E6.04 | E6 | done | drop | Motion cue conditioning robustness (quantiles vs fixed bins) + logistic controls | IndEgo human gold, 150 windows / 7 clips (one labeller), sam / 480p/10fps IndEgo mp4_480 |

## E6.01 — Motion cue conditioning on ego-motion magnitude (terciles + Spearman partial)
- **Why:** The residual-motion cue is documented inverted (signed AUROC ~0.20). If it is only 'the wearer is walking', it is an ego-motion magnitude proxy and cannot separate usable windows within a motion level.
- **Hypothesis:** The residual-motion cue survives within ego-motion terciles and its Spearman partial correlation with the label controlling for shift_px stays substantially negative.
- **Data / label source:** IndEgo human gold, 150 windows / 7 clips (one labeller); cue recomputed from video, ego-motion from caches/perpixel/series_gold.npz shift_px
- **Regime:** 480p/10fps IndEgo mp4_480, ~100 frames per 10 s window
- **Method:** per-window mean residual motion after translation-only phase-correlation compensation at 160x90; ego-motion = mean shift_px; within-tercile AUROC + Spearman partial correlation; clip-bootstrap CI (2000)
- **Owner / started / finished:** experimenter / 2026-10-10 15:21 / 2026-10-10 15:48
- **Verdict:** drop — n=150 IndEgo human windows (100 good/50 bad, 7 clips, one labeller). Strongest motion scalar = ego-motion magnitude (series_gold shift_px mean), oriented AUROC 0.805 [0.675,0.893] (unoriented 0.195). Residual-motion cue (translation-compensated, 160x90) oriented AUROC 0.692; Spearman partial correlation with the label controlling for ego-motion = -0.102 [-0.347,+0.072] (CI includes 0). Low-motion tercile is 49 good/1 bad, so the cue is largely 'the wearer is not walking'.
- **Decision:** Do not use the residual-motion cue as an independent usable-window signal; use ego-motion magnitude as a regime/stability descriptor in profiles, not as evidence of active work.
- **Cost:** CPU-only: 100 frames/window, 150 windows in 98.6 s (3 workers) = 657 ms/window, 0 MB VRAM; bootstraps 2000 draws; device A40 (GPU idle).
- **Results:** eval/results/exp_motion_cue/E6.01.json

## E6.02 — Per-clip sign calibration from unlabelled windows only (LOCO)
- **Why:** pallet is a per-clip sign inversion (raw 0.27) that puts some clips at chance. If the sign can be fixed at inference from the clip's own unlabelled windows, the cue becomes usable without labels.
- **Hypothesis:** A label-free per-clip sign/magnitude calibration (majority-prior sign or within-clip |z|) raises the pooled LOCO AUROC above the raw oriented cue.
- **Data / label source:** IndEgo human gold, 150 windows / 7 clips (one labeller) for SCORING only; calibration uses only the clip's own unlabelled cue values
- **Regime:** 480p/10fps IndEgo mp4_480
- **Method:** leave-one-clip-out direction (6 training clips), per-clip label-free recalibration on the held-out clip's own window values (majority-prior sign; within-clip -|z|); pooled AUROC + clip-bootstrap CI
- **Owner / started / finished:** experimenter / 2026-10-10 15:21 / 2026-10-10 15:48
- **Verdict:** drop — n=150 IndEgo human windows, 7 clips, one labeller. LOCO with per-clip sign from the clip's OWN unlabelled windows: raw pooled 0.692 [0.477,0.839]; within-clip -|z| 0.606 [0.525,0.708]; majority-prior sign 0.610 [0.449,0.767]. Mean per-clip AUROC raw 0.664 [0.486,0.809] vs majority-prior 0.691 [0.535,0.808] (CIs overlap); pallet 0.36->0.64 and organise 0.31->0.69 repaired but desk 0.73->0.27 broken. Oracle per-clip sign (uses test labels, UPPER BOUND only) mean 0.757.
- **Decision:** The pallet inversion cannot be fixed at inference without labels: the label-free sign rule trades one broken clip for two repaired and lowers the pooled AUROC. No per-clip sign calibration in the product.
- **Cost:** no extra extraction (cached cue); 2000-draw clip bootstraps x 4 variants x 4 rules; device A40 (CPU).
- **Results:** eval/results/exp_motion_cue/E6.02.json

## E6.03 — Motion cue out-of-domain on 60 human-labelled Egocentric-10K windows
- **Why:** Test whether the inverted motion cue even points the right way outside IndEgo; the PoC cascade was at chance out of domain (0.529).
- **Hypothesis:** The IndEgo-frozen cue direction is at or near chance out of domain.
- **Data / label source:** Egocentric-10K queues/label_queue_random.json 480p/30fps transcodes, 60 windows human-labelled (51 good / 9 bad, one labeller; n_bad=9 -> EXPLORATORY)
- **Regime:** 480p/30fps transcode of 1080p/30fps HEVC factory clips, stride 3 -> ~10 fps effective
- **Method:** same cue extraction (160x90, translation-only compensation); direction frozen from IndEgo; pooled AUROC + window-bootstrap CI (optimistic)
- **Owner / started / finished:** experimenter / 2026-10-10 15:21 / 2026-10-10 15:48
- **Verdict:** drop — n=60 human-labelled Egocentric-10K windows (51 good/9 bad, one labeller; EXPLORATORY). Cue on the labelled 480p/30 fps transcode (stride 3 ~10 fps). IndEgo-frozen direction: resid_mean AUROC 0.281 [0.145,0.434] (window bootstrap, optimistic) - worse than chance; Spearman cue vs label +0.271, opposite sign to in-domain (-0.314). Post-hoc best orientation 0.719 (not usable).
- **Decision:** Do not apply the IndEgo motion-cue direction out of domain. Per-source sign calibration is not applicable (one labelled window per source).
- **Cost:** 60 windows, 95.6 s wall (3 workers) = 1593 ms/window, 0 MB VRAM; device A40 (CPU-only extraction).
- **Results:** eval/results/exp_motion_cue/E6.03.json

## E6.04 — Motion cue conditioning robustness (quantiles vs fixed bins) + logistic controls
- **Why:** The E6.01 conclusion must not depend on one arbitrary stratification; a vanishing effect must be logged as a negative result.
- **Hypothesis:** Within-stratum AUROC and the partial correlation are stable across quantile and fixed-bin strata, and cue+ego logistic does not beat ego alone.
- **Data / label source:** IndEgo human gold, 150 windows / 7 clips (one labeller), same cue/ego vectors as E6.01
- **Regime:** 480p/10fps IndEgo mp4_480
- **Method:** same strata under 4 rules (terciles, quintiles, fixed 5/15 px, fixed 1/5/20 px) x 4 cue variants; LOCO logistic controls cue-only vs ego-only vs cue+ego
- **Owner / started / finished:** experimenter / 2026-10-10 15:21 / 2026-10-10 15:48
- **Verdict:** drop — n=150 IndEgo human windows, 7 clips. Within-stratum oriented AUROC of the residual cue under 4 stratification rules x 4 cue variants: interpretable strata (>=3 bad) span 0.503-0.756 (terciles 0.517/0.682; quintiles 0.503-0.756; fixed 1/5/20 px 0.569-0.806; fixed 5/15 px only 1 interpretable stratum). LOCO logistic control: cue only 0.660, ego-motion magnitude only 0.798, cue+ego 0.793.
- **Decision:** Negative result logged: the within-motion-level effect is weak and stratum-dependent, and the cue adds nothing on top of ego-motion magnitude (-0.005). Keep the cue only as a global stability descriptor.
- **Cost:** no extra extraction; 4 schemes x 4 variants x 2000-draw bootstraps; device A40 (CPU).
- **Results:** eval/results/exp_motion_cue/E6.04.json
