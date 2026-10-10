# E6.03 — Motion cue out-of-domain on 60 human-labelled Egocentric-10K windows

**Why:** Test whether the inverted motion cue even points the right way outside IndEgo; the PoC cascade was at chance out of domain (0.529).

**Hypothesis:** The IndEgo-frozen cue direction is at or near chance out of domain.

**Data / label source:** Egocentric-10K queues/label_queue_random.json 480p/30fps transcodes, 60 windows human-labelled (51 good / 9 bad, one labeller; n_bad=9 -> EXPLORATORY)

**Data regime:** 480p/30fps transcode of 1080p/30fps HEVC factory clips, stride 3 -> ~10 fps effective

**Method:** same cue extraction (160x90, translation-only compensation); direction frozen from IndEgo; pooled AUROC + window-bootstrap CI (optimistic)

**Owner:** experimenter  **Started:** 2026-10-10 15:21

## Results
(pending — `exp.py finish` fills the registry; put figures/JSON in this folder)

**Verdict (2026-10-10 15:48): drop.** n=60 human-labelled Egocentric-10K windows (51 good/9 bad, one labeller; EXPLORATORY). Cue on the labelled 480p/30 fps transcode (stride 3 ~10 fps). IndEgo-frozen direction: resid_mean AUROC 0.281 [0.145,0.434] (window bootstrap, optimistic) - worse than chance; Spearman cue vs label +0.271, opposite sign to in-domain (-0.314). Post-hoc best orientation 0.719 (not usable).

Results file: eval/results/exp_motion_cue/E6.03.json

Decision: Do not apply the IndEgo motion-cue direction out of domain. Per-source sign calibration is not applicable (one labelled window per source).

Cost: 60 windows, 95.6 s wall (3 workers) = 1593 ms/window, 0 MB VRAM; device A40 (CPU-only extraction).
