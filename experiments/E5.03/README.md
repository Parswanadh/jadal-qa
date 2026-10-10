# E5.03 — Per-feature signed AUROC IndEgo gold vs Egocentric-10K OOD (the what-transfers table)

**Why:** The domain gap is per-feature; a table of which of the 26 per-pixel features keep their sign from IndEgo to Egocentric-10K tells us what transfers and what flips.

**Hypothesis:** Sharpness/light/duplicate features keep their sign; motion and temporal-band features flip between IndEgo (480p ~10fps) and Egocentric-10K (1080p/30fps source).

**Data / label source:** 150 IndEgo human gold windows (manifest_gold.json) + 60 Egocentric-10K human-labelled windows (51 good / 9 bad, ood_labels.json)

**Data regime:** IndEgo 480p ~10fps vs Egocentric-10K 1080p/30fps-source, both decoded 160x90; IndEgo native 10fps vs OOD subsampled to 10fps effective

**Method:** signed AUROC per feature (label1=usable); clip-bootstrap CI (IndEgo, 7 clips) and window-bootstrap CI (OOD, optimistic given ~1 window/factory); classify stable/flip/chance

**Owner:** experimenter  **Started:** 2026-10-10 15:17

## Results
(pending — `exp.py finish` fills the registry; put figures/JSON in this folder)

**Verdict (2026-10-10 15:40): maybe.** 26 per-pixel features, signed AUROC IndEgo(gold,n=150,clip-CI) vs OOD(n=60,n_bad=9,window-CI): 8 sign-stable (radial_slope 0.633/0.645, hf_ratio 0.618/0.693, dct_energy 0.631/0.702, luma_p05, global_shift, illum_r, illum_b, zncc_self); 14 flip (incl. motion_energy 0.308->0.721, temp_b0 0.534->0.240, skin_frac 0.667->0.257); 4 chance. Sharpness/light are the transferable cues.

Results file: eval/results/exp_domain/E5.03.json

Decision: Use the stable sharpness/light features for regime routing; do not reuse motion/temporal-band signs across domains.

Cost: CPU ; 150+60 windows x 26 features
