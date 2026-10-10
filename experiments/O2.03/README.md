# O2.03 — E5.03 refresh: per-feature signed AUROC on the 160 OOD labels

**Why:** The what-transfers table was built on 60 OOD windows (n_bad=9); refresh it against the 160 labels (n_bad=22) and check whether the stable/flip/chance calls hold.

**Hypothesis:** The sharpness/spatial-spectrum features keep their sign; motion/temporal-band features still flip; counts barely move with more labels.

**Data / label source:** 150 IndEgo human gold windows (manifest_gold) + 160 Egocentric-10K human windows (ood160_labels.json); 26 per-pixel features, frozen 160x90 recipe

**Data regime:** IndEgo native ~10fps vs OOD 10 fps-effective

**Method:** signed AUROC per feature; IndEgo clip-bootstrap CI, OOD160 factory-grouped CI; classify_transfer stable/flip/chance; compare class vs the old 60-label table

**Owner:** O2  **Started:** 2026-10-10 17:40

## Results
(pending — `exp.py finish` fills the registry; put figures/JSON in this folder)

**Verdict (2026-10-10 17:40): maybe.** 160-label table: 8 stable / 14 flip / 4 chance (same counts as the 60). Only 2/26 change class: frozen_frac flip->stable, global_shift stable->flip. Sharpness trio stable (radial_slope 0.633/0.635, hf_ratio 0.618/0.657, dct_energy 0.631/0.700 with grouped CI [0.55,0.83]); motion_energy still flips (0.308->0.646). Transferable subset (stable & >=0.10): dct_energy, hf_ratio, radial_slope, luma_p05, illum_b, illum_r, frozen_frac.

Results file: eval/results/exp_fewshot/O2.03.json

Decision: Use the sharpness/spatial-spectrum features for regime routing; do not reuse motion/temporal-band signs. Table robust to doubling labels.

Cost: CPU; 150+160 windows x 26 features
