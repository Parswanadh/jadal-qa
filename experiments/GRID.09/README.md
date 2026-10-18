# GRID.09 — Brightness-calibrated blur rule on the shipped S0 probe: catch synthetic blur without rejecting good windows

**Why:** The shipped pipeline's S0 probe has no sharpness statistic, so a strongly blurred clip is accepted; GRID.09 adds Laplacian variance of the grey probe divided by mean luma (lapvar_cal) as a sixth S0 rule and measures the catch/false-reject trade-off.

**Hypothesis:** lapvar_cal separates Gaussian-blurred windows (sigma 1.5/3/8, applied at 480p before the 160x90 working downsample) from clean good windows with AUROC > 0.9 at sigma>=3, and a threshold chosen on a held-out CLIP split of clean data rejects < 5% of good windows at the production S0 stage.

**Data / label source:** IndEgo-150 human gold windows (100 good / 50 bad, ONE labeller) + 160 human-labelled Egocentric-10K windows (138 good / 22 bad, 47 factories); decoded from the 480p sources, blur applied in memory, no frames committed.

**Data regime:** 480p source, IndEgo 10 fps and Egocentric-10K 30 fps, decoded to the production 160x90 working resolution, probe = first 60 frames (2 s)

**Method:** Reuse scripts/poc_run.py::s0_from_luma (production probe) extended with lapvar_cal; threshold by Youden J on a grouped CALIBRATION split (clips/factories), evaluated on the disjoint EVAL split; group-bootstrap 95% CI on AUROC; catch rate per sigma and false-reject rate on good windows

**Owner:** dsh  **Started:** 2026-10-18 06:50

## Results
(pending — `exp.py finish` fills the registry; put figures/JSON in this folder)

**Verdict (2026-10-18 07:30): adopt.** Reproduced the motivating hole: on 16 eval-split human-good windows the shipped five S0 rules reject 0/16 blurred windows and the shipped head accepts 6/16 at sigma=3 (max p=0.9999, median p=0.5116) and 4/16 at sigma=8 (max p=0.9889). Adopted statistic: lapvar_cal at 160x90 (the 80x45 probe buffer scored only 0.757 AUROC at sigma=3 vs 0.955 at 160x90). Threshold 0.021092022956273312 = 2nd percentile of the CLEAN CALIB split (grouped even/odd by clip and by factory: calib n=106/26 groups, eval n=132/24 groups). At that threshold on the held-out EVAL split: catch 18.9% [5.8,40.4] at sigma=1.5 (n=132), 72.0% [38.8,89.4] at sigma=3, 100% [100,100] at sigma=8; AUROC 0.741/0.955/1.000; per-dataset sigma=3 IndEgo 63/63=100% vs OOD-160 32/69=46.4% [21.1,70.9]. Clean false-reject on 238 human-good windows (100 IndEgo + 138 OOD-160): 5.04% [1.92,9.96] total but 0/238 MARGINAL (all 12 exits were already rejected by the shipped five rules); IndEgo-150 good 3/100=3.00%, OOD-160 good 9/138=6.52%. Brightness controls CONFIRM the calibration does work but is not gain-invariant: sigma=3 + x1.4 gain AUROC 0.927 for lapvar_cal vs 0.831 for the uncalibrated 160x90 Laplacian variance, while the gain-only x1.4 control rejects 1.6% IndEgo / 7.2% OOD-160 and the gain-only x0.5 control rejects 22.2% IndEgo / 15.9% OOD-160 of GOOD windows. CAVEATS: all blur positives are SYNTHETIC Gaussian applied at 480p on human-good windows (no real defocus footage exists); one pooled threshold over two heterogeneous datasets; 16-window head arm; no container/level defects.

Results file: eval/results/exp_grid/GRID.09.json

Decision: Add lapvar_cal (Laplacian var / (mean luma+eps) on the decoded 160x90 luma) to the S0 probe and the BLUR rule to cascade.S0_RULES, SHIPPED DISABLED (BLUR_LAPVAR_CAL_MIN=0.0) so no shipped verdict changes; enabling needs one constant flip to 0.021092022956273312. Flip recommended with the OOD-160 catch gap and dark-window control known.

Cost: 238 window decodes x 8 variants + 16-window full-head arm, CPU only, 2 workers, nice -n 10, ~4 min wall; no pixels written
