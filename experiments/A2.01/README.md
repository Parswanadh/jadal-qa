# A2.01 — New full-res signals alone: flow / hand / full-res per-pixel

**Why:** The cached 26 features are computed at 160x90; test whether dense optical flow,
hand/active-work segmentation, and native-resolution per-pixel statistics carry NEW signal.

**Hypothesis:** Dense flow (divergence/curl/stability) and hand-motion fraction separate
NO_ACTIVE_WORK / HANDS_NOT_VISIBLE better than the 160x90 features, in-domain and OOD.

**Data / label source:** IndEgo gold 150 + Egocentric-10K OOD-160 human labels

**Method:** AUROC of each new feature as a bad-score; sign fit on gold, applied to OOD
(factory-grouped 95% CI). Extracted at full-res decode, all frames.

**Owner:** A2  **Started:** 2026-10-10 20:50

## Results
See `eval/results/exp_acc/A2.01.json`.

**Verdict:** pending
