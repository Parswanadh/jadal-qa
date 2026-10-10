# F40.01 — Per-pixel features 26 -> 40: accuracy gain (LOCO + OOD grouped CI)

**Why:** Test whether 14 new per-pixel features (Z3 transferable-6 + finer spectral bands + colour ratios + temporal lags + noise floor) improve the 26-feature per-pixel view in-domain and out-of-domain.

**Hypothesis:** The added transferable/spectral/colour features raise in-domain LOCO AUROC above 0.810 and the adapted-recipe OOD AUROC above the frozen 0.781 pooled / 0.676 hold-out baselines.

**Data / label source:** gold 150 (human, LOCO) + OOD 160 (human, factory-grouped) + hold-out 108 (human, scored once)

**Data regime:** IndEgo 480p + Egocentric-10K 1080p/30 + 854x480/10 transcode

**Method:** 26 cached + 6 frozen Z3 + 8 fresh pixel features; LOCO logistic C=0.3 balanced; adapted per-domain-z-score equal-weight mean; clip/factory bootstrap 2000

**Owner:** F40  **Started:** 2026-10-10 21:20

## Results
(pending — `exp.py finish` fills the registry; put figures/JSON in this folder)
