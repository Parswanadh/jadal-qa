# F40.01 — Per-pixel features 26 -> 40: accuracy gain (LOCO + OOD grouped CI)

**Why:** Test whether 14 new per-pixel features (Z3 transferable-6 + finer spectral bands + colour ratios + temporal lags + noise floor) improve the 26-feature per-pixel view in-domain and out-of-domain.

**Hypothesis:** The added transferable/spectral/colour features raise in-domain LOCO AUROC above 0.810 and the adapted-recipe OOD AUROC above the frozen 0.781 pooled / 0.676 hold-out baselines.

**Data / label source:** gold 150 (human, LOCO) + OOD 160 (human, factory-grouped) + hold-out 108 (human, scored once)

**Data regime:** IndEgo 480p + Egocentric-10K 1080p/30 + 854x480/10 transcode

**Method:** 26 cached + 6 frozen Z3 + 8 fresh pixel features; LOCO logistic C=0.3 balanced; adapted per-domain-z-score equal-weight mean; clip/factory bootstrap 2000

**Owner:** F40  **Started:** 2026-10-10 21:20

## Results
(pending — `exp.py finish` fills the registry; put figures/JSON in this folder)

**Verdict (2026-10-10 21:32): drop.** 26->40 does not help: in-domain LOCO 0.810->0.775; frozen adapted recipe OOD 0.781 / hold-out 0.676 not beaten (all-40 0.535/0.455; expanded-22 0.730/0.653). Added features carry some OOD signal vs all-26 (+0.150 grouped) but the selected frozen 14 stays best.

Results file: eval/results/exp_feat40/eval_feat40.json

Decision: do not adopt the 40-feature view; keep the frozen stable-8+Z3-6 recipe

Cost: 8 fresh features: 377-4129 ms/window end-to-end (8-way pool); 0.3-2.2 s/window single-thread
