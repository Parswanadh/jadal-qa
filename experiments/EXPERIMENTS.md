# Experiment log

Every experiment: why it was run, on what data, what came out, and the decision. Generated from `experiments/registry.jsonl` (append-only). Negative results are kept.

| id | track | status | verdict | title | data / regime |
|---|---|---|---|---|---|
| F50.01 | F50 | done | drop | Per-pixel features 26 -> 50: in-domain LOCO | IndEgo gold 150 windows (7 clips), manifest_gold.json + cach / in-domain, leave-one-clip-out |
| F50.02 | F50 | done | drop | Per-pixel features 26 -> 50: OOD adapted | OOD-160 human labels (138/22, 47 factories); adapted per-dom / out-of-domain, factory-grouped CI |
| F50.03 | F50 | done | drop | Ablation of the 24 added features | same as F50.02 / OOD-160 |
| F50.04 | F50 | done | drop | Hold-out108 scored once (26 vs 50) | round-3 confirmatory hold-out 108 (92/16, 44 factories) / confirmatory, scored once |
| F50.05 | F50 | done | adopt | Cost of the 24 added features | one real 10-s window + synthetic frames / CPU |

## F50.01 — Per-pixel features 26 -> 50: in-domain LOCO
- **Why:** Does expanding the per-pixel feature vector from 26 to 50 improve IndEgo LOCO accuracy?
- **Hypothesis:** Added features raise in-domain AUROC above 0.810
- **Data / label source:** IndEgo gold 150 windows (7 clips), manifest_gold.json + caches/gpu/features.json
- **Regime:** in-domain, leave-one-clip-out
- **Method:** logistic C=0.3 balanced, clip-bootstrap CI
- **Owner / started / finished:** F50 / 2026-10-10 21:22 / 2026-10-10 21:34
- **Verdict:** drop — In-domain LOCO 0.810 (26) -> 0.787 (50), delta -0.023 (ns). No in-domain gain.
- **Decision:** Keep the 26-feature set in-domain.
- **Cost:** CPU, <1 min
- **Results:** eval/results/exp_feat50/in_domain.json

## F50.02 — Per-pixel features 26 -> 50: OOD adapted
- **Why:** Does the 50-feature set improve OOD-160 adapted z-score over 0.781?
- **Hypothesis:** Additional transferable features raise OOD AUROC
- **Data / label source:** OOD-160 human labels (138/22, 47 factories); adapted per-domain z-score
- **Regime:** out-of-domain, factory-grouped CI
- **Method:** per-domain z-score, IndEgo sign, equal-weight mean; factory-grouped bootstrap
- **Owner / started / finished:** F50 / 2026-10-10 21:22 / 2026-10-10 21:34
- **Verdict:** drop — OOD-160 adapted: documented 14-feat 0.781; +18 new 0.657 (-0.123 [-0.224,-0.035], sig. worse); all-50 0.512. No transferable gain.
- **Decision:** Do not adopt the 50-feature set.
- **Cost:** CPU, cached features
- **Results:** eval/results/exp_feat50/ood.json

## F50.03 — Ablation of the 24 added features
- **Why:** Which added feature families carry the gain; is there saturation?
- **Hypothesis:** A few families dominate; saturation after ~6-10 features
- **Data / label source:** same as F50.02
- **Regime:** OOD-160
- **Method:** leave-one-family-out + family-only + cumulative
- **Owner / started / finished:** F50 / 2026-10-10 21:22 / 2026-10-10 21:34
- **Verdict:** drop — Ablation: only dct_bands (0.728 alone) and edge_lap (0.680) carry OOD signal; saturation immediate (0.781->0.779->...->0.657).
- **Decision:** No family worth adding.
- **Cost:** 
- **Results:** eval/results/exp_feat50/ablation.json

## F50.04 — Hold-out108 scored once (26 vs 50)
- **Why:** Honest generalisation check of the 50-feature recipe
- **Hypothesis:** Gain replicates weakly or not at all
- **Data / label source:** round-3 confirmatory hold-out 108 (92/16, 44 factories)
- **Regime:** confirmatory, scored once
- **Method:** frozen per-domain z-score recipe, no tuning
- **Owner / started / finished:** F50 / 2026-10-10 21:22 / 2026-10-10 21:34
- **Verdict:** drop — Hold-out108 scored once: documented 0.676; +18 new 0.612; all-50 0.457. Gain does not replicate.
- **Decision:** Do not adopt.
- **Cost:** 
- **Results:** eval/results/exp_feat50/holdout.json

## F50.05 — Cost of the 24 added features
- **Why:** Edge budget: ms/window of the added kernels
- **Hypothesis:** Added kernels < 250 ms/window CPU
- **Data / label source:** one real 10-s window + synthetic frames
- **Regime:** CPU
- **Method:** wall-clock repeats
- **Owner / started / finished:** F50 / 2026-10-10 21:22 / 2026-10-10 21:34
- **Verdict:** adopt — Added cost 148 ms/window (18 kernels 43 ms + ffprobe 104 ms); decode 157 ms shared.
- **Decision:** Documented for the edge budget.
- **Cost:** 148 ms/window CPU
- **Results:** eval/results/exp_feat50/cost.json
