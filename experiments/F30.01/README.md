# F30.01 — Per-pixel feature vector 26 -> 30 (4 Z3 transferable invariants)

**Why:** Test whether 4 more per-pixel features improve in-domain LOCO and OOD transfer accuracy

**Hypothesis:** Adding inv_clip_frac/inv_laplacian_var/inv_hist_entropy/inv_dct_hf_lf raises OOD transfer at little in-domain cost

**Data / label source:** 150 IndEgo gold (LOCO) + 160 Egocentric-10K OOD (grouped CI) + 108 round-3 hold-out (scored once); human labels

**Data regime:** in-domain LOCO + OOD zero-shot adapted z-score

**Method:** LOCO logistic C=0.3 balanced; per-domain z-score, IndEgo sign, equal-weight mean; factory-grouped bootstrap; per-feature ablation

**Owner:** F30  **Started:** 2026-10-10 21:32

## Results
(pending — `exp.py finish` fills the registry; put figures/JSON in this folder)

**Verdict (2026-10-10 21:32): maybe.** In-domain LOCO 26->30: 0.8100->0.8118 (+0.0018 ns). OOD adapted z-score full vector 0.3864->0.4954 (+0.109, paired grouped CI [0.027,0.199], 99.4%>0); hold-out 108 0.3580->0.3995. Frozen logistic unchanged. The 4 per-pixel invariants alone score OOD 0.793 / hold-out 0.692, matching the curated 14-feature baseline (0.781/0.676). Cost +57 ms/window (+25%).

Results file: eval/results/exp_feat30/F30.01.json

Decision: Keep the 4 invariants in the per-pixel extractor (free in-domain, large adapted-OOD gain); do not claim it beats the frozen ensemble.

Cost: +57.4 ms/window (+25.1% over the 26-feature cost, CPU, 160x90 ~10fps)
