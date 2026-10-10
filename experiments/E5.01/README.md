# E5.01 — In-domain EG2 frame heads (10k AI FACTORY frames) pooled to IndEgo human windows

**Why:** All IndEgo-trained models failed out-of-domain on Egocentric-10K (0.4-0.53 AUROC). Before adapting, measure how much window-level signal a head trained on in-domain (factory, AI-labelled) frames carries on the 150 human-labelled IndEgo windows.

**Hypothesis:** Frame heads trained on the 10k AI-labelled FACTORY frames separate IndEgo usable/bad windows when pooled (LOCO AUROC > 0.5), but this is model-vs-model agreement, NOT ground-truth accuracy.

**Data / label source:** Egocentric-10K-Evaluation 10k AI-labelled FACTORY frames (hand_count, active_labor; AI, gemini-2.5-flash, NOT human) + 150 IndEgo human gold windows (manifest_gold.json, single labeller)

**Data regime:** head frames: 1080p/30fps factory stills; IndEgo eval: 480p ~10fps H.264, mixed lighting/sharpness

**Method:** EG2 frame embeddings -> logistic heads (5-fold CV, C=1) -> pool per-window (mean/max/p25/p75/frac>0.5/var) -> LOCO over 7 clips + clip-bootstrap CI

**Owner:** experimenter  **Started:** 2026-10-10 15:17

## Results
(pending — `exp.py finish` fills the registry; put figures/JSON in this folder)

**Verdict (2026-10-10 15:40): maybe.** Pooled AI-frame heads on 150 IndEgo human windows (LOCO over 7 clips, clip-bootstrap CI): work-prob mean 0.813 [0.695,0.864]; hands-prob mean 0.818 [0.685,0.893]; both 0.826 [0.717,0.881]; 12-feat LOCO logistic 0.789 [0.686,0.862]. Below the 0.85 in-domain EG2-window-embedding baseline. Model-vs-model (AI head labels), NOT accuracy.

Results file: eval/results/exp_domain/E5.01.json

Decision: Keep pooled AI-frame heads as a cheap prior only; they do not beat the in-domain EG2 embedding baseline. Do not claim transfer/ground truth.

Cost: CPU ~2 min; 10k AI frames + 150 windows; no GPU encode
