# E9.01 — standard eval harness for PR comparison

**Why:** PR-triggered agents need one deterministic number to compare a stranger's PR against main; reproduces the documented IndEgo baselines

**Hypothesis:** LOCO logistic on the 26 per-pixel features reproduces ~0.81 and the EG2 window embedding reproduces ~0.85; the frozen IndEgo model does not transfer OOD (0.4-0.53)

**Data / label source:** IndEgo gold manifest_gold.json n=150 (100 good/50 bad, 7 clips, human, one labeller); OOD ood_labels.json n=60 (51 good/9 bad, human)

**Data regime:** IndEgo 480p 10fps H264 in-domain; Egocentric-10K 1080p30 HEVC out-of-domain

**Method:** LOCO logistic + clip-bootstrap CI; frozen-model OOD with window-bootstrap CI; CPU sklearn, no network

**Owner:** infra  **Started:** 2026-10-10 15:22

## Results
(pending — `exp.py finish` fills the registry; put figures/JSON in this folder)
