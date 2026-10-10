# E9.01 — standard eval harness for PR comparison

**Why:** PR-triggered agents need one deterministic number to compare a stranger's PR against main; reproduces the documented IndEgo baselines

**Hypothesis:** LOCO logistic on the 26 per-pixel features reproduces ~0.81 and the EG2 window embedding reproduces ~0.85; the frozen IndEgo model does not transfer OOD (0.4-0.53)

**Data / label source:** IndEgo gold manifest_gold.json n=150 (100 good/50 bad, 7 clips, human, one labeller); OOD ood_labels.json n=60 (51 good/9 bad, human)

**Data regime:** IndEgo 480p 10fps H264 in-domain; Egocentric-10K 1080p30 HEVC out-of-domain

**Method:** LOCO logistic + clip-bootstrap CI; frozen-model OOD with window-bootstrap CI; CPU sklearn, no network

**Owner:** infra  **Started:** 2026-10-10 15:22

## Results
(pending — `exp.py finish` fills the registry; put figures/JSON in this folder)

**Verdict (2026-10-10 15:23): adopt.** Reproduces both documented baselines on n=150 IndEgo human windows (100 good/50 bad, 7 clips): 26 per-pixel features LOCO logistic C=0.3 -> AUROC 0.8100 (clip-bootstrap 95% CI 0.693-0.887); EG2 window embedding LOCO logistic C=0.01 -> 0.8480 (0.734-0.909). OOD (n=60, n_bad=9, window-bootstrap, optimistic): frozen IndEgo pixel26 model does NOT transfer, 0.4314 at fps=10 (E1.03 protocol); the features.json-trained variant saturates (all p=1.0, reported as 0.5).

Results file: eval/results/standard/run_all.json

Decision: Use eval/standard/run_all.py as the single comparison number for PR-triggered review; exec tier runs it in a throwaway clone behind the run-tests label.

Cost: CPU sklearn, no GPU, ~14-36 s, no network

**Fix (A6b, M4):** the OOD 26-feature matrix is now reordered into
`FEATURE_NAMES` order (or the harness raises) before `predict_proba`, so a
regenerated shard cannot silently misalign it; `run_all.json` records
`feature_names` and `feature_order_reordered` (currently `false`, the shard
order already matches). Numbers above are unchanged by this fix.
