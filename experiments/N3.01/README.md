# N3.01 — Quality-vs-semantic split: does quality transfer and semantic not?

**Why:** Fuse/route the available scorers; decide where deterministic physics rules are valid

**Hypothesis:** Quality sub-problems (dark/blur/overexposed/frozen/duplicate) transfer out-of-domain; semantic sub-problems (no-active-work/hands-not-visible) do not

**Data / label source:** IndEgo gold 150 (LOCO) + OOD-160 human labels (factory-grouped)

**Data regime:** n/a

**Method:** AUROC of {good} vs {bad windows with reason in category} for rules (total/quality-only/semantic-only), frozen ensemble_v1, adapted Z3 model

**Owner:** N3  **Started:** 2026-10-10 19:47

## Results
(pending — `exp.py finish` fills the registry; put figures/JSON in this folder)
