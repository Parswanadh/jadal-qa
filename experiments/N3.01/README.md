# N3.01 — Quality-vs-semantic split: does quality transfer and semantic not?

**Why:** Fuse/route the available scorers; decide where deterministic physics rules are valid

**Hypothesis:** Quality sub-problems (dark/blur/overexposed/frozen/duplicate) transfer out-of-domain; semantic sub-problems (no-active-work/hands-not-visible) do not

**Data / label source:** IndEgo gold 150 (LOCO) + OOD-160 human labels (factory-grouped)

**Data regime:** n/a

**Method:** AUROC of {good} vs {bad windows with reason in category} for rules (total/quality-only/semantic-only), frozen ensemble_v1, adapted Z3 model

**Owner:** N3  **Started:** 2026-10-10 19:47

## Results
(pending — `exp.py finish` fills the registry; put figures/JSON in this folder)

**Verdict (2026-10-10 19:59): inconclusive.** IndEgo LOCO: quality rules near-perfect (AUROC 0.957, 11/13 quality-bad REJECT), semantic hard (rules 0.775, frozen 0.821). OOD-160 has 0 quality-bad windows (all 22 bad semantic/other) -> quality transfer untestable; rules invert on OOD quality-only (0.312) and fire on 93/138 good; semantic OOD solved only by adapted (0.749). Hypothesis 'quality transfers, semantic not' not supported as stated.

Results file: eval/results/exp_acc/N3.01.json

Decision: Treat the split as 'where each scorer wins', not a transfer law; route in-domain->rules/ensemble, OOD->adapted

Cost: CPU, cached features, ~70s
