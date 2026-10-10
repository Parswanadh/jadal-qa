# Experiment log

Every experiment: why it was run, on what data, what came out, and the decision. Generated from `experiments/registry.jsonl` (append-only). Negative results are kept.

| id | track | status | verdict | title | data / regime |
|---|---|---|---|---|---|
| SKIP.01 | SKIP | done | maybe | Section-drop rules anywhere (start/middle/end/evenly) on IndEgo gold | IndEgo human gold 150 windows/7 clips (labels/manifest_gold. / 480p/10fps IndEgo assembly+logistics |
| SKIP.02 | SKIP | done | drop | Stratified vs contiguous probe + dense-where-uncertain | IndEgo human gold 150 windows/7 clips + cached oof_probs.npz / 480p/10fps IndEgo |
| SKIP.03 | SKIP | done | adopt | Best-cell selection: cheapest policy within verdict/error tolerance | SKIP.01+SKIP.02 result JSONs / IndEgo |

## SKIP.01 — Section-drop rules anywhere (start/middle/end/evenly) on IndEgo gold
- **Why:** Q2: E3 only tested skipping the START and it failed; test whether dropping 1-2 of 5 equal sections ANYWHERE keeps the clip verdict at lower cost
- **Hypothesis:** Dropping middle/end/evenly biases the clip bad-fraction less than dropping the start, and may match the verdict at lower cost
- **Data / label source:** IndEgo human gold 150 windows/7 clips (labels/manifest_gold.json) + cached oof_probs.npz
- **Regime:** 480p/10fps IndEgo assembly+logistics
- **Method:** 5 equal sections per clip; drop 1/2 under START/MIDDLE/END/EVENLY (seeds 0-2); error vs full coverage + verdict agreement @0.30; clip bootstrap CI
- **Owner / started / finished:** X2 / 2026-10-10 16:27 / 2026-10-10 16:27
- **Verdict:** maybe — drop1-start err 0.110 (worst, confirms E3) vs drop1-end 0.057 / drop1-evenly 0.033-0.041; drop2-start 0.189. START-drop biases high, END-drop low (bad rate sec1 0.09 -> sec5 0.58). END/MIDDLE/EVENLY keep 7/7 verdicts but every variant is dominated by uniform-every-2 (0.049 @ 51%). No section drop adopted.
- **Decision:** Do not adopt any section-drop rule; keep every-2. Direction (start worst, end/even safest) is evidence for time-spread sampling.
- **Cost:** CPU-only simulation on cached labels+scores, no GPU
- **Results:** eval/results/exp_skip/SKIP.01.json

## SKIP.02 — Stratified vs contiguous probe + dense-where-uncertain
- **Why:** Q2: cheap sparse probe first window of each kept section; compare to equal number of contiguous windows and to densifying the section closest to threshold
- **Hypothesis:** Stratified beats contiguous at equal budget; densifying the closest-to-threshold section beats uniform dropping at equal budget
- **Data / label source:** IndEgo human gold 150 windows/7 clips + cached oof_probs.npz
- **Regime:** 480p/10fps IndEgo
- **Method:** stratified first-window probe; contiguous first-m control; CUSUM/SPRT-style densify of nearest-threshold section; equal-budget comparison vs uniform every-k grid
- **Owner / started / finished:** X2 / 2026-10-10 16:27 / 2026-10-10 16:27
- **Verdict:** drop — strat-probe(1/section) err 0.142 @ 33.1% beats equal-budget contiguous probe 0.211 @ 33.1% and lifts agreement 0.57->0.86, but is not better than uniform-every-3 (0.117 @ 34.9%). dense-where-uncertain x1 0.100 @ 45.9% loses to uniform at equal budget (~0.072); x2 0.057 @ 62.5% also loses. Reproduces E3 CUSUM rejection.
- **Decision:** Drop dense-where-uncertain and stratified-as-primary; uniform every-k grid stays.
- **Cost:** CPU-only, cached scores + labels, no GPU
- **Results:** eval/results/exp_skip/SKIP.02.json

## SKIP.03 — Best-cell selection: cheapest policy within verdict/error tolerance
- **Why:** Q2 deliverable: pick the cheapest policy keeping clip-verdict agreement >=90% and bad-fraction error <=0.05
- **Hypothesis:** A section-skip policy is the cheapest cell; otherwise E3 every-2 stands
- **Data / label source:** SKIP.01+SKIP.02 result JSONs
- **Regime:** IndEgo
- **Method:** tolerance-based selection on the cost-error frontier; per-clip detail at 0.30 and 0.50 thresholds
- **Owner / started / finished:** X2 / 2026-10-10 16:27 / 2026-10-10 16:27
- **Verdict:** adopt — Best cell = uniform-every-2: 51.5% processed, error 0.049 [0.019,0.087], 7/7 verdicts @0.30. Only policy meeting agreement>=0.90 AND error<=0.05; still wins when relaxing error to 0.07/0.10. No section-skip policy is cheaper within tolerance.
- **Decision:** Re-affirm E3.01 uniform-every-2 (optionally + E3.03 4s early exit). No skip-anywhere rule adopted.
- **Cost:** CPU-only synthesis, no GPU
- **Results:** eval/results/exp_skip/SKIP.03.json
