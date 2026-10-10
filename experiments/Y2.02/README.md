# Y2.02 — Deterministic stage-0 triage + combined review pipeline

**Why:** Decide additional REVIEW windows from cached quality signals (dark/over-exposed/frozen/undecodable) without re-decoding.

**Hypothesis:** Deterministic quality rules decide a meaningful share of REVIEW windows with high precision on human-labelled subsets.

**Data / label source:** caches/gpu/features.json (150 gold + 60 synthetic degradations); 1,546 windows_rescored.jsonl features

**Data regime:** mixed

**Method:** Fixed a-priori descriptor-space quality rules; validated on synthetic degradations and 150 gold; applied to the 1,546 review windows; combined gating+stage-0+calibration pipeline review share

**Owner:** monitor  **Started:** 2026-10-10 16:47

## Results
(pending — `exp.py finish` fills the registry; put figures/JSON in this folder)

**Verdict (2026-10-10 16:48): adopt.** Deterministic stage-0 rules fire on 6/6 of each synthetic black/darken/overexpose/frozen and 0/6 clean/blur/noise/shake/lowbitrate/static; on 150 gold they fire 3x (precision_bad 0.67, n=3, matching the funnel's no-certified-rule result); on the 1,546 they fire 39x (2.5%) and drop review 55.0%->54.0%. Combined pipeline reaches 54.0% review. Calibration (gold-LOCO temperature) is a no-op on share and hurts gold BA -> dropped.

Results file: eval/results/exp_review/Y2.02.json

Decision: adopt stage-0 reject overlay; temperature off

Cost: CPU, <5 s, cached only
