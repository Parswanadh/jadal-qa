# Experiment log

Every experiment: why it was run, on what data, what came out, and the decision. Generated from `experiments/registry.jsonl` (append-only). Negative results are kept.

| id | track | status | verdict | title | data / regime |
|---|---|---|---|---|---|
| A4.01 | A4 | done | maybe | Coverage at >=95% decided-accuracy: deterministic abstain rule (in-domain + OOD + hold-out) | gold150 LOCO; OOD160 grouped (138/22); hold-out108 (92/16, e / in-domain LOCO + OOD factory-grouped + hold-out one-shot |
| A4.02 | A4 | done | adopt | Deterministic integer rule + per-factory calibration + receipt path | same cached scores; hold-out108 features / deterministic integer rule; receipt round-trip |

## A4.01 — Coverage at >=95% decided-accuracy: deterministic abstain rule (in-domain + OOD + hold-out)
- **Why:** The pod reports review-rate frontiers at ~85% BA; nobody has asked what coverage survives a 95% decided-accuracy bar.
- **Hypothesis:** A rank/quantile abstain rule on the frozen two-view and adapted O4 score reaches >=95% decided balanced-accuracy only at low coverage; the CI lower bound is unreachable at n_bad<=22.
- **Data / label source:** gold150 LOCO; OOD160 grouped (138/22); hold-out108 (92/16, evaluate only); frozen ensemble + adapted O4/Z3 cached scores
- **Regime:** in-domain LOCO + OOD factory-grouped + hold-out one-shot
- **Method:** sweep symmetric band, global/within-factory quantile, asymmetric top/bottom-k, conformal; decided BA/acc vs coverage; grouped bootstrap CI
- **Owner / started / finished:** A4 / 2026-10-10 20:51 / 2026-10-10 20:57
- **Verdict:** maybe — Coverage at >=95% decided balanced-accuracy: in-domain frozen 0.120 (honest CI-LB 0.027); OOD-160 adapted 0.300 (CI-LB unreachable, n_bad=22); hold-out108 adapted 0.269 (CI-LB 0.148). Plain-accuracy bar is much looser but decides almost no bad windows.
- **Decision:** Adopt top/bottom-k abstention for the 95% bar; report coverage with grouped CI lower bound, never accuracy alone.
- **Cost:** CPU only, ~1 min
- **Results:** eval/results/exp_acc/A4.01.json

## A4.02 — Deterministic integer rule + per-factory calibration + receipt path
- **Why:** A coverage claim is useless unless the rule is deterministic, integer-comparable and receipt-verifiable.
- **Hypothesis:** Per-factory (within-factory) calibration does not raise coverage at 95% because OOD factories hold 1-4 windows.
- **Data / label source:** same cached scores; hold-out108 features
- **Regime:** deterministic integer rule; receipt round-trip
- **Method:** express rule as integer permille thresholds + top/bottom-k counts; make_receipt/verify_receipt; within-factory vs global quantile
- **Owner / started / finished:** A4 / 2026-10-10 20:51 / 2026-10-10 20:57
- **Verdict:** adopt — Deterministic integer rule (accept top-k / reject bottom-k, permille band) + byte-identical receipt (sha a65b0701). Per-factory calibration does NOT raise OOD coverage (1-4 windows/factory); per-clip helps in-domain only.
- **Decision:** Ship top/bottom-k integer rule; do not use per-factory quantiles OOD.
- **Cost:** CPU only, <1 min
- **Results:** eval/results/exp_acc/A4.02.json
