# A4.02 — Deterministic integer rule + per-factory calibration + receipt path

**Why:** A coverage claim is useless unless the rule is deterministic, integer-comparable and receipt-verifiable.

**Hypothesis:** Per-factory (within-factory) calibration does not raise coverage at 95% because OOD factories hold 1-4 windows.

**Data / label source:** same cached scores; hold-out108 features

**Data regime:** deterministic integer rule; receipt round-trip

**Method:** express rule as integer permille thresholds + top/bottom-k counts; make_receipt/verify_receipt; within-factory vs global quantile

**Owner:** A4  **Started:** 2026-10-10 20:51

## Results
(pending — `exp.py finish` fills the registry; put figures/JSON in this folder)

**Verdict (2026-10-10 20:57): adopt.** Deterministic integer rule (accept top-k / reject bottom-k, permille band) + byte-identical receipt (sha a65b0701). Per-factory calibration does NOT raise OOD coverage (1-4 windows/factory); per-clip helps in-domain only.

Results file: eval/results/exp_acc/A4.02.json

Decision: Ship top/bottom-k integer rule; do not use per-factory quantiles OOD.

Cost: CPU only, <1 min
