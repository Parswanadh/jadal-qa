# Q5.01 — Deterministic physics rules: full-corpus run + atomic latency

**Why:** Q1 proved the deterministic physics rule list on 150 in-domain IndEgo
gold windows. Q5 asks what it does at scale: run it over the whole 1,546-window
Egocentric-10K corpus with atomic per-stage timing, verify receipt determinism,
and quantify the determinism/OOD trade-off against the logistic runs (L2/L3) and
E8.01.

**Hypothesis:** The rules are cheap; swapping the decision stage out of the L3
latency runner leaves decode/features/EG2 untouched, so the full corpus runs at
the same feature-bound throughput with a near-zero decision cost, and receipts
are byte-identical across repeats.

**Data / label source:** `queues/manifest_scale.json` — 1,546 windows / 58 clips
/ 4.404 h, 1080p30 HEVC, `label_source: none` (verdicts DESCRIPTIVE).

**Method:** L3 runner (`eval/exp_latency_fast/run_fast.py`) with the decision
stage swapped to `eval/exp_phys/phys_rules.py` + frozen `rule_table.json`; an
extra atomic `phys_features` stage computes the 16 physics features. Adopted
optimizations: pixel 3.33 fps, EG2 1 fps, 4 s early exit, mosaic off; 7 workers,
`JADAL_GPU_SLOTS=6`, `OMP_NUM_THREADS=1`, A40 NVDEC. `receipt_sha256` hashes the
contract line excluding `latency_ms`, so repeats are comparable.

**Owner:** Q5  **Started:** 2026-10-10 19:10

## Results
(see `eval/results/exp_phys_full/Q5.01.json` and
`docs/reports/exp_phys_full.md`)
