# Q3.01 — Byte-identical receipt proof for the deterministic rule path (10 runs)

**Why:** FACTS marks the real-module end-to-end 10-identical-receipt proof as PENDING; the product requirement is that receipts are SHA-256 of canonical JSON.

**Hypothesis:** A pure, integer-quantised rule scorer over a fixed real input set emits the same receipt SHA-256 for every window on all 10 runs, with no float/time/random/GPU in the path.

**Data / label source:** 150 gold IndEgo windows (bundle perpixel caches) + 30 OOD Egocentric-10K label_windows; labels human; features hist_ent/ten_norm/blur_norm

**Data regime:** in-domain gold + out-of-domain (Egocentric-10K, 47 factories); rule table frozen from configs/perpixel_v1.json

**Method:** Q=4 fixed-point integer comparisons (decimal ROUND_HALF_EVEN); canonical JSON (sorted keys, compact); receipt = sha256(canonical_json(body)); 10 fresh-process runs (one interpreter per run, PYTHONHASHSEED 0/1/42 cycling); negative control (perturbed feature/threshold must change the receipt); fail-closed on missing features

**Owner:** Q3  **Started:** 2026-10-10 18:52

## Results
(pending — `exp.py finish` fills the registry; put figures/JSON in this folder)

**Verdict (2026-10-10 18:52): adopt.** PASS: all 180 windows (150 gold + 30 OOD) produce byte-identical receipt SHA-256 and bytes across 10 fresh-process runs (same machine); no dict-order/time/random/GPU dependence; PYTHONHASHSEED 0/1/42 determinism holds; negative control changes the receipt; missing features fail closed. rule_table_sha256=cf3ab757... code_sha256=02486b4f...

Results file: eval/results/exp_phys/Q3.01.json

Decision: Deterministic rule path proven byte-identical; use as the receipt-hash baseline. Rule list is Q1's frozen scorer.py quality_stage tripwires (lowent/blur/ten); cascade S0 probe features (spatial_var/temp_e) were not in the pod/base caches.

Cost: CPU only; 10 runs over 180 windows in <1 s; OOD feature extraction one-shot ~103 s
