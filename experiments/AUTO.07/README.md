# AUTO.07 — Audit policy: audit rate vs detected-error rate vs human hours per 10,000 video hours

**Why:** The brief's recommended deployment form is auto-decide everything plus a random audit sample plus all low-confidence windows; the human workload and the detection power of that audit must be stated as arithmetic over measured shares.

**Hypothesis:** A 2-10% audit of a 100%-automated stream detects only a proportionate share of escaped errors, so the band review share drives the human workload, not the audit.

**Data / label source:** measured review/band shares and auto-decision error rates from AUTO.00/AUTO.01 on IndEgo-150 and OOD-160

**Data regime:** arithmetic over measured shares; no new measurement

**Method:** closed-form audit arithmetic; every row labelled arithmetic, not measurement; review time per window is an explicit ASSUMED parameter because no measured value exists in the bundle

**Owner:** dsh-s4-auto  **Started:** 2026-10-18 06:42

## Results
(pending — `exp.py finish` fills the registry; put figures/JSON in this folder)

**Verdict (2026-10-18 06:58): adopt.** Audit policy specified (auto-decide everything, review the low-confidence band plus a uniform random audit sample, track error and verdict mix per site, alarm on drift) with the required table; every row labelled arithmetic. Measured inputs: IndEgo band share 0.480 [0.347,0.570] clip CI, in-band forced error 0.333, out-of-band 0.192; OOD-160 band 0.138, in-band error 0.545, out-of-band 0.457. At T_review = 30 s/window (ASSUMED - no measured review time exists in the bundle): human hours per 10,000 video hours = 15,000/15,900/17,400 on IndEgo and 4,725/5,625/7,125 on OOD at audit rates 2/5/10 %; detected-error rate 0.623/0.635/0.654 (IndEgo) and 0.177/0.202/0.244 (OOD); 2 % catches 20 (IndEgo) and 79 (OOD) of the escaped errors per 10,000 windows. The band, not the audit, dominates the human workload. Drift arithmetic: 15 audited windows give a 95 % chance of at least one detected error at the IndEgo out-of-band rate, 5 at the OOD rate; estimating a site's error to +-5 pp needs 239 (IndEgo) / 382 (OOD) audited windows. Decision-rule check: OOD-160 100 %-coverage bal-acc lower bound 0.448 < 0.70 -> tiered automation with audit.

Results file: eval/results/auto/AUTO.07.json

Decision: Adopt the audit-shaped policy spec for the AUTO.08 memo; the hours columns are arithmetic over an assumed T_review and must be re-derived if a measured review time appears.

Cost: 0 GPU, ~1 s CPU (arithmetic over measured shares)
