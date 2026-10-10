# O4.02 — Per-feature IndEgo vs OOD-160 transfer

**Why:** Report per-feature transfer class at n_bad=22 for the 6 Z3 and 8 stable features

**Hypothesis:** The 6 transferable candidates stay stable at 160; some stable-8 features may flip

**Data / label source:** same as O4.01

**Data regime:** OOD Egocentric-10K, human-labelled

**Method:** signed AUROC + factory-grouped bootstrap

**Owner:** O4  **Started:** 2026-10-10 17:41

## Results
(pending — `exp.py finish` fills the registry; put figures/JSON in this folder)

**Verdict (pre-review note, 2026-10-10 17:43): adopt.** All 6 Z3 transferable candidates
stable on OOD-160; 9 stable/10 flip/1 chance across 20 candidates; selection set unchanged.
Note: `global_shift` is classified **flip** (IndEgo 0.376 / OOD-160 0.509) yet sits in the
E5 stable-8 set; recipe (b2) in the report drops it and the conclusion is unchanged. O2.03
later re-ran the sign table on all 160 labels and changed `global_shift` (stable→flip) and
`frozen_frac` (flip→stable); O4 kept the pre-O2.03 set (disclosed in the report).

Results file: eval/results/exp_adapt160/O4.02.json

Decision: confirm

Cost: CPU only
