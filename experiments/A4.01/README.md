# A4.01 — Coverage at >=95% decided-accuracy: deterministic abstain rule (in-domain + OOD + hold-out)

**Why:** The pod reports review-rate frontiers at ~85% BA; nobody has asked what coverage survives a 95% decided-accuracy bar.

**Hypothesis:** A rank/quantile abstain rule on the frozen two-view and adapted O4 score reaches >=95% decided balanced-accuracy only at low coverage; the CI lower bound is unreachable at n_bad<=22.

**Data / label source:** gold150 LOCO; OOD160 grouped (138/22); hold-out108 (92/16, evaluate only); frozen ensemble + adapted O4/Z3 cached scores

**Data regime:** in-domain LOCO + OOD factory-grouped + hold-out one-shot

**Method:** sweep symmetric band, global/within-factory quantile, asymmetric top/bottom-k, conformal; decided BA/acc vs coverage; grouped bootstrap CI

**Owner:** A4  **Started:** 2026-10-10 20:51

## Results
(pending — `exp.py finish` fills the registry; put figures/JSON in this folder)

**Verdict (2026-10-10 20:57): maybe.** Coverage at >=95% decided balanced-accuracy: in-domain frozen 0.120 (honest CI-LB 0.027); OOD-160 adapted 0.300 (CI-LB unreachable, n_bad=22); hold-out108 adapted 0.269 (CI-LB 0.148). Plain-accuracy bar is much looser but decides almost no bad windows.

Results file: eval/results/exp_acc/A4.01.json

Decision: Adopt top/bottom-k abstention for the 95% bar; report coverage with grouped CI lower bound, never accuracy alone.

Cost: CPU only, ~1 min
