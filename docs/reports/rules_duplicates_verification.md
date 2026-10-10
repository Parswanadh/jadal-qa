# Deterministic-rule and duplicate-detection claims — pod verification (LB1-rules)

Verifier: b/verify-rules worker (CPU only). Claims quoted from
`/workspace/jadal_pod_bundle/docs/FACTS.md` (lines 22, 23, 41, 46); the bundle
is read-only. Machine-readable results:
`eval/results/pod_verify/rules_and_duplicates.json`.
Rerun: `OMP_NUM_THREADS=2 python3 eval/verify/pod_rules_duplicates.py`
(deterministic, reads only committed caches; reruns `eval/perpixel/rules_fix.py`
into a temp dir and never writes a committed artifact).

| # | Claim (abridged) | Verdict | Published | Recomputed |
|---|---|---|---|---|
| C1a | 150 labels: no L0 rule can be certified; the correction needs ~86 clean fires; the CERTIFY split has 84 | **REPRODUCED** | 86 / 84 | floor 86, `n_certify=84` (3 clips), all 8 reject rules disabled, `accept_conf=null`; rerun matches `funnel_gold_calibration.json` on every compared field |
| C1b | Descriptively TOO_DARK catches 6/6 at 1 false alarm | **REPRODUCED** | 6/6, 1 FA | 6/6, 1 good window fires (+1 HANDS_NOT_VISIBLE, +1 NO_ACTIVE_WORK); equals the artifact's `rule_fired_x_gold.TOO_DARK` row |
| C2a | 64-pair held-out F1: spectral 0.947, EG2 cosine 0.909, pHash 0.865 | **CANNOT VERIFY** | 0.947 / 0.909 / 0.865 | F1s are internally consistent with the published P/R (0.9474 / 0.9089 / 0.8648), but the pair manifest and per-detector scores are absent from this pod |
| C2b | "Not yet tested on the 6 real human DUPLICATE labels (in progress)" | **NOT REPRODUCED** | untested | stale: the test exists (`m2.json`, `docs/reports/perpixel_accuracy.md`); independent recompute matches exactly — recall@1 = recall@3 = 0.5, precision of top matches 0.0476, 63/150 above the constructed threshold |
| C3a | bound arithmetic: ~86 clean windows (59 with fixed-sequence testing alone) | **REPRODUCED** (with convention caveat) | 86 / 59 | 86 at `delta_rule=0.0125`; 59 at family `delta=0.05`; the shipped default `delta=0.1` makes the fixed-sequence opt-out floor **45**, not 59 |
| C3b | effective sample size ~20-43 (within-clip correlation) | **CANNOT VERIFY** | 20-43 | no computation in-repo; the pod's own sync note gives ~11-43 from an out-of-repo "research4 note". Labelled cross-check (ANOVA ICC over 7 clip clusters): `n_eff` 31.0 (bad rate), 40.7 (TOO_DARK rate) |
| C4 | deterministic threshold rule list is degenerate (rejects nearly everything) and was rejected | **REPRODUCED** | degenerate / rejected | rerun of the producing script is bit-identical: sens 1.0, `frr_good` 0.97 (97/100 good windows rejected), `bal_acc` 0.515, GOOD kept 0.03; pre-fix objective kept 0.0 good |

## Arithmetic shown

- Support floor (`jadal/funnel.py:1216-1217`):
  `n_power = ceil(ln(delta_rule) / ln(1-beta))`, `beta = 0.05` (target_beta).
  - shipped defaults (`delta=0.1`, `m = len(REJECT_RULES) = 8`):
    `delta_rule = 0.0125` -> `ceil(ln 0.0125 / ln 0.95) = ceil(85.43) = 86`.
  - fixed-sequence at family `delta=0.05`: `ceil(ln 0.05 / ln 0.95) = 59`.
  - shipped opt-out (`--no-bonferroni`, `delta=0.1`): 45
    (`tests/test_funnel_round2.py:318` asserts 45; `:308` asserts 86).
- Why 84 works: exact one-sided Clopper-Pearson UCB for 0 fires in n at
  confidence `1-delta_rule` is `1 - delta_rule^(1/n)`:
  `UCB(0/84)@0.0125 = 0.05083 > 0.05` (fails), `UCB(0/85) = 0.05024` (fails),
  `UCB(0/86) = 0.04968 <= 0.05` (passes). The CERTIFY split holds 84 windows,
  one short of 86, so no rule can be certified in principle — matching the
  artifact's `reason` on all 8 disabled rules.
- CERTIFY split reproduced by `funnel._split_select_certify(records, 1234)`:
  SELECT 66 windows / 4 clips (`desk, ecu_late, logistics4, pallet2`),
  CERTIFY 84 / 3 clips (`ecu_full, organise, pallet`).
- Duplicate F1 from published P/R: `F1 = 2PR/(P+R)` -> 0.9474, 0.9089, 0.8648
  (rounds to the published 0.947 / 0.909 / 0.865).

## Unverifiable, and what is missing

- **C2a (three constructed-pair F1s)**: no `manifest_dups.json`, no
  `spectral_data/arms/{A,B,E}/*.json`, no `spectral_study.md` anywhere under
  `/workspace` (bounded search, maxdepth 5), and the pod gold manifest
  (`/workspace/jadal_pod_bundle/labels/manifest_gold.json`) has no `pairs` key.
  The source is out-of-repo (`spectral-harness` worktree, L1.08 in
  `legacy_models.md`). Related but different pair set: the repo's own L1.09 rerun
  reports P 0.952 / R 1.0 / F1 0.976 on 128 pairs — the snapshot itself marks the
  two as not directly comparable.
- **C3b (effective sample size)**: no in-repo computation (grep for effective
  sample / n_eff / design effect / ICC finds none); `docs/reports/claims_poc.md`
  records the 59 / 20-43 figures as unbacked and deleted them from README.
  Provenance is the out-of-repo "research4 risk-control note", which the pod sync
  note cites as ~11-43 — a different range. My ICC cross-check is a plausibility
  check only, not the published protocol.

## Claim C4 and the shipped rule list

The degenerate list is recorded with evidence (`eval/results/perpixel/rulelist.json`,
`eval/results/perpixel/theory.json` `rule_list` bal_acc 0.47 / AUROC 0.589,
`eval/perpixel/rules_fix.py` docstring: the unconstrained objective "chose
thresholds rejecting every good window out-of-clip"), and I reproduced the
committed rule list exactly by rerunning its producer offline. It is **not** what
ships: `configs/funnel_thresholds.json` is `funnel-v1-placeholder`
(status `placeholder`, `hard_reject_policy: integrity_only`, `accept_conf: null`),
and the round-2 calibration disabled all 8 tunable reject rules. Descriptively
under `evidence_reject` the shipped thresholds reject 12/100 good windows
(`funnel_gold_descriptive.json`), not "nearly everything" — so "was rejected" is
accurate, "is the shipped rule list" is not.
