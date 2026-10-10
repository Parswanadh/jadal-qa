# exp_phys_ready — deterministic physics model, ready to ship

**Q4.01.** Package the Q1 deterministic physics rule-list model as a runnable,
documented entry point, and quantify pure rules vs a rules + logistic-fallback
hybrid.

* Package: **`tools/phys/`** (rules table, scorer, CLI, extractor, tests, README)
* Results: `eval/results/exp_phys/Q4.01.json`
* Rule table sha256 `52355d72…`, decision-code sha256 `1996c145…`, 16 rules, reject point 14, Q=4.
* Frozen inputs: Q1 `m/phys-rules` @ `fd01b6d` (`eval/results/exp_phys/rule_table.json`,
  `eval/exp_phys/phys_rules.py`); ensemble_v1 (`caches/ensemble/oof_probs.npz`,
  `caches/ensemble/ood_scores{,2}.json`); gold `labels/manifest_gold.json`;
  OOD `labels/ood160_labels.json`. CPU only, cached features.

## How to run the package

```bash
python3 tools/phys/score_window.py --features-json rec.json --window-id g000 --pretty
python3 tools/phys/score_window.py --clip clip.mp4 --start-s 0 --duration 10 --window-id w1
python3 tools/phys/score_window.py --features-json rec.json --window-id q000 \
    --fallback-json caches/ensemble/ood_scores2.json      # hybrid mode
python3 tools/phys/tests/test_phys_model.py                # 12 deterministic tests
```

Each call prints the canonical receipt
`{window_id, decision, evidence[{rule,feature,value,op,threshold}], feature_digest,
rule_table_sha256, code_sha256}` (plus `score`, `fired_rules`; sorted keys,
compact) and exits `0` ACCEPT / `1` REJECT / `2` REVIEW. The **rule decision path
is integer-only**: every value and threshold is quantised to `10**4` before
comparison, the score is an integer count, and the decision is a pure function of
integers. The *optional* logistic fallback is the one float comparison
(`usable_prob >= 0.5`) and is recorded in the receipt's `fallback` block together
with the sha256 of its source file (`fallback.source_sha256`). See
`tools/phys/README.md` for the full contract and the threshold table with its
derivation.

## Pure rules vs hybrid

Modes: **(a) pure rules** = the packaged deterministic scorer;
**(b) hybrid** = REVIEW windows are resolved deterministically by the frozen
ensemble_v1 (`accept iff prob >= 0.5`); ACCEPT/REJECT stay deterministic.

### Gold — 150 IndEgo windows (100 good / 50 bad, 7 clips, leave-one-clip-out)

| mode | AUROC [95% CI] | BA [95% CI] | REVIEW | REJECT | ACCEPT | decided det. / by model |
|---|---|---|---|---|---|---|
| pure rules | 0.8222 [0.715, 0.860] | 0.710 (review→accept) / 0.590 (review→reject); **0.808** decided-only | 60.0% | 24.0% | 16.0% | 40% / 60% |
| rules + fallback | 0.8285 hybrid-score [0.730, 0.866] | **0.790** [0.691, 0.827] | 0% (60% before fallback) | 41.3% | 58.7% | 40% / 60% |
| frozen ensemble_v1 (OOF) | 0.8336 | — | — | — | — | — |

**Significance (paired clip-grouped bootstrap, 2,000 shared resamples, seed 0).**
The rules−ensemble AUROC difference is **−0.0114, CI [−0.136, +0.060]** — the CI
contains 0, so **no AUROC difference is established** at n=150. The fallback's
BA gain over pure review-as-accept is **+0.080, CI [+0.016, +0.131]** (does not
contain 0). Fallback resolves 90 REVIEW windows: 64 → ACCEPT, 26 → REJECT.

### OOD-160 — 160 Egocentric-10K windows (138 good / 22 bad, 47 factories)

| mode | AUROC [95% CI, factory-grouped] | BA [95% CI] | REVIEW | REJECT | ACCEPT | decided det. / by model |
|---|---|---|---|---|---|---|
| pure rules | 0.4272 [0.275, 0.561] | 0.390 (review→accept) / 0.500 (review→reject) | 35.6% | 64.4% | 0.0% | 64.4% / 35.6% |
| rules + fallback | 0.4264 [0.280, 0.554] | **0.468** [0.384, 0.544] | 0% (35.6% before) | 91.9% | 8.1% | 64.4% / 35.6% |
| frozen ensemble_v1 (usable_prob) | 0.566 [0.424, 0.697] | — | — | — | — | — |

Fallback resolves 57 REVIEW windows: 13 → ACCEPT, 44 → REJECT. The OOD AUROC CIs
all contain 0.5 (not distinguishable from chance); the rules−ensemble paired
difference is −0.139, CI [−0.366, +0.063] (contains 0, not established).

## What determinism costs, what the fallback buys

| quantity | gold LOCO | OOD-160 |
|---|---|---|
| determinism cost, AUROC (ensemble − pure) | **+0.0114** (CI contains 0 → not established) | **+0.1388** (CI contains 0) |
| fallback gain, BA (hybrid − review-as-accept) | **+0.080** [+0.016, +0.131] | **+0.078** [−0.057, +0.243] |
| fallback gain, BA (hybrid − review-as-reject) | +0.200 | −0.032 |
| share decided deterministically | 40% | 64% |
| share decided by the model | 60% | 36% |

* In-domain, the deterministic score ranks within 0.011 AUROC of the learned
  ensemble, but that gap is **not established** (paired CI [−0.136, +0.060]
  contains 0). The fallback recovers the whole REVIEW band to BA 0.790 at full
  coverage (vs 0.710 treating REVIEW as accept), a +0.080 gain whose paired CI
  excludes 0.
* Out-of-domain, the frozen table is **not distinguishable from chance**
  (AUROC 0.427, CI [0.275, 0.561]) and the ensemble only reaches 0.566; the
  fallback nudges BA from 0.390 to 0.468 but its paired CI contains 0. Absolute
  IndEgo thresholds do not carry across the resolution/codec shift (consistent
  with Q2.01; that shift is a hypothesis, not tested here).

## What to ship

* **On IndEgo-comparable footage**, ship the pure deterministic rules
  (`tools/phys`, `reject_threshold = 14`, the 5%-FR operating point) as the
  product's deterministic mode. It is bit-reproducible with no learned weights,
  and in-domain it trails the frozen ensemble by 0.011 AUROC (not a resolved
  difference). The optional fallback is *not* deterministic (it needs the learned
  ensemble and a hashed source file), so it cannot be part of a
  deterministic-only product; keep it as an explicit "assisted" mode only if the
  product accepts a model dependency.
* **Do not ship the deterministic table on OOD/factory footage.** It is not
  distinguishable from chance and flags ~100% of windows (0% ACCEPT). A
  re-derived per-regime model is needed; this PR does **not** establish one
  (the Q2 "O2/O4 0.65–0.70" pointer is not a committed result).

## Leakage and provenance (inherited from Q1)

* The candidate rule list, the feature choice and the directions in
  `tools/phys/rules.json` come from Q1 and were selected by the author **with all
  150 IndEgo labels visible**. Only the fold thresholds and the reject operating
  point are nested in LOCO, so the gold AUROC is an **optimistic in-domain
  estimate**. The shipped `rules.json` is derived from all 7 clips; its
  **in-sample** AUROC (0.8239) is not a held-out number.
* The optional fallback is a frozen learned model; the hybrid result is therefore
  **not** a deterministic-product result. Its source is bound into the result and
  into each receipt (`fallback.source_sha256`).

## Consistency with the Q1 report (same model, two implementations)

The same rule list is reported as AUROC **0.8222** here and **0.8224** in the Q1
report (`m/phys-rules`). The 0.0002 difference is an implementation artefact:
this package compares **integer-quantised** thresholds (`model.quantise`,
half-up), while Q1 compares the 4-dp **float** thresholds; the bootstrap also now
uses the same `numpy.default_rng(seed=0)` in both. The difference is immaterial
and the two point estimates should be quoted as the same model; the packaged
integer scorer's number (**0.8222 [0.715, 0.860]**) is the shipping number.

## Traceability

`Q4.01.json:gold_loco.per_window_scores` (150 rows) and
`Q4.01.json:ood160.per_window_scores` (160 rows) carry `{id, clip/factory, y,
score, decision, p_ens/usable_prob, hybrid_decision}`; every metric above is
re-derivable from those rows plus the paired-bootstrap blocks
(`gold_loco.paired_clip_bootstrap`, `ood160.paired_factory_bootstrap`).

## Residual accuracy gap

* In-domain: deterministic AUROC 0.822 vs ensemble 0.834 → **0.011**, not a
  resolved difference (paired CI contains 0); at the shipped operating point the
  model decides only 40% of windows and returns REVIEW for 60%. The gap is in
  coverage/operating point, not in a demonstrated ranking difference.
* Out-of-domain: deterministic 0.427 vs ensemble 0.566 → 0.139 (paired CI
  contains 0), both at/below chance. This is the honest residual gap that a
  deterministic product must accept.

## Caveats

* Thresholds are derived from **7 IndEgo clips**; they transfer only to
  comparable 480p/10fps footage.
* OOD features for the 160 windows were extracted with the same vendored kernels
  (`eval/exp_phys/extract_ood_full.py`, 160 windows, 2 workers, 99 s wall clock,
  0 errors); gold uses the cached Q1 features.
* The frozen `rule_table.json` records the fr5 operating point; Q1 also derives a
  BA-maximising point (BA 0.678 on 62% coverage). This package ships the fr5 one.
* `tools/phys/vendor/` duplicates the extraction kernels that also live under
  `jadal/`; re-sync deliberately (drift risk).
