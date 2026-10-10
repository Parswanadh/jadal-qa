# exp_phys_ready — deterministic physics model, ready to ship

**Q4.01.** Package the Q1 deterministic physics rule-list model as a runnable,
documented entry point, and quantify pure rules vs a rules + logistic-fallback
hybrid.

* Package: **`tools/phys/`** (rules table, scorer, CLI, extractor, tests, README)
* Results: `eval/results/exp_phys/Q4.01.json`
* Rule table sha256 `52355d72…`, decision-code sha256 `5627b112…`, 16 rules, reject point 14, Q=4.
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
compact) and exits `0` ACCEPT / `1` REJECT / `2` REVIEW. The decision path is
integer-only: every value and threshold is quantised to `10**4` before
comparison, the score is an integer count, and the decision is a pure function of
integers. See `tools/phys/README.md` for the full contract and the threshold
table with its derivation.

## Pure rules vs hybrid

Modes: **(a) pure rules** = the packaged deterministic scorer;
**(b) hybrid** = REVIEW windows are resolved deterministically by the frozen
ensemble_v1 (`accept iff prob >= 0.5`); ACCEPT/REJECT stay deterministic.

### Gold — 150 IndEgo windows (100 good / 50 bad, 7 clips, leave-one-clip-out)

| mode | AUROC [95% CI] | BA [95% CI] | REVIEW | REJECT | ACCEPT | decided det. / by model |
|---|---|---|---|---|---|---|
| pure rules | 0.8222 [0.713, 0.855] | 0.710 (review→accept) / 0.590 (review→reject); **0.808** decided-only | 60.0% | 24.0% | 16.0% | 40% / 60% |
| rules + fallback | 0.8285 hybrid-score [0.730, 0.861] | **0.790** [0.692, 0.828] | 0% (60% before fallback) | 41.3% | 58.7% | 40% / 60% |
| frozen ensemble_v1 (OOF) | 0.8336 | — | — | — | — | — |

Fallback resolves 90 REVIEW windows: 64 → ACCEPT, 26 → REJECT.

### OOD-160 — 160 Egocentric-10K windows (138 good / 22 bad, 47 factories)

| mode | AUROC [95% CI, factory-grouped] | BA [95% CI] | REVIEW | REJECT | ACCEPT | decided det. / by model |
|---|---|---|---|---|---|---|
| pure rules | 0.4272 [0.269, 0.562] | 0.390 (review→accept) / 0.500 (review→reject) | 35.6% | 64.4% | 0.0% | 64.4% / 35.6% |
| rules + fallback | 0.4264 [0.275, 0.556] | **0.468** [0.381, 0.543] | 0% (35.6% before) | 91.9% | 8.1% | 64.4% / 35.6% |
| frozen ensemble_v1 (usable_prob) | 0.566 [0.414, 0.697] | — | — | — | — | — |

Fallback resolves 57 REVIEW windows: 13 → ACCEPT, 44 → REJECT.

## What determinism costs, what the fallback buys

| quantity | gold LOCO | OOD-160 |
|---|---|---|
| determinism cost, AUROC (ensemble − pure) | **+0.0114** | **+0.1388** |
| fallback gain, BA (hybrid − review-as-accept) | **+0.080** | **+0.078** |
| fallback gain, BA (hybrid − review-as-reject) | +0.200 | −0.032 |
| share decided deterministically | 40% | 64% |
| share decided by the model | 60% | 36% |

* In-domain, determinism is cheap: the integer score ranks within 0.011 AUROC of
  the learned ensemble, and the fallback recovers the whole REVIEW band to
  BA 0.790 at full coverage (vs 0.710 treating REVIEW as accept).
* Out-of-domain, determinism is expensive: the frozen table scores **below
  chance** (0.427) and the ensemble only reaches 0.566; the fallback nudges BA
  from 0.390 to 0.468 — still below chance. Absolute IndEgo thresholds do not
  carry across the resolution/codec shift (confirms Q2.01).

## What to ship

* **Ship the pure deterministic rules** (`tools/phys`, `reject_threshold = 14`,
  the 5%-FR operating point) as the product's deterministic mode. It is
  bit-reproducible with no learned weights, and in-domain it trails the frozen
  ensemble by only **0.011 AUROC**. The optional fallback is *not* deterministic
  (it needs the learned ensemble), so it cannot be part of a deterministic-only
  product; keep it as an explicit "assisted" mode only if the product accepts a
  model dependency.
* **Do not ship the deterministic table on OOD/factory footage.** It is below
  chance and flags 100% of windows (0% ACCEPT). Gate OOD with a re-derived
  per-regime model, as Q2.01 concluded.

## Residual accuracy gap

* In-domain: deterministic AUROC 0.822 vs ensemble 0.834 → **0.011**; at the
  shipped operating point the model decides only 40% of windows and returns
  REVIEW for 60%. The gap is in coverage/operating point, not in ranking.
* Out-of-domain: deterministic 0.427 vs ensemble 0.566 → **0.139**, both weak.
  This is the honest residual gap that a deterministic product must accept.

## Caveats

* Thresholds are derived from **7 IndEgo clips**; they transfer only to
  comparable 480p/10fps footage.
* OOD features for the 160 windows were extracted with the same vendored kernels
  (`eval/exp_phys/extract_ood_full.py`, 160 windows, 2 workers, 99 s, 0 errors);
  gold uses the cached Q1 features.
* The frozen `rule_table.json` records the fr5 operating point; Q1 also derives a
  BA-maximising point (BA 0.678 on 62% coverage). This package ships the fr5 one.
