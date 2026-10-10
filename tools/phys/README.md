# Deterministic physics model (`tools/phys`)

A runnable, dependency-light entry point for the **deterministic physics rule-list
scorer** (Q1 rule table). Given a window it emits a canonical JSON receipt and one
of three decisions — `ACCEPT`, `REJECT`, `REVIEW` — with the exact evidence that
produced it.

There are **no learned weights**. The **rule decision path compares quantised
integers only**: feature values and thresholds are converted to integers at
`10**4` *before* any comparison, the score is an integer count, and the decision
is a pure function of integers. The float stages are feature extraction
(decode + fixed kernels, identical to the kernels that produced the frozen
thresholds) and — only in the optional hybrid mode — the logistic fallback
comparison `usable_prob >= 0.5`, which is recorded in the receipt's `fallback`
block with the source sha256.

## Quick start

Score a cached feature record:

```bash
python3 tools/phys/score_window.py --features-json rec.json --window-id g000 --pretty
# exit code: 0 = ACCEPT, 1 = REJECT, 2 = REVIEW
```

Score a clip (decodes with ffmpeg, CPU only):

```bash
python3 tools/phys/score_window.py --clip clip.mp4 --start-s 0 --duration 10 --window-id w1
```

Extract features only:

```bash
python3 tools/phys/extract_features.py --clip clip.mp4 --start-s 0 --duration 10 \
    --out features.json
```

Hybrid mode — resolve a `REVIEW` with the frozen logistic fallback:

```bash
python3 tools/phys/score_window.py --features-json rec.json --window-id g000 \
    --fallback-prob 0.73
python3 tools/phys/score_window.py --features-json rec.json --window-id q000 \
    --fallback-json caches/ensemble/ood_scores2.json
```

Run the tests (fixed inputs → fixed outputs):

```bash
python3 tools/phys/tests/test_phys_model.py          # or: python3 -m pytest tools/phys/tests
```

## Files

| file | role |
|---|---|
| `rules.json` | frozen rule table: 16 (rule, feature, op, strength) rows, integer thresholds, reject point |
| `model.py` | integer-quantised scorer + canonical receipt (the decision code; `code_sha256` hashes this file) |
| `score_window.py` | CLI entry point |
| `extract_features.py` | clip → `full.*` + `phys.*` feature record |
| `vendor/` | the exact extraction kernels, vendored verbatim (`jadal/gpu/features.py`, `jadal/perpixel/{physics,theory,series}.py`) |
| `tests/test_phys_model.py` | deterministic unit tests |

Accepted feature-record shapes for `--features-json`: `{"features": {...}}`, a
flat `{"full.luma_mean": ...}` dict, a `run_features`-style `{"full": {...},
"phys": {...}}` record, or the whole `features.json` (`{"windows": {...}}`).

## Canonical receipt

```json
{
  "code_sha256": "<sha256 of model.py>",
  "decision": "ACCEPT | REJECT | REVIEW",
  "evidence": [{"rule": "...", "feature": "...", "value": <int>, "op": "<|>", "threshold": <int>}],
  "feature_digest": "<sha256 of the quantised values of every feature the table reads>",
  "fired_rules": ["..."],
  "rule_table_sha256": "<sha256 of rules.json>",
  "score": <int>,
  "window_id": "..."
}
```

* Keys are **sorted** and the CLI prints compact canonical JSON by default
  (`--pretty` for indentation). The same window always yields byte-identical
  output.
* `value` and `threshold` are **integers** at `10**4` (`op` is the strict
  comparison that fired).
* `feature_digest` binds the receipt to the exact input features; the two
  `*_sha256` fields bind it to the rule table and to the decision code.
* `decision_exit_code`: `0` ACCEPT, `1` REJECT, `2` REVIEW.

With `--fallback-prob` / `--fallback-json` the receipt additionally carries
`decision_deterministic` (the pre-fallback verdict) and `fallback`
(`{model, prob, rule, decision, source_sha256}`). The fallback only ever runs on
`REVIEW`. `source_sha256` is the sha256 of the `--fallback-json` file (null when
only `--fallback-prob` is given), so the hybrid decision is bound to a hashed
source; the README and `docs/reports/exp_phys_ready.md` state that dependency.

## Decision logic (integer path)

```
score = number of crossed thresholds across all rules   (+ 10**6 if undecodable)
if any QUALITY rule crosses its EXTREME threshold (index 0) -> REJECT
elif score >= reject_threshold   -> REJECT     (reject_threshold = 14)
elif score >= 1                  -> REVIEW
else                             -> ACCEPT
```

A quality rule forces `REJECT` only when it crosses the *extreme* threshold —
`p01` for a low-is-bad feature, `p99` for a high-is-bad feature (index 0 in
`thresholds_q`). Softer crossings of that same quality feature only add to the
integer score and can lead to `REVIEW`. `UNDECODABLE` fires when any of
`full.luma_mean`, `full.motion_energy`, `full.luma_p05` is missing or
non-finite; it is a hard quality rule and forces `REJECT`.

## The rule table and its derivation

Thresholds are **quantiles of the training-clip feature distribution**, quantised
to 4 decimal places (`thresholds_q` are the authoritative integers). For a
low-is-bad feature (`side: "low"`, `op: "<"`) the four thresholds are
`p01, p05, p10, p25`; for a high-is-bad feature (`side: "high"`, `op: ">"`) they
are the mirror `p99, p95, p90, p75` (index 0 is the extreme). A value crossing
*k* thresholds adds *k* to the score; a QUALITY feature that crosses its
**extreme** threshold forces `REJECT`. The reject point is the smallest integer
score whose false-reject rate on training-good windows is `<= 0.05` (the
`fr5_conservative` operating point; Q1 also derives a BA-maximising point but
this package ships the 5%-FR one, `reject_threshold = 14`).

| rule | feature | op | strength | thresholds (`×10⁴`) |
|---|---|---|---|---|
| TOO_DARK | full.luma_mean | < | quality | 723, 1845, 2360, 2678 |
| TOO_DARK | full.dark_frac | > | quality | 7398, 3709, 2394, 1949 |
| OVEREXPOSED | full.bright_frac | > | quality | 979, 459, 251, 57 |
| BLUR | full.hf_ratio | < | quality | 47, 77, 97, 134 |
| BLUR | full.dct_energy | < | quality | 2, 35, 49, 68 |
| FROZEN_STATIC | full.temp_b0 | < | quality | 8563, 47671, 76483, 97065 |
| FROZEN_STATIC | full.frozen_frac | > | quality | 223, 41, 24, 3 |
| NO_ACTIVE_WORK | phys.flow_act | > | weak | 8914, 7970, 7661, 6394 |
| NO_ACTIVE_WORK | phys.flow_med | > | weak | 22787, 20763, 19475, 13544 |
| NO_ACTIVE_WORK | full.motion_energy | > | weak | 734, 669, 624, 502 |
| HANDS_NOT_VISIBLE | full.skin_frac | < | weak | 3, 46, 66, 161 |
| DUPLICATE | phys.dhash_loop_min | > | weak | 120000, 100000, 91000, 70000 |
| DUPLICATE | phys.mad_mean | > | weak | 197433, 158866, 141781, 121200 |
| CODING_NOISE | phys.frm_bytes_med | > | weak | 100000700, 89142500, 82470500, 69572500 |
| CODING_NOISE | phys.p_size_med | > | weak | 99047400, 88492000, 81206000, 67588750 |
| CODING_NOISE | phys.snr_db | > | weak | 189924, 168766, 160496, 148676 |

Source: Q1 frozen table `eval/results/exp_phys/rule_table.json` on `m/phys-rules`
(the exact float thresholds are preserved in `rules.json` as `thresholds`).
`rules.json` records the table sha256 at load time and the CLI echoes it in every
receipt.

## Determinism guarantees

* No randomness, no time, no threads in the rule decision path; the rule
  comparisons are on integers only (the optional logistic fallback is one float
  comparison, `usable_prob >= 0.5`, and is recorded with its source sha256).
* `model.quantise` uses fixed half-up rounding for both signs.
* Comparisons are strict (`<`, `>`), so a value exactly on a threshold never
  crosses it.
* Feature values are read from a fixed, sorted feature list; missing/non-finite
  values are treated as non-crossing, except the `UNDECODABLE` tripwire.
* The receipt is emitted with sorted keys and compact separators, so a given
  `(features, rules.json, model.py)` triple is byte-stable.

## Provenance caveat (leakage)

The candidate rule list, the feature choice and the directions in `rules.json`
were selected by the author with all 150 IndEgo labels visible (see the Q1
report). Only the fold thresholds and the reject operating point are nested, so
the in-domain AUROC is an **optimistic** estimate. The shipped `rules.json` is
derived from all 7 clips; its in-sample AUROC (0.8239) is **not** a held-out
number. Do not quote either as a generalisation estimate.

## Vendored kernels

`vendor/` is a verbatim copy of the extraction kernels that also live under
`jadal/` (and the pod stack). It is duplicated so the package is self-contained;
if the originals change, `vendor/` can **drift**. Re-sync deliberately and re-run
the tests after any kernel change.

## Scope / caveats

* The rule table is derived from **7 IndEgo training clips**; thresholds transfer
  only if the deployment clips are comparable. See
  `docs/reports/exp_phys_ready.md` for the pure-vs-hybrid comparison and the
  measured determinism cost.
* Extraction uses ffmpeg (CPU) at 160×90 and the vendored fixed kernels. It is
  deterministic but not cheap; cache the feature record for repeated scoring.
