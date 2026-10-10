# Funnel report — deterministic accept/reject/review router (issue #26)

Branch `feat/funnel`. Worktree `/home/parshu/projects/jadal-qa-wt/funnel`.
Python `PYTHONNOUSERSITE=1 /home/parshu/envs/jadal/bin/python` throughout; nothing installed.

## What this is

Awign's human QA team should only see clips worth their time. `jadal/funnel.py:route()`
is the single place a verdict is decided: L0 hard rules reject deterministically with
evidence, L1 heads can only demote to `review`, clean + confident clips auto-accept,
everything else queues for QA ranked by uncertainty. Voice untouched.

## Phase 1 — interface (DONE, pushed)

`route(signals, decisions, thresholds, allow_uncalibrated_accept=False)` is pure
(no I/O, randomness, clock). Safety rule (C1): every REQUIRED L0 field
(`decodable` plus each measurement whose rule is configured) goes through a
strict path — missing, non-numeric, NaN/inf or garbage yields `review` with
`LOW_CONFIDENCE` and a `missing:<field>` evidence entry, never accept, never
reject. Only `decodable` explicitly false rejects. Static motion =>
`NO_ACTIVE_WORK`, too-short/undecodable => `UNDECODABLE` (no new reason codes).

Auto-accept additionally requires `thresholds['calibration']['status']` in
(`calibrated`, `provisional`), unless the caller passes
`allow_uncalibrated_accept=True` (demo pattern; the flag lands in evidence).
The shipped config has `accept_conf: null` and status `placeholder`, with a
real pinned `sha256` that `load_thresholds()` verifies (tampering raises).

Choices, all documented in the module docstring: `answer_conf` = min of
available L1 head probs; `queue_priority` = mean per-head `1 - |2p - 1|` for
review (`0.0` accept/reject, `1.0` missing L1); ties break by segment_id.

Tests: `tests/test_funnel.py`, 50 tests, all pass
(`python -m unittest tests.test_funnel` → OK), including a 1000-case seeded
fuzz proving accept is unreachable unless every required L0 field is valid
and passing. Full-suite note: `discover -s tests` has 2 pre-existing
`test_voicebank` failures (Sarvam API 403, needs credentials; unrelated).

Integration owner: `from jadal.funnel import route, load_thresholds`;
`route(signals, decisions, load_thresholds())`. `null` rule thresholds disable
that rule; `null` accept_conf disables auto-accept.

## Phase 2 — two-sided calibration (machinery DONE, gold numbers PENDING)

`calibrate_two_sided(records, targets)` + `eval/calibrate_funnel.py`. Per
label source, never pooled: group-based SELECT/CERTIFY split (same clip never
spans the split), select with fixed-sequence on SELECT, WRITE only on CERTIFY
gates (reject: bound <= beta and support >= min_support; accept: bound <=
alpha and accepted >= min_support), else threshold null + reason. Union
false-reject rate of route() measured on CERTIFY with its exact bound
(no Bonferroni; per-rule bounds do not bound the union — stated).

Gold path (proven end-to-end in a throwaway worktree with real
`jadal.signals` from `origin/feat/signals`): `--from-gold` cuts exactly
[start_s, end_s), runs `compute_signals`, joins labels → human_gold records
(group=clip, decisions None). Verified on 2 segments: full SPEC signals
(blur_var ~830-930 normalised, motion px/sec, hands proxy, phash). After the
labelling session:

```
PYTHONNOUSERSITE=1 /home/parshu/envs/jadal/bin/python eval/calibrate_funnel.py --from-gold --source human_gold --write-config
```

`--write-config` sets status `calibrated` only for human_gold N>=100 with all
CERTIFY bounds met, else `provisional` (source stated).

## Phase 3 — gold labels (tool DONE + verified; labelling session PENDING)

- Manifest: `eval/gold/segments.json`, 150 x 10 s segments, deterministic:
  ecu_full 48, ecu_late 14, desk 29, pallet2 6, pallet 17, organise 19,
  logistics4 17. Clips read-only, never copied into the repo.
- Labelling: `tools/gold_label/label_server.py` (stdlib only, untouched by fix
  round 1). Start with:

```
PYTHONNOUSERSITE=1 /home/parshu/envs/jadal/bin/python tools/gold_label/label_server.py
```

  then open http://127.0.0.1:8811/ — 6-frame contact sheets, GOOD/BAD +
  g/b/s/u keys, reason dropdown (required for BAD), note, progress + ETA, NO
  model verdict shown. Clicks save atomically to `eval/gold/labels.json`.
- Verified live: label POST → 200, undo round-trip, frame JPEGs, Playwright
  screenshots (system Chrome) inspected.

## Phase 4 — scale + funnel report (DONE on fixtures)

Scale (`eval/scale_bench.py --limit 150`; L0 = documented PROXY since
`jadal.signals` is not on main; `jadal.decision` absent so L1 skipped after
`nvidia-smi` showed GPU 17% from another worker, untouched):

| workers | seg/h | decode mean | l0_proxy mean | route mean |
|---|---|---|---|---|
| 1 | 12961.0 | 273.663 ms | 3.332 ms | 0.074 ms |
| 2 | 18179.5 | 391.156 ms | 4.04 ms | 0.091 ms |
| 4 | 23179.1 | 612.987 ms | 5.147 ms | 0.112 ms |
| 8 | 24901.1 | 1135.069 ms | 6.321 ms | 0.162 ms |

Decode-bound; little gain past P=4. Cross-process determinism: IDENTICAL
route() hashes, 150 segments x 4 worker counts. Scale-out by processes, never
batching. Bench verdict mix (146 reject / 4 review) is a FIXTURE artifact
(144 SHAKY: motion on 1 s-apart frames vs placeholder gate). Saved to
`eval/results/funnel_scale.json`. No QA-cost illustration (no user-supplied
minutes-per-review; `--minutes-per-review` supported).

## NOT DONE

- Human gold labels (user session) → then gold calibration + config update.
- Real `jadal.signals` / `jadal.decision` swap-ins (bench + report rerun).
- Funnel wiring into `jadal/pipeline.py` (integration owner; do NOT merge this
  branch — PR only).

---

## Fix round 1 (review of PR #29: SPEC FAIL → fixes; all numbers below are
generated from the JSON artifacts, not hand-typed)

Rulings addressed: C1 strict missing-L0 path + 1000-case fuzz; C2
SELECT/CERTIFY disjoint gating (rules now DISABLE when CERTIFY fails — the
previously-hidden held-out failures gate the output); C3 artifacts
regenerated, tables below are `--markdown` output; C4 per-source only, status
`calibrated` solely for human_gold N>=100 with bounds met; I1 real sha256 +
verify-on-load; I2 `accept_conf: null` + placeholder accept gate with
`allow_uncalibrated_accept` flag; I3 `--from-gold` join proven with real
signals; I4 measured union bound on CERTIFY (no Bonferroni — stated).

### Calibration artifact: eval/results/funnel_calibration_synthetic.json

`eval/calibrate_funnel.py --synthetic 1200 --seed 7`
(provenance=synthetic-fixtures — machinery check, never human-grade):

records=1200 sources={'synthetic': 1200} seed=7

## source=synthetic (N=1200, select=600, certify=600)
| side | threshold | certify fired | certify errors | certify UCB | gate |
|---|---|---|---|---|---|
| accept | 0.4 | 364 | 9 | 0.0387 | WRITTEN |
| rule | threshold | disabled | certify fired | certify errors | certify UCB |
|---|---|---|---|---|---|---|
| blur_var_min | 90.0458 | False | 91 | 0 | 0.025 |
| brightness_min | 80.1498 | False | 88 | 0 | 0.0258 |
| under_exposed_frac_max | None | True | 0 | 0 | None |
| over_exposed_frac_max | 0.0497 | False | 95 | 1 | 0.0403 |
| motion_max | None | True | 70 | 3 | 0.093 |
| motion_min | 2.0011 | False | 69 | 0 | 0.0328 |
| hands_visible_frac_min | None | True | 82 | 2 | 0.0636 |
| min_duration_s | None | True | 0 | 0 | None |
union CERTIFY: rejected=223 errors=1 UCB=0.0173 (n_certify=600)

Read it plainly: motion_max (CERTIFY 3/70 → 0.093) and
hands_visible_frac_min (CERTIFY 2/82 → 0.0636) are now DISABLED — the old
report's hidden held-out failures gate the config. under_exposed_frac_max
and min_duration_s never fire in these fixtures (disabled, said so).
The live `configs/funnel_thresholds.json` was NOT overwritten (placeholders
stay until human gold). Support floor for beta=0.05/delta=0.1 is 45.

### Funnel report artifact: eval/results/funnel_report_synthetic.json

`eval/funnel_report.py --synthetic 1200 --seed 7 --allow-uncalibrated`
(shipped placeholder config: accept_conf null → confident goods review;
per-source section, labels synthetic, N=1200):
split accept=0 / reject=489 (40.75%) / review=711 (59.25%);
false_reject 0/489 → UCB 0.0047; defect among accepted n/a (accepted=0);
queue defect rate 0.0 vs unfiltered 0.4075; QA workload reduction
1 - review share = 0.4075. Enrichment is vacuous on these separable
fixtures — real gold labels will populate all three bins.

### Verification commands (all ran)

- `python -m unittest tests.test_funnel` → Ran 50 tests, OK.
- `discover -s tests`: only the 2 pre-existing voicebank API-credential
  failures (unrelated, out of scope).
- Throwaway-worktree `--from-gold` with real `jadal.signals`: 2/2 records
  built with full SPEC signals (fake labels lived only in /tmp; the branch's
  `eval/gold/` untouched, no `git clean`, no resets).
