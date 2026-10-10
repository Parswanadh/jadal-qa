# Funnel report — deterministic accept/reject/review router (issue #26)

Branch `feat/funnel`. Worktree `/home/parshu/projects/jadal-qa-wt/funnel`.
Python `PYTHONNOUSERSITE=1 /home/parshu/envs/jadal/bin/python` throughout; nothing installed.

## What this is

Awign's human QA team should only see clips worth their time. `jadal/funnel.py:route()`
is the single place a verdict is decided: L0 hard rules reject deterministically with
evidence, L1 heads can only demote to `review`, clean + confident clips auto-accept,
everything else queues for QA ranked by uncertainty. Voice untouched.

## Phase 1 — interface (DONE, pushed)

`route(signals, decisions, thresholds)` is pure (no I/O, randomness, clock).
Choices, all documented in the module docstring:

- `answer_conf` = min of available L1 head probs (weakest head governs accept).
- `queue_priority` = mean per-head `1 - |2p - 1|` for review (higher = more
  uncertain), `0.0` for accept/reject, `1.0` when L1 is entirely missing;
  ties break by segment_id via `queue_rank_key()`.
- Missing L0 measurement or missing L1 decision => `review` (`LOW_CONFIDENCE`),
  never reject, never accept, never crash. Static motion => `NO_ACTIVE_WORK`,
  too-short/undecodable => `UNDECODABLE` (no new reason codes).
- `thresholds_hash()` pins the config minus its own `sha256` field.
- `configs/funnel_thresholds.json` = PLACEHOLDERS from `jadal.config.DEFAULT`
  (`accept_conf` 0.9, `motion_min` 0.5, `under_exposed_frac_max` 0.35,
  `min_duration_s` 5.0) + `calibration.status: placeholder`. No bound claimed.

Tests: `tests/test_funnel.py`, 35 tests, all pass
(`python -m unittest tests.test_funnel` → OK). Covers one test per hard rule,
L1-cannot-reject x4, multi-failure stable ordering, determinism hash,
boundaries, missing fields, calibration (6 tests), threshold helpers.

Integration owner: `from jadal.funnel import route, load_thresholds`;
`route(signals, decisions, load_thresholds())`. Thresholds without
`accept_conf` disable auto-accept; `null` rule thresholds disable that rule.

## Phase 2 — two-sided calibration (machinery DONE, gold numbers PENDING)

`calibrate_two_sided(records, targets)` + `eval/calibrate_funnel.py`.
Stratified seeded calib/heldout split; accept via `pick_threshold` on confs
floored to a fixed `accept_step` (default 0.05) grid — without this,
continuous confs stop the fixed-sequence scan at the first candidate
(documented in code); reject side scans strictest→loosest per rule with a
label-independent support floor `max(min_support, ceil(log(delta)/log(1-beta)))`
(the exact bound with 0 errors on fewer clips already exceeds beta).
Every bound reported overall AND per label source; unknown sources raise.

Synthetic machinery check (labels synthetic, MUST NOT be quoted as real):

```
python eval/calibrate_funnel.py --synthetic 1200 --seed 7
records=1200 (calib=600, heldout=600) sources={'synthetic': 1200}
accept_conf=0.4 heldout defect errors=9/364, upper_bound=0.0387 (delta=0.1)
blur_var_min=90.5 heldout FR 0/88 bound=0.0274 | brightness_min=80.9 1/92 0.0435
over_exposed_frac_max=0.0497 1/89 0.043 | motion_max=7.97 1/71 0.0537
motion_min=2.03 2/68 0.0764 | hands_visible_frac_min=0.70 8/92 0.1374
under_exposed_frac_max + min_duration_s: disabled (never fire in fixtures)
```

Saved to `eval/results/funnel_calibration_synthetic.json`. The live
`configs/funnel_thresholds.json` was deliberately NOT overwritten with
synthetic numbers — placeholders stay until human gold labels exist, then:

```
python eval/calibrate_funnel.py --records eval/gold/labels.json --write-config
```

which writes `eval/results/funnel_calibration.json` and updates the config
(calibration block: sources, N per source, bounds, seed, date, disabled rules,
fresh `sha256`). Two bugs found and fixed while validating (both covered by
tests): accept-side heldout counted good-among-accepted as errors (inverted —
defects are bad-among-accepted); `%`-formatting in the label page.

## Phase 3 — gold labels (tool DONE + verified; labelling session PENDING)

- Manifest: `eval/gold/segments.json`, 150 x 10 s segments, deterministic
  (largest-remainder counts over durations, evenly spaced starts, no RNG):
  ecu_full 48, ecu_late 14, desk 29, pallet2 6, pallet 17, organise 19,
  logistics4 17. Clips read-only, never copied into the repo.
- Labelling: `tools/gold_label/label_server.py` (stdlib only). Start with:

```
PYTHONNOUSERSITE=1 /home/parshu/envs/jadal/bin/python tools/gold_label/label_server.py
```

  then open http://127.0.0.1:8811/ — 6-frame contact sheets (OpenCV JPEGs,
  cached `data_runtime/gold/`), GOOD/BAD + keys g/b/s/skip/u/undo, reason
  dropdown (existing codes, required for BAD), note, progress + ETA, NO model
  verdict shown. Clicks save atomically to `eval/gold/labels.json`
  (append-only history + latest map; resume-safe).
- Verified live: label POST → 200, undo round-trip, frame JPEGs 360x360, and
  Playwright screenshots (system Chrome — no MCP playwright tool in this
  environment) inspected: header, progress, buttons above the grid, 6 frames.

## Phase 4 — scale + funnel report (DONE on fixtures)

`eval/scale_bench.py --limit 150` (all 150 manifest segments, 10 fixed frames
each, real IndEgo decode on THIS laptop; L0 = documented PROXY stats since
`jadal.signals` is not on main; `jadal.decision` not importable so L1 skipped
after checking `nvidia-smi` — GPU showed 17% from another worker, untouched):

| workers | seg/h | decode mean | l0_proxy mean | route mean |
|---|---|---|---|---|
| 1 | 16579.5 | 213.5 ms | 2.9 ms | 0.06 ms |
| 2 | 24419.8 | 289.8 ms | 3.9 ms | 0.08 ms |
| 4 | 28917.8 | 490.2 ms | 4.7 ms | 0.10 ms |
| 8 | 29424.2 | 960.6 ms | 6.0 ms | 0.13 ms |

Decode-bound; little gain past P=4 on this disk. Cross-process determinism:
IDENTICAL route() hashes for all 150 segments across P={1,2,4,8}. Scale-out is
by processes, never batching, one segment at a time. Saved to
`eval/results/funnel_scale.json`. Bench verdict mix (146 reject / 4 accept) is
a FIXTURE artifact: motion is differenced on 1 s-apart frames, so 144/150 trip
the placeholder SHAKY gate — speed numbers only, no quality claim.

`eval/funnel_report.py --synthetic 1200 --seed 7` (placeholder thresholds,
labels synthetic, N=1200): split 59.25% accept / 40.75% reject / 0% review;
false-reject bound 0/489 → 0.0047; defect bound 0/711 → 0.0032 (delta=0.1).
Enrichment is vacuous on these separable fixtures (queue defect rate 0) —
real gold labels will populate all three bins. Saved to
`eval/results/funnel_report_synthetic.json`. No QA-cost illustration (no
user-supplied minutes-per-review; `--minutes-per-review` supported).

## NOT DONE

- Human gold labels (user session) → then gold calibration + config update.
- Real `jadal.signals` / `jadal.decision` swap-ins (bench + report rerun).
- Funnel wiring into `jadal/pipeline.py` (integration owner; do NOT merge this
  branch — PR only).
