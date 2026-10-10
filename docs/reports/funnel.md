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

Auto-accept additionally requires `thresholds['calibration']['status']` ==
`calibrated`, unless the caller passes `allow_provisional_accept=True`
(provisional: synthetic/Gemini labels; every such accept carries
`accept_status` evidence with the label source) or
`allow_uncalibrated_accept=True` (placeholder/demo pattern; the flag lands
in evidence). The shipped config has `accept_conf: null` and status
`placeholder`, with a real pinned `sha256` that `load_thresholds()` verifies
(tampering raises). `hard_reject_policy` defaults to `integrity_only`
(only undecodable/too-short reject; quality rules review — a false reject
costs worker pay plus a re-shoot); `evidence_reject` enables calibrated
quality rejects. The hands rule is task-scoped via `profiles`
(`assembly` requires hands; `logistics`/`default` review instead).

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
label source, never pooled: clip-group-based SELECT/CERTIFY split (same clip
never spans the split — with 7 clips a 4-clip SELECT / 3-clip CERTIFY
partition by hash order), select with fixed-sequence on SELECT, WRITE only
on CERTIFY gates (reject: bound <= beta at Bonferroni delta/m and support
>= min_support; accept: bound <= alpha and accepted >= min_support), else
threshold null + reason (certification stopped a bad rule). Beside EVERY
bound the output stores provenance: n_segments / n_clips / n_wearers
(parsed from clip names, else `unknown`) / n_labellers (gold file ids;
synthetic/gemini => 0) / label_source / per-rule delta. Union
false-reject rate of route() measured on CERTIFY with its exact bound
(per-rule bounds do not bound the union — stated). I5: `min_accepted` can
leave `accept_conf` null on low-prevalence data — when SELECT holds fewer
than `min_accepted` high-confidence clips, no candidate is selected and
auto-accept stays disabled (threshold null + note); that is the honest
output, not a failure to look.

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
- Verified live: label POST → 200, undo round-trip, frame JPEGs served.
  (M9: the earlier draft claimed Playwright screenshots were inspected —
  no screenshot is on the branch, so the claim is removed. The server is
  another worker's file; only the join contract above is funnel-owned.)

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

---

## Round 2 (audit L3/L5/L8/L9, N1; open items I5, M3, M4, M9)

Shipped config `configs/funnel_thresholds.json` now carries
`hard_reject_policy: integrity_only` and a `profiles` map
(`assembly` requires hands; `logistics`/`default` do not), still
`accept_conf: null` + status `placeholder`, sha
`9a707f41b65a5ff3f453e15ff2660585d1b4723b98631a946b934bb32dd0dc8a`
(verified by `load_thresholds()`).

1. PROVISIONAL ACCEPT SAFETY (N1): `route()` auto-accepts by default only
   on `status == 'calibrated'`. `provisional` reviews unless
   `allow_provisional_accept=True` is passed, and every such accept
   carries `{check: 'accept_status', value: 'provisional', label_source}`
   evidence (pinned by test). `--write-config` from a non-human source
   refuses to set `accept_conf` unless `--allow-provisional` is given
   (then status stays `provisional`, source stated, `accept_conf` written).
2. TASK-SCOPED HANDS RULE (L8): `route(..., profile=...)` hard-rejects a
   hands violation only where the profile requires hands; elsewhere it
   reviews with evidence, never rejects (walking with a pallet IS the
   task). No `profiles` map keeps legacy enforcement.
3. POLICY SWITCH (L9): `integrity_only` (default — only
   undecodable/too-short reject; a false reject costs worker pay plus a
   re-shoot) vs `evidence_reject` (calibrated L0 quality rejects).
   Unknown policy values raise `ValueError`.
4. DUPLICATE (L5): never rejects on a degenerate 64-bit phash (all-zero
   hex, popcount <= 4 / >= 60) and never in `integrity_only` mode —
   review with evidence either way.
5. CALIBRATION PROVENANCE (L3/L7): split is by CLIP GROUP (7 clips ->
   deterministic 4-SELECT / 3-CERTIFY partition, pinned by test);
   `provenance` (n_segments / n_clips / n_wearers / n_labellers /
   label_source / delta_per_rule) is printed and stored beside EVERY
   bound; reject rules are tested at Bonferroni `delta/m` by default
   (`delta_per_rule=0.0125`, support floor 86; `--no-bonferroni` recovers
   round-1, stated as not family-wise). A rule uncertifiable on held-out
   clips is DISABLED and the tables say `STOPPED (...)` plainly.
   `eval/calibrate_funnel.py` takes `--gold PATH` (default
   `eval/gold/labels.json`); with no labels it exits saying `no human
   gold labels yet` and everything stays provisional/synthetic.
   M3: `produced_utc` moved to a `sidecar` field excluded from every hash
   (`calibration_content_hash`, pinned by test). M4: reject `reasons` are
   `sorted(hard | soft)` — L1 soft codes ride along as uncertainty
   context, documented in `route()` (pinned by test).

### Calibration artifact (regenerated): eval/results/funnel_calibration_synthetic.json

`eval/calibrate_funnel.py --synthetic 1200 --seed 7`
(provenance=synthetic-fixtures — machinery check, never human-grade;
Bonferroni default, `delta_per_rule=0.0125`):

records=1200 sources={'synthetic': 1200} seed=7 bonferroni=True delta_per_rule=0.0125

## source=synthetic (N=1200, select=600, certify=600; n_segments=1200, n_clips=0, n_wearers=unknown, n_labellers=0, label_source=synthetic, delta_per_rule=0.0125)
| side | threshold | certify fired | certify errors | certify UCB | gate |
|---|---|---|---|---|---|
| accept | 0.4 | 364 | 9 | 0.0387 | WRITTEN |
| rule | threshold | disabled | certify fired | certify errors | certify UCB |
|---|---|---|---|---|---|
| blur_var_min | None | True | 0 | 0 | None |
| brightness_min | 80.1498 | False | 88 | 0 | 0.0486 |
| under_exposed_frac_max | None | True | 0 | 0 | None |
| over_exposed_frac_max | None | True | 0 | 0 | None |
| motion_max | None | True | 0 | 0 | None |
| motion_min | None | True | 0 | 0 | None |
| hands_visible_frac_min | None | True | 0 | 0 | None |
| min_duration_s | None | True | 0 | 0 | None |
union CERTIFY (policy=evidence_reject, profile=default): rejected=88 errors=0 UCB=0.0258 (n_certify=600, n_segments=1200, n_clips=0, n_wearers=unknown, n_labellers=0, label_source=synthetic)
union_integrity_only CERTIFY (policy=integrity_only, profile=default): rejected=0 errors=0 UCB=None (n_certify=600, n_segments=1200, n_clips=0, n_wearers=unknown, n_labellers=0, label_source=synthetic)

Read it plainly: family-wise control at this N certifies almost nothing
— only the accept side and brightness survive; every other rule is
DISABLED with its reason (certification stopped bad rules, and the
support floor 86 says this N cannot support family-wise quality
rejects). The union is measured under BOTH policies, labelled: under the
certified `evidence_reject` policy the calibrated rule set rejects 88/600
with 0 false rejects (UCB 0.0258); under the shipped `integrity_only`
default it is structurally 0 (quality rules review).

### Funnel report artifact (regenerated): eval/results/funnel_report_synthetic.json

`eval/funnel_report.py --synthetic 1200 --seed 7 --allow-uncalibrated`
(shipped placeholder config, `integrity_only`): split accept=0 /
reject=0 / review=1200 (100%). Placeholders reject nothing and accept
nothing — the whole queue goes to humans (FR n/a, defect n/a, QA
workload reduction 0.0). That is the safe default working as designed.

### Scale artifact (regenerated): eval/results/funnel_scale.json

`eval/scale_bench.py --limit 150` (L0 = documented PROXY;
`jadal.decision` absent so L1 skipped; GPU at 2% from another worker):

| workers | seg/h | decode mean | l0_proxy mean | route mean |
|---|---|---|---|---|
| 1 | 17417.5 | 203.539 ms | 2.683 ms | 0.072 ms |
| 2 | 25962.4 | 273.031 ms | 3.392 ms | 0.087 ms |
| 4 | 31125.8 | 455.452 ms | 4.124 ms | 0.108 ms |
| 8 | 31953.4 | 885.227 ms | 5.2 ms | 0.131 ms |

Decode-bound; little gain past P=4. Cross-process determinism: IDENTICAL
route() hashes, 150 segments x 4 worker counts. Bench verdict mix is
150 review (safe-default artifact: placeholder + `integrity_only`).

### Verification commands (all ran)

- `python -m unittest discover -s tests -p 'test_funnel*'` → Ran 90
  tests, OK (50 round-1 + 40 round-2; three round-1 calibration tests now
  run with explicit `bonferroni: False` as mechanics tests, and the
  provisional test asserts the new N1 gating — noted, not hidden).
- `python -m unittest discover -s tests` → Ran 344 tests, OK (whole
  suite green in this worktree).

---

## Fix round 1 (review of PR #42: SPEC FAIL → fixes; all numbers below are
generated from the committed JSON artifacts, not hand-typed)

Rulings addressed, with evidence:

- Critical 1 (garbage `decodable` accepted): `route()` now treats
  `decodable` as a STRICT boolean — only a real `True` passes, an
  explicit `False` rejects `UNDECODABLE`, anything else (`0.0`, `"0"`,
  `2`, `0.5`, `None`, ...) reviews with `invalid:decodable` evidence
  (never a numeric parse, never accept). Same strictness everywhere else
  stands: bools/strings/NaN/inf never parse as numbers. New 5000-case
  seeded fuzz (type confusion: numeric strings, bools-as-numbers, nested
  dicts, huge ints, NaN) proves accept is impossible unless every
  required field is valid and passing. Minor 8 folded in: evidence
  values are always `float|int|str` (bools/`None`/non-finite/garbage
  encoded). Minor 9 documented: `dup_of` with no usable `phash` stands
  alone (rejects under `evidence_reject`).
- Important 2 (union under the wrong policy): the union false-reject
  bound is now measured under BOTH policies, labelled, from
  `eval/results/funnel_calibration_synthetic.json`: `union`
  (policy=`evidence_reject`) rejected=88 errors=0 UCB=0.0258
  (n_certify=600); `union_integrity_only` rejected=0 errors=0 UCB=None.
  The certificate now describes the calibrated rule set.
- Important 3 (provenance beside every bound): each reject-rule entry,
  the accept certify block, and both union blocks carry their own
  `provenance` (`n_segments=1200, n_clips=0, n_wearers=unknown,
  n_labellers=0, label_source=synthetic, delta_per_rule=0.0125`), copied
  into the config's `accept_certify`/`union_certify` by `--write-config`.
- Important 4 (`--write-config` did not refuse): a certified non-human
  `accept_conf` without `--allow-provisional` is now REFUSED — non-zero
  exit with `REFUSING ... Nothing was written` before any write
  (verified: config file byte-identical after the refusal). Nothing to
  refuse (uncertified accept) still writes with auto-accept off.
- Important 5 (label source trusted from the file): `--from-gold`
  schema-checks the labels file (history + latest, verdicts good|bad —
  the exact shape `tools/gold_label/label_server.py` writes) and REFUSES
  anything else; labeller ids are collected when present (the current
  tool records none → `unknown`, said plainly). `--records` files can
  never establish human provenance: a records file that self-declares
  `human_gold` is REFUSED, and `--human-gold` is REFUSED with `--records`
  (SystemExit pointing to `--from-gold`). A records file therefore reaches
  at most `status: provisional`. `status: calibrated` requires the
  schema-checked `--from-gold` path (human provenance established).
- Important 6 (demo numbers not from JSON): the "Demo-only illustration"
  paragraph is DELETED (its numbers lived only in /tmp). No number in
  this report comes from outside a committed artifact.
- Minor 7: a round-1 test now pins the shipped default explicitly
  (no `hard_reject_policy` key → `integrity_only`).
- Merge-gate blocker: `test_missing_signals_module_raises_clearly` now
  SIMULATES the absence (`sys.modules['jadal.signals']=None` in a
  context manager restoring by key diff), so it passes with and without
  the real module. Verified both ways: this worktree (no
  `jadal.signals`) AND a throwaway worktree with `origin/feat/signals`
  merged (no `data_runtime/`) — funnel suite green in both.

### Verification commands (all ran)

- `python -m unittest discover -s tests -p 'test_funnel*'` → Ran 107
  tests, OK (90 round-1/2 + 17 fix-round-1: strict-decodable, 5000-fuzz,
  dual-union, per-bound provenance, gold schema, CLI refusal).
- Review probe `reviews/funnel2_probe.py` (51 checks): the 10 round-2
  failures are fixed. Two probe expectations are superseded by these
  rulings, not by code gaps — 1g asserted the old silent-null behaviour
  (ruling 4 now demands refusal; the probe exits there with
  `REFUSING ... Nothing was written`), and 5j hard-codes `False`
   (unpassable as written). A byte-identical copy of the probe with only
   those two checks updated to the post-fix contracts passes 51/51
   (`/tmp` copy, reviewer's file untouched).

---

## Round 3 (issue #56: autonomous mode + calibrate on the 150 human gold labels)

Branch `feat/funnel3`. Every number below is generated from the committed
JSON artifacts, not hand-typed. Reproduce with the commands shown
(`PYTHONNOUSERSITE=1 /home/parshu/envs/jadal/bin/python` throughout,
CPU only, `JADAL_MODELS_DIR=/home/parshu/projects/jadal-qa/models`
for the signals step).

### Autonomous mode (DONE, shipped default unchanged)

`thresholds['mode']`: `'supervised'` (DEFAULT when absent — every existing
test passes unchanged) or `'autonomous'` (anything else raises
`ValueError`; a `mode=` argument overrides the file). Autonomous has no
`review` bucket: confident accepts and evidence rejects are unchanged,
and every clip supervised mode would review instead rejects with
`LOW_CONFIDENCE` (soft L1 codes ride along as context, plus an
`autonomous` marker evidence entry with a value/op/threshold triple).
L1 heads still never hard-reject; non-boolean `decodable` and missing L0
measurements still fail closed (now to reject, never accept). The shipped
`configs/funnel_thresholds.json` gains `"mode": "supervised"` (sha
`304b96a418e8819d840989bc0cc5ad336fe32441c5205aee65b45b158b7b1e3d`).
`calibrate_two_sided` additionally measures `union_autonomous` on CERTIFY
at full delta under BOTH policies (fix round 1: `union_autonomous` for
`evidence_reject`, `union_autonomous_integrity_only` for `integrity_only`,
mirroring the supervised dual-union — `route()` permits
autonomous+integrity_only, whose larger fallback reject set needs its own
bound): evidence rejects PLUS fallback rejects — these bounds INCLUDE the
LOW_CONFIDENCE count and ARE the safety claim with no human backstop.
New `tests/test_funnel_autonomous.py` (13 tests): 3000-case
seeded fuzz (autonomous never emits `review`; every reject carries
evidence with check/value/op/threshold/code), mode switch,
mode-argument-wins-over-file precedence, L1-never-rejects,
garbage-decodable fails closed, `union_autonomous` fired >= `union` fired
under both policies, tampered-thresholds-file raises, forged `human_gold`
records without the schema-checked gold path stay `provisional`.

### Human gold snapshot (READ-ONLY source, copied once)

Live file `/home/parshu/projects/jadal-qa-wt/funnel/eval/gold/labels.json`
(never written by this worker) snapshotted to
`eval/results/gold_labels_snapshot.json`
(sha256 `43ab44e5…cab062dc`, identical before/after):
`latest` = 150 entries: 100 good / 50 bad; bad reasons NO_ACTIVE_WORK 23,
HANDS_NOT_VISIBLE 14, DUPLICATE 6, TOO_DARK 6, UNDECODABLE 1.
Manifest `eval/gold/segments.json` (identical to the funnel worktree's):
150 x 10 s windows from 7 clips (ecu_full 48, desk 29, organise 19,
pallet 17, logistics4 17, ecu_late 14, pallet2 6).

### Real L0 signals (DONE): eval/results/gold_l0_signals.json

```
JADAL_MODELS_DIR=/home/parshu/projects/jadal-qa/models PYTHONNOUSERSITE=1 \
/home/parshu/envs/jadal/bin/python eval/compute_gold_l0.py --jobs 6
```

Cut exactly [start_s, end_s) per window (cached under git-ignored
`data_runtime/gold_cuts/`, sources read-only) + real
`jadal.signals.compute_signals`, 6 processes, then the exact
`build_gold_records` join (decisions None — no L1 yet; dup by exact phash
in manifest order). Backends recorded in the JSON: faces mediapipe,
hands mediapipe, mediapipe 1.1.0, opencv 5.0.0, face/hand model sha256
pinned. 150/150 windows.

### Certified calibration: eval/results/funnel_gold_calibration.json
(NB: named `funnel_gold_*`, not `funnel_calibration_*`, so the service's
`funnel_calibration*.json` fallback glob keeps surfacing the shipped
synthetic-labelled file — verified by
`test_stats_and_sop_endpoints_work`.)

Same command as above (calibration half: `calibrate_two_sided` defaults —
alpha/beta 0.05, delta 0.1, Bonferroni delta/m = 0.0125, seed 1234 — with
schema-checked `human_provenance` attached):

```
## source=human_gold (N=150, select=66 (4 clips), certify=84 (3 clips);
n_segments=150, n_clips=7, n_wearers=3, n_labellers=unknown,
label_source=human_gold, delta_per_rule=0.0125)
| side | threshold | certify fired | certify errors | certify UCB | gate |
|---|---|---|---|---|---|
| accept | None | 0 | 0 | None | WRITTEN |
(every reject rule DISABLED: strictest supported candidate already exceeds
target_beta — support floor 86)
union CERTIFY (evidence_reject): rejected=0 errors=0 UCB=None (n_certify=84)
union_autonomous CERTIFY: rejected=84 errors=58 UCB=0.757 (n_certify=84)
union_autonomous_integrity_only CERTIFY: rejected=84 errors=58 UCB=0.757
(n_certify=84; identical here because decisions None rejects everything
under both policies)
```

Read it plainly. **Nothing certifies at N=150 with 7 clips**: the
Bonferroni support floor (86 clean fires) exceeds the whole CERTIFY split
(84 segments on 3 clips), so all 8 rules are DISABLED and no rule could
have passed in principle. `accept_conf` is null and uncalibratable — no
decider score exists for these windows (`spectral_data/arms/E/gold.json`
carries 79 embeddings, every score null; said so, not worked around).
The evidence-reject union fires 0 times (UCB None: no measurable claim).
The autonomous union rejects all 84 CERTIFY clips with 58 false
(UCB 0.757): with decisions None there is no confidence signal, so
autonomous mode rejects everything, goods included. The 7-clip split
means this openly: SELECT is 4 clips, CERTIFY 3 — one odd clip swings
every bound, and wearer coverage is 3 ids with unknown labeller.
`configs/funnel_thresholds.gold150.provisional.json` is therefore written
with status **`provisional`** (all quality thresholds nulled/disabled,
`load_thresholds()`-verified) and says why in its note — no placeholder
is presented as calibrated. The filename states the content: gold150 +
provisional, never "calibrated". Regenerated ONLY through the fixed gate
(`eval/compute_gold_l0.py --write-config`, which calls
`apply_to_config`); the exact command is in the verification list below.

### Descriptive only (NO certificate): eval/results/funnel_gold_descriptive.json

```
PYTHONNOUSERSITE=1 /home/parshu/envs/jadal/bin/python eval/analyze_gold_l0.py
```

Routing all 150 gold windows at SHIPPED placeholder thresholds under
`evidence_reject` (decisions None, so 0 accepts anywhere — description,
not evaluation): bad 19 reject / 31 review; good 12 reject / 88 review.
L0-checkable bads (TOO_DARK+DUPLICATE+UNDECODABLE = 13): 7 reject at
placeholders — the TOO_DARK rule catches 6/6 TOO_DARK bads at 1 false
(brightness medians 19.0 bad vs 72.1 good: separable); exact-phash
duplicate catches 0/6 human DUPLICATEs (near-dups, not bit-copies).
HANDS_NOT_VISIBLE: 13/14 reject under the assembly profile
(requires_hands). NO_ACTIVE_WORK (23, not L0-checkable per the brief):
motion medians good 9.32 vs bad 9.32 — pixel motion does not separate
human activity judgement at all; this 46% of the bad set is what the
learned/spectral tier must catch. Blur never fires (good median 738 vs
placeholder 60; zero BLUR-labelled bads). SHAKY at placeholder 12.0 is
noisy (11 goods + 11 bads fire) — exactly the kind of rule certification
stops, and here it did.

### Verification commands (all ran)

- `python -m unittest discover -s tests -p 'test_funnel*'` → Ran 123
  tests, OK (53 round-1 + 55 round-2/fix-round-1 + 15 autonomous/gate).
- `PYTHONNOUSERSITE=1 timeout 250 /home/parshu/envs/jadal/bin/python -m
  unittest discover -s tests` → Ran 568 tests, OK (skipped=8), 54.4 s.
  Two service tests needed attention after the legitimate
  thresholds change and are green again: `receipt.json` in
  `docs/api_examples/` regenerated via the documented
  `scripts/gen_api_examples.py` (receipt hashes cover the thresholds, and
  batch receipts list `ORDER BY receipt_id`, so the `mode` key reshuffles
  which receipt is first — the regeneration is the design working, not
  drift; `pico_status.json` hardware values left untouched), and the gold
  calibration file renamed out of the service glob (above).
- End-to-end gold commands above; all outputs committed under
  `eval/results/` with snapshot sha `43ab44e5…` and mediapipe backends
  pinned beside the signals.
- Fix-round-1 regeneration (the committed provisional config's provenance):
  `cp configs/funnel_thresholds.json
  configs/funnel_thresholds.gold150.provisional.json`, then
  `JADAL_MODELS_DIR=/home/parshu/projects/jadal-qa/models
  PYTHONNOUSERSITE=1 /home/parshu/envs/jadal/bin/python
  eval/compute_gold_l0.py --jobs 6 --write-config
  configs/funnel_thresholds.gold150.provisional.json` → status=provisional
  ("no threshold certified on CERTIFY: accept null, every reject rule
  disabled, and the union fired 0 with bound None (target 0.05)"),
  `load_thresholds()`-verified on load.

### NOT DONE / caveats

- No certified L0 threshold exists at this N (provisional config only);
  autonomous operation has no safety bound below UCB 0.757 until a
  confidence signal (decider scores on these windows) enables accept
  calibration and more clips lift the support floor.
- Single labelling session, labeller id unknown; 7 IndEgo clips, 3
  wearers — nothing here transfers to Awign footage without re-calibration.
- Learned/spectral tier for the 23 NO_ACTIVE_WORK bads is other workers'
  scope; this report only bounds what L0 cannot do.

---

## Fix round 1 (review of PR #59: REFUSE → fixes; all numbers below are
generated from the committed JSON artifacts, not hand-typed)

Two Important findings blocked approval; both are fixed in owned files
only (protected files untouched — verified: no diff in `docs/SPEC.md`,
`docs/BASELINE.md`, `jadal/config.py`, `jadal/__init__.py`; the service
`receipt.json` regen was mechanical, see above).

- Important 1 (vacuous `calibrated` gate, proven by the reviewer's
  `probe1_vacuous_gate.py`): `written_bounds_ok`
  (`eval/calibrate_funnel.py`) returned `(True, [])` when NOTHING was
  written. It now requires ≥1 written threshold (each meeting its bound),
  or — with nothing written — a measured union (fired > 0) whose
  false-reject bound meets target_beta; else it fails with "no threshold
  certified on CERTIFY: accept null, every reject rule disabled, and the
  union fired 0 …". Pinned by
  `TestWriteConfigGating.test_vacuous_result_stays_provisional` (watched
  fail with `'calibrated' != 'provisional'` before the fix). The committed
  provisional config is now reproducible by a committed command (see the
  regeneration command above): the reviewer's "NOT reproducible"
  consequence is closed — the file the gate emits carries status
  `provisional` with exactly that note.
- Important 2 (autonomous bound covered one operating point):
  `_calibrate_source` (`jadal/funnel.py`) now measures
  `union_autonomous` AND `union_autonomous_integrity_only` (dual-policy,
  mirroring the supervised dual-union); both carry provenance and ride
  into the config as `union_autonomous_certify` /
  `union_autonomous_integrity_only_certify` and into `--markdown` tables.
  Pinned by
  `TestAutonomousBounds.test_union_autonomous_measured_under_both_policies`
  (watched fail, key missing, before the fix). On the gold data both
  autonomous unions read rejected=84 errors=58 UCB=0.757 (identical here:
  decisions None rejects everything under both policies).
- Minor 1: `configs/funnel_thresholds.calibrated.json` (status
  provisional, name claiming "calibrated") renamed to
  `configs/funnel_thresholds.gold150.provisional.json`, regenerated via
  the fixed gate (not hand-edited).
- Minor 2: (a) `eval/analyze_gold_l0.py` uses ROOT-anchored paths
  (verified runnable with cwd outside the repo); (b)
  `human_provenance.file` is repo-relative
  (`eval/results/gold_labels_snapshot.json`, sha still pins bytes);
  (c) `gold_l0_signals.json` logs all 7 source clips' size_bytes +
  mtime_utc beside `labels_snapshot_sha256`, so reused cuts under
  git-ignored `data_runtime/` cannot go stale silently.
- Minor 3: `TestModeDefault.test_mode_argument_wins_over_file` pins that
  the `mode=` argument wins over the file both directions (plus bogus
  argument raises).

### Verification commands (all ran)

- `python -m unittest discover -s tests -p 'test_funnel*'` → Ran 123
  tests, OK.
- `PYTHONNOUSERSITE=1 timeout 250 /home/parshu/envs/jadal/bin/python -m
  unittest discover -s tests 2>&1 | grep -E '^(Ran|OK|FAILED|ERROR|FAIL:)'
  ` → Ran 568 tests / OK (skipped=8).
- `git fetch origin && git merge origin/main` → clean merge (dashboard3
  screenshots only), suite re-run after the merge.
