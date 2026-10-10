# Claims PoC docs report (docs/claims-poc, issue #74)

Docs-only: README.md, docs/ARCHITECTURE.md, docs/ATTRIBUTIONS.md.
Goal: remove the certified-bound claim, record the edge-first / whole-video
direction, keep every number honest (n + label source + backing pointer).

## Fix round 1 (2026-10-10, answer to PR #74 review)

Review verdict was SPEC: FAIL. All four blocking findings verified correct
against the repo — no pushback, all fixed in owned files only
(README.md, docs/ARCHITECTURE.md; ATTRIBUTIONS.md untouched, no findings there;
jadal/config.py, jadal/__init__.py, docs/SPEC.md and other workers' files untouched).

- C1 (measurement table + speed/VLM/10K numbers unbacked) — confirmed:
  `eval/results/BASELINE.json` does not exist; grep over eval/ + docs/reports/
  finds no backing for any table number (only incidental `0.81` substrings in
  unrelated signal/coverage fields). Fix (b) applied: README section renamed to
  "Status and preliminary builder notes (unverified here)", intro states no
  backing file or re-runnable command exists for any row, all intervals/ranges
  removed (0.71-0.87, 0.63-0.86, 0.85-0.90, 0.47-0.59 gone), point estimates
  kept as "about 0.X, preliminary", every row labelled unverified/unmerged,
  counts (58 clips, 4.404 h, 47 factories, 1,546 windows, 1,214-window 37%,
  60-window, 43-55/620-730 fps, ~700 h/h, 79 labels, 0.57, 1.4 s, 58 GPU-days,
  26 features) deleted or withheld. ARCH CUSUM/SPRT "1.4 s" gate reworded to
  planned direction with timing withheld as unverified.
- I1 (59 / 20-43 / 2.7%->46% unbacked) — confirmed: `docs/reports/funnel.md`
  backs only the 86 floor (lines 228, 267, 460); no hits for the other three
  fragments. Fix applied in README + ARCH: kept "about 86 clean windows (see
  docs/reports/funnel.md: support floor 86)", deleted the 59 / 20-43 /
  2.7%->46% clauses.
- I2 (ARCH inverted SPEC hierarchy) — confirmed: SPEC.md:1 is "the contract
  every module codes against" with fixed frame sampling (SPEC.md:15).
  Fix applied: ARCH intro restored to "where this page and SPEC.md disagree,
  SPEC.md governs" + whole-video/no-sampling marked "planned, pending a
  SPEC.md amendment"; determinism rule 2 reworded the same way. README
  Direction header carries the same one-line guard.
- I3 (off-branch work described as measured) — fixed: table intro + every
  per-pixel/GPU/10K row marked unmerged off-branch work (open PRs #52, #57,
  #61, #66-#73; PoC demo on `integration/poc`, pending merge) with no
  commit/file/reproduce command on this branch; intervals removed per C1.
- M1 (STUB_THRESHOLD pointer) — confirmed: defined in `jadal/stubs.py:25`,
  no such symbol in `jadal/certify.py`. Fix applied: ARCH now cites
  `jadal/stubs.py` explicitly.

Verification probes (all short, real outputs):

```
$ ls eval/results/BASELINE.json
ls: cannot access 'eval/results/BASELINE.json': No such file or directory

$ grep -n "support floor 86" docs/reports/funnel.md
228:   (`delta_per_rule=0.0125`, support floor 86; `--no-bonferroni` recovers
267:support floor 86 says this N cannot support family-wise quality
460:target_beta — support floor 86)

$ grep -n "STUB_THRESHOLD" jadal/stubs.py jadal/certify.py
jadal/stubs.py:25:STUB_THRESHOLD = 0.9  # placeholder until pick_threshold has calibration data
(no hits in jadal/certify.py)

$ grep -n -e "0\.71-0\.87" -e "0\.63-0\.86" -e "0\.85-0\.90" -e "0\.47-0\.59" \
  -e "4\.404" -e "1,546" -e "1,214" -e "60-window" -e "43-55" -e "620-730" \
  -e "20-43" -e "2\.7%" -e "59 with fixed" -e "1\.4 s" README.md docs/ARCHITECTURE.md
(no hits — all interval-carrying/unbacked numbers removed)
```

Tests (real command output):

```
$ PYTHONNOUSERSITE=1 timeout 250 /home/parshu/envs/jadal/bin/python -m unittest discover -s tests 2>&1 | grep -E '^(Ran|OK|FAILED|ERROR|FAIL:)'
Ran 582 tests in 34.555s
OK (skipped=8)
```

(582 vs the review's 568: +14 from the merged origin/main spectral-features tests.)
Suite fast and hang-free (unittest with timeout; no waits added — docs-only diff).
