# Claims PoC docs report (docs/claims-poc, issue #74)

Docs-only: README.md, docs/ARCHITECTURE.md, docs/ATTRIBUTIONS.md.
Goal: remove the certified-bound claim, record the edge-first / whole-video
direction, keep every number honest (n + label source + backing pointer).

## Fix round 1 (2026-10-10, answer to PR #74 review)

Review verdict was SPEC: FAIL. All four blocking findings verified correct
against the repo — no pushback, all fixed in owned files only
(README.md, docs/ARCHITECTURE.md; ATTRIBUTIONS.md: 2-row wording alignment only
(EmbeddingGemma 2 → "Optional helper, not the core", Laya → "Not in the pipeline");
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

## Fix round 2 (2026-10-10, answer to PR #74 REFUSE review)

Review verdict was SPEC: FAIL, QUALITY: CHANGES NEEDED. Every Critical and
Important finding verified correct against the repo — no pushback, all fixed
in owned files only (README.md, docs/ARCHITECTURE.md, docs/reports/claims_poc.md;
ATTRIBUTIONS.md wording from round 1 kept; jadal/config.py, jadal/__init__.py,
docs/SPEC.md and other workers' files untouched).

- C1 ("Measured numbers" contradicts own disclaimer) — confirmed: README:5
  claimed "Measured numbers are in the table below" while README:22 disclaimed
  every row as unverified with no backing file, and `eval/results/BASELINE.json`
  still does not exist. Fix applied: line 5 replaced with "No measured numbers
  are claimed on this branch (`eval/results/BASELINE.json` does not exist, so
  per `docs/BASELINE.md` there is nothing citable); the table below lists
  unverified builder notes with their stated n and label source."
- C2 (table point estimates unbacked; merged spectral-harness numbers disagree)
  — confirmed: no file under eval/ or docs/reports/ backs any table AUROC, and
  docs/reports/spectral_features.md:218 gives T0 0.725 / T1 0.748 for a
  different (spectral-harness) method. Fix applied (delete option): every point
  estimate removed from the table (11 rows now read "direction tested, result
  pending"); the "about 0.8 AUROC honest ceiling" sentence in "What this means"
  removed; table intro now states "No point estimates are given and no
  intervals are claimed until backing files land."
- I1 (unbacked "10 repeated runs" digest claim) — confirmed: grep finds
  "quantised digest" only in my own two lines; no 10-run log anywhere (closest
  backing is docs/reports/edge.md:81-83: identical across 3 repeats and 1-vs-4
  cores, different artifact). Fix applied in README + ARCH: replaced with the
  edge.md claim verbatim with pointer plus "the 10x receipt-hash proof through
  the real modules is pending".
- I2 (early-reject "1-2 seconds" stated as fact) — confirmed. Fix applied:
  README:14 now carries "(timing unverified — no backing file or measurement
  in this repo, cf. ARCHITECTURE.md Stage 0 note)"; ARCH:22 reworded to
  "planned to reject … (timing unverified …; see the Stage 0 note below)",
  joining the existing ARCH Stage 0 qualifier.
- I3 (unpointed pallet/chance, 1080p/HEVC, SmolVLM2, RTX 4070 claims) —
  confirmed: "smolvlm" occurs nowhere in docs/ or eval/; no report backs a
  1080p/30fps-HEVC PoC run or pallet "at chance" (pallet/pallet2 names exist
  only in eval/gold/segments.json). Fix applied: SmolVLM2 was already qualified
  and is kept; pallet/chance reworded to an unverified builder note; 1080p/HEVC
  reworded to "footage described in an unverified builder note … as 1080p/30
  fps HEVC"; ARCH hardware line now opens "(unverified builder note — no
  backing measurement file in this repo)".
- I4 (label breakdown without pointer) — confirmed backing exists
  (docs/reports/funnel.md:421-422; eval/results/gold_labels_snapshot.json).
  Fix applied: pointer appended to the README human-set bullet.
- M1 (report said "ATTRIBUTIONS.md untouched" but the diff touches it) —
  confirmed: the diff has the 2-row wording alignment. Fix applied: report
  sentence corrected above.

Verification probes (all short, real outputs):

```
$ grep -n -e "Measured numbers are in the table" -e "AUROC about" \
  -e "balanced accuracy about" -e "near chance, preliminary" \
  -e "0.8 AUROC as an honest ceiling" -e "quantised digest over 10" \
  README.md docs/ARCHITECTURE.md
(no hits — exit 1: all point estimates and the false measurement claim gone)

$ grep -n "first 1-2 seconds" README.md docs/ARCHITECTURE.md
README.md:14: ... rejected in the first 1-2 seconds (timing unverified — no backing file or measurement in this repo, cf. ARCHITECTURE.md Stage 0 note).
docs/ARCHITECTURE.md:22: ... Stage 0 is planned to reject ... in the first 1-2 seconds, with evidence (timing unverified — no backing file or measurement in this repo; see the Stage 0 note below).

$ grep -rn -i "smolvlm" docs/ eval/ | head
(no hits — SmolVLM2 sentence stays qualified as an unverified builder note)

$ git status --short
 M README.md
 M docs/ARCHITECTURE.md
 M docs/reports/claims_poc.md
(owned files only; no protected or other-worker files)
```

Tests (real command output):

```
$ PYTHONNOUSERSITE=1 timeout 250 /home/parshu/envs/jadal/bin/python -m unittest discover -s tests 2>&1 | grep -E '^(Ran|OK|FAILED|ERROR|FAIL:)'
Ran 582 tests in 38.668s
OK (skipped=8)
```

Suite fast and hang-free (unittest with timeout; every wait already timed out
— docs-only diff, no waits added).

## Fix round 3 (2026-10-10, answer to PR #74 REFUSE review)

Review verdict was SPEC: FAIL, QUALITY: CHANGES NEEDED with no Criticals and
two Importants. Both Importants plus all three Minors verified correct against
the repo — no pushback, all fixed in owned files only (README.md,
docs/ARCHITECTURE.md, docs/reports/claims_poc.md (this report); jadal/config.py,
jadal/__init__.py, docs/SPEC.md and other workers' files untouched).
`git fetch origin && git merge origin/main` first: already up to date
(`origin/main` HEAD `68f85cc`, merge of PR #52, already an ancestor).

- I1 (stale main pin: PR #52 merged, still listed as open) — confirmed:
  `origin/main` HEAD is `68f85cc` (2026-10-10) while README:69 / ARCH:3 pinned
  `f58e0a1`, and README:22 / README:111 listed #52 as open. Fix applied: pin
  moved to `68f85cc` in both files; #52 dropped from both open-PR ranges;
  Modules table re-verified against the new tree (`git ls-tree origin/main`
  shows the 14 new #52 files) — new row "Spectral features arm
  (`jadal/spectral/`, `eval/spectral_features/`,
  `docs/reports/spectral_features.md`, `tests/test_spectral_*.py`) | merged,
  PR #52", remainder row narrowed to open PRs #57, #61, #66-#73; repo-layout
  sketch gained `eval/` + spectral lines so "Matches `origin/main`" is checkable
  at `68f85cc`. Table intro reworded to carve the merged spectral arm out of
  the "no backing file for any row / unmerged off-branch" sentence (the merged
  files exist on-branch; the row still claims no point estimate).
- I2 (unqualified "scorer was calibrated on 480p/10 fps IndEgo") — confirmed:
  the clause read as fact while the branch's own disclaimer admits no backing
  file for the calibration (provenance of 480p/10fps itself is backed,
  edge.md:69). Fix applied verbatim as suggested: README now reads "The scorer
  is described in an unverified builder note (no backing file in this repo) as
  calibrated on 480p/10 fps IndEgo, so the domain shift is untested"; ARCH:152
  now reads "The domain shift from the 480p/10 fps calibration data (unverified
  builder note — no backing file in this repo for the calibration) is untested."
- M1 ("2-second chunks" bare fact) — confirmed. Fix applied: both README:14
  and ARCH:22 now carry "(chunk size per unverified builder note, no backing
  measurement on this branch)".
- M2 (RTX 4070 specs under an "unverified" note) — qualifier retained, no
  `nvidia-smi` backing added deliberately: this machine has `nvidia-smi`
  (`/usr/bin/nvidia-smi`) but its output would describe the wrong machine, not
  the measurement laptop, so citing it here would be false backing.
- M3 (480p/10fps clause lacks pointer) — confirmed. Fix applied: README human-set
  bullet now cites `docs/reports/edge.md:69` ("480x480, 10 fps IndEgo").

Verification probes (all short, real outputs):

```
$ git rev-parse origin/main && git log -1 --format='%ad' --date=short origin/main
68f85cc9c546aa8faa3c21ec3a300d3dbc194768
2026-10-10

$ grep -n "f58e0a1\|#52, #57\|was calibrated on 480p\|from the 480p/10 fps calibration data is" README.md docs/ARCHITECTURE.md
(no hits — exit 1: stale pin, stale open-PR range, and both bare calibration clauses gone)

$ sed -n '69p' docs/reports/edge.md
Conditions: 30 segments x 10 s (480x480, 10 fps IndEgo, read-only),

$ git status --short
 M README.md
 M docs/ARCHITECTURE.md
(owned files only; no protected or other-worker files)
```

Tests (real command output):

```
$ PYTHONNOUSERSITE=1 timeout 250 /home/parshu/envs/jadal/bin/python -m unittest discover -s tests 2>&1 | grep -E '^(Ran|OK|FAILED|ERROR|FAIL:)'
Ran 582 tests in 31.855s
OK (skipped=8)
```

Suite fast and hang-free (unittest with timeout; no waits added — docs-only diff).
