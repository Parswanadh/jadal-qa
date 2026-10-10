# Demo clips report — known-defect generator (issue #38)

Branch `feat/democlips`. Worktree `/home/parshu/projects/jadal-qa-wt/democlips`.
Python `PYTHONNOUSERSITE=1 /home/parshu/envs/jadal/bin/python` throughout; nothing installed.
PR: https://github.com/Parswanadh/jadal-qa/pull/40 (open, NOT merged).

## What exists

- `scripts/make_demo_clips.py` — deterministic generator. Reads 2 IndEgo
  windows read-only (ECU assembly `User_15_410_0411_2_480.mp4` [60,80)s, inside
  the annotated `02_attach_ram_chip` step; logistics organise
  `User_1_2110_Organisation_480.mp4` [30,50)s, inside `01_organize`), emits 9
  variants each (good, blurred, too_dark, overexposed, frozen, wrong_fps at
  15 fps, truncated, garbage, reupload_duplicate) = 18 clips + `manifest.json`
  under `data_runtime/demo_clips/` (git-ignored; default is repo-anchored so
  it works from any CWD). Every entry has defect, variant, parameters, fps,
  `actual_duration_s` (20.0, 13.333 for wrong_fps), source clip + window,
  sha256, label (`SYNTHETIC-GOOD` only for the clean copy, else
  `SYNTHETIC-DEFECT`; duplicate carries defect `duplicate` + `copy_of` /
  `good_sha256`). Encoder: `cv2.VideoWriter`, fourcc `mp4v`, 10 fps
  (15 for wrong_fps), actual 480x480 recorded. Flags: `--out`, `--seed`
  (default 42), `--list`, debug-only `--allow-outside-data-runtime` (temp
  dirs always refused per SPEC). Inputs: `JADAL_INDEGO_ROOT` (default the
  awing IndEgo dir). Missing sources: clear stderr message, exit 2, nothing
  written; bad `--out`: exit 3, nothing written.
- `tests/test_make_demo_clips.py` — 7 tests on tiny synthetic sources in temp
  dirs (never the real clips): all 9 variants with defect codes/labels,
  manifest sha match + sha-redacted determinism compare, byte-op sha
  stability across runs vs pixel-closeness for re-encoded, duplicate
  byte-identical to good, wrong_fps at ~15 fps, good-family decodes while
  garbage/truncated fail, frozen closeness, dark/bright pixel direction,
  5 fps resample, short-read error, `--out` guard (incl. /tmp-with-flag),
  exit-2 path, `--list` (18 lines, subprocess with timeout). Adds ~0.9 s.
- Labels are true by construction (we injected the defect) and MUST NOT be
  presented as human-labelled data (stated in script docstring + manifest).

## How to run

```
PYTHONNOUSERSITE=1 /home/parshu/envs/jadal/bin/python scripts/make_demo_clips.py --list
PYTHONNOUSERSITE=1 /home/parshu/envs/jadal/bin/python scripts/make_demo_clips.py [--out DIR] [--seed 42]
PYTHONNOUSERSITE=1 /home/parshu/envs/jadal/bin/python -m unittest tests.test_make_demo_clips -v
```

## REAL test output tails (current; older round sections below are history)

```
PYTHONNOUSERSITE=1 /home/parshu/envs/jadal/bin/python -m unittest tests.test_make_demo_clips -v
Ran 7 tests in 0.833s
OK
timeout 250 env PYTHONNOUSERSITE=1 /home/parshu/envs/jadal/bin/python -m unittest discover -s tests 2>&1 | grep -E '^(Ran|OK|FAILED|ERROR|FAIL:)'
Ran 329 tests in 10.018s
OK
```

Real run on IndEgo sources (~4 s): 18 clips + manifest; all manifest sha256s
recomputed OK; good/blurred/too_dark/overexposed/frozen/wrong_fps/duplicate
decode, truncated/garbage fail (`moov atom not found`); rerun byte-identical
(clips + manifest bytes); missing-sources path exits 2; `--out` outside
data_runtime exits 3 (incl. under /tmp even with the override flag). Sizes
e.g. `ecu_good.mp4` 2255914 B, `logistics_good.mp4` 2716271 B, garbage
65536 B each.

Determinism note: byte-identical clips + manifest across reruns on this
machine + OpenCV build (verified). Garbage/truncated/duplicate are pure byte
ops (stable everywhere given same inputs + seed); re-encoded mp4s may differ
across OpenCV builds — documented in the script; tests therefore compare
re-encoded variants by decoded pixels and manifests with hashes redacted.

## Gaps

- Windows are hand-picked from VIA annotations, not auto-detected for hands.
- ECU window assumed hands-in-view from the step label; not verified by a detector.
- No service/dashboard wiring; clips are standalone files for other workers.

## Fix round 1 (review findings C1, I1-I3, M1-M6 — all in owned files)

- C1 wrong-FPS: added 9th variant `wrong_fps` (same frames at 15 fps,
  `fps: 15.0` in manifest entry). Now 18 clips; `--list` test updated 16->18.
- I1 labels: `good` is now defect `none` + `SYNTHETIC-GOOD`; `reupload_duplicate`
  is defect `duplicate` + `SYNTHETIC-DEFECT` with `copy_of` + `good_sha256`
  params. All other variants keep their defect code + `SYNTHETIC-DEFECT`.
- I2 containment: `--out` outside `data_runtime/` is refused (exit 3, nothing
  written) unless `--allow-outside-data-runtime` is passed.
- I3 manifest determinism: dropped `created_utc` from `manifest.json` (time now
  only printed to stdout). Manifest bytes identical across two runs (verified).
- M1/M2: good encoded once via `with`-block; truncated/duplicate derive from
  those bytes. M3: `is_decodable` now a postcondition assert in
  `build_source_set`. M4: encoder records actual `width/height` (480x480).
  M5: `read_window` resamples non-10fps sources to 10 fps (nearest-neighbour).
  M6: tests assert byte-stability only for byte-ops; re-encoded variants match
  by decoded pixels (first-frame mean-abs-diff < 5.0); frozen uses closeness
  (< 10.0; measured lossy noise ~2.07 vs ~45-50 for real motion).
- New tests: `test_read_window_resamples_to_fps` (5 fps source -> 20 frames),
  `test_out_guard` (exit 3, nothing written). No finding disputed.

REAL outputs (fix round 1):

```
PYTHONNOUSERSITE=1 /home/parshu/envs/jadal/bin/python -m unittest tests.test_make_demo_clips -v
Ran 4 tests in 0.564s
OK
timeout 250 env PYTHONNOUSERSITE=1 /home/parshu/envs/jadal/bin/python -m unittest discover -s tests 2>&1 | grep -E '^(Ran|OK|FAILED|ERROR|FAIL:)'
Ran 326 tests in 10.107s
OK
```

Real run: `wrote 18 clips + manifest.json to data_runtime/demo_clips`
(~4.4 s); clip shas stable across runs True; manifest bytes identical True;
`ecu_wrong_fps.mp4`/`logistics_wrong_fps.mp4` report 15.0 fps and decode;
guard probe exits 3 with nothing written.

## Fix round 2 (review findings I1, M1-M5 — all in owned files)

- I1 CWD-relative default: `DEFAULT_OUT` is now
  `os.path.join(DATA_RUNTIME, "demo_clips")` (repo-anchored, absolute), so a
  default run works from any CWD. Proven: default run with CWD=/tmp exits 0
  and writes to the worktree's `data_runtime/demo_clips/` (19 files), nothing
  under `/tmp/data_runtime`.
- M1 short reads: `read_window` raises `RuntimeError` when fewer than 90% of
  wanted frames are read (no more silent linspace-stretch of truncated
  inputs). New `test_short_read_raises` (5-frame source, 2 s request).
- M2 containment scope: documented on `build_source_set` — containment under
  `data_runtime/` is a CLI (`main()`) property; the API writes wherever
  `out_dir` points (tests use temp dirs).
- M3 duration honesty: each manifest entry now has `actual_duration_s`
  (`n_frames/fps`): 20.0 for 10 fps variants, 13.333 for `wrong_fps`.
- M4 encoder size: `main` collects per-source sizes and raises on a mix
  (both real sources 480x480; per-entry `window.frame_size` was already
  per-source).
- M5 test overclaim: manifest determinism now compares sha-redacted dicts
  (holds on any OpenCV build); byte-op sha equality kept as the stability
  check. No finding disputed.

REAL outputs (fix round 2):

```
PYTHONNOUSERSITE=1 /home/parshu/envs/jadal/bin/python -m unittest tests.test_make_demo_clips -v
Ran 5 tests in 0.533s
OK
timeout 250 env PYTHONNOUSERSITE=1 /home/parshu/envs/jadal/bin/python -m unittest discover -s tests 2>&1 | grep -E '^(Ran|OK|FAILED|ERROR|FAIL:)'
Ran 327 tests in 10.360s
OK
```

Live manifest check: 18 files, no `created_utc`, good `actual_duration_s`
20.0, wrong_fps 13.333, encoder 480x480 mp4v @ 10 fps.

## Fix round 3 (review findings I1, M1-M3 — all in owned files)

- I1 stale report counts: rewrote "What exists" (9 variants = 18 clips incl.
  wrong_fps @ 15 fps, good = `SYNTHETIC-GOOD`, byte-stability scoped to
  byte-ops), replaced the superseded top-level tails with current ones, and
  dropped the done wrong-FPS gap. No false count remains.
- M1 old tails: deleted the superseded "Ran 2 / Ran 306" block; only current
  tails at top, per-round history kept in its own labeled section.
- M2 /tmp escape via flag: temp dirs (`/tmp`, plus `$TMPDIR` when set) are
  now refused with exit 3 even with `--allow-outside-data-runtime` (SPEC
  "Never write results to /tmp"); flag documented debug-only. New
  `test_tmp_refused_even_with_flag`.
- M3 env override + exit-2 test: `INDEGO_ROOT` reads `JADAL_INDEGO_ROOT`
  (default unchanged); error message prints the effective root. New
  `test_missing_sources_exit_2` (subprocess, rc 2, "not found" in stderr,
  nothing written). No finding disputed.

REAL outputs (fix round 3):

```
PYTHONNOUSERSITE=1 /home/parshu/envs/jadal/bin/python -m unittest tests.test_make_demo_clips
Ran 7 tests in 0.833s
OK
timeout 250 env PYTHONNOUSERSITE=1 /home/parshu/envs/jadal/bin/python -m unittest discover -s tests 2>&1 | grep -E '^(Ran|OK|FAILED|ERROR|FAIL:)'
Ran 329 tests in 10.018s
OK
```

Real run: `wrote 18 clips + manifest.json to .../data_runtime/demo_clips`
(~4.2 s, 19 files with manifest).
