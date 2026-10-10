# Demo clips report — known-defect generator (issue #38)

Branch `feat/democlips`. Worktree `/home/parshu/projects/jadal-qa-wt/democlips`.
Python `PYTHONNOUSERSITE=1 /home/parshu/envs/jadal/bin/python` throughout; nothing installed.
PR: https://github.com/Parswanadh/jadal-qa/pull/40 (open, NOT merged).

## What exists

- `scripts/make_demo_clips.py` — deterministic generator. Reads 2 IndEgo
  windows read-only (ECU assembly `User_15_410_0411_2_480.mp4` [60,80)s, inside
  the annotated `02_attach_ram_chip` step; logistics organise
  `User_1_2110_Organisation_480.mp4` [30,50)s, inside `01_organize`), emits 8
  variants each (good, blurred, too_dark, overexposed, frozen, truncated,
  garbage, reupload_duplicate) = 16 clips + `manifest.json` under
  `data_runtime/demo_clips/` (git-ignored). Every entry has defect,
  parameters, source clip + window, sha256, `label: SYNTHETIC-DEFECT`.
  Encoder: `cv2.VideoWriter`, fourcc `mp4v`, 10 fps, source frame size.
  Flags: `--out`, `--seed` (default 42), `--list`. Missing sources: clear
  stderr message, exit 2, nothing written.
- `tests/test_make_demo_clips.py` — builds a tiny synthetic source in a temp
  dir (never the real clips); asserts all 8 variants produced, manifest
  sha256s match files, all 8 sha256s stable across two runs, duplicate
  byte-identical to good, good-family decodes while garbage/truncated fail,
  frozen frames all equal, dark dimmer / bright brighter than good; plus a
  `--list` subprocess test (with timeout). Adds ~0.4 s to the suite.
- Labels are true by construction (we injected the defect) and MUST NOT be
  presented as human-labelled data (stated in script docstring + manifest).

## How to run

```
PYTHONNOUSERSITE=1 /home/parshu/envs/jadal/bin/python scripts/make_demo_clips.py --list
PYTHONNOUSERSITE=1 /home/parshu/envs/jadal/bin/python scripts/make_demo_clips.py [--out DIR] [--seed 42]
PYTHONNOUSERSITE=1 /home/parshu/envs/jadal/bin/python -m unittest tests.test_make_demo_clips -v
```

## REAL test output tails

New tests (2 tests, 0.356 s, OK):
```
Ran 2 tests in 0.356s
OK
```

Full suite after the change:
```
Ran 306 tests in 15.598s
OK
```

Real run on IndEgo sources (~7 s): 16 clips + manifest; all manifest sha256s
recomputed OK; good/blurred/too_dark/overexposed/frozen/duplicate decode,
truncated/garbage fail (`moov atom not found`); second run to a different
`--out` byte-identical on all 16; missing-sources path exits 2. Sizes e.g.
`ecu_good.mp4` 2255914 B, `logistics_good.mp4` 2716271 B, garbage 65536 B each.

Determinism note: all 8 variants byte-identical across reruns on this
machine + OpenCV build (verified). Garbage/truncated/duplicate are pure byte
ops (stable everywhere given same inputs + seed); re-encoded mp4s may differ
across OpenCV builds — documented in the script.

## Gaps

- No wrong-FPS variant (issue text mentions it; the worker spec's variant list
  does not — implemented the worker spec list; easy to add as a 9th variant).
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
