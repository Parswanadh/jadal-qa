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
