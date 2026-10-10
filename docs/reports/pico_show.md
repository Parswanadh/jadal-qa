# picoshow report (issue #37, PR #39, branch `feat/picoshow`)

## What exists (new files only; nothing else touched)
- `scripts/pico_show.py` — CLI over `jadal.pico_link.PicoLink` (real API read
  first; pattern table ported from `/home/parshu/projects/awing/prep/pico_hw/show.py`,
  which was not edited). Subcommands: `verdict accept|review|reject`, `demo`
  (all six patterns incl. SOS, double-pulse heartbeat, countdown), `status`,
  `pattern HEX8`. Pure `pattern_for(verdict) -> hex` for later service reuse.
- `tests/test_pico_show.py` — 11 tests, fake serial only (`FakePicoSerial`);
  never opens `/dev/ttyACM0`.

Behaviour: waits up to 2.5 s for the first heartbeat after connect (announced;
the autodetect probe swallows the first HB so `status()` is `connected False`
immediately after connect), prints what it is doing, always sends `LED OFF`
and disconnects in a `finally` block, `--port`/`--fake` flags, exit 2 with
`error: no Pico answers (...)` when no Pico answers.

Verdict patterns (FLASH = 32 bits, 100 ms/bit, MSB first):
accept `FFFFFFFF` (solid), review `F0F0F0F0` (slow blink),
reject `AAAAAAAA` (fast blink). Demo extras: SOS `A87770A8`, heartbeat
`D800D800`, countdown `8A150AA0`.

NOTE: the board's onboard LED is a single colour — accept/review/reject are
told apart by blink *pattern*, not colour.

## How to run
```
PYTHONNOUSERSITE=1 /home/parshu/envs/jadal/bin/python scripts/pico_show.py --fake demo
PYTHONNOUSERSITE=1 /home/parshu/envs/jadal/bin/python scripts/pico_show.py --fake verdict accept
PYTHONNOUSERSITE=1 /home/parshu/envs/jadal/bin/python scripts/pico_show.py --port /dev/ttyACM0 status
PYTHONNOUSERSITE=1 /home/parshu/envs/jadal/bin/python -m unittest tests.test_pico_show -v
```

## REAL test output tails (what was actually run)
`python -m unittest tests.test_pico_show -v` (fake only):
```
Ran 11 tests in 0.148s
OK
```
`python scripts/pico_show.py --fake demo` (no-hardware demo, synthetic — no board attached):
```
ACCEPT   - solid on                FLASH FFFFFFFF
REVIEW   - slow blink (400 ms)     FLASH F0F0F0F0
REJECT   - fast blink (100 ms)     FLASH AAAAAAAA
SOS      - ... --- ...             FLASH A87770A8
HEARTBEAT- double pulse            FLASH D800D800
COUNTDOWN- 1,2,3,4 pulses          FLASH 8A150AA0
status: {'connected': True, 'last_hb_age_s': 0.301..., 'last_hb_led': 'OFF'}
LED OFF sent.
done - LED off
```
Error paths (real output): `--fake verdict bogus` -> argparse exit 2
(invalid choice); `--port /dev/ttyNOPE0 status` ->
`error: no Pico answers (could not open serial port '/dev/ttyNOPE0': ...)` exit 2.
Full suite: `python -m unittest discover -s tests` -> `Ran 315 tests ... OK`
(~13 s; this branch adds ~0.15 s).

## Gaps (NOT DONE)
- No real-hardware run yet: no Pico was attached in this worktree session, so
  the `--fake demo` above is synthetic and the 100 ms/bit timing is unverified
  on the board. Needs one `--port /dev/ttyACM0 demo` eyes-on run.
- No camera-in-the-loop check that the flashed pattern is visible to the camera.
- `pattern_for()` is not wired into the service yet (another worker owns
  `jadal/service.py`); reuse is future work.

## Fix round 1 (review of PR #39 -> SPEC: FAIL; no Critical findings)
Fixed in owned files only (`scripts/pico_show.py`, `tests/test_pico_show.py`,
this report):
- Important #1 (e2e never asserted what was flashed): the verdict e2e test now
  drives `main()` through a `_make_link` seam with a shared inspectable
  `FakePicoSerial` and asserts, per verdict, `flashed == [<expected>]` plus
  `LED OFF` sent; a new demo e2e asserts all six patterns in order. Verified
  the reviewer's regression (force-flash accept for every verdict) is now
  caught: probe flashed `['ffffffff']` for `verdict review`, which the test
  rejects (expects `['f0f0f0f0']`).
- Minor #2: `--pause` uses `_nonneg_float`; `--fake --pause -1 verdict accept`
  -> `error: argument --pause: --pause must be >= 0`, exit 2 (no traceback).
- Minor #3: `--fake --port ...` -> `error: --port is ignored with --fake ...`,
  exit 2.
- Minor #4: `hex_to_bits` docstring documents `ValueError` on bad input;
  `pattern` help now says `8 hex digits with optional 0x prefix`.
- Minor #5: `verdict`/`demo`/`pattern` exit 3 with a stderr warning when no
  heartbeat is seen (pattern still attempted, LED OFF still sent); `status`
  still exits 0 since the disconnected status is its answer.

REAL command outputs (this round):
```
$ PYTHONNOUSERSITE=1 /home/parshu/envs/jadal/bin/python -m unittest tests.test_pico_show
Ran 16 tests in 0.660s
OK
$ PYTHONNOUSERSITE=1 timeout 250 /home/parshu/envs/jadal/bin/python -m unittest discover -s tests | grep -E '^(Ran|OK|FAILED|ERROR|FAIL:)'
Ran 338 tests in 10.875s
OK
```
(338 = 315 pre-existing incl. main-merge additions + 16 pico_show + rest of
suite growth from other workers' merged work; pico_show adds ~0.66 s.)
