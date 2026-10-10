# picoshow report (issue #37, PR #39, branch `feat/picoshow`)

## What exists (new files only; nothing else touched)
- `scripts/pico_show.py` — CLI over `jadal.pico_link.PicoLink` (real API read
  first; pattern table ported from `/home/parshu/projects/awing/prep/pico_hw/show.py`,
  which was not edited). Subcommands: `verdict accept|review|reject`, `demo`
  (all six patterns incl. SOS, double-pulse heartbeat, countdown), `status`,
  `pattern HEX8`. Pure `pattern_for(verdict) -> hex` for later service reuse.
- `tests/test_pico_show.py` — 21 tests, fake serial only (`FakePicoSerial`);
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

## REAL test output tails (round 0, before any fix — historical; current figures are in the Fix round sections below)
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

## Fix round 2 (round-2 review -> SPEC: FAIL; no Critical findings)
Fixed in owned files only (`scripts/pico_show.py`, `tests/test_pico_show.py`,
this report):
- Important #1 (valid-`pattern` path had no positive flash assertion): new
  `test_main_pattern_valid_fake` drives `main()` through the `_make_link`
  seam for `F0F0F0F0` and `0xabcdef01`, asserting `flashed == [expected]` +
  LED OFF. Proved it catches the reviewer's probe4 bug (validation kept but
  always flashing `00000000`): the new test fails 2/2 subtests under the
  patch, suite green without it.
- Minor #1/#2 (helper widths): `hex_to_bits` raises `ValueError` unless the
  input is exactly 8 hex digits; `_bits_to_hex` raises on >32 bits or non-bit
  chars. Covered by `test_helpers_reject_off_spec_widths`.
- Minor #3: bad-hex test tightened to `assertEqual(rc, 2)` + LED OFF sent.
- Minor #4: the round-0 output block above is now labelled historical.

REAL command outputs (this round):
```
$ PYTHONNOUSERSITE=1 /home/parshu/envs/jadal/bin/python -m unittest tests.test_pico_show
Ran 18 tests in 0.784s
OK
$ PYTHONNOUSERSITE=1 timeout 250 /home/parshu/envs/jadal/bin/python -m unittest discover -s tests | grep -E '^(Ran|OK|FAILED|ERROR|FAIL:)'
Ran 340 tests in 10.335s
OK
```
(340 = 338 after round 1 + 2 new pattern/width tests; pico_show adds ~0.8 s.)

## Fix round 3 (round-3 review -> SPEC: FAIL; no Critical findings)
Fixed in owned files only (`scripts/pico_show.py`, `tests/test_pico_show.py`,
this report):
- Important #1 (wait-phase exception skipped cleanup): `wait_for_heartbeat`
  now runs inside the `try` that owns `cleanup()` in all four subcommands, so
  a `KeyboardInterrupt` during the ~2.5 s wait still sends LED OFF and
  disconnects (the interrupt itself still propagates — only cleanup is
  guaranteed). New `test_wait_phase_interrupt_still_cleans_up` raises a
  synthetic `KeyboardInterrupt` inside the wait and asserts LED OFF +
  disconnected.
- Important #2 (mid-pattern `PicoLinkError` escaped as traceback, exit 1):
  `verdict`/`demo`/`pattern` now catch `PicoLinkError` around the flash/play
  section and return 2 with `error: Pico failed mid-pattern (...)` on stderr
  (cleanup still runs in `finally`). New
  `test_flash_failure_returns_2_and_cleans_up` uses a raising-flash seam and
  asserts rc 2, the message, nothing flashed, and LED OFF sent.
- Minor #1: header above now says 21 tests (was stale "11" from round 0).
- Minor #2: `hex_to_bits` accepts an optional `0x`/`0X` prefix (documented),
  matching the `pattern` subcommand; new
  `test_hex_to_bits_accepts_optional_0x_prefix`.

REAL command outputs (this round):
```
$ PYTHONNOUSERSITE=1 /home/parshu/envs/jadal/bin/python -m unittest tests.test_pico_show
Ran 21 tests in 0.824s
OK
$ PYTHONNOUSERSITE=1 timeout 250 /home/parshu/envs/jadal/bin/python -m unittest discover -s tests | grep -E '^(Ran|OK|FAILED|ERROR|FAIL:)'
Ran 343 tests in 10.839s
OK
```
(343 = 340 after round 2 + 3 new tests; pico_show adds ~0.8 s.)
