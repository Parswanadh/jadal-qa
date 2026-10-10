# Pico attract/idle mode — hardware report (issue #41)

Board: Raspberry Pi Pico (RP2040, non-W), MicroPython 1.29.0, `/dev/ttyACM0`.
Branch `feat/picoidle`. All outputs below are REAL captures, not predictions.

## What changed (owned files only)

- `pico/idle.py` (new): pure attract-mode logic, imports on MicroPython and
  CPython. Tunables at the top: `IDLE_AFTER_S = 20`, 3 s breathing x3,
  ACCEPT solid 1 s, REVIEW slow blink (400 ms half-period) 2 s, REJECT fast
  blink (100 ms half-period) 1.5 s, dark gap 0.5 s, 14 s total cycle.
  Helpers: `idle_state / idle_brightness / tour_phase / phase_rgb /
  ticks_diff` (2**30 wraparound-safe, same modulus as `time.ticks_*`).
- `pico/main.py`: non-blocking idle drive in the existing loop (1 ms sleep
  unchanged); ANY received line (even unknown/malformed/overlong) exits idle
  BEFORE handling, restores plain-output pins + last host colour; idle
  restarts only after 20 s silence and never while a FLASH plays. PWM on
  GPIO25 (deinit + pin rebuild on exit); PWM on RGB pins when `RGB_PINS` set.
- `tests/test_pico_idle.py` (new, 25 tests): timeout boundary, deactivation
  at every tour phase, brightness range/periodicity, exact tour schedule
  (1 ms scan of the 14 s cycle), RGB colours, wraparound vectors, import
  side-effect freedom (incl. a clean-subprocess import of both files).
- `pico/README.md`: idle section, `IDLE` heartbeat note, power-up note.

One visible protocol addition (read-only, documented in `main.py` + README):
while idle, the `HB` line reports `led_state` as `IDLE`. The host can never
*set* `IDLE` (`encode_led` unchanged); `parse_hb` accepts any non-empty state
so `PicoLink` and `tests/test_pico_link.py` accept it (they only ever see
fake-device states anyway). Protocol otherwise UNCHANGED.

## Host tests (no hardware)

```
$ PYTHONNOUSERSITE=1 /home/parshu/envs/jadal/bin/python -m unittest discover -s tests
Ran 329 tests in 9.861s
OK
```
(304 baseline incl. `test_pico_protocol.py` + `test_pico_link.py`, all still
green; 25 new idle tests.) `py_compile` clean on `pico/main.py`,
`pico/idle.py`, `tests/test_pico_idle.py`.

## Upload + on-board import (flock-held mpremote throughout)

First `fs cp` attempts failed with `could not complete raw paste: b''` /
`could not enter raw repl`. Cause found with `fuser`: a sibling worker's
`python -m unittest discover -s tests` (cwd `/tmp/jadal_baseline`, PID
1128527) was holding `/dev/ttyACM0` open with ~25 fds without taking
`tools/pico.lock`, garbling the REPL handshake. Waited for it to exit, port
free, upload then succeeded first try:

```
cp pico/main.py :main.py ; MAIN-CP-OK
cp pico/idle.py :idle.py ; IDLE-CP-OK
```

Readback sha256 — local == board, both files:

```
69ab96f807677cbddba1ab9bdacceba02f0fa075324842102ea14e41f7d4cca2  pico/main.py == :main.py
13eee8da88d6f922bcd3b03caf0bf551f8d8c11bcbcd04a35d755862d7005a32  pico/idle.py == :idle.py
```

On-board import smoke test (real MicroPython 1.29.0):

```
IDLE-IMPORT-OK 20 14000
('BREATHE', 0.0) ('ACCEPT', 1) ('REVIEW', 1) ('REJECT', 1) ('GAP', 0)
```

then `mpremote connect /dev/ttyACM0 reset` → `RESET-OK`, `/dev/ttyACM0` back.

## Host regression, real PicoLink (after reset)

```
port: /dev/ttyACM0
status#1: {'connected': True, 'port': '/dev/ttyACM0', 'last_hb_age_s': 0.0387, 'last_hb_ticks_ms': 28235, 'last_hb_led': 'OFF'}
status#2: {'connected': True, ... 'last_hb_age_s': 0.2342, ...}   (1.2 s apart, age < 2 s)
set_led G -> last_hb_led=OFF age=0.48
set_led A -> last_hb_led=OFF age=0.73
set_led R -> last_hb_led=R age=0.00
set_led OFF -> last_hb_led=R age=0.25
flash A5A5A5A5 sent (3.2 s pattern running)
mid-flash status: {'connected': True, ... 'last_hb_age_s': 0.7508, ... 'last_hb_led': 'R'}
estimate_offset: n=8 rtt_min=6.32 rtt_med=15.04 offset=-13432125.20 bound=3.16
final status: {'connected': True, ... 'last_hb_age_s': 0.8892, 'last_hb_led': 'OFF'}
REGRESSION-PASS
```
(HB `led_state` lags commands by up to one 1 s beat — normal sampling, not a
bug. Initial `status` right after connect can legitimately show
`connected: False` until the first HB arrives; the script waits up to 5 s.)

## Idle proof: HB reports IDLE after 20+ s silence, PING wakes (real capture)

Port opened, nothing sent for 26 s, then passive read (reads are not host
frames, cannot wake the board). Board tick of last host frame ≈ 33xxx;
`IDLE` first appears at tick 54240 — i.e. ~21 s of silence, matching
`IDLE_AFTER_S = 20` within one HB beat:

```
captured: HB 46239 OFF
captured: HB 47238 OFF
captured: HB 48238 OFF
captured: HB 49239 OFF
captured: HB 50239 OFF
captured: HB 51239 OFF
captured: HB 52239 OFF
captured: HB 53240 OFF
captured: HB 54240 IDLE
captured: HB 55240 IDLE
... (9x IDLE total)
sent: PING
captured: PONG 70404
captured: HB 71246 OFF
wake OK: 'PONG 70404' -> host mode 'HB 71246 OFF'
IDLE-PROOF-PASS
```

Notes, stated exactly: (1) In that first run the PONG arrived ~6 s after the
PING write. A tight retest (drain 2 s, PING, measure) answered in one read
quantum — `0.100s PONG 158415` — so the 6 s was host-side backlog (26 s of
unread HBs queued in kernel buffers, drained in bursts; plus two more
lock-less `unittest discover` holders from sibling worktrees `verify` and
`picoshow` appeared on the port during the session), not firmware latency.
The firmware wakes on the next tick by construction (input is polled every
~1 ms before the idle drive step). (2) A lone `captured: IDLE` fragment in
the raw log is a reader-script split artifact (buffer reset mid-line), not a
firmware frame — the firmware only ever sends `HB <ticks> IDLE`.

## User action: watch the LED

`main.py` auto-runs on power-up with no host frame ever → idle starts at
once. Power-cycle the Pico (or `mpremote connect /dev/ttyACM0 reset`) and
watch the onboard LED with no laptop attached: breathing, then
solid / slow-blink / fast-blink tour, 0.5 s dark, repeat.

## Fix round 1 (review of PR #45 — all Critical/Important addressed)

Review: `/home/parshu/projects/jadal-qa-wt/reviews/pr45-review.md`
(SPEC: FAIL — 4 Important, 3 Minor; Security/Determinism/Honesty PASS).

### Important #1 — BREATHE RGB desync after cycle 1: CONFIRMED, FIXED

Reproduced before the fix (`elapsed=14000` gives teal, not the red restart):

```
14000 BREATHE (0.0, 0.33, 0.67)
```

Fix (`pico/idle.py`, `phase_rgb`): the BREATHE fade now reduces mod
`IDLE_CYCLE_MS` first, so it restarts with the tour cycle however the caller
passes time — strictly stronger than the suggested caller-side `tcyc`, and
`main.py` (which passes raw elapsed to both `tour_phase` and `phase_rgb`)
is consistent with no change. After the fix:

```
0 BREATHE (1.0, 0.0, 0.0)
14000 BREATHE (1.0, 0.0, 0.0)
28000 BREATHE (1.0, 0.0, 0.0)
42000 BREATHE (1.0, 0.0, 0.0)
cycle-check: True
```

Regression test added (`test_breathe_colour_restarts_with_cycle`: equality
across 3 cycles at 211 ms steps + the exact 14000/red assertion). Onboard
PWM was never affected (reviewer agrees); RGB still has no hardware
observation (nothing wired).

### Important #2 — README deploy dropped idle.py: FIXED

`pico/README.md` step 4 now copies BOTH files (`:main.py` + `:idle.py`),
states both are required (without `:idle.py` the firmware boots with attract
mode silently off), notes `run` needs `idle.py` beside `main.py`, and
documents the `flock -w 900 <repo>/tools/pico.lock` discipline.

### Important #3 — lock file missing from repo: FIXED

Committed empty `tools/pico.lock` (0 bytes, `LOCK-OK` via
`flock -w 5 tools/pico.lock true`) and documented it in the README flashing
section. All round-1 port commands used `flock -w <60..120> tools/pico.lock`
(repo path this time, not the worktree-parent path). Residual risk, stated
plainly: sibling workers (`verify`, `picoshow`, …) open `/dev/ttyACM0`
without flock, so the lock disciplines only those who take it.

### Important #4 — PWM failure killed firmware: FIXED (+ docs corrected)

- New `_idle_step()` helper (`pico/main.py`): one drive tick, returns
  `(idle_on, idle_start_ms, ok)`, NEVER raises; any PWM/driver exception
  best-effort stops idle and reports `ok=False`, and the main loop then sets
  `use_idle=False` permanently while HB/LED/FLASH/PING continue. New
  `_idle_stop()` similarly never raises (used on the receive path, so a
  broken PWM state cannot kill the loop when a host frame arrives).
- `idle_begin` initialises handles to `None` first and releases/re-raises on
  partial failure; `idle_end` guards every `deinit`, clears `_pwm` (also
  fixes Minor #2, stale handle) and is safe to call twice.
- "RP2350 should also work" claims removed (module docstring +
  `ONBOARD_LED_PIN` comment): RP2040 non-W tested; Pico W (LED on CYW43, not
  GPIO25) / RP2350 not tested, and on such boards attract mode now degrades
  to off instead of killing the firmware.
- Tests: 5 new stub-driven cases (`TestIdleStep`: start/drive, host-frame
  stop, FLASH suppression, begin/write/end failure → `ok=False` without
  raising, `_idle_stop` never raises). A W-board failure itself is not
  reproducible on CPython — stated, not simulated beyond the stubs.

### Minor #1 — any-byte wakes (adopted), #2 — done above, #3 — not mine

- Minor #1: `last_host_ms` now updates on ANY received byte (`rx += chunk`),
  so a slow sender dribbling a frame without newline still counts as a live
  host; documented in the `main.py` comment (SPEC frames stay
  newline-terminated).
- Minor #3 (`web/api.js` IDLE→OFF mapping) is outside owned files;
  follow-up only, no action taken.

### Verification (real outputs)

```
$ PYTHONNOUSERSITE=1 /home/parshu/envs/jadal/bin/python -m py_compile pico/main.py pico/idle.py tests/test_pico_idle.py && echo COMPILE-OK
COMPILE-OK
$ flock -w 5 tools/pico.lock true && echo LOCK-OK
LOCK-OK
$ PYTHONNOUSERSITE=1 timeout 250 /home/parshu/envs/jadal/bin/python -m unittest discover -s tests 2>&1 | grep -E '^(Ran|OK|FAILED|ERROR|FAIL:)'
Ran 584 tests in 26.859s
OK (skipped=8)
```

(`tests/test_pico_idle.py` now 32 tests; `test_pico_protocol.py` /
`test_pico_link.py` untouched and green. The 584 total vs 329 in round 0 is
the `origin/main` merge, not this diff.)

### Hardware re-verify: BLOCKED — board silent, exact failure stated

The fix-round-1 image is NOT on the board (it still holds the round-0
image, flash assumed intact). Re-upload was attempted and failed, documented
exactly:

- `flock … mpremote … fs cp pico/main.py :main.py` → `TransportError:
  could not enter raw repl` (two attempts), then `exec "print('STOP-OK')`
  → same error. `mpremote … reset` → same error path (`do_exec`/
  `ensure_raw_repl`, exit 1) plus `OSError: [Errno 71] Protocol error` on
  close RTS ioctl.
- Direct pyserial probe (flock-held): 3 s passive read → zero bytes (no HB);
  `PING` → 4 s → zero bytes (no PONG). Raw `Ctrl-C Ctrl-C` → `b''`,
  `Ctrl-A` → `b''`. Post-"reset" 5 s read → `lines=0`.
- `lsusb` still shows `Bus 003 Device 013: ID 2e8a:0005 MicroPython Board in
  FS mode` (enumeration healthy); `fuser` shows no holder. `dmesg`
  unavailable (Operation not permitted). Conclusion: the on-board runtime is
  hung — reachable via neither REPL nor protocol. Sibling `picoshow` tooling
  only sends PING/LED/FLASH (no foreign image; read-only grep of their
  script), so the flash should still be the round-0 build, but its death is
  unexplained from here (a PWM-path hang after hours of idle is one
  unproven suspect — notably the class of failure Important #4 now guards
  against; the guard itself is host-tested only).
- No host-side USB reset possible (uid 1000, sysfs `authorized` root-owned).
  Recovery needs a physical power-cycle (user action) — after which the
  board will boot the round-0 image (attract mode starts at once) until the
  fix-round-1 image is uploaded and re-verified in a later session.

Shipped without a fresh on-board boot test: mitigated by `py_compile` +
CPython import-freedom test + 32 unit tests (incl. stub-driven `_idle_step`
failure paths); the refactor preserves the round-0 call sequence
(begin→write…/end) and adds only guards. Not claimed otherwise.

## NOT claimed

- The LED was not visually observed (no camera on the bench); proven via the
  `HB IDLE` diagnostic + on-board `tour_phase` outputs + PWM path review.
- No external RGB LED is wired, so the RGB fade/colours are code- and
  unit-tested only, not hardware-observed.
- Port contention: sibling workers open `/dev/ttyACM0` without `flock`;
  every command in this report that opens the port used
  `flock -w 900 .../tools/pico.lock`.
