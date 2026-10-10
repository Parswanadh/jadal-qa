# Pico module report (issue #2)

Owner branch: `feat/pico` (worktree `.../jadal-qa-wt/pico`). Implements the Pico 2
firmware + host serial link per the protocol in [`docs/SPEC.md`](../SPEC.md).
USB serial only, no Wi-Fi.

## What exists

- `pico/main.py` — MicroPython firmware for Raspberry Pi Pico 2 (RP2350).
  - `HB <ticks_ms> <led_state>` every 1.0 s (non-blocking schedule).
  - `LED G|A|R|OFF` — onboard LED plus an **optional external RGB LED** via
    `RGB_PINS` at the top of the file (G=green, A=red+green amber, R=red).
  - `FLASH <8 hex>` — 32-bit pattern, one bit per 100 ms, **MSB first**, LED
    facing the camera; time-sliced so heartbeats keep flowing during a flash,
    then the previous LED colour is restored.
  - `PING` -> `PONG <ticks_ms>`.
- `jadal/pico_link.py` — host side.
  - `PicoLink.connect()` auto-detects `/dev/ttyACM*` (then `/dev/ttyUSB*`);
    probes each candidate with `PING`/`PONG` and keeps the one that answers.
  - Background daemon reader thread parses frames; `status()` returns
    `connected`, `port`, `last_hb_age_s`, `last_hb_ticks_ms`, `last_hb_led`.
  - `set_led`, `flash`, `ping`.
  - `estimate_offset()` — NTP-style PING/PONG; returns `rtt_ms_{min,median,max}`,
    `offset_ms` (min-RTT sample), `offset_ms_median` and a conservative
    `sync_error_bound_ms = min_rtt/2`. **No claim better than one frame.**
  - Fake-serial mode for tests: `FakePico` (protocol emulator) +
    `FakePicoSerial` (pyserial-like transport, optional symmetric latency).
  - `pyserial` is imported lazily; nothing touches the network at import time;
    no secrets in code.
- `tests/test_pico_link.py` — 22 `unittest`s, all hardware-free.
- `pico/README.md` — wiring, protocol, and mpremote flashing steps.

Env note: the `jadal` venv **does** ship `pyserial` 3.5 (checked), but the module
still imports it lazily so the suite runs on hosts without it. No `pip install`
was done.

## How to run

```bash
# host tests (no Pico needed)
/home/parshu/envs/jadal/bin/python -m unittest discover -s tests -v

# firmware syntax check + import-safety (no hardware)
/home/parshu/envs/jadal/bin/python -m py_compile pico/main.py

# deploy to a real Pico 2 (see pico/README.md)
mpremote fs cp pico/main.py :main.py && mpremote reset
```

## Test output (real, `/home/parshu/envs/jadal/bin/python`)

```
$ /home/parshu/envs/jadal/bin/python -m unittest discover -s tests -v
...
test_parse_pong_valid (test_pico_link.TestProtocolParsing.test_parse_pong_valid) ... ok

----------------------------------------------------------------------
Ran 22 tests in 1.564s

OK
```
Stable across 5 repeated runs (each `OK`). A throwaway stub-based smoke check
(not committed) also exercised the firmware's pure logic on CPython: LED colour
mapping, `PING`->`PONG` ticks, MSB-first FLASH playback and colour restore — all
passed.

## Known gaps / NOT DONE

- **Firmware never run on a real Pico 2 from this repo** (no device in CI). Only
  syntax-checked (`py_compile`) and logic-checked against stubbed
  `machine`/`uselect`. Board-level behaviour (pin mapping, USB-CDC framing under
  load, actual `mpremote` flash) is unverified.
- **No measured sync accuracy.** `estimate_offset` is an NTP-style *estimate*
  whose uncertainty is bounded by half the measured round-trip
  (`sync_error_bound_ms`). We deliberately do not claim anything tighter than one
  camera frame. There is no `eval/results/` number behind it.
- `HB`/`PONG` timestamps are the raw `time.ticks_ms()` counter and wrap at
  ~12.4 days; the inner timers use `time.ticks_diff` (wrap-safe), but no epoch is
  tracked across reboots.
- Only micro-USB CDC (`/dev/ttyACM*`) is handled; no data-CDC/vendor framing, no
  flow control, no reconnect/rescan for a port that vanishes mid-session.
- `estimate_offset` correlates by "one PING -> next PONG" ordering; concurrent
  writers on the same link are not supported.

## Contract change proposals

None. `jadal/config.py` (read-only, untouched) exposes `pico_port` and
`pico_baud`; since fix round 1, `PicoLink` reads them via
`from jadal.config import DEFAULT` — explicit-port and scan opens use the
configured baud, and when no explicit port is given and the `/dev/ttyACM*`
scan finds no answering Pico, `connect()` falls back to opening
`DEFAULT.pico_port`, so `JADAL_PICO_PORT` is honoured. No config edit is
needed. If the service wants to *use* this link, `jadal/service.py` should
add a `GET /pico/status` that returns `PicoLink.status()`.

## Fix round 1 (review of PR #10)

Fixed all 7 Important findings, in owned files only:

- **I-7** — deleted `from __future__ import annotations` from `pico/main.py`
  and removed all function annotations, so boot needs nothing but stdlib
  `sys`/`time` before the guarded hardware import.
- **I-1** — `connected` is now derived from heartbeat freshness
  (`HB_STALE_S = 2.5 s`): True only while the reader runs *and* the last HB is
  fresh. New test advances a fake clock past the threshold and asserts
  `connected is False` with no error involved.
- **I-2** — `disconnect()` drops a factory-opened handle (`self.ser = None`,
  injected test doubles are kept) and checks `is_open` before reusing one, so
  `connect(port=...)` after `disconnect()` really opens the new port. Tested
  with a two-port factory asserting both opens.
- **I-3** — every open failure (`SerialException`/`OSError`/...) is wrapped in
  `PicoLinkError` by `_open_wrapped`. Tested with an `open_serial` stub that
  raises `OSError`.
- **I-4** — firmware loop reworked: the `uselect.poll` object is created once,
  the loop sleeps 1 ms when idle, and input is buffered one byte at a time and
  split on newline, so a partial line can never stall HB/FLASH timing.
  Overlong lines (>256 B, no newline) are dropped.
- **I-5** — new committed `tests/test_pico_protocol.py` (13 tests: 6 bit-order,
  3 LED mapping, 4 command routing) imports the pure helpers straight from
  `pico/main.py`: `flash_bits(0xDEADBEEF)` asserted
  against a literal MSB-first bit list (plus roundtrip guard),
  `FLASH_BIT_MS == 100`, `led_levels` onboard+RGB mapping, and `_handle`
  routing (LED set/ignore/cancels-flash, FLASH 8-hex validation). Host tests
  now also assert `FakePico.flashed == ["deadbeef"]` and `led_commands`.
- **I-6** — `PicoLink` now reads `from jadal.config import DEFAULT` (file
  itself untouched): opens use the configured baud, and with no explicit port
  and an empty scan it falls back to opening `DEFAULT.pico_port`, so
  `JADAL_PICO_PORT` is honoured. Tested by monkeypatching the config default.
  The inaccurate sentence in the previous "Contract change proposals" section
  above has been corrected.

Cheap minors also fixed: reader-exception path stops the thread and writes are
wrapped as `PicoLinkError` behind a write lock (M-1/M-2); `t3` poll bias and
tick-wrap caveats documented on `offset_sample` (M-3/M-4); scan defaults to
ACM-only with USB opt-in (M-5); onboard LED pin defaults to GPIO 25 (M-6);
latency test asserts `|offset| <= sync_error_bound_ms`, disconnect test asserts
the thread exited, and the probe write to a mute device is asserted (M-7/M-8);
unknown-frame behaviour documented (M-10). Skipped: moving the fake serial to
a new `jadal/` module (M-9, outside owned files) and any on-device verification
of I-4/I-7 (no Pico in CI — still NOT DONE).

The fake serial now also emits the device's ~1 s periodic heartbeat (this fell
out of I-1 testing: the probe consumes the first HB, so the fake must keep
beating like real firmware).

Command and real output tail
(`cd /home/parshu/projects/jadal-qa-wt/pico`):

```
$ /home/parshu/envs/jadal/bin/python -m unittest discover -s tests -v 2>&1 | tail -30
...
test_unknown_led_ignored (test_pico_protocol.TestHandleCommands.test_unknown_led_ignored) ... ok
test_onboard_levels (test_pico_protocol.TestLedMapping.test_onboard_levels) ... ok
test_rgb_mapping (test_pico_protocol.TestLedMapping.test_rgb_mapping) ... ok
test_unknown_colour_is_dark (test_pico_protocol.TestLedMapping.test_unknown_colour_is_dark) ... ok

----------------------------------------------------------------------
Ran 39 tests in 2.257s

OK
```
Stable across 5 repeated runs (each `OK`). Firmware `py_compile` passes and the
stub-based logic smoke check still passes after the loop rewrite.

## Fix round 2 (re-review of PR #10)

One new Important item, fixed in owned files only:

- **N-1** — the `DEFAULT.pico_port` fallback in `connect()` is now PING-probed
  with `_probe(ser, probe_timeout)` before adoption. A mute fallback is closed
  and `connect()` raises `PicoLinkError` instead of returning a link that would
  write `LED`/`FLASH` to an unverified device. New test
  `test_fallback_mute_port_raises` (fake mute port; asserts the raise, the
  close, and that no handle leaks onto the link). Written first and watched
  fail (`PicoLinkError not raised`), then fixed.

Also: corrected the `test_pico_protocol.py` count to 13 (6 bit-order + 3 LED +
4 routing; D-1), and tightened the latency-test offset assertion to the strict
`abs(offset) < sync_error_bound_ms` the M-7 follow-up asked for (holds with
~one-way-latency margin under the symmetric fake; D-2).

No real serial hardware was touched: every test uses injected fakes/factories
only (the plugged-in `/dev/ttyACM0` is unreadable to this user and is never
opened — the fallback tests monkeypatch both the candidate list and the
config default).

Command and real output tail
(`cd /home/parshu/projects/jadal-qa-wt/pico`):

```
$ PYTHONNOUSERSITE=1 /home/parshu/envs/jadal/bin/python -m unittest discover -s tests -v 2>&1 | tail -20
...
test_unknown_led_ignored (test_pico_protocol.TestHandleCommands.test_unknown_led_ignored) ... ok
test_onboard_levels (test_pico_protocol.TestLedMapping.test_onboard_levels) ... ok
test_rgb_mapping (test_pico_protocol.TestLedMapping.test_rgb_mapping) ... ok
test_unknown_colour_is_dark (test_pico_protocol.TestLedMapping.test_unknown_colour_is_dark) ... ok

----------------------------------------------------------------------
Ran 40 tests in 3.561s

OK
```
Stable across 5 repeated runs (each `OK`).
