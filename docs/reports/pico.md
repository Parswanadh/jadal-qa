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

None. `jadal/config.py` already exposes `pico_port` and `pico_baud`; the module
honours `JADAL_PICO_PORT` via those defaults and needs no config edit. If the
service wants to *use* this link, `jadal/service.py` should add a
`GET /pico/status` that returns `PicoLink.status()`.
