# Review: PR #10 (feat/pico) for issue #2 (Pico 2 firmware + host serial link)

SPEC: PASS, QUALITY: CHANGES NEEDED

Diff: `reviews/pico.diff` (origin/feat/pico @ 55ca014, 5 files, +1174). Unit tests: `Ran 22 tests ... OK` (1.57 s, no hardware). No MicroPython binary is installed locally, so firmware findings are from reading the code and are not device-verified.

## SPEC compliance

| # | Requirement | Result | Evidence |
|---|---|---|---|
| 1 | Firmware is MicroPython for Pico 2, USB serial only, no network | ✅ (one unverified risk) | No network imports. `pico/main.py:27` uses `from __future__ import annotations`; I could not confirm whether stock MicroPython accepts it (see I-7). |
| 2 | Protocol: `HB <ticks_ms> <led_state>` every 1.0 s; `LED G\|A\|R\|OFF`; `FLASH <8 hex>` 100 ms/bit MSB first; `PING`→`PONG <ticks_ms>` | ✅ | HB period `pico/main.py:36`, emit `:186-188`; LED `:147-151`; FLASH length/hex check `:153-155`; MSB-first `:124`; PING `:156-157`. Host encoders `jadal/pico_link.py:100-124` (`FLASH %08X`) match. |
| 3 | PicoLink: auto-detect `/dev/ttyACM*`, reader thread, `status()` with HB age and connected, `set_led`, `flash`, `estimate_offset` via PING/PONG, fake-serial mode | ⚠️ partial | Autodetect `jadal/pico_link.py:284-285, 361-372`; reader thread `:395-411`; fake serial `:158-260`. `connected` is True as soon as `connect()` runs, before any HB (`:343`), and never goes stale (see I-1). |
| 4 | Tests in `tests/test_pico_link.py`, hardware-free | ✅ | 22 tests pass, no hardware. Several are weak (see Minor). Firmware has no tests. |
| 5 | `pico/README.md` documents mpremote flashing | ✅ | `pico/README.md:25-60` (install, BOOTSEL, `mpremote fs cp`, `mpremote reset`). |
| 6 | No sync accuracy claim better than one camera frame | ✅ | `docs/reports/pico.md:67-76`, `pico/README.md:89-94`, `pico/main.py:18-25`, `jadal/pico_link.py:16-19`. |
| 7 | Only owned files touched | ✅ | Diff touches only `pico/*`, `jadal/pico_link.py`, `tests/test_pico_link.py`, `docs/reports/pico.md`. No edits to `jadal/config.py`, `jadal/__init__.py`, `docs/SPEC.md`. But see I-6 (report claims config integration that does not exist). |
| 8 | No secrets; no network at import | ✅ | `pico_link.py:24-32` imports only stdlib. `pyserial` is imported lazily in `_open_real_serial` (`:272-280`). No secrets. |

Overall SPEC: PASS. Row 1 is unverified rather than failing: MicroPython `__future__` support could not be confirmed from the web or locally (no micropython binary). Row 3's `connected` semantics are a quality issue, since the SPEC only requires the field to exist.

## Findings

### Critical

None.

### Important

**I-7. `pico/main.py:27`: `from __future__ import annotations` may crash the firmware at boot (unverified).**
MicroPython has no `__future__` module, so the import raises ImportError before the hardware `try` block. The device would never emit HB.
MicroPython may not ship a `__future__` module, and a web search could not settle it. The import is unnecessary here, so delete line 27 regardless. Confirm with `mpremote run pico/main.py`.

**I-1. `jadal/pico_link.py:343` and `:439-449`: `connected` never goes stale.**
`connect()` sets `_connected = True` immediately. Only the reader's exception path (`:410`) clears it. If the Pico hangs, resets, or the cable is pulled without an exception, `status()["connected"]` stays True forever, and `last_hb_age_s` grows silently.
Fix: derive `connected` from HB freshness, e.g. `connected = running and last_hb_time is not None and (clock() - last_hb_time) < 2.5 * HEARTBEAT_S`. Set `_connected=False` in `_reset_state`. Add a test with a fake clock that advances past the threshold and asserts `connected is False`.

**I-2. `jadal/pico_link.py:325-344`: `connect()` ignores a new `port` after any earlier connection, and `disconnect()` never clears `self.ser`.**
`if self.ser is not None` (`:334`) runs before the `port` branch. After `disconnect()` (`:519-534`), a second `connect(port="/dev/ttyACM1")` reuses the old, closed handle and never opens the new port. Autodetect is also skipped. Same for `connect()` on an injected serial that was closed.
Fix: in `disconnect()`, set `self.ser = None` unless the serial was injected (keep an `_injected` flag). Also check `is_open` before reusing a handle. Test: connect, disconnect, connect with another port, and assert the second port is opened.

**I-3. `jadal/pico_link.py:337`: `connect(port=...)` raises raw `serial.SerialException`, not `PicoLinkError`.**
`_open_real_serial` wraps only ImportError (`:278-280`), and `connect()` does not wrap `_open_serial` errors (`:337`). Callers are documented to get `PicoLinkError` for connection failures (`:52-53`).
Fix: wrap `serial.SerialException` and `OSError` in `PicoLinkError` in `connect()` around `self._open_serial(port, BAUD)`. Add a test with an `open_serial` that raises `OSError` and assert `PicoLinkError`. Also add a test for a missing explicit port.

**I-4. `pico/main.py:160-170` and `:173-192`: the main loop blocks and allocates on every pass.**
- `_readline()` builds a new `uselect.poll()` on every iteration (`:162-163`). That is GC churn in a tight loop.
- The loop has no sleep, so it spins at 100% CPU.
- `sys.stdin.readline()` (`:165`) blocks until `\n`. A partial line from the host stalls the loop and stops HB and FLASH timing.
Fix: create the poll object once in `main()` and pass it in. Add `time.sleep_ms(1)` when no input is available. For partial lines, buffer bytes yourself (read with `sys.stdin.read(1)` after `poll` says ready, accumulate until `\n`), so the loop never blocks mid-line.

**I-5. `pico/main.py` has no committed tests, and the host tests never check FLASH bit order or LED colour mapping.**
`Flasher` (MSB-first, restore-colour) and `Leds` mapping are the pieces most likely to be wrong, and nothing in the repo exercises them. `FakePico.flashed` and `led_commands` (`jadal/pico_link.py:164-165, 176-177`) are recorded but never asserted (grep finds no reads in tests).
Fix: move the pure logic (bit extraction, LED state→pin mapping) into a helper importable on CPython, and add tests that assert `[bit31..bit0]` for `0xDEADBEEF` and the amber mapping. Or assert `FakePico.flashed == ["deadbeef"]` in the existing FLASH test (`tests/test_pico_link.py:166-170`).

**I-6. `docs/reports/pico.md:88-90` says the module "honours `JADAL_PICO_PORT` via those defaults" through `jadal/config.py`. This is false.**
`jadal/pico_link.py` never imports `jadal.config` (grep finds no reference). The env var has no effect.
Fix: correct the report to say config is not wired. Either add `port` / `baud` defaults from `jadal.config` in `connect()` (no edit to config.py is needed, since it already exposes `pico_port` / `pico_baud`), or delete the sentence.

### Minor

- **M-1. `jadal/pico_link.py:395-411` (`_run`), `:410`:** on a reader exception `_connected` is set False but `_running` stays True. `set_led` then writes to a dead handle and raises a raw pyserial error, not `PicoLinkError`. Fix: set `_running = False` in the exception path and wrap writes in `try/except` that re-raises `PicoLinkError`.
- **M-2. `jadal/pico_link.py:396`, `:343-344`, `:351`: shared-state locking.** `_running` is read and written without `_lock`. `_connected` is cleared without the lock at `:410` and `:351`. `ser.write` (`:461`, `:466`, `:470`) runs concurrently with `ser.read` in the reader thread, with no lock. Fix: guard writes with a `threading.Lock` and move all `_connected` / `_running` reads and writes under `_lock`.
- **M-3. `jadal/pico_link.py:427`: reader-thread timestamps add up to `read_poll_s` (20 ms) of bias to `t3`.** `estimate_offset` takes `t3` from the reader after its sleep, not at receipt. RTT is inflated, so `sync_error_bound_ms` stays conservative, but `offset_ms` shifts by about -d/2. Fix: document it, or reduce `read_poll_s` / use a blocking read with a short timeout. The docs should not present `offset_ms` as better than the ~10 ms bias.
- **M-4. `jadal/pico_link.py:151`: `offset_ms` mixes a wrapping device tick (2^30 ms) with a host monotonic clock.** Any device wrap or reboot shifts the estimate by an arbitrary amount. Fix: say so in the docstring, or normalise modulo 2^30 and report `epoch_ok`.
- **M-5. `jadal/pico_link.py:284-285`: `_default_candidates` also probes `/dev/ttyUSB*`.** SPEC says `/dev/ttyACM*` only. Sending `PING` to unrelated USB-UART adapters is harmless but unspecified. Fix: keep ACM only, or make ttyUSB opt-in.
- **M-6. `pico/main.py:33`: `ONBOARD_LED_PIN = "LED"`.** The `"LED"` alias is not guaranteed on every Pico 2 MicroPython build. GPIO 25 is the documented pin. Fix: default to `25`.
- **M-7. `tests/test_pico_link.py:201-212` (latency test) and `:216-219` (disconnect test):** the latency test's `offset` assertion is `abs(offset) <= rtt_median`, which is nearly vacuous. The disconnect test checks `connected` only and never checks that the thread has exited. Fix: assert `abs(offset) < sync_error_bound_ms` and `not link._thread.is_alive()` (or add a public `is_running`).
- **M-8. Dead / write-only fake state:** `_MuteSerial.writes` (`tests/test_pico_link.py:39-42`), `FakePico.flashed` / `led_commands` (`jadal/pico_link.py:164-165`). Remove, or assert on them (see I-5).
- **M-9. `jadal/pico_link.py` is 540 lines and ships the fake serial in production code.** This is acceptable for the spec's fake-serial mode, but consider moving `FakePico` / `FakePicoSerial` to `jadal/pico_fake.py` to keep the module small.
- **M-10. `pico/main.py:148-150`:** an unknown LED colour is silently ignored. Fine for firmware, but the host cannot tell. Consider a `ERR` frame, or document it.

## Test run
```
Ran 22 tests in 1.566s
OK
```
Reproduced. The `docs/reports/pico.md:59-61` output matches. The "stable across 5 runs" claim was not re-checked.

## Files reviewed
- `<worktrees>/reviews/pico.diff`
- `pico/main.py`, `pico/README.md`, `jadal/pico_link.py`, `tests/test_pico_link.py`, `docs/reports/pico.md` (all in the diff)
- `docs/SPEC.md` (sections: Determinism rules, Pico serial protocol); issue #2
