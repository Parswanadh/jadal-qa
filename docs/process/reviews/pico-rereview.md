ALL ADDRESSED: yes, NEW BREAKAGE: 1 Important (see N-1)

# Scoped re-review: PR #10 (feat/pico), fix round 1

Scope: fix diff `55ca014..48a188a` (`reviews/pico-fix1.diff`, 1176 lines). Head verified at `48a188a`, matching `origin/feat/pico`. Nothing modified, merged, pushed or commented.

## Test run (step 5)

`PYTHONNOUSERSITE=1 <env>/bin/python -m unittest discover -s tests`: `Ran 39 tests in 2.251s`, `OK`. Repeated 5 times, all `OK`. `py_compile pico/main.py`: OK.

## Original findings

| ID | Verdict | Evidence (current code, `48a188a`) |
|---|---|---|
| I-1 `connected` never goes stale | ADDRESSED | `jadal/pico_link.py:503-511` (`connected` property): `running and last_hb within HB_STALE_S (2.5 s)`, using `self._clock()`. `_connected` latch removed. Test `test_connected_goes_stale_without_heartbeat` advances the fake clock and asserts `connected is False`. |
| I-2 `connect()` ignores new port / `disconnect()` keeps handle | ADDRESSED | `pico_link.py:365-400` (`connect`): reuse only if `self._injected and is_open`. `pico_link.py:612-620` (`disconnect`): `self.ser = None` for factory-opened handles. Test `test_connect_after_disconnect_opens_new_port` asserts `opened == ["ACM0", "ACM1"]`. |
| I-3 raw serial errors escape `connect()` | ADDRESSED | `pico_link.py:355-363` `_open_wrapped` converts any exception to `PicoLinkError`, used in `connect()` at `:383` and `:397`. Test `test_connect_open_error_wrapped`. |
| I-4 firmware loop blocks / allocates / spins | ADDRESSED (not device-verified) | `pico/main.py:195`: `poll` built once. `:215-222`: one byte at a time, split on `\n`, so partial lines cannot block. `:219`: overlong-line drop against `MAX_LINE`. `:226`: `time.sleep_ms(1)` when idle. |
| I-5 firmware untested, bit order / LED map unasserted | ADDRESSED | New `tests/test_pico_protocol.py` imports `pico/main.py` pure helpers. `DEADBEEF_BITS` literal (`:1052-1061`) checked against `flash_bits`. LED map and `_handle` routing covered. Host tests assert `FakePico.flashed == ["deadbeef"]` (`test_pico_link.py:896`) and `led_commands`. |
| I-6 false `JADAL_PICO_PORT` claim | ADDRESSED | `pico_link.py:40` imports `DEFAULT` read-only. `:390-400` fallback to `DEFAULT.pico_port`. `docs/reports/pico.md` "Contract change proposals" corrected (lines 84-90). Test `test_default_port_fallback_comes_from_config`. Note the fallback is only used when the scan finds nothing (see N-1). |
| I-7 `from __future__` in firmware | ADDRESSED | `pico/main.py`: no `__future__` import and no function annotations (grep hits are in comments and docstrings only). |

Minors that were in scope and fixed (M-1, M-2, M-3, M-4, M-5, M-6, M-7, M-8, M-10) were spot-checked and present. M-7 is only partly done (see D-2). M-9 was skipped on purpose, as the report says.

## New breakage (step 4)

**N-1 (Important). Silent fallback to an unverified port (`jadal/pico_link.py:386-400`).**
When autodetect finds no answering Pico and no explicit `candidates` list is given, `connect()` now opens `DEFAULT.pico_port` without a PING probe and marks the link as running. `connect()` returns successfully with `connected == False`. Any later `set_led` or `flash` writes to whatever device sits on that port (for example another board or a USB-UART adapter on `/dev/ttyACM0`). Before this change, a missing Pico raised `PicoLinkError`.
Suggested fix: keep the fallback, but probe it with `_probe(ser, probe_timeout)`. If it does not answer, close it and raise `PicoLinkError`. Add a test where the fallback port is mute and `connect()` raises.

## Deferred minors (out of scope or non-blocking)

- D-1. Report says the new `tests/test_pico_protocol.py` has 14 tests. It has 13 (6 bit-order, 3 LED, 4 command routing). The 39 total is correct (26 in `test_pico_link.py` + 13). Fix the count in `docs/reports/pico.md`.
- D-2. M-7 asked for a strict `abs(offset) < sync_error_bound_ms`. The test uses `<=` (`tests/test_pico_link.py:251`), and the comment "holds by construction" is weak. Test is near-vacuous.
- D-3. MicroPython support for `bytes.split(sep, 1)` and `bytes.decode(..., "ignore")` in `pico/main.py` is not confirmed. No device run.
- D-4. Firmware behaviour changed: onboard LED is now driven in RGB mode as well (`led_levels`), and FLASH "on" lights all three RGB channels (white). Both are documented in the file docstring but not in the report.
- D-5. Firmware still has no on-device verification (acknowledged in the report as NOT DONE).
