ALL ADDRESSED: yes, NEW BREAKAGE: none

# Scoped re-review: PR #10 (feat/pico), fix round 2

Scope: fix diff `48a188a..origin/feat/pico` (`reviews/pico-fix2.diff`, 146 lines). Worktree HEAD `2ca2497` ("pico: fix round 2 — probe the config fallback port (N-1)"). Nothing modified, merged, pushed or commented. No serial device opened (`/dev/ttyACM0` untouched).

## Test run

`cd jadal-qa-wt/pico && PYTHONNOUSERSITE=1 <env>/bin/python -m unittest discover -s tests`: `Ran 40 tests`, `OK`. Repeated 5 times, all `OK`. Test counts: 27 in `test_pico_link.py`, 13 in `test_pico_protocol.py`.

Old-code check (scratch copy only): `jadal/pico_link.py` from `48a188a` with the new tests gives `FAIL: test_fallback_mute_port_raises`, so the new test does exercise N-1.

## N-1 (Important): verified fixed

`jadal/pico_link.py:392-414`: the `DEFAULT.pico_port` fallback is opened, then `self._probe(fallback, probe_timeout)` is called before adoption. On a mute port it closes the handle and raises `PicoLinkError`. `self.ser` stays `None` on that path, so no handle leaks onto the link. Explicit `candidates` still raise strictly, as before.

## Doc claims (`docs/reports/pico.md`, "Fix round 2")

- D-1 count (13 = 6 bit-order + 3 LED + 4 routing): matches the file.
- D-2 strict `assertLess(abs(offset), sync_error_bound_ms)` at `tests/test_pico_link.py:251`: matches. The test passes.
- "Ran 40 tests ... OK": matches.
- "Written first and watched fail": consistent with the old-code run above.
- "No real serial hardware touched": consistent with the injected fakes in the diff.

## New breakage

None in the fix diff. No new Critical or Important items.

## Deferred (non-blocking, carried from round 1)

- D-3 MicroPython `bytes.split(sep, 1)` / `decode(..., "ignore")` unconfirmed on device.
- D-4 onboard LED RGB-mode behaviour not in the report.
- D-5 firmware not verified on device.
