# Raspberry Pi Pico (RP2040 non-W tested; Pico W / RP2350 not tested) status-LED module

Firmware for a Raspberry Pi Pico (RP2040 non-W tested; Pico W / RP2350 not tested) that drives a status LED, plus the
host-side Python link in [`jadal/pico_link.py`](../../jadal/pico_link.py).
USB serial only — **no Wi-Fi**. Protocol is defined in
[`docs/SPEC.md`](../../docs/SPEC.md) at **115200 baud, newline-terminated ASCII**.

```
Pico -> host  every 1.0 s :  HB <ticks_ms> <led_state>
Host -> Pico             :  LED G|A|R|OFF
Host -> Pico             :  FLASH <8 hex digits>   (32 bits, 100 ms each, MSB first, facing the camera)
Host -> Pico             :  PING                   -> Pico replies  PONG <ticks_ms>
```

Unknown or malformed frames are silently ignored by the firmware (an unknown
`LED` colour never latches the LED on).

## Idle / attract mode

With no laptop attached the board still shows it is alive: after 20 s
(`IDLE_AFTER_S` in `pico/idle.py`) with no received host frame, `main.py`
plays a repeating ambient animation — 3× PWM breathing (3 s each) on the
onboard LED, then a verdict tour (ACCEPT solid 1 s, REVIEW slow blink
400 ms for 2 s, REJECT fast blink 100 ms for 1.5 s), then a 0.5 s dark gap.
With an external RGB LED wired (`RGB_PINS`): green/amber/red for the tour
plus a slow colour fade while breathing. Every duration is tunable at the top
of `pico/idle.py`; the pure animation logic there is unit-tested in
`tests/test_pico_idle.py`.

Host protocol always wins: **any** received frame — even an unknown or
malformed one — stops the animation immediately (pins restored to plain
outputs, last host-set colour back) and idle restarts only after another 20 s
of silence. Heartbeats keep flowing during the animation; a `FLASH` arriving
mid-animation starts on the next tick. The one visible protocol addition is
read-only: while idle, the `HB` line reports `led_state` as `IDLE`
(the host can never *set* `IDLE`; `parse_hb` accepts it like any other state).

Power up with no host and watch the LED: idle starts at once (no host frame
has ever arrived), so after `mpremote reset` — or a power-cycle — the board
breathes and tours verdicts on its own.

## Hardware

- Raspberry Pi Pico (RP2040 non-W tested; Pico W / RP2350 not tested) running **MicroPython**.
- Onboard LED (GP25) is always used and is visible to the camera.
- Optional **external RGB LED**: wire three GPIOs and set `RGB_PINS = (r, g, b)`
  at the top of `main.py`. Colour mapping: `G` = green, `A` = red+green (amber),
  `R` = red, `OFF` = off. When `RGB_PINS` is `None` the onboard LED is used
  (on for `G|A|R`, off for `OFF`).

## Flashing (mpremote)

1. Install MicroPython tooling (host, not the firmware):

   ```bash
   python -m pip install --user mpremote
   # or:  pipx install mpremote
   ```

2. Put the Pico into **bootloader mode**: hold the **BOOTSEL** button, plug in
   USB, then release. It mounts as a `RPI-RP2` mass-storage drive. (If
   MicroPython is already installed, you can skip the drag-drop below and flash
   straight to `/dev/ttyACM*` with mpremote.)

3. First time only — install the MicroPython firmware:

   - Download the latest RP2040 `.uf2` from <https://micropython.org/download/RPI_PICO/>.
     (Pico 2 / RP2350 is a different chip and is NOT tested with this firmware.)
   - Copy it onto the `RPI-RP2` drive. The board reboots into MicroPython.

4. Install the app and reset (both files are required — without `:idle.py`
   the firmware boots but attract mode stays off with no error):

   ```bash
   mpremote run pico/main.py          # quick test run (Ctrl-C to stop; host-protocol only — run-mode does NOT exercise attract mode unless :idle.py is already on the board, so deploy :idle.py first to test the animation)
   # deploy persistently:
   mpremote fs cp pico/main.py :main.py
   mpremote fs cp pico/idle.py :idle.py
   mpremote reset                     # reboot into main.py
   ```

   Port access is serialised with `flock`: wrap every `mpremote connect
   /dev/ttyACM0 ...` command in
   `flock -w 900 <repo>/tools/pico.lock <command>` (the empty `tools/pico.lock`
   file is committed so all workers share one lock).

5. Watch the serial console:

   ```bash
   mpremote                           # opens an interactive REPL/serial on /dev/ttyACM*
   # you should see a heartbeat line once per second, e.g.  HB 1234567 OFF
   ```

Press **Ctrl-D** in the mpremote REPL for a soft reset, or **Ctrl-C** to interrupt.

## Host link

```python
from jadal.pico_link import PicoLink

link = PicoLink().connect()        # auto-detects /dev/ttyACM* (needs pyserial)
link.set_led("A")
link.flash(0xDEADBEEF)             # 32-bit pattern, MSB first
print(link.status())               # connected, port, last heartbeat age, ...
est = link.estimate_offset()       # NTP-style PING/PONG: rtt + clock offset
link.disconnect()
```

Held open in fake-serial mode for tests (no hardware):

```python
from jadal.pico_link import PicoLink, FakePicoSerial
link = PicoLink(serial=FakePicoSerial()).connect()
print(link.status())
link.disconnect()
```

Run the host tests (no device needed):

```bash
/home/parshu/envs/jadal/bin/python -m unittest discover -s tests -v
```

## Timing / accuracy caveat

`estimate_offset()` returns an **estimate** only, together with a conservative
`sync_error_bound_ms` (half the measured round-trip). A flash edge is **not**
synchronised to any camera frame better than about one frame — do not treat the
LED blink as a frame-accurate trigger.

## Status

Firmware (`pico/main.py`) has been run on a real Raspberry Pi Pico (RP2040) with
MicroPython 1.29.0: heartbeat, LED, FLASH and PING/PONG verified over USB serial;
see `docs/reports/pico_hardware.md`. The onboard LED's colour output and the FLASH
pattern were not observed by the agent.
The host link and its fake serial are unit-tested. See `main.py`'s module
docstring (LIMITS section) and `docs/reports/pico.md` for the open points.
