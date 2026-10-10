# Pico 2 status-LED module

Firmware for a Raspberry Pi Pico 2 (RP2350) that drives a status LED, plus the
host-side Python link in [`jadal/pico_link.py`](../../jadal/pico_link.py).
USB serial only — **no Wi-Fi**. Protocol is defined in
[`docs/SPEC.md`](../../docs/SPEC.md) at **115200 baud, newline-terminated ASCII**.

```
Pico -> host  every 1.0 s :  HB <ticks_ms> <led_state>
Host -> Pico             :  LED G|A|R|OFF
Host -> Pico             :  FLASH <8 hex digits>   (32 bits, 100 ms each, MSB first, facing the camera)
Host -> Pico             :  PING                   -> Pico replies  PONG <ticks_ms>
```

## Hardware

- Raspberry Pi Pico 2 (RP2350) running **MicroPython**.
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

2. Put the Pico 2 into **bootloader mode**: hold the **BOOTSEL** button, plug in
   USB, then release. It mounts as a `RPI-RP2` mass-storage drive. (If
   MicroPython is already installed, you can skip the drag-drop below and flash
   straight to `/dev/ttyACM*` with mpremote.)

3. First time only — install the MicroPython firmware:

   - Download the latest `.uf2` from <https://micropython.org/download/RPI_PICO2/>.
   - Copy it onto the `RPI-RP2` drive. The board reboots into MicroPython.

4. Install the app and reset:

   ```bash
   mpremote run pico/main.py          # quick test run (Ctrl-C to stop)
   # deploy persistently:
   mpremote fs cp pico/main.py :main.py
   mpremote reset                     # reboot into main.py
   ```

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

Firmware (`pico/main.py`) has **not been run on real hardware from this repo**
(no device in CI); it is syntax-checked and logic-verified against stubs only.
The host link and its fake serial are unit-tested. See `main.py`'s module
docstring and `docs/reports/pico.md` for the NOT-DONE list.
