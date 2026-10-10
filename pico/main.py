"""Jadal QA Pico 2 firmware (MicroPython, RP2350, USB serial only - no Wi-Fi).

Implements the firmware half of the protocol in ``docs/SPEC.md`` at 115200 baud,
newline-terminated ASCII:

    -> host, every HEARTBEAT_S :  ``HB <ticks_ms> <led_state>``
    host -> ``LED G|A|R|OFF``   : set the status LED colour
    host -> ``FLASH <8 hex>``   : blink the 32-bit pattern, one bit per
                                   FLASH_BIT_MS, MSB first, LED facing the camera
    host -> ``PING``            : reply ``PONG <ticks_ms>``

Configuration is at the top of the file: the *onboard* LED is always used, and an
*optional external RGB LED* can be wired and enabled via ``RGB_PINS`` (G drives
green, A = red+green amber, R = red). Heartbeats are emitted on a non-blocking
schedule while a FLASH pattern plays out, so heartbeats keep flowing during a
flash.

NOT DONE (no hardware in CI, so none of this is measured/verified on-device):
  * Never executed on a real Pico 2 from this repo; only syntax-checked.
  * ``time.ticks_*`` wraparound (~12.4 days) is handled for the flash/HB timers,
    but the absolute ``ticks_ms`` reported in HB/PONG is raw and wraps the same way.
  * A blocking 3.2 s ``FLASH`` is instead time-sliced in the main loop; if the
    camera frame rate is high, the effective on/off timing has jitter of one
    scheduler tick. We make NO claim that a flash edge is synchronised to better
    than one camera frame.
"""
from __future__ import annotations

import sys
import time

# ---- configuration (edit to taste) --------------------------------------- #
ONBOARD_LED_PIN = "LED"      # Pico 2 built-in LED ("LED" or GPIO number e.g. 25)
RGB_PINS = None              # None = onboard LED only; else (r_pin, g_pin, b_pin)
                             #   e.g. (17, 16, 18) wires an external RGB LED.
HEARTBEAT_S = 1.0            # seconds between HB frames
FLASH_BIT_MS = 100           # ms per bit while playing a FLASH pattern (32 bits)

try:
    import machine  # noqa: F401  (only present on real MicroPython hardware)
    import uselect  # noqa: F401
    _ON_HARDWARE = True
except ImportError:           # allows import/py_compile on a normal CPython box
    _ON_HARDWARE = False

VALID_LED = ("G", "A", "R", "OFF")
_HEX = "0123456789abcdef"


def _send(line: str) -> None:
    """Write a newline-terminated ASCII frame to the USB CDC stdout."""
    sys.stdout.write(line)


class Leds:
    """Onboard LED plus an optional external RGB LED.

    Colour mapping (RGB): G -> green, A -> red+green (amber), R -> red, OFF -> off.
    Onboard-only fallback: any of G/A/R -> ON, OFF -> OFF.
    """

    def __init__(self):
        self._onboard = machine.Pin(ONBOARD_LED_PIN, machine.Pin.OUT)
        self._r = self._g = self._b = None
        if RGB_PINS:
            self._r = machine.Pin(RGB_PINS[0], machine.Pin.OUT)
            self._g = machine.Pin(RGB_PINS[1], machine.Pin.OUT)
            self._b = machine.Pin(RGB_PINS[2], machine.Pin.OUT)
        self._state = "OFF"
        self._write_state("OFF")

    def _write_state(self, state: str) -> None:
        if self._r is not None:
            self._r.value(1 if state in ("R", "A") else 0)
            self._g.value(1 if state in ("G", "A") else 0)
            self._b.value(0)
        else:
            self._onboard.value(1 if state in ("G", "A", "R") else 0)
        self._state = state

    def set(self, state: str) -> None:
        self._write_state(state)

    def on(self) -> None:
        if self._r is not None:
            self._r.value(1)
            self._g.value(1)
            self._b.value(0)
        else:
            self._onboard.value(1)

    def off(self) -> None:
        if self._r is not None:
            self._r.value(0)
            self._g.value(0)
            self._b.value(0)
        else:
            self._onboard.value(0)

    def state(self) -> str:
        return self._state


class Flasher:
    """Non-blocking MSB-first bit player for FLASH patterns.

    A 32-bit pattern is shown one bit per ``FLASH_BIT_MS``; ``tick()`` advances
    it using ``time.ticks_diff`` so the wraparound of ``ticks_ms`` is safe. On
    completion the previously set LED colour is restored.
    """

    def __init__(self, leds: "Leds"):
        self.leds = leds
        self.pattern = None
        self.index = 0
        self._next_ms = 0

    def start(self, pattern: int) -> None:
        self.pattern = pattern & 0xFFFFFFFF
        self.index = 0
        self._show_bit()

    def _show_bit(self) -> None:
        bit = (self.pattern >> (31 - self.index)) & 1
        if bit:
            self.leds.on()
        else:
            self.leds.off()
        self.index += 1
        self._next_ms = time.ticks_add(time.ticks_ms(), FLASH_BIT_MS)

    def tick(self) -> None:
        if self.pattern is None:
            return
        if time.ticks_diff(time.ticks_ms(), self._next_ms) >= 0:
            if self.index < 32:
                self._show_bit()
            else:
                self.pattern = None
                self.leds.set(self.leds.state())  # restore last colour


def _handle(text: str, leds: "Leds", flasher: "Flasher") -> None:
    text = text.strip()
    if not text:
        return
    if text.startswith("LED "):
        colour = text[4:].strip().upper()
        if colour in VALID_LED:
            flasher.pattern = None           # an explicit LED cancels a running flash
            leds.set(colour)
    elif text.startswith("FLASH "):
        h = text[6:].strip().lower()
        if len(h) == 8 and all(c in _HEX for c in h):
            flasher.start(int(h, 16))
    elif text == "PING":
        _send("PONG %d\n" % time.ticks_ms())


def _readline():
    """Return one line of input if available, else None (non-blocking)."""
    poll = uselect.poll()
    poll.register(sys.stdin, uselect.POLLIN)
    if poll.poll(0):
        raw = sys.stdin.readline()
        if raw:
            if isinstance(raw, (bytes, bytearray)):
                return bytes(raw).decode("ascii", "ignore")
            return raw
    return None


def main() -> None:
    if not _ON_HARDWARE:
        raise SystemExit("pico/main.py must run on MicroPython hardware")
    leds = Leds()
    flasher = Flasher(leds)

    next_hb = time.ticks_add(time.ticks_ms(), 0)
    hb_period_ms = int(HEARTBEAT_S * 1000)

    while True:
        now = time.ticks_ms()
        flasher.tick()

        if time.ticks_diff(now, next_hb) >= 0:
            _send("HB %d %s\n" % (time.ticks_ms(), leds.state()))
            next_hb = time.ticks_add(now, hb_period_ms)

        line = _readline()
        if line:
            _handle(line, leds, flasher)


if __name__ == "__main__":
    main()
