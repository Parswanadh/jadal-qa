"""Jadal QA Raspberry Pi Pico firmware (MicroPython, RP2040 tested; RP2350 should also work; USB serial only - no Wi-Fi).

Implements the firmware half of the protocol in ``docs/SPEC.md`` at 115200 baud,
newline-terminated ASCII:

    -> host, every HEARTBEAT_S :  ``HB <ticks_ms> <led_state>``
    host -> ``LED G|A|R|OFF``   : set the status LED colour
    host -> ``FLASH <8 hex>``   : blink the 32-bit pattern, one bit per
                                   FLASH_BIT_MS, MSB first, LED facing the camera
    host -> ``PING``            : reply ``PONG <ticks_ms>``

Configuration is at the top of the file: the *onboard* LED is always driven, and
an *optional external RGB LED* can be wired and enabled via ``RGB_PINS``
(G drives green, A = red+green amber, R = red). Heartbeats are emitted on a
non-blocking schedule while a FLASH pattern plays out, so heartbeats keep
flowing during a flash. Unknown or malformed frames are silently ignored.

(No ``from __future__`` import and no function annotations anywhere in this
file: stock MicroPython must be able to boot it with zero extra modules.)

NOT DONE (no hardware in CI, so none of this is measured/verified on-device):
  * Never executed on a real Raspberry Pi Pico (RP2040 tested; RP2350 should also work) from this repo; only syntax-checked and
    logic-verified against stubs on the host (see tests/test_pico_protocol.py
    for the pure helpers, which run on CPython).
  * ``time.ticks_*`` wraparound (~2**30 ms) is handled for the flash/HB timers
    via ``time.ticks_diff``, but the absolute ``ticks_ms`` reported in HB/PONG
    is raw and wraps the same way.
  * We make NO claim that a flash edge is synchronised to better than one
    camera frame.
"""
import sys
import time

# ---- configuration (edit to taste) --------------------------------------- #
ONBOARD_LED_PIN = 25             # Raspberry Pi Pico (RP2040 tested; RP2350 should also work) built-in LED (GPIO 25)
RGB_PINS = None                  # None = onboard LED only; else (r_pin, g_pin, b_pin)
                                 #   e.g. (17, 16, 18) wires an external RGB LED.
HEARTBEAT_S = 1.0                # seconds between HB frames
FLASH_BIT_MS = 100               # ms per bit while playing a FLASH pattern (32 bits)
MAX_LINE = 256                   # drop an input line longer than this (never block)

try:
    import machine  # noqa: F401  (only present on real MicroPython hardware)
    import uselect  # noqa: F401
    _ON_HARDWARE = True
except ImportError:           # allows import/py_compile on a normal CPython box
    _ON_HARDWARE = False

VALID_LED = ("G", "A", "R", "OFF")
_HEX = "0123456789abcdef"


# ---- pure helpers (no hardware, no ticks: unit-tested on the host) -------- #
def flash_bit(pattern, index):
    """Bit ``index`` of a 32-bit pattern, MSB first (index 0 = bit 31)."""
    return (pattern >> (31 - index)) & 1


def flash_bits(pattern):
    """Expand a 32-bit pattern to its 32 on/off bits, MSB first."""
    pattern = pattern & 0xFFFFFFFF
    return [flash_bit(pattern, i) for i in range(32)]


def led_levels(state):
    """Map a colour to pin levels.

    Returns ``(onboard, (r, g, b))``. Onboard: any of G/A/R -> 1, else 0.
    RGB: G -> green, A -> red+green (amber), R -> red, anything else -> dark.
    Unknown colours map to dark so a bad command can never latch the LED on.
    """
    s = state.strip().upper()
    onboard = 1 if s in ("G", "A", "R") else 0
    if s == "G":
        rgb = (0, 1, 0)
    elif s == "A":
        rgb = (1, 1, 0)
    elif s == "R":
        rgb = (1, 0, 0)
    else:
        rgb = (0, 0, 0)
    return onboard, rgb


def _send(line):
    """Write a newline-terminated ASCII frame to the USB CDC stdout."""
    sys.stdout.write(line)


class Leds:
    """Onboard LED plus an optional external RGB LED (colour map: led_levels)."""

    def __init__(self):
        self._onboard = machine.Pin(ONBOARD_LED_PIN, machine.Pin.OUT)
        self._r = self._g = self._b = None
        if RGB_PINS:
            self._r = machine.Pin(RGB_PINS[0], machine.Pin.OUT)
            self._g = machine.Pin(RGB_PINS[1], machine.Pin.OUT)
            self._b = machine.Pin(RGB_PINS[2], machine.Pin.OUT)
        self._state = "OFF"
        self._write_state("OFF")

    def _write_state(self, state):
        level, (r, g, b) = led_levels(state)
        self._onboard.value(level)
        if self._r is not None:
            self._r.value(r)
            self._g.value(g)
            self._b.value(b)
        self._state = state

    def set(self, state):
        self._write_state(state)

    def on(self):
        # FLASH "on" bit: everything lit, most visible to the camera.
        self._onboard.value(1)
        if self._r is not None:
            self._r.value(1)
            self._g.value(1)
            self._b.value(1)

    def off(self):
        self._onboard.value(0)
        if self._r is not None:
            self._r.value(0)
            self._g.value(0)
            self._b.value(0)

    def state(self):
        return self._state


class Flasher:
    """Non-blocking MSB-first bit player for FLASH patterns.

    A 32-bit pattern is shown one bit per ``FLASH_BIT_MS``; ``tick()`` advances
    it using ``time.ticks_diff`` so the wraparound of ``ticks_ms`` is safe. On
    completion the previously set LED colour is restored.
    """

    def __init__(self, leds):
        self.leds = leds
        self.pattern = None
        self.index = 0
        self._next_ms = 0

    def start(self, pattern):
        self.pattern = pattern & 0xFFFFFFFF
        self.index = 0
        self._show_bit()

    def _show_bit(self):
        if flash_bit(self.pattern, self.index):
            self.leds.on()
        else:
            self.leds.off()
        self.index += 1
        self._next_ms = time.ticks_add(time.ticks_ms(), FLASH_BIT_MS)

    def tick(self):
        if self.pattern is None:
            return
        if time.ticks_diff(time.ticks_ms(), self._next_ms) >= 0:
            if self.index < 32:
                self._show_bit()
            else:
                self.pattern = None
                self.leds.set(self.leds.state())  # restore last colour


def _handle(text, leds, flasher):
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


def main():
    if not _ON_HARDWARE:
        raise SystemExit("pico/main.py must run on MicroPython hardware")
    leds = Leds()
    flasher = Flasher(leds)

    poll = uselect.poll()
    poll.register(sys.stdin, uselect.POLLIN)

    rx = b""
    next_hb = time.ticks_add(time.ticks_ms(), 0)
    hb_period_ms = int(HEARTBEAT_S * 1000)

    while True:
        now = time.ticks_ms()
        flasher.tick()

        if time.ticks_diff(now, next_hb) >= 0:
            _send("HB %d %s\n" % (time.ticks_ms(), leds.state()))
            next_hb = time.ticks_add(now, hb_period_ms)

        # Non-blocking input: only read when poll says bytes are ready, one
        # byte at a time, buffering until a newline. A partial line never
        # stalls the loop, so HB and FLASH timing keep running.
        if poll.poll(0):
            chunk = sys.stdin.read(1)
            if chunk:
                if isinstance(chunk, str):
                    chunk = chunk.encode("ascii", "ignore")
                rx += chunk
                if len(rx) > MAX_LINE:
                    rx = b""  # overlong line without newline: drop, stay live
                else:
                    while b"\n" in rx:
                        line, rx = rx.split(b"\n", 1)
                        _handle(line.decode("ascii", "ignore"), leds, flasher)
        else:
            time.sleep_ms(1)


if __name__ == "__main__":
    main()
