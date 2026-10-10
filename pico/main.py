"""Jadal QA Raspberry Pi Pico firmware (MicroPython, RP2040 non-W tested; Pico W
and RP2350 not tested; USB serial only - no Wi-Fi).

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

Idle/attract mode (pure logic in ``pico/idle.py``): after ``IDLE_AFTER_S``
(see ``idle.py``) of host silence the board plays a visible ambient animation
(PWM breathing on the onboard LED, then an ACCEPT/REVIEW/REJECT verdict tour;
colour fade/patterns on the external RGB LED when wired). ANY received host
frame -- even an unknown or malformed one -- stops the animation immediately,
restores the plain-output pins and the last host-set LED colour, and idle mode
restarts only after another ``IDLE_AFTER_S`` of silence. While idle, the HB
line reports ``led_state`` as ``IDLE`` (read-only diagnostic; the host can
never *set* IDLE and ``parse_hb`` accepts it like any other state).

(No ``from __future__`` import and no function annotations anywhere in this
file: stock MicroPython must be able to boot it with zero extra modules.)

LIMITS (the pure helpers are logic-tested on the host, see tests/test_pico_protocol.py, which runs on CPython):
  * Exercised on a real Raspberry Pi Pico (RP2040) with MicroPython 1.29.0: heartbeat, LED, FLASH, PING/PONG
    verified over USB serial; see docs/reports/pico_hardware.md. LED colours: the onboard LED is a single
    colour, G/A/R are distinguishable only with an external RGB LED.
  * ``time.ticks_*`` wraparound (~2**30 ms) is handled for the flash/HB timers
    via ``time.ticks_diff``, but the absolute ``ticks_ms`` reported in HB/PONG
    is raw and wraps the same way.
  * We make NO claim that a flash edge is synchronised to better than one
    camera frame.
"""
import sys
import time

# ---- configuration (edit to taste) --------------------------------------- #
ONBOARD_LED_PIN = 25             # Raspberry Pi Pico (RP2040 non-W tested; Pico W / RP2350 not tested) built-in LED (GPIO 25)
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

try:
    import idle               # board layout: main.py + idle.py side by side
except ImportError:           # repo checkout: pico/idle.py (needs repo root on sys.path)
    try:
        from pico import idle
    except ImportError:
        idle = None           # degraded: host protocol still works, no attract mode

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

    # -- idle/attract mode: PWM drive, hardware only (never called on CPython).
    # Tested on the RP2040 non-W Pico, where GPIO25 supports machine.PWM (RGB
    # pins too when wired). NOT tested on Pico W (its onboard LED hangs off
    # the CYW43 chip, not GPIO25) or RP2350: there idle_begin may raise, which
    # the main loop treats as "no attract mode" (host protocol keeps running).
    def idle_begin(self):
        self._pwm = None
        self._pwm_rgb = None
        rgb = []
        try:
            self._pwm = machine.PWM(self._onboard, freq=1000)
            if self._r is not None:
                # One at a time: a tuple constructor would strand the
                # already-attached handles if a later one raises.
                for pin in (self._r, self._g, self._b):
                    rgb.append(machine.PWM(pin, freq=1000))
                self._pwm_rgb = tuple(rgb)
        except Exception:
            for pwm in rgb:
                try:
                    pwm.deinit()
                except Exception:
                    pass
            try:
                self.idle_end()  # releases onboard PWM if attached
            except Exception:
                pass
            raise
        self.idle_write(0.0, (0.0, 0.0, 0.0))

    def idle_write(self, brightness, rgb):
        duty = int(brightness * 65535)
        if duty < 0:
            duty = 0
        elif duty > 65535:
            duty = 65535
        self._pwm.duty_u16(duty)
        if self._pwm_rgb is not None:
            for pwm, level in zip(self._pwm_rgb, rgb):
                d = int(level * 65535)
                if d < 0:
                    d = 0
                elif d > 65535:
                    d = 65535
                pwm.duty_u16(d)

    def idle_end(self):
        # Back to plain outputs + the last host-set colour (host always wins).
        # Never leaves a deinit'd handle behind; safe to call twice or after
        # a half-failed idle_begin.
        if getattr(self, "_pwm", None) is not None:
            try:
                self._pwm.deinit()
            except Exception:
                pass
            self._pwm = None
        if getattr(self, "_pwm_rgb", None) is not None:
            for pwm in self._pwm_rgb:
                try:
                    pwm.deinit()
                except Exception:
                    pass
            self._pwm_rgb = None
        self._onboard = machine.Pin(ONBOARD_LED_PIN, machine.Pin.OUT)
        if self._r is not None:
            self._r = machine.Pin(RGB_PINS[0], machine.Pin.OUT)
            self._g = machine.Pin(RGB_PINS[1], machine.Pin.OUT)
            self._b = machine.Pin(RGB_PINS[2], machine.Pin.OUT)
        self._write_state(self._state)


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


def _idle_step(leds, idle_mod, now_ms, last_host_ms, flasher_busy,
               idle_on, idle_start_ms):
    """Advance attract mode by one tick; NEVER raises.

    Returns ``(idle_on, idle_start_ms, ok)``. ``ok=False`` means the LED
    driver (PWM) failed: the caller must stop using idle mode, while the host
    protocol loop keeps running. A failed stop is best-effort (pins are
    restored if possible) and likewise never raises.
    """
    try:
        if not idle_mod.idle_state(now_ms, last_host_ms) or flasher_busy:
            if idle_on:
                leds.idle_end()
            return (False, idle_start_ms, True)
        if not idle_on:
            leds.idle_begin()
            idle_on = True
            idle_start_ms = now_ms
        elapsed = idle_mod.ticks_diff(now_ms, idle_start_ms)
        name, level = idle_mod.tour_phase(elapsed)
        r, g, b = idle_mod.phase_rgb(name, elapsed)
        leds.idle_write(level, (r * level, g * level, b * level))
        return (True, idle_start_ms, True)
    except Exception:
        try:
            leds.idle_end()
        except Exception:
            pass
        return (False, idle_start_ms, False)


def _idle_stop(leds):
    """Leave idle mode if active; never raises (broken PWM must not kill HB)."""
    try:
        leds.idle_end()
    except Exception:
        pass


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

    # Idle/attract mode: last_host_ms tracks ANY received host frame (even an
    # unknown/malformed one); None = no frame ever (fresh power-up -> idle now).
    use_idle = idle is not None
    last_host_ms = None
    idle_on = False
    idle_start_ms = 0

    while True:
        now = time.ticks_ms()
        flasher.tick()

        if time.ticks_diff(now, next_hb) >= 0:
            if idle_on:
                hb_state = "IDLE"
            else:
                hb_state = leds.state()
            _send("HB %d %s\n" % (time.ticks_ms(), hb_state))
            next_hb = time.ticks_add(now, hb_period_ms)

        # Non-blocking input: only read when poll says bytes are ready, one
        # byte at a time, buffering until a newline. A partial line never
        # stalls the loop, so HB and FLASH timing keep running. ANY received
        # byte counts as host activity (a slow sender dribbling a frame
        # without a newline yet is still a live host, and the host always
        # wins); every completed line stops idle mode BEFORE the frame is
        # handled, so a FLASH starting here drives plain output pins with no
        # added latency beyond this tick.
        if poll.poll(0):
            chunk = sys.stdin.read(1)
            if chunk:
                if isinstance(chunk, str):
                    chunk = chunk.encode("ascii", "ignore")
                rx += chunk
                last_host_ms = now
                if len(rx) > MAX_LINE:
                    rx = b""  # overlong line without newline: drop, stay live
                    if idle_on:
                        _idle_stop(leds)
                        idle_on = False
                else:
                    while b"\n" in rx:
                        line, rx = rx.split(b"\n", 1)
                        if idle_on:
                            _idle_stop(leds)
                            idle_on = False
                        _handle(line.decode("ascii", "ignore"), leds, flasher)
        else:
            time.sleep_ms(1)

        # Idle animation: non-blocking, one drive step per tick; never runs
        # while a FLASH pattern is playing (belt-and-braces: a FLASH command
        # always implies a host frame well inside IDLE_AFTER_S anyway). A PWM
        # failure (e.g. Pico W, whose LED is not GPIO25) disables attract
        # mode permanently; the host loop is unaffected.
        if use_idle:
            idle_on, idle_start_ms, ok = _idle_step(
                leds, idle, now, last_host_ms, flasher.pattern is not None,
                idle_on, idle_start_ms)
            if not ok:
                use_idle = False


if __name__ == "__main__":
    main()
