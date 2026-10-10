"""Committed host-side tests for the Pico firmware's pure protocol logic.

``pico/main.py`` is MicroPython device code and never runs on the host, but its
pure helpers (FLASH bit expansion, LED colour mapping, command routing) are
imported straight from the file and tested here. Anything touching
``machine.Pin`` or ``time.ticks_ms`` stays on the device (``_handle("PING")``
is therefore *not* covered here — the ``PONG <ticks>`` wire format is covered
by ``parse_pong`` in ``test_pico_link.py`` instead).

Run from the repo root:
    /home/parshu/envs/jadal/bin/python -m unittest discover -s tests -v
"""
import importlib.util
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_firmware():
    spec = importlib.util.spec_from_file_location(
        "pico_main_fw", os.path.join(ROOT, "pico", "main.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


fw = _load_firmware()

# 0xDEADBEEF, MSB first: D=1101 E=1110 A=1010 D=1101 B=1011 E=1110 E=1110 F=1111
DEADBEEF_BITS = [
    1, 1, 0, 1,
    1, 1, 1, 0,
    1, 0, 1, 0,
    1, 1, 0, 1,
    1, 0, 1, 1,
    1, 1, 1, 0,
    1, 1, 1, 0,
    1, 1, 1, 1,
]


class TestFlashBitOrder(unittest.TestCase):
    def test_deadbeef_msb_first(self):
        self.assertEqual(fw.flash_bits(0xDEADBEEF), DEADBEEF_BITS)

    def test_literal_not_mistyped(self):
        # guards against a typo in the literal above, independent of flash_bits
        self.assertEqual(int("".join(map(str, DEADBEEF_BITS)), 2), 0xDEADBEEF)

    def test_all_zeros_and_ones(self):
        self.assertEqual(fw.flash_bits(0), [0] * 32)
        self.assertEqual(fw.flash_bits(0xFFFFFFFF), [1] * 32)

    def test_msb_and_lsb_positions(self):
        msb = fw.flash_bits(0x80000000)
        self.assertEqual(msb[0], 1)
        self.assertEqual(msb[1:], [0] * 31)
        lsb = fw.flash_bits(0x00000001)
        self.assertEqual(lsb[-1], 1)
        self.assertEqual(lsb[:-1], [0] * 31)

    def test_flash_bit_period_is_100ms(self):
        self.assertEqual(fw.FLASH_BIT_MS, 100)  # SPEC: 100 ms per bit

    def test_pattern_masked_to_32_bits(self):
        self.assertEqual(fw.flash_bits(0x1FFFFFFFF), [1] * 32)


class TestLedMapping(unittest.TestCase):
    def test_onboard_levels(self):
        # onboard-only fallback: any colour -> ON, OFF -> OFF.
        self.assertEqual(fw.led_levels("G"), (1, (0, 1, 0)))
        self.assertEqual(fw.led_levels("A"), (1, (1, 1, 0)))
        self.assertEqual(fw.led_levels("R"), (1, (1, 0, 0)))
        self.assertEqual(fw.led_levels("OFF"), (0, (0, 0, 0)))

    def test_rgb_mapping(self):
        # G -> green, A -> red+green (amber), R -> red, OFF -> dark.
        _, rgb = fw.led_levels("G")
        self.assertEqual(rgb, (0, 1, 0))
        _, rgb = fw.led_levels("A")
        self.assertEqual(rgb, (1, 1, 0))
        _, rgb = fw.led_levels("R")
        self.assertEqual(rgb, (1, 0, 0))
        _, rgb = fw.led_levels("OFF")
        self.assertEqual(rgb, (0, 0, 0))

    def test_unknown_colour_is_dark(self):
        level, rgb = fw.led_levels("PURPLE")
        self.assertEqual((level, rgb), (0, (0, 0, 0)))


class _StubLeds:
    def __init__(self):
        self.states = []
        self._state = "OFF"

    def set(self, state):
        self.states.append(state)
        self._state = state

    def state(self):
        return self._state

    def on(self):
        pass

    def off(self):
        pass


class _StubFlasher:
    def __init__(self):
        self.started = []
        self.pattern = None

    def start(self, pattern):
        self.started.append(pattern)
        self.pattern = pattern


class TestHandleCommands(unittest.TestCase):
    def test_led_routing(self):
        leds, fl = _StubLeds(), _StubFlasher()
        fw._handle("LED G", leds, fl)
        self.assertEqual(leds.states, ["G"])
        fw._handle("LED off", leds, fl)
        self.assertEqual(leds.states[-1], "OFF")

    def test_unknown_led_ignored(self):
        leds, fl = _StubLeds(), _StubFlasher()
        fw._handle("LED PURPLE", leds, fl)
        fw._handle("LED", leds, fl)
        self.assertEqual(leds.states, [])

    def test_led_cancels_running_flash(self):
        leds, fl = _StubLeds(), _StubFlasher()
        fw._handle("FLASH DEADBEEF", leds, fl)
        self.assertEqual(fl.started, [0xDEADBEEF])
        fw._handle("LED R", leds, fl)
        self.assertIsNone(fl.pattern)
        self.assertEqual(leds.states, ["R"])

    def test_flash_validation(self):
        leds, fl = _StubLeds(), _StubFlasher()
        fw._handle("FLASH 123", leds, fl)        # too short
        fw._handle("FLASH ZZZZZZZZ", leds, fl)   # not hex
        self.assertEqual(fl.started, [])
        fw._handle("FLASH deadbeef", leds, fl)
        self.assertEqual(fl.started, [0xDEADBEEF])


if __name__ == "__main__":
    unittest.main()
