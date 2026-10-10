"""Unit tests for scripts/pico_show.py — fake serial only, no hardware.

Nothing here opens /dev/ttyACM0: every link is a PicoLink over an injected
FakePicoSerial, and the missing-port path uses an open_serial stub that
raises PicoLinkError without touching the real serial stack.

Run from the repo root:
    PYTHONNOUSERSITE=1 /home/parshu/envs/jadal/bin/python -m unittest tests.test_pico_show -v
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.pico_show import (  # noqa: E402
    DEMO_PATTERNS,
    hex_to_bits,
    main,
    pattern_for,
)
from jadal.pico_link import FakePicoSerial, PicoLink, PicoLinkError  # noqa: E402


class PatternBitsTest(unittest.TestCase):
    def test_exact_verdict_hex(self):
        self.assertEqual(pattern_for("accept"), "FFFFFFFF")
        self.assertEqual(pattern_for("review"), "F0F0F0F0")
        self.assertEqual(pattern_for("reject"), "AAAAAAAA")

    def test_bit_expansion_msb_first(self):
        # accept = solid on: all 32 bits set.
        self.assertEqual(hex_to_bits("FFFFFFFF"), "1" * 32)
        # review = slow blink: 11110000 repeated 4x (400 ms on/off at 100 ms/bit).
        self.assertEqual(hex_to_bits("F0F0F0F0"), "11110000" * 4)
        # reject = fast blink: alternating bits (100 ms period).
        self.assertEqual(hex_to_bits("AAAAAAAA"), "10" * 16)

    def test_pattern_for_matches_expansion(self):
        for verdict, bits in (
            ("accept", "1" * 32),
            ("review", "11110000" * 4),
            ("reject", "10" * 16),
        ):
            self.assertEqual(hex_to_bits(pattern_for(verdict)), bits)

    def test_demo_covers_all_verdicts(self):
        flashed = [h for _, h in DEMO_PATTERNS]
        for verdict in ("accept", "review", "reject"):
            self.assertIn(pattern_for(verdict), flashed)
        self.assertGreaterEqual(len(DEMO_PATTERNS), 6)  # verdicts + SOS/heartbeat/countdown

    def test_unknown_verdict_raises(self):
        with self.assertRaises(ValueError):
            pattern_for("maybe")


class FakeCliTest(unittest.TestCase):
    def test_verdict_accept_flashes_solid_and_cleans_up(self):
        ser = FakePicoSerial()
        link = PicoLink(serial=ser)
        link.connect()
        try:
            link.flash(pattern_for("accept"))
        finally:
            try:
                link.set_led("OFF")
            finally:
                link.disconnect()
        self.assertEqual(ser.pico.flashed, ["ffffffff"])
        self.assertIn("LED OFF", ser.pico.led_commands)
        self.assertFalse(link.connected)

    def test_cleanup_in_finally_on_error(self):
        """A flash failure must still send LED OFF and disconnect."""
        ser = FakePicoSerial()
        link = PicoLink(serial=ser)
        link.connect()
        with self.assertRaises(PicoLinkError):
            try:
                link.flash("ZZZZZZZZ")  # invalid pattern -> PicoLinkError
            finally:
                try:
                    link.set_led("OFF")
                finally:
                    link.disconnect()
        self.assertIn("LED OFF", ser.pico.led_commands)
        self.assertEqual(link.status()["connected"], False)

    def test_main_verdict_fake_end_to_end(self):
        rc = main(["--fake", "verdict", "review"])
        self.assertEqual(rc, 0)

    def test_main_pattern_bad_hex_exits_nonzero(self):
        rc = main(["--fake", "pattern", "XYZ"])
        self.assertNotEqual(rc, 0)

    def test_main_status_fake(self):
        rc = main(["--fake", "status"])
        self.assertEqual(rc, 0)


class MissingPortTest(unittest.TestCase):
    def test_no_pico_answers_is_clear_and_nonzero(self):
        def _refuse(port, baud):
            raise PicoLinkError(f"could not open serial port {port!r}: missing (synthetic)")

        link = PicoLink(open_serial=_refuse)
        with self.assertRaises(PicoLinkError) as ctx:
            link.connect(port="/dev/ttyNOPE0")
        self.assertIn("could not open serial port", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
