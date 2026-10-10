"""Unit tests for scripts/pico_show.py — fake serial only, no hardware.

Nothing here opens /dev/ttyACM0: every link is a PicoLink over an injected
FakePicoSerial, and the missing-port path uses an open_serial stub that
raises PicoLinkError without touching the real serial stack.

Run from the repo root:
    PYTHONNOUSERSITE=1 python3 -m unittest tests.test_pico_show -v
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
import scripts.pico_show as psmod  # noqa: E402  (seam patched in the e2e tests)
from jadal.pico_link import FakePicoSerial, PicoLink, PicoLinkError  # noqa: E402


def _run_main_with_fake(argv):
    """Run main() with a shared, inspectable FakePicoSerial. Returns (rc, ser)."""
    ser = FakePicoSerial()
    orig = psmod._make_link
    psmod._make_link = lambda args: PicoLink(serial=ser)  # noqa: E731
    try:
        rc = main(argv)
    finally:
        psmod._make_link = orig
    return rc, ser


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

    def test_hex_to_bits_accepts_optional_0x_prefix(self):
        # Same input the `pattern` subcommand accepts must work for service reuse.
        self.assertEqual(hex_to_bits("0xF0F0F0F0"), "11110000" * 4)
        self.assertEqual(hex_to_bits("0XF0F0F0F0"), "11110000" * 4)
        self.assertEqual(hex_to_bits("F0F0F0F0"), hex_to_bits("0xF0F0F0F0"))

    def test_helpers_reject_off_spec_widths(self):
        import scripts.pico_show as psmod_inner

        with self.assertRaises(ValueError):
            hex_to_bits("ABC")  # short: must not silently pad
        with self.assertRaises(ValueError):
            hex_to_bits("1FFFFFFFF")  # long: exceeds 32-bit FLASH width
        with self.assertRaises(ValueError):
            psmod_inner._bits_to_hex("1" * 33)  # must not silently truncate
        with self.assertRaises(ValueError):
            psmod_inner._bits_to_hex("10102")  # non-bit chars


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
        # Core wiring: main() must flash the pattern FOR the verdict asked,
        # then send LED OFF. (Catches a regression that flashes a fixed
        # pattern regardless of the verdict.)
        for verdict, expected_hex in (
            ("accept", "FFFFFFFF"),
            ("review", "F0F0F0F0"),
            ("reject", "AAAAAAAA"),
        ):
            with self.subTest(verdict=verdict):
                rc, ser = _run_main_with_fake(["--fake", "verdict", verdict])
                self.assertEqual(rc, 0)
                self.assertEqual(ser.pico.flashed, [expected_hex.lower()])
                self.assertIn("LED OFF", ser.pico.led_commands)

    def test_main_demo_fake_flashes_all_patterns_in_order(self):
        rc, ser = _run_main_with_fake(["--fake", "demo"])
        self.assertEqual(rc, 0)
        self.assertEqual(
            ser.pico.flashed, [hexpat.lower() for _, hexpat in DEMO_PATTERNS]
        )
        self.assertIn("LED OFF", ser.pico.led_commands)

    def test_no_heartbeat_means_nonzero_exit_but_still_cleans_up(self):
        orig = psmod.wait_for_heartbeat
        psmod.wait_for_heartbeat = lambda link, timeout=2.5: False  # noqa: E731
        try:
            rc, ser = _run_main_with_fake(["--fake", "verdict", "accept"])
        finally:
            psmod.wait_for_heartbeat = orig
        self.assertEqual(rc, 3)
        self.assertEqual(ser.pico.flashed, ["ffffffff"])  # pattern still attempted
        self.assertIn("LED OFF", ser.pico.led_commands)

    def test_status_still_exits_zero_without_heartbeat(self):
        orig = psmod.wait_for_heartbeat
        psmod.wait_for_heartbeat = lambda link, timeout=2.5: False  # noqa: E731
        try:
            rc, ser = _run_main_with_fake(["--fake", "status"])
        finally:
            psmod.wait_for_heartbeat = orig
        self.assertEqual(rc, 0)

    def test_wait_phase_interrupt_still_cleans_up(self):
        # Important #1 (round 3): an exception during the heartbeat wait must
        # still send LED OFF and disconnect (wait is inside try/finally).
        import io
        from contextlib import redirect_stderr

        ser = FakePicoSerial()
        made = {}
        orig_make, orig_wait = psmod._make_link, psmod.wait_for_heartbeat

        def _factory(args):
            link = PicoLink(serial=ser)
            made["link"] = link
            return link

        def _boom(link, timeout=2.5):
            raise KeyboardInterrupt("synthetic interrupt during wait")

        psmod._make_link, psmod.wait_for_heartbeat = _factory, _boom
        try:
            with self.assertRaises(KeyboardInterrupt):
                main(["--fake", "verdict", "accept"])
        finally:
            psmod._make_link, psmod.wait_for_heartbeat = orig_make, orig_wait
        self.assertIn("LED OFF", ser.pico.led_commands)
        self.assertFalse(made["link"].connected)

    def test_flash_failure_returns_2_and_cleans_up(self):
        # Important #2 (round 3): a device failing mid-pattern -> rc 2 with a
        # clear message (not an uncaught traceback), LED OFF still sent.
        import io
        from contextlib import redirect_stderr

        def _fail(self, pattern):
            raise PicoLinkError("write failed (synthetic unplug)")

        orig_flash = PicoLink.flash
        PicoLink.flash = _fail
        try:
            err = io.StringIO()
            with redirect_stderr(err):
                rc, ser = _run_main_with_fake(["--fake", "verdict", "accept"])
        finally:
            PicoLink.flash = orig_flash
        self.assertEqual(rc, 2)
        self.assertIn("Pico failed mid-pattern", err.getvalue())
        self.assertEqual(ser.pico.flashed, [])
        self.assertIn("LED OFF", ser.pico.led_commands)

    def test_negative_pause_rejected(self):
        with self.assertRaises(SystemExit) as ctx:
            main(["--fake", "--pause", "-1", "verdict", "accept"])
        self.assertEqual(ctx.exception.code, 2)

    def test_fake_with_port_rejected(self):
        with self.assertRaises(SystemExit) as ctx:
            main(["--fake", "--port", "/dev/ttyNOPE0", "status"])
        self.assertEqual(ctx.exception.code, 2)

    def test_main_pattern_valid_fake(self):
        # Same gap class as the round-1 verdict hole: a bug that flashes
        # fixed bits for every valid pattern input must fail the suite.
        for given, expected in (("F0F0F0F0", "f0f0f0f0"), ("0xabcdef01", "abcdef01")):
            with self.subTest(given=given):
                rc, ser = _run_main_with_fake(["--fake", "pattern", given])
                self.assertEqual(rc, 0)
                self.assertEqual(ser.pico.flashed, [expected])
                self.assertIn("LED OFF", ser.pico.led_commands)

    def test_main_pattern_bad_hex_exits_nonzero(self):
        rc, ser = _run_main_with_fake(["--fake", "pattern", "XYZ"])
        self.assertEqual(rc, 2)
        self.assertIn("LED OFF", ser.pico.led_commands)

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
