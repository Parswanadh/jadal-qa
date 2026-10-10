"""Host-side unit tests for jadal/pico_link.py.

These run WITHOUT any Pico hardware: PicoLink is exercised against an in-memory
fake serial device (FakePicoSerial) that emulates the firmware protocol from
docs/SPEC.md, and against a fake clock for the offset math.

Run from the repo root:
    /home/parshu/envs/jadal/bin/python -m unittest discover -s tests -v
"""
import os
import sys
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jadal.pico_link import (  # noqa: E402
    BAUD,
    VALID_LED,
    FakePico,
    FakePicoSerial,
    PicoLink,
    PicoLinkError,
    encode_flash,
    encode_led,
    offset_sample,
    parse_hb,
    parse_pong,
)


class _MuteSerial:
    """A serial device that never answers PING (i.e. not our Pico)."""

    def __init__(self, port):
        self.port = port
        self.baudrate = BAUD
        self.is_open = True
        self.writes = []

    def write(self, data):
        self.writes.append(bytes(data))

    @property
    def in_waiting(self):
        return 0

    def read(self, n=1):
        return b""

    def close(self):
        self.is_open = False


class TestProtocolParsing(unittest.TestCase):
    def test_parse_hb_valid(self):
        self.assertEqual(parse_hb(b"HB 12345 G\n"), (12345, "G"))
        self.assertEqual(parse_hb("HB 0 OFF"), (0, "OFF"))
        self.assertEqual(parse_hb(b"HB 900 A"), (900, "A"))

    def test_parse_hb_invalid(self):
        self.assertIsNone(parse_hb(b"PONG 1\n"))
        self.assertIsNone(parse_hb(b"HB abc G"))   # non-int ticks
        self.assertIsNone(parse_hb(b"HB 12"))      # missing state
        self.assertIsNone(parse_hb(b"garbage"))

    def test_parse_pong_valid(self):
        self.assertEqual(parse_pong(b"PONG 4242\n"), 4242)

    def test_parse_pong_invalid(self):
        self.assertIsNone(parse_pong(b"HB 1 G\n"))
        self.assertIsNone(parse_pong(b"PONG xx"))
        self.assertIsNone(parse_pong(b"PONG 1 2"))

    def test_encode_led_valid(self):
        for c in ("G", "A", "R", "OFF"):
            self.assertEqual(encode_led(c), b"LED %s\n" % c.encode())
        # case insensitive
        self.assertEqual(encode_led("off"), b"LED OFF\n")
        self.assertEqual(encode_led(" g "), b"LED G\n")

    def test_encode_led_invalid(self):
        with self.assertRaises(PicoLinkError):
            encode_led("X")
        with self.assertRaises(PicoLinkError):
            encode_led("GREEN")

    def test_encode_flash_int(self):
        self.assertEqual(encode_flash(0), b"FLASH 00000000\n")
        self.assertEqual(encode_flash(0xFFFFFFFF), b"FLASH FFFFFFFF\n")
        self.assertEqual(encode_flash(0xDEADBEEF), b"FLASH DEADBEEF\n")

    def test_encode_flash_hex_string(self):
        self.assertEqual(encode_flash("deadbeef"), b"FLASH DEADBEEF\n")
        self.assertEqual(encode_flash("0xDEADBEEF"), b"FLASH DEADBEEF\n")

    def test_encode_flash_invalid(self):
        with self.assertRaises(PicoLinkError):
            encode_flash("xyz12345")     # bad hex / wrong len
        with self.assertRaises(PicoLinkError):
            encode_flash(2 ** 32)        # too big
        with self.assertRaises(PicoLinkError):
            encode_flash(-1)


class TestOffsetMath(unittest.TestCase):
    def test_offset_sample_exact(self):
        # host monotonic seconds; pico ticks are ms on its own clock.
        # t0=1.0s t3=1.5s -> rtt 500ms ; midpoint=1.25s=1250ms
        rtt, off = offset_sample(t0_s=1.0, t1_ticks_ms=4250.0, t3_s=1.5)
        self.assertAlmostEqual(rtt, 500.0, places=6)
        self.assertAlmostEqual(off, 4250.0 - 1250.0, places=6)

    def test_offset_zero_rtt(self):
        rtt, off = offset_sample(t0_s=10.0, t1_ticks_ms=10.0 * 1000, t3_s=10.0)
        self.assertAlmostEqual(rtt, 0.0, places=6)
        self.assertAlmostEqual(off, 0.0, places=6)


class TestPicoLinkWithFake(unittest.TestCase):
    def _connect(self, latency_s=0.0, epoch_offset_ms=12345.0):
        ser = FakePicoSerial(latency_s=latency_s, epoch_offset_ms=epoch_offset_ms)
        link = PicoLink(serial=ser)
        link.connect()
        return link, ser

    def _await(self, cond, timeout=2.0):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if cond():
                return True
            time.sleep(0.005)
        return cond()

    def test_connect_fake_sets_connected_and_port(self):
        link, ser = self._connect()
        try:
            self.assertTrue(link.connected)
            self.assertEqual(link.port, "FAKE")
        finally:
            link.disconnect()

    def test_reader_parses_initial_heartbeat(self):
        link, ser = self._connect()
        try:
            self.assertTrue(
                self._await(lambda: link.status()["last_hb_ticks_ms"] is not None),
                "reader never saw the initial heartbeat",
            )
            st = link.status()
            self.assertTrue(st["connected"])
            self.assertIn(st["last_hb_led"], VALID_LED)
            self.assertIsNotNone(st["last_hb_age_s"])
        finally:
            link.disconnect()

    def test_set_led_writes_correct_bytes(self):
        link, ser = self._connect()
        try:
            link.set_led("A")
            time.sleep(0.03)
            self.assertIn(b"LED A\n", ser._out_accum)
        finally:
            link.disconnect()

    def test_flash_writes_correct_bytes(self):
        link, ser = self._connect()
        try:
            link.flash(0xDEADBEEF)
            time.sleep(0.03)
            self.assertIn(b"FLASH DEADBEEF\n", ser._out_accum)
        finally:
            link.disconnect()

    def test_send_without_serial_raises(self):
        link = PicoLink()  # no serial, not connected
        with self.assertRaises(PicoLinkError):
            link.set_led("G")

    def test_bad_led_rejected_before_write(self):
        link, ser = self._connect()
        try:
            with self.assertRaises(PicoLinkError):
                link.set_led("PURPLE")
        finally:
            link.disconnect()

    def test_estimate_offset_zero_latency_recovers_epoch(self):
        # With no artificial latency the offset estimator should recover the
        # fake device's epoch_offset_ms almost exactly.
        link, ser = self._connect(latency_s=0.0, epoch_offset_ms=7777.0)
        try:
            est = link.estimate_offset(samples=8, timeout=2.0)
            self.assertEqual(est["n"], 8)
            self.assertAlmostEqual(est["offset_ms"], 7777.0, delta=25.0)
            self.assertLess(est["rtt_ms_median"], 150.0)
            self.assertGreaterEqual(est["sync_error_bound_ms"], 0.0)
        finally:
            link.disconnect()

    def test_estimate_offset_with_latency_rtt_and_bound(self):
        # Symmetric one-way latency L -> round trip ~2L; offset still centred,
        # but the honest error bound grows with latency. We assert the RTT is in
        # a loose band and that the offset is within half the measured RTT.
        L = 0.03
        link, ser = self._connect(latency_s=L, epoch_offset_ms=0.0)
        try:
            est = link.estimate_offset(samples=8, timeout=3.0)
            self.assertGreater(est["rtt_ms_min"], 0.0)
            self.assertLess(est["rtt_ms_median"], 500.0)
            self.assertLessEqual(est["sync_error_bound_ms"], est["rtt_ms_median"])
            self.assertLessEqual(abs(est["offset_ms"]), est["rtt_ms_median"])
        finally:
            link.disconnect()

    def test_disconnect_stops_thread(self):
        link, ser = self._connect()
        link.disconnect()
        self.assertFalse(link.connected)


class TestAutoDetect(unittest.TestCase):
    def test_autodetect_picks_first_that_pongs(self):
        mute = _MuteSerial("MUTE")
        pico = FakePicoSerial(port="PICO")
        factory = {
            "MUTE": lambda port, baud: mute,
            "PICO": lambda port, baud: pico,
        }
        link = PicoLink(open_serial=lambda port, baud: factory[port](port, baud))
        try:
            link.connect(port=None, probe_timeout=0.4, candidates=["MUTE", "PICO"])
            self.assertTrue(link.connected)
            self.assertEqual(link.port, "PICO")
        finally:
            link.disconnect()

    def test_autodetect_raises_when_none_answer(self):
        factory = {"MUTE": lambda port, baud: _MuteSerial(port)}
        link = PicoLink(open_serial=lambda port, baud: factory[port](port, baud))
        with self.assertRaises(PicoLinkError):
            link.connect(port=None, probe_timeout=0.15,
                         candidates=["MUTE"])


if __name__ == "__main__":
    unittest.main()
