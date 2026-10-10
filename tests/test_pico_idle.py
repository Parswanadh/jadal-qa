"""Host-side tests for the Pico idle/attract-mode pure logic (``pico/idle.py``).

Like ``test_pico_protocol.py``, these run on CPython: ``pico/idle.py`` (and
``pico/main.py``) are imported straight from the files, no hardware needed.

Run from the repo root:
    /home/parshu/envs/jadal/bin/python -m unittest discover -s tests -v
"""
import importlib.util
import os
import subprocess
import sys
import threading
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IDLE_PATH = os.path.join(ROOT, "pico", "idle.py")
MAIN_PATH = os.path.join(ROOT, "pico", "main.py")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


idle = _load("pico_idle_fw", IDLE_PATH)
fw = _load("pico_main_idle_fw", MAIN_PATH)


class TestIdleScheduleConstants(unittest.TestCase):
    def test_timeout_is_20s(self):
        self.assertEqual(idle.IDLE_AFTER_S, 20)

    def test_tour_durations_exact(self):
        self.assertEqual(idle.BREATHE_PERIOD_MS, 3000)
        self.assertEqual(idle.BREATHE_CYCLES, 3)
        self.assertEqual(idle.BREATHE_TOTAL_MS, 9000)
        self.assertEqual(idle.ACCEPT_MS, 1000)
        self.assertEqual(idle.REVIEW_HALF_MS, 400)
        self.assertEqual(idle.REVIEW_MS, 2000)
        self.assertEqual(idle.REJECT_HALF_MS, 100)
        self.assertEqual(idle.REJECT_MS, 1500)
        self.assertEqual(idle.GAP_MS, 500)
        self.assertEqual(idle.IDLE_CYCLE_MS, 9000 + 1000 + 2000 + 1500 + 500)


class TestIdleState(unittest.TestCase):
    def test_inactive_within_timeout(self):
        self.assertFalse(idle.idle_state(0, 0))
        self.assertFalse(idle.idle_state(19999, 0))
        self.assertFalse(idle.idle_state(15000, 0))

    def test_active_at_and_after_timeout(self):
        self.assertTrue(idle.idle_state(20000, 0))
        self.assertTrue(idle.idle_state(20001, 0))
        self.assertTrue(idle.idle_state(10 ** 6, 0))

    def test_active_immediately_with_no_host_ever(self):
        # Fresh power-up: no host frame has ever arrived -> attract at once.
        self.assertTrue(idle.idle_state(0, None))
        self.assertTrue(idle.idle_state(12345, None))

    def test_host_frame_deactivates_at_any_phase(self):
        # A host frame "now" kills idle no matter where the tour is.
        for t in range(0, idle.IDLE_CYCLE_MS + 2000, 137):
            self.assertFalse(idle.idle_state(t, t), "t=%d" % t)
            self.assertFalse(idle.idle_state(t + 1, t), "t=%d" % t)

    def test_stays_off_while_host_active(self):
        # Host chatter every 5 s keeps idle off forever.
        last = 0
        for now in range(0, 120000, 5000):
            if now > 0:
                last = now  # frame arrives every 5 s
            self.assertFalse(idle.idle_state(now, last), "t=%d" % now)


class TestTicksWraparound(unittest.TestCase):
    WRAP = 1 << 30

    def test_diff_basic(self):
        self.assertEqual(idle.ticks_diff(10, 3), 7)
        self.assertEqual(idle.ticks_diff(3, 10), -7)
        self.assertEqual(idle.ticks_diff(5, 5), 0)

    def test_diff_wraps_like_ticks_ms(self):
        self.assertEqual(idle.ticks_diff(0, self.WRAP - 1), 1)
        self.assertEqual(idle.ticks_diff(19000, self.WRAP - 1000), 20000)
        self.assertEqual(idle.ticks_diff(100, self.WRAP - 1000), 1100)

    def test_idle_state_across_wrap(self):
        W = self.WRAP
        self.assertFalse(idle.idle_state(100, W - 1000))      # 1.1 s of silence
        self.assertFalse(idle.idle_state(0, W - 1))           # 1 ms of silence
        self.assertTrue(idle.idle_state(19000, W - 1000))     # 20 s of silence
        self.assertTrue(idle.idle_state(20000, W - 1000))

    def test_brightness_and_tour_wrap_safe(self):
        # Millisecond counters near the wrap still give sane outputs.
        W = self.WRAP
        b = idle.idle_brightness(W - 7)
        self.assertTrue(0.0 <= b <= 1.0)
        name, level = idle.tour_phase(W - 7)
        self.assertIn(name, ("BREATHE", "ACCEPT", "REVIEW", "REJECT", "GAP"))
        self.assertTrue(0.0 <= level <= 1.0)


class TestBreathing(unittest.TestCase):
    def test_range(self):
        for t in range(0, 3000, 37):
            b = idle.idle_brightness(t)
            self.assertTrue(0.0 <= b <= 1.0, "t=%d b=%r" % (t, b))

    def test_shape(self):
        self.assertAlmostEqual(idle.idle_brightness(0), 0.0)
        self.assertAlmostEqual(idle.idle_brightness(1500), 1.0)
        self.assertAlmostEqual(idle.idle_brightness(750), 0.5)
        self.assertAlmostEqual(idle.idle_brightness(2250), 0.5)

    def test_periodic(self):
        for t in range(0, 6000, 211):
            self.assertAlmostEqual(idle.idle_brightness(t),
                                   idle.idle_brightness(t + 3000))


class TestTour(unittest.TestCase):
    def test_phase_boundaries_exact(self):
        seen = []
        prev = None
        for t in range(0, idle.IDLE_CYCLE_MS + 1):
            name, _ = idle.tour_phase(t)
            if name != prev:
                seen.append((t, name))
                prev = name
        self.assertEqual([s[0] for s in seen],
                         [0, 9000, 10000, 12000, 13500, 14000])
        self.assertEqual([s[1] for s in seen],
                         ["BREATHE", "ACCEPT", "REVIEW", "REJECT", "GAP",
                          "BREATHE"])

    def test_accept_is_solid(self):
        for t in range(9000, 10000, 50):
            name, level = idle.tour_phase(t)
            self.assertEqual((name, level), ("ACCEPT", 1))

    def test_review_is_slow_blink(self):
        self.assertEqual(idle.tour_phase(10000), ("REVIEW", 1))
        self.assertEqual(idle.tour_phase(10400), ("REVIEW", 0))
        self.assertEqual(idle.tour_phase(10800), ("REVIEW", 1))
        self.assertEqual(idle.tour_phase(11999), ("REVIEW", 1))
        self.assertEqual(idle.tour_phase(11998), ("REVIEW", 1))
        levels = set(idle.tour_phase(t)[1] for t in range(10000, 12000, 25))
        self.assertEqual(levels, {0, 1})

    def test_reject_is_fast_blink(self):
        self.assertEqual(idle.tour_phase(12000), ("REJECT", 1))
        self.assertEqual(idle.tour_phase(12100), ("REJECT", 0))
        self.assertEqual(idle.tour_phase(12200), ("REJECT", 1))
        self.assertEqual(idle.tour_phase(13499), ("REJECT", 1))
        self.assertEqual(idle.tour_phase(13450), ("REJECT", 1))
        levels = set(idle.tour_phase(t)[1] for t in range(12000, 13500, 25))
        self.assertEqual(levels, {0, 1})

    def test_gap_is_dark(self):
        for t in range(13500, 14000, 50):
            self.assertEqual(idle.tour_phase(t), ("GAP", 0))

    def test_cycle_repeats(self):
        for t in range(0, idle.IDLE_CYCLE_MS, 173):
            self.assertEqual(idle.tour_phase(t),
                             idle.tour_phase(t + idle.IDLE_CYCLE_MS))


class TestPhaseRgb(unittest.TestCase):
    def test_verdict_colours(self):
        self.assertEqual(idle.phase_rgb("ACCEPT", 0), (0.0, 1.0, 0.0))
        self.assertEqual(idle.phase_rgb("REVIEW", 0), (1.0, 1.0, 0.0))
        self.assertEqual(idle.phase_rgb("REJECT", 0), (1.0, 0.0, 0.0))
        self.assertEqual(idle.phase_rgb("GAP", 0), (0.0, 0.0, 0.0))
        self.assertEqual(idle.phase_rgb("BOGUS", 0), (0.0, 0.0, 0.0))

    def test_breathe_fade_bounded_and_moving(self):
        seen = set()
        for t in range(0, idle.BREATHE_TOTAL_MS, 100):
            r, g, b = idle.phase_rgb("BREATHE", t)
            for v in (r, g, b):
                self.assertTrue(0.0 <= v <= 1.0, "t=%d rgb=%r" % (t, (r, g, b)))
            seen.add((round(r, 3), round(g, 3), round(b, 3)))
        self.assertGreater(len(seen), 10)  # actually fades, not static

    def test_breathe_colour_restarts_with_cycle(self):
        # Regression: tour_phase is periodic over IDLE_CYCLE_MS but the fade
        # used elapsed % BREATHE_TOTAL_MS, so cycle 1+ showed wrong colours
        # (e.g. elapsed=14000 gave teal instead of the red restart).
        for t in range(0, 3 * idle.IDLE_CYCLE_MS, 211):
            self.assertEqual(idle.phase_rgb("BREATHE", t),
                             idle.phase_rgb("BREATHE", t % idle.IDLE_CYCLE_MS),
                             "t=%d" % t)
        self.assertEqual(idle.phase_rgb("BREATHE", 14000), (1.0, 0.0, 0.0))
        self.assertEqual(idle.phase_rgb("BREATHE", 0),
                         idle.phase_rgb("BREATHE", 28000))


class TestIdleStep(unittest.TestCase):
    """The main-loop drive step: host-wins gating + PWM-failure survival."""

    class _StubLeds(object):
        def __init__(self, fail_at=None):
            self.calls = []
            self.fail_at = fail_at

        def idle_begin(self):
            self.calls.append("begin")
            if self.fail_at == "begin":
                raise RuntimeError("no PWM (simulated Pico W)")

        def idle_write(self, brightness, rgb):
            self.calls.append(("write", brightness, rgb))
            if self.fail_at == "write":
                raise RuntimeError("duty failed")

        def idle_end(self):
            self.calls.append("end")
            if self.fail_at == "end":
                raise RuntimeError("deinit failed")

    def test_step_starts_and_drives_when_silent(self):
        leds = self._StubLeds()
        on, start, ok = fw._idle_step(leds, idle, 0, None, False, False, 0)
        self.assertEqual((on, start, ok), (True, 0, True))
        self.assertEqual(leds.calls[0], "begin")
        self.assertEqual(leds.calls[1][0], "write")

    def test_step_stops_on_host_frame(self):
        leds = self._StubLeds()
        on, start, ok = fw._idle_step(leds, idle, 5000, 5000, False, True, 1000)
        self.assertEqual((on, start, ok), (False, 1000, True))
        self.assertIn("end", leds.calls)

    def test_step_suppressed_while_flashing(self):
        leds = self._StubLeds()
        on, _, ok = fw._idle_step(leds, idle, 10 ** 6, None, True, False, 0)
        self.assertEqual((on, ok), (False, True))
        self.assertEqual(leds.calls, [])

    def test_step_pwm_failure_disables_idle_without_raising(self):
        # (begin, last_host, busy, idle_on): "end" fails on the stop path, so
        # it needs an active host to take that path.
        cases = {
            "begin": (None, False, False),
            "write": (None, False, False),
            "end": (10 ** 6, False, True),
        }
        for fail_at, (last, busy, was_on) in cases.items():
            leds = self._StubLeds(fail_at=fail_at)
            on, _, ok = fw._idle_step(leds, idle, 10 ** 6, last, busy,
                                      was_on, 0)
            self.assertFalse(ok, fail_at)
            self.assertFalse(on, fail_at)

    def test_idle_stop_never_raises(self):
        leds = self._StubLeds(fail_at="end")
        self.assertIsNone(fw._idle_stop(leds))  # must not raise


class TestImportSideEffects(unittest.TestCase):
    def test_idle_import_is_clean(self):
        self.assertFalse(hasattr(idle, "machine"))
        for name in ("IDLE_AFTER_S", "idle_state", "idle_brightness",
                     "tour_phase", "phase_rgb", "ticks_diff"):
            self.assertTrue(hasattr(idle, name), name)

    def test_main_import_has_no_side_effects(self):
        # Not on hardware, idle logic found, main loop NOT running.
        self.assertFalse(fw._ON_HARDWARE)
        self.assertIsNot(fw.idle, None)
        self.assertTrue(callable(fw.main))

    def test_fresh_import_spawns_nothing_and_prints_nothing(self):
        before = threading.active_count()
        code = (
            "import sys; sys.path.insert(0, %r);"
            "import importlib.util;"
            "spec = importlib.util.spec_from_file_location('x_idle', %r);"
            "m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m);"
            "spec2 = importlib.util.spec_from_file_location('x_main', %r);"
            "m2 = importlib.util.module_from_spec(spec2); spec2.loader.exec_module(m2);"
            "assert m2.idle is not None and m2._ON_HARDWARE is False;"
            "print('IMPORT-OK')"
        ) % (ROOT, IDLE_PATH, MAIN_PATH)
        proc = subprocess.run([sys.executable, "-c", code],
                              capture_output=True, text=True, timeout=60)
        self.assertEqual(proc.returncode, 0, proc.stderr[-2000:])
        self.assertEqual(proc.stdout, "IMPORT-OK\n")
        self.assertEqual(proc.stderr, "")
        self.assertEqual(threading.active_count(), before)


if __name__ == "__main__":
    unittest.main()
