"""Pico attract/idle-mode animation: pure, hardware-free logic.

When no host frame has arrived for ``IDLE_AFTER_S`` the firmware plays a
visible ambient animation (the point is that people walking past SEE the Pico
is alive with no laptop attached). ANY host frame stops it immediately and it
restarts only after another ``IDLE_AFTER_S`` of host silence. The host
protocol itself (HB/LED/FLASH/PING, see ``pico/main.py``) is UNCHANGED and
always wins.

Animation cycle (repeats forever while idle), all constants tunable below:

    1. PWM "breathing" (triangle 0..1 over ``BREATHE_PERIOD_MS``) x ``BREATHE_CYCLES``
    2. Verdict tour of the three patterns used by the project:
         ACCEPT solid 1 s, REVIEW slow blink (400 ms half-period) 2 s,
         REJECT fast blink (100 ms half-period) 1.5 s
    3. Dark gap 0.5 s, then repeat.

With an external RGB LED (``RGB_PINS`` in ``main.py``): green = ACCEPT,
amber (red+green) = REVIEW, red = REJECT, and a slow colour fade while
breathing. The onboard LED is single-colour, so there the tour shows as
solid/slow-blink/fast-blink.

This module is deliberately pure (no ``machine``, no ``time`` ticks calls, no
function annotations, no f-strings) so it imports on stock MicroPython AND on
CPython, where ``tests/test_pico_idle.py`` exercises every helper, including
``ticks_ms`` wraparound (~2**30 ms, same modulus ``time.ticks_*`` uses).
"""
# ---- tunables ------------------------------------------------------------ #
IDLE_AFTER_S = 20              # host silence before idle mode may (re)start
BREATHE_PERIOD_MS = 3000       # one PWM breathing cycle
BREATHE_CYCLES = 3             # breathing cycles before the verdict tour
ACCEPT_MS = 1000               # ACCEPT verdict: solid on
REVIEW_HALF_MS = 400           # REVIEW verdict: slow blink half-period ...
REVIEW_MS = 2000               # ... for this long
REJECT_HALF_MS = 100           # REJECT verdict: fast blink half-period ...
REJECT_MS = 1500               # ... for this long
GAP_MS = 500                   # dark gap before the cycle repeats

# ---- derived schedule (phase start offsets inside one cycle) -------------- #
BREATHE_TOTAL_MS = BREATHE_PERIOD_MS * BREATHE_CYCLES
ACCEPT_START_MS = BREATHE_TOTAL_MS
REVIEW_START_MS = ACCEPT_START_MS + ACCEPT_MS
REJECT_START_MS = REVIEW_START_MS + REVIEW_MS
GAP_START_MS = REJECT_START_MS + REJECT_MS
IDLE_CYCLE_MS = GAP_START_MS + GAP_MS

# MicroPython ticks_ms wraps at 2**30 on the RP2040 port.
TICKS_WRAP = 1 << 30
TICKS_HALF = 1 << 29


def ticks_diff(new_ms, old_ms):
    """Signed (new - old), wraparound-safe like ``time.ticks_diff``.

    Valid while the true gap is under half a wrap (~2**29 ms); the idle
    timeout (20 s) is far inside that.
    """
    d = (new_ms - old_ms) % TICKS_WRAP
    if d >= TICKS_HALF:
        d = d - TICKS_WRAP
    return d


def idle_state(now_ms, last_host_ms):
    """True when the idle animation may run at ``now_ms``.

    ``last_host_ms`` is the ticks_ms of the last received host frame (ANY
    frame, even unknown/malformed, counts). ``None`` means no host frame has
    ever arrived (fresh power-up): idle starts immediately so the board shows
    life with no host attached.
    """
    if last_host_ms is None:
        return True
    return ticks_diff(now_ms, last_host_ms) >= IDLE_AFTER_S * 1000


def idle_brightness(t_ms):
    """Breathing curve at ``t_ms``: triangle 0..1, period BREATHE_PERIOD_MS."""
    p = (t_ms % BREATHE_PERIOD_MS) / BREATHE_PERIOD_MS
    if p < 0.5:
        return p * 2.0
    return 2.0 - p * 2.0


def tour_phase(t_ms):
    """Map idle-elapsed ``t_ms`` to ``(phase_name, level)``.

    ``phase_name`` is one of BREATHE/ACCEPT/REVIEW/REJECT/GAP; ``level`` is
    0..1 (float during BREATHE, 1/0 for the blink phases). Periodic over
    ``IDLE_CYCLE_MS``.
    """
    t = t_ms % IDLE_CYCLE_MS
    if t < ACCEPT_START_MS:
        return ("BREATHE", idle_brightness(t))
    t = t - ACCEPT_START_MS
    if t < ACCEPT_MS:
        return ("ACCEPT", 1)
    t = t - ACCEPT_MS
    if t < REVIEW_MS:
        if (t // REVIEW_HALF_MS) % 2 == 0:
            return ("REVIEW", 1)
        return ("REVIEW", 0)
    t = t - REVIEW_MS
    if t < REJECT_MS:
        if (t // REJECT_HALF_MS) % 2 == 0:
            return ("REJECT", 1)
        return ("REJECT", 0)
    return ("GAP", 0)


def phase_rgb(name, t_ms):
    """Saturated (r, g, b) colour for a tour phase, floats 0..1.

    ACCEPT = green, REVIEW = amber (red+green), REJECT = red, GAP = dark;
    BREATHE = slow colour fade cycling red->green->blue->red over the three
    breaths. The caller scales by the phase level for actual PWM duties.
    """
    if name == "ACCEPT":
        return (0.0, 1.0, 0.0)
    if name == "REVIEW":
        return (1.0, 1.0, 0.0)
    if name == "REJECT":
        return (1.0, 0.0, 0.0)
    if name == "BREATHE":
        # Cycle-relative: tour_phase is periodic over IDLE_CYCLE_MS, so the
        # fade must restart with the cycle (elapsed=14000 is BREATHE offset 0,
        # red) rather than running on elapsed % BREATHE_TOTAL_MS (which
        # desyncs the colour from cycle 1 on).
        t = (t_ms % IDLE_CYCLE_MS) % BREATHE_TOTAL_MS
        seg = t // BREATHE_PERIOD_MS
        f = (t % BREATHE_PERIOD_MS) / BREATHE_PERIOD_MS
        if seg == 0:
            return (1.0 - f, f, 0.0)
        if seg == 1:
            return (0.0, 1.0 - f, f)
        return (f, 0.0, 1.0 - f)
    return (0.0, 0.0, 0.0)
