"""Host side of the Jadal QA Pico link (USB serial only).

Implements the serial protocol from ``docs/SPEC.md`` (115200 baud,
newline-terminated ASCII):

    Pico -> host every 1.0 s :  ``HB <ticks_ms> <led_state>``
    Host -> Pico            :  ``LED G|A|R|OFF``
    Host -> Pico            :  ``FLASH <8 hex digits>``  (32 bits, one bit per 100 ms, MSB first)
    Host -> Pico            :  ``PING``      -> Pico answers ``PONG <ticks_ms>``

The :class:`PicoLink` class opens a real pyserial port (auto-detected from
``/dev/ttyACM*``) OR an injected fake serial device so the whole host stack is
testable without hardware. A background daemon thread reads frames, tracks the
heartbeat, and correlates ``PING``/``PONG`` pairs for :meth:`PicoLink.estimate_offset`.

NOT DONE:
  * No measurement of true end-to-end sync accuracy. ``estimate_offset`` returns
    an *estimate* with a conservative error bound (half the measured round-trip);
    we never claim anything better than one camera frame.
  * Real-hardware behaviour (port quirks, framing under load) is not exercised in
    CI - only the :class:`FakePicoSerial` emulator is.
  * ``pyserial`` is imported lazily so this module imports on machines without it.
"""
from __future__ import annotations

import glob
import queue
import threading
import time
from statistics import median
from typing import Callable, List, Optional, Tuple

__all__ = [
    "BAUD",
    "VALID_LED",
    "PicoLink",
    "PicoLinkError",
    "FakePico",
    "FakePicoSerial",
    "parse_hb",
    "parse_pong",
    "encode_led",
    "encode_flash",
    "offset_sample",
]

BAUD = 115200
VALID_LED = ("G", "A", "R", "OFF")
_HEX_DIGITS = "0123456789abcdef"


class PicoLinkError(RuntimeError):
    """Raised for bad commands, connection failures, or missing pyserial."""


# --------------------------------------------------------------------------- #
# Pure, easily testable protocol helpers                                      #
# --------------------------------------------------------------------------- #
def _to_text(line) -> str:
    if isinstance(line, (bytes, bytearray)):
        return bytes(line).decode("ascii", "ignore")
    return str(line)


def parse_hb(line) -> Optional[Tuple[int, str]]:
    """Parse a heartbeat frame ``HB <ticks_ms> <led_state>``.

    Returns ``(ticks_ms, led_state)`` or ``None`` if it is not a valid HB line.
    """
    text = _to_text(line).strip()
    if not text.startswith("HB "):
        return None
    parts = text.split()
    if len(parts) != 3:
        return None
    try:
        ticks = int(parts[1])
    except ValueError:
        return None
    state = parts[2].upper()
    if not state:
        return None
    return ticks, state


def parse_pong(line) -> Optional[int]:
    """Parse a ``PONG <ticks_ms>`` frame, returning the ticks or ``None``."""
    text = _to_text(line).strip()
    if not text.startswith("PONG "):
        return None
    parts = text.split()
    if len(parts) != 2:
        return None
    try:
        return int(parts[1])
    except ValueError:
        return None


def encode_led(color: str) -> bytes:
    """Encode a ``LED <color>`` command. Case-insensitive. Raises on bad colour."""
    text = str(color).strip().upper()
    if text not in VALID_LED:
        raise PicoLinkError(f"invalid LED colour {color!r}; expected one of {VALID_LED}")
    return b"LED %s\n" % text.encode("ascii")


def encode_flash(pattern) -> bytes:
    """Encode a ``FLASH <8 hex>`` command from a 32-bit int or an 8-hex string."""
    if isinstance(pattern, str):
        s = pattern.strip().lower()
        if s.startswith("0x"):
            s = s[2:]
        if len(s) != 8 or any(c not in _HEX_DIGITS for c in s):
            raise PicoLinkError("FLASH pattern must be exactly 8 hex digits")
        value = int(s, 16)
    else:
        try:
            value = int(pattern)
        except (TypeError, ValueError):
            raise PicoLinkError(f"FLASH pattern not an int or hex string: {pattern!r}")
    if not (0 <= value < 2 ** 32):
        raise PicoLinkError("FLASH pattern must be a 32-bit value")
    return b"FLASH %08X\n" % value


def offset_sample(t0_s: float, t1_ticks_ms: float, t3_s: float) -> Tuple[float, float]:
    """NTP-style offset/RTT from one PING/PONG pair.

    All timestamps are in seconds (host clock) here so caller/test doubles are exact.

    * ``t0_s``  : host send time of PING.
    * ``t3_s``  : host receive time of PONG.
    * ``t1_ticks_ms`` : the tick counter (in ms) the device stamped in the PONG.

    This is the NTP symmetric-delay model with the device's reply treated as
    instantaneous (t1 == t2). Returns ``(rtt_ms, offset_ms)`` where

        rtt_ms   = (t3 - t0) * 1000
        offset_ms = t1_ticks_ms - (t0 + t3) / 2 * 1000

    With no artificial latency and identical clocks, ``offset_ms`` is ~0.

    The *pico clock* is assumed to be the same virtual clock as ``t0/t3`` (host
    monotonic expressed in ms) plus a constant offset; if the real device's epoch
    differs, ``offset_ms`` recovers that constant epoch difference instead. Because
    the device reply is stamped once (t1 == t2), the *interpretation* of
    ``offset_ms`` also absorbs any forward-path delay asymmetry.
    """
    rtt_ms = (t3_s - t0_s) * 1000.0
    offset_ms = t1_ticks_ms - ((t0_s + t3_s) / 2.0) * 1000.0
    return rtt_ms, offset_ms


# --------------------------------------------------------------------------- #
# Fake device (no hardware)                                                   #
# --------------------------------------------------------------------------- #
class FakePico:
    """In-process Pico emulator: computes HB/PONG ticks, applies LED/FLASH."""

    def __init__(self, epoch_offset_ms: float = 1000.0, led: str = "OFF"):
        self._offset = float(epoch_offset_ms)
        self._led = led
        self.flashed: List[str] = []
        self.led_commands: List[str] = []

    def now_ms(self, host_time_s: Optional[float] = None) -> int:
        base = time.monotonic() if host_time_s is None else host_time_s
        return round(base * 1000.0 + self._offset)

    def set_led(self, color: str) -> None:
        c = color.strip().upper()
        if c in VALID_LED:
            self._led = c

    def do_flash(self, hex_pattern: str) -> None:
        self.flashed.append(hex_pattern.strip().lower())

    def hb_line(self, host_time_s: Optional[float] = None) -> bytes:
        return b"HB %d %s\n" % (self.now_ms(host_time_s), self._led.encode("ascii"))

    def pong_line(self, host_time_s: Optional[float] = None) -> bytes:
        return b"PONG %d\n" % self.now_ms(host_time_s)


class FakePicoSerial:
    """A pyserial-like byte transport wrapping a :class:`FakePico`.

    ``latency_s`` is a *symmetric* one-way delay: a PING written at host time T
    is answered with a PONG stamped at ``T + latency_s`` and delivered to the
    host at ``T + 2*latency_s``. Set ``latency_s = 0`` for an instantaneous
    (in-memory) device used by the determinising tests.

    The transport mimics the pySerial surface PicoLink relies on: ``write``,
    ``in_waiting``, ``read`` and ``close`` - so no real hardware is needed.
    """

    def __init__(
        self,
        epoch_offset_ms: float = 1000.0,
        latency_s: float = 0.0,
        baud: int = BAUD,
        port: str = "FAKE",
        pico: Optional[FakePico] = None,
    ):
        self.baudrate = baud
        self.port = port
        self.is_open = True
        self.pico = pico or FakePico(epoch_offset_ms=epoch_offset_ms)
        self.latency_s = latency_s
        self._ready = b""            # bytes delivered and readable now
        self._pending = []           # list of (due_monotonic_s, bytes)
        self._out_accum = b""        # every byte the host has written
        # The real Pico emits an HB about once per second on connect; emit one
        # immediately so a freshly-connected host sees a heartbeat quickly.
        self._pending.append((time.monotonic(), self.pico.hb_line()))

    # -- pyserial-like transport ------------------------------------------- #
    def write(self, data) -> int:
        self._out_accum += bytes(data)
        now = time.monotonic()
        for chunk in bytes(data).split(b"\n"):
            text = chunk.strip()
            if not text:
                continue
            s = text.decode("ascii", "ignore")
            if s == "PING":
                proc = now + self.latency_s
                due = now + 2 * self.latency_s
                self._pending.append((due, self.pico.pong_line(proc)))
            elif s.startswith("LED"):
                parts = s.split()
                if len(parts) == 2 and parts[1].upper() in VALID_LED:
                    self.pico.led_commands.append(s)
                    self.pico.set_led(parts[1])
            elif s.startswith("FLASH"):
                self.pico.led_commands.append(s)
                self.pico.do_flash(s[6:])
        return len(data)

    def _pump(self) -> None:
        now = time.monotonic()
        still = []
        for due, payload in self._pending:
            if due <= now:
                self._ready += payload
            else:
                still.append((due, payload))
        self._pending = still

    @property
    def in_waiting(self) -> int:
        self._pump()
        return len(self._ready)

    def read(self, n: int = 1) -> bytes:
        self._pump()
        if not self._ready:
            return b""
        out, self._ready = self._ready[:n], self._ready[n:]
        return out

    def flush(self) -> None:
        pass

    def close(self) -> None:
        self.is_open = False
        self._ready = b""
        self._pending = []


def _open_real_serial(port: str, baud: int):
    """Open a real pyserial port (pyserial is imported lazily)."""
    try:
        import serial  # pyserial
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise PicoLinkError(
            "pyserial is required to talk to real hardware; "
            "use FakePicoSerial for tests"
        ) from exc
    return serial.Serial(port=port, baudrate=baud, timeout=0)


def _default_candidates() -> List[str]:
    return sorted(glob.glob("/dev/ttyACM*")) + sorted(glob.glob("/dev/ttyUSB*"))


# --------------------------------------------------------------------------- #
# Host link                                                                    #
# --------------------------------------------------------------------------- #
class PicoLink:
    """Host controller for a Jadal Pico status LED.

    Two ways to obtain one:

    * Real hardware - ``link.connect()`` auto-detects a ``/dev/ttyACM*`` port
      (pyserial is opened lazily and required only for this path).
    * Tests/fake - ``PicoLink(serial=FakePicoSerial(...))`` then ``connect()``.
    """

    def __init__(
        self,
        serial=None,
        *,
        clock: Callable[[], float] = time.monotonic,
        open_serial: Callable[[str, int], object] = _open_real_serial,
        read_poll_s: float = 0.02,
    ):
        self._clock = clock
        self._open_serial = open_serial
        self._read_poll_s = read_poll_s
        self.ser = serial
        self.pong_q: "queue.Queue[Tuple[int, float]]" = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._lock = threading.Lock()
        self._buf = b""
        self._connected = False
        self._port: Optional[str] = None
        self._last_hb_time: Optional[float] = None
        self._last_hb_ticks: Optional[int] = None
        self._last_hb_led: Optional[str] = None

    # -- connection -------------------------------------------------------- #
    def connect(
        self,
        port: Optional[str] = None,
        *,
        probe_timeout: float = 1.0,
        candidates: Optional[List[str]] = None,
    ) -> "PicoLink":
        if self._running:
            return self
        if self.ser is not None:                       # injected fake-serial mode
            used_port = getattr(self.ser, "port", "FAKE")
        elif port is not None:
            self.ser = self._open_serial(port, BAUD)
            used_port = port
        else:
            self.ser, used_port = self._autodetect(probe_timeout, candidates)
        self._reset_state()
        self._port = used_port
        self._connected = True
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def _reset_state(self) -> None:
        self._buf = b""
        self._connected = False
        self._last_hb_time = None
        self._last_hb_ticks = None
        self._last_hb_led = None
        while not self.pong_q.empty():
            try:
                self.pong_q.get_nowait()
            except queue.Empty:
                break

    def _autodetect(self, probe_timeout: float, candidates: Optional[List[str]]):
        cands = candidates if candidates is not None else _default_candidates()
        for candidate in cands:
            try:
                ser = self._open_serial(candidate, BAUD)
            except Exception:
                continue
            if self._probe(ser, probe_timeout):
                return ser, candidate
            try:
                ser.close()
            except Exception:
                pass
        raise PicoLinkError(f"no Pico found on any of: {cands}")

    def _probe(self, ser, timeout: float) -> bool:
        end = time.monotonic() + timeout
        buf = b""
        try:
            ser.write(b"PING\n")
            while time.monotonic() < end:
                waiting = getattr(ser, "in_waiting", 0)
                if waiting:
                    buf += ser.read(waiting)
                    while b"\n" in buf:
                        line, buf = buf.split(b"\n", 1)
                        if parse_pong(line) is not None:
                            return True
                else:
                    time.sleep(0.01)
        except Exception:
            return False
        return False

    def _run(self) -> None:
        while self._running:
            ser = self.ser
            if ser is None:
                break
            try:
                waiting = getattr(ser, "in_waiting", 0)
                if waiting:
                    self._buf += ser.read(waiting)
                    while b"\n" in self._buf:
                        line, self._buf = self._buf.split(b"\n", 1)
                        self._process_line(line)
                else:
                    time.sleep(self._read_poll_s)
            except Exception:
                self._connected = False
                break

    def _process_line(self, line: bytes) -> None:
        hb = parse_hb(line)
        if hb is not None:
            ticks, state = hb
            with self._lock:
                self._last_hb_time = self._clock()
                self._last_hb_ticks = ticks
                self._last_hb_led = state
                self._connected = True
            return
        pong_ticks = parse_pong(line)
        if pong_ticks is not None:
            with self._lock:
                self._connected = True
            self.pong_q.put((pong_ticks, self._clock()))

    # -- status ------------------------------------------------------------ #
    @property
    def port(self) -> Optional[str]:
        return self._port

    @property
    def connected(self) -> bool:
        with self._lock:
            return self._connected and self._running

    def status(self) -> dict:
        with self._lock:
            last = self._last_hb_time
            age = None if last is None else max(0.0, self._clock() - last)
            return {
                "connected": self._connected and self._running,
                "port": self._port,
                "last_hb_age_s": age,
                "last_hb_ticks_ms": self._last_hb_ticks,
                "last_hb_led": self._last_hb_led,
            }

    # -- commands ---------------------------------------------------------- #
    def _require_up(self) -> None:
        if getattr(self.ser, "is_open", True) is False:
            raise PicoLinkError("link is closed")
        if not self._running:
            raise PicoLinkError("not connected")

    def set_led(self, color: str) -> None:
        payload = encode_led(color)          # validate before touching the wire
        self._require_up()
        self.ser.write(payload)

    def flash(self, pattern) -> None:
        payload = encode_flash(pattern)       # validate before touching the wire
        self._require_up()
        self.ser.write(payload)

    def ping(self) -> None:
        self._require_up()
        self.ser.write(b"PING\n")

    # -- clock offset estimation ------------------------------------------ #
    def estimate_offset(self, samples: int = 8, timeout: float = 2.0) -> dict:
        """Estimate round-trip time and the Pico<->host clock offset.

        Sends ``samples`` PINGs, correlates each with the next PONG, and uses the
        minimum-RTT sample (least sensitivity to asymmetric delay) as the best
        offset estimate. Returns a dict with per-sample detail plus a conservative
        ``sync_error_bound_ms (= min_rtt / 2)``.

        This is only an *estimate*: do not trust it to better than one frame.
        """
        self._drain_pongs()
        recs: List[Tuple[float, float]] = []
        for _ in range(max(1, samples)):
            t0 = self._clock()
            self.ping()
            try:
                ticks, t3 = self.pong_q.get(timeout=timeout)
            except queue.Empty:
                break
            rtt_ms, offset_ms = offset_sample(t0, ticks, t3)
            recs.append((rtt_ms, offset_ms))
            time.sleep(0.005)
        if not recs:
            raise PicoLinkError("no PONG received during estimate_offset")
        rtts = sorted(r for r, _ in recs)
        offsets = [o for _, o in recs]
        best_rtt, best_offset = min(recs, key=lambda r: r[0])
        return {
            "n": len(recs),
            "rtt_ms_min": rtts[0],
            "rtt_ms_median": median(rtts),
            "rtt_ms_max": rtts[-1],
            "offset_ms": best_offset,          # from min-RTT sample
            "offset_ms_median": median(offsets),
            "sync_error_bound_ms": best_rtt / 2.0,
            "samples": [{"rtt_ms": r, "offset_ms": o} for r, o in recs],
        }

    def _drain_pongs(self) -> None:
        while not self.pong_q.empty():
            try:
                self.pong_q.get_nowait()
            except queue.Empty:
                break

    # -- lifecycle --------------------------------------------------------- #
    def disconnect(self) -> None:
        self._running = False
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=1.0)
        self._thread = None
        if self.ser is not None:
            try:
                self.ser.close()
            except Exception:
                pass
        with self._lock:
            self._connected = False
            self._last_hb_time = None
            self._last_hb_ticks = None
            self._last_hb_led = None

    def __enter__(self) -> "PicoLink":
        return self

    def __exit__(self, *exc) -> None:
        self.disconnect()
