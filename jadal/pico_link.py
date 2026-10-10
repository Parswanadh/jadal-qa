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

Connection semantics: ``connected`` is derived from heartbeat freshness — it is
True only while the reader is running *and* the last heartbeat is younger than
``HB_STALE_S``. A silent/hung device therefore reads as disconnected. The
explicit port and baud defaults come from the read-only shared
``jadal.config.DEFAULT`` (so ``JADAL_PICO_PORT`` is honoured); this module never
edits that file. All open failures surface as :class:`PicoLinkError`.

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

from jadal.config import DEFAULT as _CONFIG_DEFAULT  # read-only shared defaults

__all__ = [
    "BAUD",
    "HB_PERIOD_S",
    "HB_STALE_S",
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

BAUD = _CONFIG_DEFAULT.pico_baud   # protocol baud; single source is jadal.config
HB_PERIOD_S = 1.0                  # firmware emits HB this often (docs/SPEC.md)
HB_STALE_S = 2.5 * HB_PERIOD_S     # no HB within this -> link reads disconnected
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
    """RTT/offset from one PING/PONG pair (NTP symmetric-delay model).

    ``t0_s`` is the host send time of PING, ``t3_s`` the host receive time of
    PONG, ``t1_ticks_ms`` the tick counter (in ms) the device stamped in the
    PONG, treating the device reply as instantaneous. Returns
    ``(rtt_ms, offset_ms)`` with ``offset_ms = t1 - mid(t0, t3)`` in ms.

    Caveats: the device tick counter wraps (MicroPython ``ticks_ms`` is ~2**30
    ms) and the device may reboot, either of which shifts ``offset_ms`` by an
    arbitrary amount — only compare offsets taken from one unbroken run. The
    reader thread also stamps ``t3`` up to one ``read_poll_s`` after the byte
    arrived, so ``rtt_ms`` is inflated by at most that poll interval (this keeps
    the error bound conservative but biases ``offset_ms`` by about half of it).
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
        hb_period_s: float = 1.0,
    ):
        self.baudrate = baud
        self.port = port
        self.is_open = True
        self.pico = pico or FakePico(epoch_offset_ms=epoch_offset_ms)
        self.latency_s = latency_s
        self.hb_period_s = hb_period_s  # <= 0 disables periodic heartbeats
        self._ready = b""            # bytes delivered and readable now
        self._pending = []           # list of (due_monotonic_s, bytes)
        self._out_accum = b""        # every byte the host has written
        # The real Pico emits an HB about once per second; emit one immediately
        # so a freshly-connected host sees a heartbeat quickly, then keep the
        # 1 s rhythm while the transport is open.
        now = time.monotonic()
        self._last_hb_emit = now
        self._pending.append((now, self.pico.hb_line(now)))

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
        if self.hb_period_s > 0:
            # keep the device's ~1 s heartbeat rhythm; if the reader falls far
            # behind, skip missed beats instead of queueing a backlog.
            while (
                self._last_hb_emit + self.hb_period_s <= now
                and len(self._pending) < 8
            ):
                self._last_hb_emit += self.hb_period_s
                emit_at = self._last_hb_emit
                self._pending.append((emit_at, self.pico.hb_line(emit_at)))
            if self._last_hb_emit + self.hb_period_s <= now:
                self._last_hb_emit = now
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


def _default_candidates(include_usb: bool = False) -> List[str]:
    # SPEC names /dev/ttyACM* only; ttyUSB adapters are opt-in (they may be
    # unrelated USB-UART dongles we should not PING by default).
    cands = sorted(glob.glob("/dev/ttyACM*"))
    if include_usb:
        cands += sorted(glob.glob("/dev/ttyUSB*"))
    return cands


# --------------------------------------------------------------------------- #
# Host link                                                                    #
# --------------------------------------------------------------------------- #
class PicoLink:
    """Host controller for a Jadal Pico status LED.

    Two ways to obtain one:

    * Real hardware - ``link.connect()`` auto-detects a ``/dev/ttyACM*`` port
      (pyserial is opened lazily and required only for this path), falling back
      to the configured ``jadal.config.DEFAULT.pico_port`` when the scan finds
      nothing (so ``JADAL_PICO_PORT`` is honoured).
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
        self._injected = serial is not None
        self.pong_q: "queue.Queue[Tuple[int, float]]" = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._lock = threading.Lock()    # guards _running/_port/_last_hb_*
        self._wlock = threading.Lock()   # serialises concurrent ser.write calls
        self._buf = b""
        self._port: Optional[str] = None
        self._last_hb_time: Optional[float] = None
        self._last_hb_ticks: Optional[int] = None
        self._last_hb_led: Optional[str] = None

    # -- connection -------------------------------------------------------- #
    def _open_wrapped(self, port: str, baud: int):
        # I-3: every open failure (SerialException/OSError/...) surfaces as
        # PicoLinkError, never a raw serial error.
        try:
            return self._open_serial(port, baud)
        except PicoLinkError:
            raise
        except Exception as exc:
            raise PicoLinkError(f"could not open serial port {port!r}: {exc}") from exc

    def connect(
        self,
        port: Optional[str] = None,
        *,
        probe_timeout: float = 1.0,
        candidates: Optional[List[str]] = None,
        include_usb: bool = False,
    ) -> "PicoLink":
        with self._lock:
            if self._running:
                return self
        if (
            self.ser is not None
            and self._injected
            and getattr(self.ser, "is_open", True)
        ):
            used_port = getattr(self.ser, "port", "FAKE")
        elif port is not None:
            self.ser = self._open_wrapped(port, _CONFIG_DEFAULT.pico_baud)
            self._injected = False
            used_port = port
        else:
            try:
                self.ser, used_port = self._autodetect(
                    probe_timeout, candidates, include_usb
                )
            except PicoLinkError:
                if candidates is not None:
                    raise  # explicit scan list: respect it strictly
                # Last resort: the configured default port, so JADAL_PICO_PORT
                # (read from jadal.config at import) is honoured.
                default_port = _CONFIG_DEFAULT.pico_port
                self.ser = self._open_wrapped(
                    default_port, _CONFIG_DEFAULT.pico_baud
                )
                used_port = default_port
            self._injected = False
        self._reset_state()
        with self._lock:
            self._port = used_port
            self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def _reset_state(self) -> None:
        with self._lock:
            self._buf = b""
            self._last_hb_time = None
            self._last_hb_ticks = None
            self._last_hb_led = None
        while not self.pong_q.empty():
            try:
                self.pong_q.get_nowait()
            except queue.Empty:
                break

    def _autodetect(
        self,
        probe_timeout: float,
        candidates: Optional[List[str]],
        include_usb: bool,
    ):
        cands = candidates if candidates is not None else _default_candidates(include_usb)
        for candidate in cands:
            try:
                ser = self._open_serial(candidate, _CONFIG_DEFAULT.pico_baud)
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
        while True:
            with self._lock:
                if not self._running:
                    break
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
                with self._lock:
                    self._running = False
                break

    def _process_line(self, line: bytes) -> None:
        hb = parse_hb(line)
        if hb is not None:
            ticks, state = hb
            with self._lock:
                self._last_hb_time = self._clock()
                self._last_hb_ticks = ticks
                self._last_hb_led = state
            return
        pong_ticks = parse_pong(line)
        if pong_ticks is not None:
            self.pong_q.put((pong_ticks, self._clock()))

    # -- status ------------------------------------------------------------ #
    @property
    def port(self) -> Optional[str]:
        with self._lock:
            return self._port

    @property
    def connected(self) -> bool:
        # I-1: derived from heartbeat freshness, never latched at connect().
        with self._lock:
            running = self._running
            last = self._last_hb_time
        if not running or last is None:
            return False
        return (self._clock() - last) < HB_STALE_S

    def status(self) -> dict:
        with self._lock:
            last = self._last_hb_time
            age = None if last is None else max(0.0, self._clock() - last)
            port = self._port
            ticks = self._last_hb_ticks
            led = self._last_hb_led
        return {
            "connected": self.connected,
            "port": port,
            "last_hb_age_s": age,
            "last_hb_ticks_ms": ticks,
            "last_hb_led": led,
        }

    # -- commands ---------------------------------------------------------- #
    def _require_up(self) -> None:
        with self._lock:
            running = self._running
        ser = self.ser
        if ser is None or getattr(ser, "is_open", True) is False or not running:
            raise PicoLinkError("not connected")

    def _write(self, payload: bytes) -> None:
        self._require_up()
        try:
            with self._wlock:
                self.ser.write(payload)
        except PicoLinkError:
            raise
        except Exception as exc:
            raise PicoLinkError(f"write failed: {exc}") from exc

    def set_led(self, color: str) -> None:
        payload = encode_led(color)          # validate before touching the wire
        self._write(payload)

    def flash(self, pattern) -> None:
        payload = encode_flash(pattern)       # validate before touching the wire
        self._write(payload)

    def ping(self) -> None:
        self._write(b"PING\n")

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
        with self._lock:
            self._running = False
            self._port = None
            self._last_hb_time = None
            self._last_hb_ticks = None
            self._last_hb_led = None
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=1.0)
        self._thread = None
        # I-2: drop a factory-opened handle so a later connect(port=...) really
        # opens the new port. An injected test double is kept (tests own it).
        ser = self.ser
        if not self._injected:
            self.ser = None
        if ser is not None:
            try:
                ser.close()
            except Exception:
                pass

    def __enter__(self) -> "PicoLink":
        return self

    def __exit__(self, *exc) -> None:
        self.disconnect()
