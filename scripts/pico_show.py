"""Pico verdict light show: CLI wrapper over :class:`jadal.pico_link.PicoLink`.

Subcommands (all print what they are doing)::

    verdict accept|review|reject   flash the verdict pattern once
    demo                           play every pattern (verdicts + SOS, heartbeat, countdown)
    status                         connect, wait for heartbeat, print the link status
    pattern HEX8                   flash one raw 8-hex-digit pattern

Verdict patterns (FLASH is 32 bits, 100 ms/bit, MSB first; see docs/SPEC.md):

    accept  solid on               FFFFFFFF
    review  slow blink (400 ms)    F0F0F0F0
    reject  fast blink (100 ms)    AAAAAAAA

The board's onboard LED is a single colour: accept/review/reject are told
apart by blink *pattern*, not colour.

Every subcommand waits for the first heartbeat after connect (up to 2.5 s):
the firmware sends HB every 1 s and PicoLink's autodetect probe swallows the
first one, so ``status()`` read immediately after connect says
``connected False``. The wait is announced on stdout.

``--fake`` uses :class:`jadal.pico_link.FakePicoSerial` so tests and a
no-hardware demo work without opening ``/dev/ttyACM0``. Every path sends
``LED OFF`` and disconnects in a ``finally`` block. When no Pico answers the
script exits 2 with a clear message; ``verdict``/``demo``/``pattern`` exit 3
with a warning when no heartbeat is seen (``status`` still exits 0 — the
disconnected status is its answer). ``--port`` is rejected together with
``--fake``, and ``--pause`` must be >= 0.

Run from the repo root, e.g.::

    PYTHONNOUSERSITE=1 /home/parshu/envs/jadal/bin/python scripts/pico_show.py --fake demo
    PYTHONNOUSERSITE=1 /home/parshu/envs/jadal/bin/python scripts/pico_show.py --port /dev/ttyACM0 status

NOT DONE:
  * No camera-in-the-loop check that the flashed pattern is actually visible.
  * Real-hardware timing (exact 100 ms/bit) is firmware-side and unmeasured here.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jadal.pico_link import FakePicoSerial, PicoLink, PicoLinkError

__all__ = [
    "ACCEPT_HEX",
    "REVIEW_HEX",
    "REJECT_HEX",
    "DEMO_PATTERNS",
    "pattern_for",
    "hex_to_bits",
    "wait_for_heartbeat",
    "cleanup",
    "build_parser",
    "main",
]

ACCEPT_HEX = "FFFFFFFF"  # solid on: 32 x 1
REVIEW_HEX = "F0F0F0F0"  # slow blink: 11110000 x 4
REJECT_HEX = "AAAAAAAA"  # fast blink: 10 x 16

_VERDICT_HEX = {
    "accept": ACCEPT_HEX,
    "review": REVIEW_HEX,
    "reject": REJECT_HEX,
}


def _bits_to_hex(bits: str) -> str:
    """Pack a bit string (MSB first) into 8 hex digits, zero-padded to 32 bits."""
    bits = (bits + "0" * 32)[:32]
    return "%08X" % int(bits, 2)


def hex_to_bits(hex8: str) -> str:
    """Expand 8 hex digits to the 32-bit MSB-first bit string the firmware plays.

    Raises :class:`ValueError` on anything that is not valid hex.
    """
    return format(int(hex8, 16), "032b")


def pattern_for(verdict: str) -> str:
    """Return the FLASH hex pattern for a verdict (pure; reusable by the service).

    Raises :class:`ValueError` for anything but accept|review|reject.
    """
    key = str(verdict).strip().lower()
    try:
        return _VERDICT_HEX[key]
    except KeyError:
        raise ValueError(
            f"unknown verdict {verdict!r}; expected one of accept|review|reject"
        ) from None


DEMO_PATTERNS: list[tuple[str, str]] = [
    ("ACCEPT   - solid on", ACCEPT_HEX),
    ("REVIEW   - slow blink (400 ms)", REVIEW_HEX),
    ("REJECT   - fast blink (100 ms)", REJECT_HEX),
    (
        "SOS      - ... --- ...",
        _bits_to_hex("101010" + "000" + "111011101110" + "000" + "101010"),
    ),
    ("HEARTBEAT- double pulse", _bits_to_hex(("11011" + "0" * 11) * 2)),
    (
        "COUNTDOWN- 1,2,3,4 pulses",
        _bits_to_hex(
            "1" + "000" + "1010" + "000" + "101010" + "000" + "10101010" + "0000"
        ),
    ),
]

HB_WAIT_S = 2.5  # firmware HB period is 1 s; wait for the first post-probe beat
FLASH_PLAY_S = 3.5  # 32 bits x 100 ms = 3.2 s, plus margin


def wait_for_heartbeat(link: PicoLink, timeout: float = HB_WAIT_S) -> bool:
    """Wait up to ``timeout`` s for the first heartbeat; announce the wait.

    Returns True once ``link.connected`` (heartbeat fresh), else False.
    """
    print(
        "waiting for first heartbeat (up to %.1f s; the autodetect probe "
        "swallows the first HB)..." % timeout,
        flush=True,
    )
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if link.connected:
            print("heartbeat seen: link connected.", flush=True)
            return True
        time.sleep(0.05)
    print("no heartbeat yet after %.1f s; continuing anyway." % timeout, flush=True)
    return False


def cleanup(link: PicoLink) -> None:
    """Send LED OFF, then disconnect. Never raises ( teardown path)."""
    try:
        link.set_led("OFF")
        print("LED OFF sent.", flush=True)
    except Exception as exc:  # noqa: BLE001 - teardown must not raise
        print(f"LED OFF failed (ignored): {exc}", flush=True)
    try:
        link.disconnect()
    except Exception as exc:  # noqa: BLE001 - teardown must not raise
        print(f"disconnect failed (ignored): {exc}", flush=True)


def _connect(link: PicoLink, port: str | None) -> PicoLink:
    try:
        link.connect(port=port) if port else link.connect()
    except PicoLinkError as exc:
        print(f"error: no Pico answers ({exc})", file=sys.stderr)
        raise SystemExit(2) from exc
    print(f"connected on {link.port}", flush=True)
    return link


def _cmd_verdict(link: PicoLink, verdict: str, pause_s: float) -> int:
    hexpat = pattern_for(verdict)
    hb_ok = wait_for_heartbeat(link)
    print(f"verdict {verdict}: FLASH {hexpat} (32 bits, 100 ms/bit, MSB first)",
          flush=True)
    try:
        link.flash(hexpat)
        print("pattern sent; playing...", flush=True)
        time.sleep(pause_s)
    finally:
        cleanup(link)
    if not hb_ok:
        print("warning: no heartbeat was seen; the board may not have "
              "played the pattern.", file=sys.stderr)
        return 3
    print("done - LED off", flush=True)
    return 0


def _cmd_demo(link: PicoLink, pause_s: float) -> int:
    hb_ok = wait_for_heartbeat(link)
    try:
        for name, hexpat in DEMO_PATTERNS:
            print(f"{name:<34} FLASH {hexpat}", flush=True)
            link.flash(hexpat)
            time.sleep(pause_s)
        st = link.status()
        print(
            "status: "
            + str({k: st.get(k) for k in ("connected", "last_hb_age_s", "last_hb_led")}),
            flush=True,
        )
    finally:
        cleanup(link)
    if not hb_ok:
        print("warning: no heartbeat was seen; the board may not have "
              "played the patterns.", file=sys.stderr)
        return 3
    print("done - LED off", flush=True)
    return 0


def _cmd_status(link: PicoLink) -> int:
    wait_for_heartbeat(link)
    try:
        print(f"status: {link.status()}", flush=True)
    finally:
        cleanup(link)
    return 0


def _cmd_pattern(link: PicoLink, hexpat: str, pause_s: float) -> int:
    hexpat = hexpat.strip()
    if hexpat.lower().startswith("0x"):
        hexpat = hexpat[2:]
    if len(hexpat) != 8 or any(
        c not in "0123456789abcdefABCDEF" for c in hexpat
    ):
        print(
            f"error: pattern must be exactly 8 hex digits, got {hexpat!r}",
            file=sys.stderr,
        )
        cleanup(link)
        return 2
    hexpat = hexpat.upper()
    hb_ok = wait_for_heartbeat(link)
    print(f"FLASH {hexpat} (32 bits, 100 ms/bit, MSB first)", flush=True)
    try:
        link.flash(hexpat)
        print("pattern sent; playing...", flush=True)
        time.sleep(pause_s)
    finally:
        cleanup(link)
    if not hb_ok:
        print("warning: no heartbeat was seen; the board may not have "
              "played the pattern.", file=sys.stderr)
        return 3
    print("done - LED off", flush=True)
    return 0


def _nonneg_float(value: str) -> float:
    """argparse type for --pause: reject negative sleep lengths with usage text."""
    try:
        result = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a number: {value!r}") from None
    if result < 0:
        raise argparse.ArgumentTypeError("--pause must be >= 0")
    return result


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Flash blink patterns on the Pico status LED via PicoLink."
    )
    ap.add_argument("--port", default=None,
                    help="serial port (default: autodetect /dev/ttyACM*).")
    ap.add_argument("--fake", action="store_true",
                    help="use the in-memory FakePicoSerial (no hardware).")
    ap.add_argument("--pause", type=_nonneg_float, default=None,
                    help="seconds to wait while a pattern plays "
                    f"(default {FLASH_PLAY_S}; 0.05 with --fake).")
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("verdict", help="flash the pattern for a verdict.")
    v.add_argument("verdict", choices=["accept", "review", "reject"])
    sub.add_parser("demo", help="play all patterns (verdicts + SOS/heartbeat/countdown).")
    sub.add_parser("status", help="connect, wait for heartbeat, print status.")
    p = sub.add_parser("pattern", help="flash one raw 8-hex-digit pattern.")
    p.add_argument("hex8", help="8 hex digits with optional 0x prefix, e.g. F0F0F0F0.")
    return ap


def _make_link(args: argparse.Namespace) -> PicoLink:
    if args.fake:
        return PicoLink(serial=FakePicoSerial())
    return PicoLink()


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.fake and args.port:
        parser.error("--port is ignored with --fake (fake link always uses FAKE).")
    pause_s = args.pause if args.pause is not None else (0.05 if args.fake else FLASH_PLAY_S)
    link = _make_link(args)
    _connect(link, args.port)
    if args.cmd == "verdict":
        return _cmd_verdict(link, args.verdict, pause_s)
    if args.cmd == "demo":
        return _cmd_demo(link, pause_s)
    if args.cmd == "status":
        return _cmd_status(link)
    if args.cmd == "pattern":
        return _cmd_pattern(link, args.hex8, pause_s)
    raise AssertionError(f"unreachable subcommand {args.cmd!r}")  # pragma: no cover


if __name__ == "__main__":
    sys.exit(main())
