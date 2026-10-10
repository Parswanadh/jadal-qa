#!/usr/bin/env python3
"""Offline verification of a Jadal receipt JSON (GitHub issue #35).

Checks, all offline unless ``--rerun`` is given:

1. **Hash check (always):** recompute ``receipt_sha256`` with
   ``jadal.receipts.receipt_hash`` and compare it with the stored
   ``receipt_sha256`` and ``receipt_id`` (``receipt_id`` = first 16 hex
   chars). On mismatch the changed field paths are named via
   ``jadal.receipts.verify_receipt`` against a re-derived copy.
2. **MAC check (only when ``--mac``/``--mac-file`` is given AND the key
   exists):** verify the HMAC-SHA256 sidecar from ``jadal.receipt_mac``
   (key from ``JADAL_RECEIPT_KEY`` or an explicit argument — never a file
   in the repo). The MAC is a sidecar, never a field inside the receipt.
3. **Clip check (only with ``--clip``):** the clip file's SHA-256 must equal
   the receipt's ``segment_sha256``.
4. **Re-run (only with ``--clip PATH --rerun``):** re-run the real pipeline
   via ``jadal.pipeline.reverify_receipt`` and report MATCH/MISMATCH with
   the differing field paths.

Exit codes: 0 only if every requested check passes; 1 on any hash / MAC /
clip / re-run mismatch; 2 on usage errors (bad arguments, unreadable or
malformed receipt, MAC requested but no key); 3 when a re-run was requested
but cannot run (real pipeline modules unavailable) — the tool fails clearly
instead of faking a result.

A hash check proves the receipt was NOT EDITED; it does NOT prove the
verdict was right. Re-verifying a VERDICT needs the clip, the pinned
weights and the pinned stack (``--clip PATH --rerun``), and even then it
only proves the receipt reproduces on this stack — not ground truth.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

# Allow `python scripts/verify_receipt.py` from the repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


HELP_EPILOG = (
    "What this proves (read carefully): a hash check proves the receipt "
    "file was not edited after it was written. It does NOT prove the "
    "verdict was right. Re-verifying a VERDICT needs the clip, the pinned "
    "weights and the pinned stack (--clip PATH --rerun), and even a MATCH "
    "only shows the receipt reproduces on this stack — not ground truth."
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify a Jadal receipt JSON (offline hash/tamper check, "
                    "optional HMAC sidecar, optional pipeline re-run).",
        epilog=HELP_EPILOG,
    )
    parser.add_argument("receipt", help="path to the RECEIPT.json file")
    mac = parser.add_mutually_exclusive_group()
    mac.add_argument("--mac", default=None,
                     help="expected HMAC-SHA256 hex sidecar value")
    mac.add_argument("--mac-file", default=None, metavar="F",
                     help="file holding the expected HMAC hex (sidecar file)")
    parser.add_argument("--clip", default=None, metavar="PATH",
                        help="original clip file (checks segment_sha256; "
                             "required for --rerun)")
    parser.add_argument("--rerun", action="store_true",
                        help="re-run the real pipeline on --clip and compare "
                             "(needs the pinned weights/stack)")
    parser.add_argument("--json", action="store_true",
                        help="emit machine-readable JSON instead of text")
    return parser


HASHED_EXCLUDED = frozenset({"receipt_sha256", "receipt_id", "created_utc"})


def _diff_bodies(old, new, prefix: str = "") -> list[str]:
    """Dotted field paths where two stripped bodies differ (sorted).

    Local differ with the same path conventions as
    ``jadal.receipts.verify_receipt`` (``a.b``, ``a[0].b``); used to compare
    the stored file body against the fresh pipeline output captured during
    ``--rerun``, which ``reverify_receipt`` itself does not return.
    """
    from collections.abc import Mapping
    paths: list[str] = []
    if isinstance(old, Mapping) and isinstance(new, Mapping):
        for key in sorted(set(old.keys()) | set(new.keys())):
            path = f"{prefix}.{key}" if prefix else str(key)
            if key not in old or key not in new:
                paths.append(path)
            else:
                paths.extend(_diff_bodies(old[key], new[key], path))
    elif isinstance(old, list) and isinstance(new, list) and len(old) == len(new):
        for i, (left, right) in enumerate(zip(old, new)):
            paths.extend(_diff_bodies(left, right, f"{prefix}[{i}]"))
    elif old != new:
        paths.append(prefix)
    return paths


def _strip(receipt: dict) -> dict:
    return {k: v for k, v in receipt.items() if k not in HASHED_EXCLUDED}


def _err(message: str) -> None:
    print(f"verify_receipt: error: {message}", file=sys.stderr)


def _file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_receipt(path: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except OSError as exc:
        raise _UsageError(f"cannot read receipt file {path!r}: {exc}") from exc
    except ValueError as exc:
        raise _UsageError(f"receipt file {path!r} is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise _UsageError(f"receipt file {path!r} must hold a JSON object")
    return payload


class _UsageError(Exception):
    """Exit-2 problem: bad invocation or unreadable/malformed input."""


class _CannotRerun(Exception):
    """Exit-3 problem: a re-run was requested but the pipeline is unavailable."""


def _rederive(receipt: dict, receipts_mod) -> dict:
    """Rebuild a receipt from its stored components (for diff naming)."""
    segment = {"sha256": receipt["segment_sha256"]}
    return receipts_mod.make_receipt(
        segment,
        receipt["signals"],
        receipt["decisions"],
        receipt["verdict"],
        receipt["config"],
    )


def check_hash(receipt: dict, receipts_mod) -> dict:
    """Offline hash + id check; names envelope paths on mismatch."""
    actual = receipts_mod.receipt_hash(receipt)
    stored_hash = receipt.get("receipt_sha256")
    stored_id = receipt.get("receipt_id")
    hash_ok = stored_hash == actual
    id_ok = stored_id is None or stored_id == actual[:16]
    diff: list[str] = []
    if not (hash_ok and id_ok):
        try:
            verdict = receipts_mod.verify_receipt(
                receipt, lambda stored: _rederive(stored, receipts_mod))
            diff = list(verdict.get("diff", []))
        except Exception:
            diff = []
        if not diff:
            if stored_hash != actual:
                diff.append("receipt_sha256")
            if "receipt_id" in receipt and stored_id != actual[:16]:
                diff.append("receipt_id")
        diff = sorted(set(diff))
    return {
        "checked": True,
        "ok": bool(hash_ok and id_ok),
        "hash_ok": bool(hash_ok),
        "id_ok": bool(id_ok),
        "expected": stored_hash,
        "actual": actual,
        "stored_id": stored_id,
        "diff": diff,
    }


def check_mac(receipt: dict, mac_hex: str | None) -> dict:
    """MAC sidecar check (only when a MAC value was supplied)."""
    from jadal import receipt_mac as mac_mod
    if mac_hex is None:
        return {"checked": False, "ok": None, "reason": "not requested"}
    key = mac_mod.load_key()
    if not key:
        raise _UsageError(
            f"MAC requested but no key is set: export {mac_mod.MAC_KEY_ENV} "
            "(refusing to report OK for a check that cannot run)")
    stored_hash = receipt.get("receipt_sha256")
    if not isinstance(stored_hash, str):
        return {"checked": True, "ok": False,
                "reason": "receipt has no receipt_sha256 to authenticate"}
    ok = mac_mod.verify_mac(stored_hash, mac_hex)
    return {"checked": True, "ok": bool(ok),
            "reason": "match" if ok else "MISMATCH"}


def check_clip(receipt: dict, clip_path: str) -> dict:
    """Clip bytes check: the file must hash to the stored segment_sha256."""
    try:
        actual = _file_sha256(clip_path)
    except OSError as exc:
        raise _UsageError(f"cannot read clip file {clip_path!r}: {exc}") from exc
    expected = receipt.get("segment_sha256")
    ok = expected == actual
    return {"checked": True, "ok": bool(ok), "expected": expected,
            "actual": actual}


def check_rerun(receipt: dict, clip_path: str) -> dict:
    """Re-run the REAL pipeline; fail clearly instead of faking (exit 3).

    Returns reverify's MATCH/MISMATCH plus ``field_diff``: the stored file
    body vs the fresh pipeline output. The extra diff matters because
    ``reverify_receipt`` compares the fresh hash with the *stored* hash, so
    an edited file whose stored hash was left intact still reports MATCH —
    while ``field_diff`` names the edited fields.
    """
    try:
        from jadal import pipeline as pipeline_mod
        from jadal import receipts as receipts_mod
        from jadal.config import Config
    except ImportError as exc:
        raise _CannotRerun(
            "cannot re-run: jadal.pipeline / jadal.config are not importable "
            f"in this tree ({exc}); refusing to fake a MATCH") from exc
    verdict = receipt.get("verdict") if isinstance(receipt.get("verdict"), dict) else {}
    segment = {
        "segment_id": verdict.get("segment_id", ""),
        "source_clip": clip_path,
        "path": clip_path,
    }
    # Capture the fresh receipt reverify builds internally (it is not part
    # of its return value). Read-only interception, always restored; if a
    # future pipeline stops using verify_receipt, field_diff is just empty.
    captured: dict = {}
    orig_verify = receipts_mod.verify_receipt

    def _spy(stored, recompute_fn):
        try:
            fresh = recompute_fn(stored)
        except Exception:
            return orig_verify(stored, recompute_fn)
        captured["fresh"] = fresh
        return orig_verify(stored, lambda _s: fresh)

    receipts_mod.verify_receipt = _spy  # type: ignore[assignment]
    try:
        result = pipeline_mod.reverify_receipt(receipt, segment, Config())
    except RuntimeError as exc:
        # e.g. "no L0 signals implementation ... JADAL_USE_STUBS is not '1'"
        raise _CannotRerun(f"cannot re-run: {exc}") from exc
    except (OSError, ValueError) as exc:
        raise _CannotRerun(f"re-run failed: {exc}") from exc
    finally:
        receipts_mod.verify_receipt = orig_verify  # type: ignore[assignment]
    if not isinstance(result, dict) or "match" not in result:
        raise _CannotRerun("re-run returned an unusable result; refusing to guess")
    field_diff: list[str] = []
    if isinstance(captured.get("fresh"), dict):
        try:
            field_diff = sorted(_diff_bodies(
                _strip(receipt), _strip(captured["fresh"])))
        except Exception:
            field_diff = []
    return {
        "checked": True,
        "ok": bool(result.get("match")) and not field_diff,
        "match": bool(result.get("match")),
        "expected": result.get("expected"),
        "actual": result.get("actual"),
        "diff": list(result.get("diff") or []),
        "field_diff": field_diff,
        "version_match": result.get("version_match"),
        "version_diff": list(result.get("version_diff") or []),
    }


def run(args: argparse.Namespace) -> tuple[dict, int]:
    """Run every requested check. Returns (report, exit_code)."""
    if args.rerun and not args.clip:
        raise _UsageError("--rerun requires --clip PATH")
    try:
        from jadal import receipts as receipts_mod
    except ImportError as exc:
        raise _UsageError(f"cannot import jadal.receipts: {exc}") from None

    receipt = _load_receipt(args.receipt)

    mac_hex = args.mac
    if args.mac_file is not None:
        try:
            with open(args.mac_file, "r", encoding="utf-8") as handle:
                mac_hex = handle.read().strip()
        except OSError as exc:
            raise _UsageError(
                f"cannot read mac file {args.mac_file!r}: {exc}") from exc
        if not mac_hex:
            raise _UsageError(f"mac file {args.mac_file!r} is empty")

    hash_rep = check_hash(receipt, receipts_mod)
    mac_rep = check_mac(receipt, mac_hex)
    clip_rep = (check_clip(receipt, args.clip) if args.clip
                else {"checked": False, "ok": None, "reason": "not requested"})
    rerun_rep = (check_rerun(receipt, args.clip) if args.rerun
                 else {"checked": False, "ok": None, "reason": "not requested"})

    ok = bool(hash_rep["ok"])
    if mac_rep["checked"]:
        ok = ok and bool(mac_rep["ok"])
    if clip_rep["checked"]:
        ok = ok and bool(clip_rep["ok"])
    if rerun_rep["checked"]:
        ok = ok and bool(rerun_rep["ok"])

    report = {
        "receipt": args.receipt,
        "hash": hash_rep,
        "mac": mac_rep,
        "clip": clip_rep,
        "rerun": rerun_rep,
        "ok": ok,
    }
    return report, (0 if ok else 1)


def render_text(report: dict) -> str:
    lines = [f"receipt: {report['receipt']}"]
    h = report["hash"]
    if h["ok"]:
        lines.append(f"hash: OK ({h['actual'][:16]}...)")
        lines.append(f"id: OK ({h['actual'][:16]})")
    else:
        if not h["hash_ok"]:
            lines.append(f"hash: MISMATCH stored={h['expected']} recomputed={h['actual']}")
        if not h["id_ok"]:
            lines.append(f"id: MISMATCH stored={h['stored_id']} "
                         f"expected={h['actual'][:16]} (first 16 of recomputed hash)")
        if h["diff"]:
            lines.append(f"changed fields: {', '.join(h['diff'])}")
            if h["diff"] == ["receipt_sha256"] or "receipt_sha256" in h["diff"]:
                lines.append("note: content hash differs but the original pipeline "
                             "output is unavailable offline, so only the envelope "
                             "path can be named; re-run with --clip PATH --rerun "
                             "to name the exact changed fields.")
    m = report["mac"]
    if not m["checked"]:
        lines.append("mac: not requested")
    else:
        lines.append(f"mac: {'OK' if m['ok'] else 'MISMATCH'} ({m['reason']})")
    c = report["clip"]
    if not c["checked"]:
        lines.append("clip: not requested")
    elif c["ok"]:
        lines.append(f"clip: OK (sha256={c['actual'][:16]}...)")
    else:
        lines.append(f"clip: MISMATCH stored segment_sha256={c['expected']} "
                     f"file sha256={c['actual']}")
    r = report["rerun"]
    if not r["checked"]:
        lines.append("rerun: not requested")
    else:
        lines.append(f"rerun: {'MATCH' if r['match'] else 'MISMATCH'} "
                     f"expected={r['expected']} actual={r['actual']}")
        if r["diff"]:
            lines.append(f"rerun diff: {', '.join(r['diff'])}")
        if r["field_diff"]:
            lines.append("stored file differs from fresh pipeline output at: "
                         f"{', '.join(r['field_diff'])}")
            if r["match"]:
                lines.append("note: the stored hash still reproduces, so the "
                             "pipeline output is unchanged — the listed "
                             "fields were edited in the stored file after it "
                             "was written.")
        elif not r["match"] and r["diff"] == ["receipt_sha256"]:
            lines.append("note: stored content matches the fresh pipeline "
                         "output; only the stored hash envelope differs.")
        if r.get("version_match") is False:
            lines.append(f"rerun note: running stack differs from stored versions "
                         f"at: {', '.join(r['version_diff'])}")
    lines.append(f"RESULT: {'PASS' if report['ok'] else 'FAIL'}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        report, code = run(args)
    except _UsageError as exc:
        _err(str(exc))
        return 2
    except _CannotRerun as exc:
        _err(str(exc))
        return 3
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(render_text(report))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
