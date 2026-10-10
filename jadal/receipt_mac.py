"""Optional HMAC-SHA256 sidecar for receipts (GitHub issue #35).

A receipt is content-hashed by ``jadal.receipts`` (anyone can recompute
``receipt_sha256`` offline). This module adds an *optional* authenticity layer:
``MAC = HMAC-SHA256(key, receipt_sha256_hex)``. The MAC is a SIDECAR — a
separate value (or ``.mac`` file) that is NEVER a field inside the hashed
receipt (adding it would change the hash it authenticates).

Key handling (no secrets in code, never in the repo):
- the key comes ONLY from the environment variable ``JADAL_RECEIPT_KEY``
  (UTF-8) or from an explicit ``key`` argument (``bytes`` or ``str``);
- this module never reads a key from any file in the repo.

NOT DONE:
- Key rotation / multi-key rings (single active key only).
- Any claim beyond "whoever holds the key saw this exact receipt hash":
  a valid MAC does not prove the verdict was right, only that the receipt
  bytes are the ones the key holder authenticated.

No network calls at import time (stdlib only).
"""
from __future__ import annotations

import hmac
import os

MAC_KEY_ENV = "JADAL_RECEIPT_KEY"
MAC_HEX_LEN = 64  # HMAC-SHA256 hex digest length


class MacKeyMissing(RuntimeError):
    """Raised when a MAC is requested but no key is available."""


def load_key(explicit: bytes | str | None = None) -> bytes | None:
    """Return the MAC key bytes, or ``None`` when no key is configured.

    Precedence: explicit argument first, then ``JADAL_RECEIPT_KEY``.
    Explicit keys are used verbatim (no stripping of the key bytes),
    except that an empty / whitespace-only explicit value counts as
    missing — exactly like an empty / whitespace-only env value.
    """
    if explicit is not None:
        if isinstance(explicit, str):
            if not explicit.strip():
                return None
            return explicit.encode("utf-8")
        key_bytes = bytes(explicit)
        try:
            text = key_bytes.decode("utf-8")
        except ValueError:
            return key_bytes  # non-text bytes are always a real key
        return key_bytes if text.strip() else None
    raw = os.environ.get(MAC_KEY_ENV)
    if raw is None or not raw.strip():
        return None
    return raw.encode("utf-8")


def require_key(explicit: bytes | str | None = None) -> bytes:
    """Return the MAC key bytes, raising :class:`MacKeyMissing` if absent."""
    key = load_key(explicit)
    if not key:
        raise MacKeyMissing(
            f"no receipt MAC key: pass an explicit key or set {MAC_KEY_ENV}"
        )
    return key


def _normalize_hash(receipt_sha256_hex: str) -> bytes:
    """Validate the receipt hash and return the exact bytes the MAC covers."""
    if not isinstance(receipt_sha256_hex, str):
        raise ValueError("receipt_sha256_hex must be a hex string")
    text = receipt_sha256_hex.strip().lower()
    if len(text) != MAC_HEX_LEN:
        raise ValueError(
            f"receipt_sha256_hex must be {MAC_HEX_LEN} hex chars, "
            f"got {len(receipt_sha256_hex.strip())}"
        )
    try:
        int(text, 16)
    except ValueError:
        raise ValueError("receipt_sha256_hex must be hexadecimal") from None
    # The MAC covers the lowercase hex digest's UTF-8 bytes (the form stored
    # in the receipt JSON), so verifiers never need to hex-decode anything.
    return text.encode("utf-8")


def mac_for(receipt_sha256_hex: str, key: bytes | str | None = None) -> str:
    """Compute the hex HMAC-SHA256 sidecar for a receipt hash.

    ``key`` defaults to the ``JADAL_RECEIPT_KEY`` env var. Raises
    :class:`MacKeyMissing` when no key is available, ``ValueError`` on a
    malformed receipt hash.
    """
    return hmac.new(require_key(key), _normalize_hash(receipt_sha256_hex),
                    "sha256").hexdigest()


def verify_mac(receipt_sha256_hex: str, mac_hex: str,
               key: bytes | str | None = None) -> bool:
    """Constant-time MAC check. ``False`` (never an exception) on mismatch.

    Raises :class:`MacKeyMissing` when no key is available and ``ValueError``
    on a malformed receipt hash; a malformed *mac* value is just ``False``.
    """
    expected = mac_for(receipt_sha256_hex, key)
    if not isinstance(mac_hex, str):
        return False
    candidate = mac_hex.strip().lower()
    if len(candidate) != MAC_HEX_LEN:
        return False
    try:
        int(candidate, 16)
    except ValueError:
        return False
    return hmac.compare_digest(expected, candidate)


__all__ = ["MAC_KEY_ENV", "MAC_HEX_LEN", "MacKeyMissing", "load_key",
           "require_key", "mac_for", "verify_mac"]
