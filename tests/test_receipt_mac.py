"""Tests for jadal/receipt_mac.py (HMAC-SHA256 receipt sidecar)."""
from __future__ import annotations

import os
import unittest
from unittest import mock

from jadal import receipt_mac as mac_mod

KEY = b"test-key-for-unit-tests-only"
HASH = "ab" * 32  # 64 hex chars


class KeyHandlingTest(unittest.TestCase):
    def setUp(self) -> None:
        self._old = os.environ.get(mac_mod.MAC_KEY_ENV)

    def tearDown(self) -> None:
        if self._old is None:
            os.environ.pop(mac_mod.MAC_KEY_ENV, None)
        else:
            os.environ[mac_mod.MAC_KEY_ENV] = self._old

    def test_missing_key_returns_none_and_require_raises(self) -> None:
        os.environ.pop(mac_mod.MAC_KEY_ENV, None)
        self.assertIsNone(mac_mod.load_key())
        with self.assertRaises(mac_mod.MacKeyMissing):
            mac_mod.require_key()
        with self.assertRaises(mac_mod.MacKeyMissing):
            mac_mod.mac_for(HASH)

    def test_empty_env_value_counts_as_missing(self) -> None:
        os.environ[mac_mod.MAC_KEY_ENV] = "   "
        self.assertIsNone(mac_mod.load_key())

    def test_key_comes_from_the_env_var_as_utf8(self) -> None:
        os.environ[mac_mod.MAC_KEY_ENV] = "s3cr3t-🔑"
        self.assertEqual(mac_mod.load_key(), "s3cr3t-🔑".encode("utf-8"))

    def test_explicit_key_wins_over_the_env(self) -> None:
        os.environ[mac_mod.MAC_KEY_ENV] = "env-key"
        self.assertEqual(mac_mod.load_key(explicit=b"explicit"), b"explicit")
        self.assertEqual(mac_mod.load_key(explicit="txt"), b"txt")
        self.assertEqual(mac_mod.mac_for(HASH, key=b"explicit"),
                         mac_mod.mac_for(HASH, key=b"explicit"))
        self.assertNotEqual(mac_mod.mac_for(HASH, key=b"explicit"),
                            mac_mod.mac_for(HASH, key=b"other"))

    def test_key_never_read_from_a_repo_file(self) -> None:
        # The module exposes no file-based key loading at all: the only
        # inputs are the explicit argument and the env var.
        os.environ.pop(mac_mod.MAC_KEY_ENV, None)
        self.assertIsNone(mac_mod.load_key())
        names = [n for n in dir(mac_mod) if not n.startswith("__")
                 and ("file" in n.lower() or "path" in n.lower())]
        self.assertEqual(names, [])


class MacTest(unittest.TestCase):
    def test_mac_is_deterministic_hex(self) -> None:
        first = mac_mod.mac_for(HASH, key=KEY)
        self.assertEqual(first, mac_mod.mac_for(HASH, key=KEY))
        self.assertEqual(len(first), 64)
        int(first, 16)

    def test_mac_matches_stdlib_hmac(self) -> None:
        import hashlib
        import hmac as std_hmac
        self.assertEqual(
            mac_mod.mac_for(HASH, key=KEY),
            std_hmac.new(KEY, HASH.encode("utf-8"), hashlib.sha256).hexdigest())

    def test_hash_hex_is_normalized_before_mac(self) -> None:
        self.assertEqual(mac_mod.mac_for(HASH.upper(), key=KEY),
                         mac_mod.mac_for(HASH, key=KEY))

    def test_malformed_receipt_hash_raises(self) -> None:
        for bad in ("xyz", "ab" * 31, "ab" * 33, "", "zz" * 32, 12345):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    mac_mod.mac_for(bad, key=KEY)  # type: ignore[arg-type]

    def test_verify_good_mac(self) -> None:
        tag = mac_mod.mac_for(HASH, key=KEY)
        self.assertTrue(mac_mod.verify_mac(HASH, tag, key=KEY))

    def test_verify_bad_mac_is_false_not_an_exception(self) -> None:
        tag = mac_mod.mac_for(HASH, key=KEY)
        flipped = ("0" if tag[-1] != "0" else "1")
        self.assertFalse(mac_mod.verify_mac(HASH, tag[:-1] + flipped, key=KEY))
        self.assertFalse(mac_mod.verify_mac("cd" * 32, tag, key=KEY))

    def test_verify_wrong_key_is_false(self) -> None:
        tag = mac_mod.mac_for(HASH, key=KEY)
        self.assertFalse(mac_mod.verify_mac(HASH, tag, key=b"wrong-key"))

    def test_verify_malformed_mac_is_false(self) -> None:
        for bad in ("", "xyz", "ab", "ab" * 33, None, 123):
            with self.subTest(bad=bad):
                self.assertFalse(mac_mod.verify_mac(HASH, bad, key=KEY))  # type: ignore[arg-type]

    def test_verify_missing_key_raises(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(mac_mod.MAC_KEY_ENV, None)
            with self.assertRaises(mac_mod.MacKeyMissing):
                mac_mod.verify_mac(HASH, "ab" * 32)

    def test_compare_is_constant_time(self) -> None:
        tag = mac_mod.mac_for(HASH, key=KEY)
        with mock.patch("hmac.compare_digest",
                        wraps=__import__("hmac").compare_digest) as spy:
            mac_mod.verify_mac(HASH, tag, key=KEY)
            self.assertTrue(spy.called)


if __name__ == "__main__":
    unittest.main()
