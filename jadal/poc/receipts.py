"""Receipts for PoC windows: content-hashed, NOT signed.

receipt_sha256 = sha256 of canonical JSON of
  {window_id, verdict, reasons, features_digest, versions}
features_digest = sha256 of canonical JSON of the rounded feature dict.
Re-verify recomputes the window from the original clip and compares.
"""
import hashlib
import json


def canonical(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False).encode("utf-8")


def sha256_hex(obj) -> str:
    return hashlib.sha256(canonical(obj)).hexdigest()


def features_digest(features) -> str:
    return sha256_hex({k: round(float(v), 4) for k, v in features.items()})


def receipt_for(window_id, verdict, reasons, feats_digest, versions) -> str:
    return sha256_hex({"window_id": window_id, "verdict": verdict,
                       "reasons": reasons, "features_digest": feats_digest,
                       "versions": versions})
