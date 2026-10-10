"""LEAD B / LB1 — verify the RECEIPT and DETERMINISM claims (load-bearing for the submission).

FACTS.md claims:
  * "Receipts are content-hashed (SHA-256 of canonical JSON), not signed."
  * "Receipts: SHA-256 of canonical JSON; re-verify = re-run the pinned pipeline on the original
     clip and compare; tampered receipt fails; not cryptographically signed (optional HMAC sidecar)."
  * "Determinism: CPU feature path gives an identical quantised digest over 10 repeated runs."

These are the trust story of the product, so they get an independent check here rather than a
re-read of the code.

Checks
  1. hash stability      - the same body hashes identically across repeated calls AND across processes
  2. tamper detection    - changing any single field changes receipt_sha256 / receipt_id
  3. canonical form      - key order and float rounding do not change the hash
  4. not signed          - no signature field inside the receipt; an HMAC sidecar is separate
  5. feature determinism - 10 repeated feature computations on one window give an identical
                           quantised digest (the claim's exact wording)
"""
import hashlib
import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)

from jadal import receipts as R                      # noqa: E402
from eval.gpu.decode import decode_window            # noqa: E402
from jadal.gpu.features import cpu_window_features   # noqa: E402
from jadal.config import DEFAULT as CFG_DEFAULT      # noqa: E402

BUNDLE = os.environ.get("JADAL_BUNDLE", "/workspace/jadal_pod_bundle")
OUT = os.path.join(ROOT, "eval", "results", "pod_verify", "receipts_determinism.json")
W, H = 160, 90



def pod_path(p):
    """The bundle manifest was restored to its pristine (source-machine) copy at 17:19Z;
    its paths do not resolve on the pod. Remap in memory only - the file is READ-ONLY."""
    if os.path.exists(p):
        return p
    for a, b in (("/home/parshu/projects/awing/data/", "/workspace/jadal_pod_bundle/data/"),
                 ("/home/parshu/projects/jadal-qa-wt/spectral_data/", "/workspace/jadal_pod_bundle/labels/"),
                 ("/home/parshu/projects/jadal-qa-wt/", "/workspace/jadal_pod_bundle/")):
        if p.startswith(a) and os.path.exists(b + p[len(a):]):
            return b + p[len(a):]
    return p

def luma_of(rgb):
    return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]


def main():
    res = {"experiment": "LB1-receipts-determinism", "verified_by": "lead-b",
           "claims": [
               "receipts are SHA-256 of canonical JSON, not signed",
               "a tampered receipt fails to verify",
               "the CPU feature path gives an identical quantised digest over 10 repeated runs",
           ]}

    seg = {"sha256": "a" * 64, "start_s": 0.0, "end_s": 10.0, "clip": "desk"}
    signals = {"motion_energy": 0.123456789, "luma_mean": 0.5, "temp_b1": 0.25}
    decisions = {"rule_dark": False, "threshold": 0.3}
    verdict = {"verdict": "accept", "score": 0.8123456789}

    cfg = CFG_DEFAULT
    r1 = R.make_receipt(seg, signals, decisions, verdict, cfg, created_utc="2026-10-10T00:00:00+00:00")
    r2 = R.make_receipt(seg, signals, decisions, verdict, cfg, created_utc="2030-01-01T00:00:00+00:00")
    res["hash_stable_across_calls"] = r1["receipt_sha256"] == r2["receipt_sha256"]
    res["timestamp_excluded"] = r1["receipt_sha256"] == r2["receipt_sha256"]
    res["receipt_sha256"] = r1["receipt_sha256"]
    res["receipt_id_is_prefix16"] = r1["receipt_id"] == r1["receipt_sha256"][:16]
    res["canonical_hash_matches_manual"] = (
        r1["receipt_sha256"] == hashlib.sha256(R.canonical_json(
            {k: r1[k] for k in ("segment_sha256", "signals", "decisions", "verdict", "config")}
        )).hexdigest())

    # 2. tamper detection on every top-level body field
    tampered = {}
    for key in ("segment_sha256", "signals", "decisions", "verdict", "config"):
        t = json.loads(json.dumps(r1))
        if isinstance(t[key], dict):
            k0 = sorted(t[key])[0]
            t[key][k0] = "TAMPERED" if not isinstance(t[key][k0], (int, float)) else t[key][k0] + 1.0
        else:
            t[key] = "b" * 64
        tampered[key] = R.receipt_hash(t) != r1["receipt_sha256"]
    res["tamper_detected_per_field"] = tampered
    res["tamper_all_detected"] = all(tampered.values())

    # 3. canonical form: key insertion order must not matter
    # NOTE: compare the ROUNDED bodies (make_receipt rounds floats via _round_floats);
    # comparing raw inputs against rounded ones would differ for a reason unrelated to key order.
    body_orig = {k: r1[k] for k in ("segment_sha256", "signals", "decisions", "verdict", "config")}
    body_flipped = dict(reversed(list(body_orig.items())))
    res["key_order_irrelevant"] = (hashlib.sha256(R.canonical_json(body_orig)).hexdigest()
                                   == hashlib.sha256(R.canonical_json(body_flipped)).hexdigest())
    res["canonical_sort_keys"] = ("sort_keys" in (R.canonical_json.__doc__ or ""))

    # 4. not signed
    # exact-ish match; "signals" is a DATA field and must not trip a substring test on "sig"
    sig_like = [k for k in r1 if k.lower() not in ("signals",)
                and any(t in k.lower() for t in ("signature", "hmac", "mac", "signed", "sig_"))]
    res["signature_fields_in_receipt"] = sig_like
    res["is_not_signed"] = len(sig_like) == 0

    # 5. feature determinism: 10 repeated runs -> identical quantised digest
    items = json.load(open(os.path.join(BUNDLE, "labels", "manifest_gold.json")))["items"]
    it = items[0]
    rgb, fps = decode_window(pod_path(it["path"]), it.get("start_s", 0.0),
                             round(it["end_s"] - it["start_s"], 2), W, H, "rgb24")
    luma = luma_of(rgb)
    digests, raw = [], []
    for _ in range(10):
        f = cpu_window_features(luma, rgb, fps=min(fps, 30.0))
        raw.append(f)
        q = {k: round(float(v), 9) for k, v in f.items()}
        digests.append(hashlib.sha256(R.canonical_json(q)).hexdigest())
    res["feature_runs"] = len(digests)
    res["feature_digest_unique"] = len(set(digests))
    res["feature_digest"] = digests[0]
    res["feature_identical_digest_10_runs"] = len(set(digests)) == 1
    res["feature_max_abs_spread"] = float(max(
        max(abs(raw[i][k] - raw[0][k]) for k in raw[0]) for i in range(1, 10)))

    # cross-process stability of the same digest
    code = (
        "import sys,json,hashlib;sys.path.insert(0,%r);"
        "from jadal import receipts as R;"
        "print(R.canonical_json(%r).hex()[:0] or hashlib.sha256(R.canonical_json(%r)).hexdigest())"
        % (ROOT, {k: round(float(v), 9) for k, v in raw[0].items()},
           {k: round(float(v), 9) for k, v in raw[0].items()})
    )
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    res["cross_process_digest_match"] = (p.stdout.strip() == digests[0]) if p.returncode == 0 else None
    res["cross_process_stderr"] = p.stderr[-200:] if p.returncode != 0 else ""

    res["conclusion"] = {
        "receipts_content_hashed": bool(res["hash_stable_across_calls"] and res["tamper_all_detected"]),
        "receipts_not_signed": bool(res["is_not_signed"]),
        "feature_path_deterministic_10_runs": bool(res["feature_identical_digest_10_runs"]),
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k not in ("claims",)}, indent=1)[:1800])
    print("wrote", OUT)


if __name__ == "__main__":
    main()
