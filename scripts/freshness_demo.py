"""Freshness demo: live / replay / trials / synthetic.

  live       issue nonce, flash Pico, capture webcam, verify, print result
  replay     verify a saved clip against a NEWLY issued nonce (-> stale)
  trials     N live + N replay trials; CIs; eval/results/freshness_trials.json
  synthetic  no hardware: decoder sweep over SNR; .../freshness_synthetic.json
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from jadal import freshness as F  # noqa: E402


def cmd_live(args) -> int:
    from jadal.pico_link import PicoLink
    nonce = F.issue_nonce()
    print(f"nonce issued (hash {F.nonce_hash(nonce)[:16]}...; value kept secret)")
    link = PicoLink().connect()
    try:
        rec = F.record_live(nonce, link, cam_index=args.cam,
                            seconds=args.seconds)
    finally:
        link.disconnect()
    print(f"captured {rec['path']}")
    res = F.verify_clip(rec["path"], nonce)
    print(json.dumps(res, indent=2))
    return 0


def cmd_replay(args) -> int:
    nonce = F.issue_nonce()  # NEW nonce: old clip must not match it
    print(f"new nonce hash {F.nonce_hash(nonce)[:16]}... vs clip {args.clip}")
    res = F.verify_clip(args.clip, nonce)
    print(json.dumps(res, indent=2))
    ok = res["status"] == "stale"
    print("REPLAY-REJECTED" if ok else f"UNEXPECTED: {res['status']}")
    return 0 if ok else 1


def cmd_trials(args) -> int:
    from jadal.certify import exact_binomial_upper
    from jadal.pico_link import PicoLink
    out = ROOT / "eval" / "results"
    out.mkdir(parents=True, exist_ok=True)
    live_ok, replay_ok = 0, 0
    live_rows, replay_rows = [], []
    link = PicoLink().connect()
    try:
        clips = []
        for i in range(args.n):
            nonce = F.issue_nonce()
            rec = F.record_live(nonce, link, cam_index=args.cam,
                                seconds=args.seconds)
            res = F.verify_clip(rec["path"], nonce)
            hit = res["status"] == "fresh"
            live_ok += hit
            live_rows.append({"clip": rec["path"], "nonce_hash": res["nonce_hash"],
                              **{k: res[k] for k in ("status", "lag_ms", "z",
                                                     "bits_matched", "bits_total")}})
            clips.append((rec["path"], nonce))
            print(f"live {i+1}/{args.n}: {res['status']} z={res['z']} "
                  f"bits={res['bits_matched']}/32")
        for i, (clip, old_nonce) in enumerate(clips[:args.n]):
            fresh_nonce = F.issue_nonce()  # replay: old clip vs new nonce
            res = F.verify_clip(clip, fresh_nonce)
            hit = res["status"] == "stale"
            replay_ok += hit
            replay_rows.append({"clip": clip, "nonce_hash": res["nonce_hash"],
                                **{k: res[k] for k in ("status", "lag_ms", "z",
                                                       "bits_matched", "bits_total")}})
            print(f"replay {i+1}/{args.n}: {res['status']} "
                  f"(want stale) z={res['z']} bits={res['bits_matched']}/32")
    finally:
        link.disconnect()
    n = args.n
    payload = {
        "n_live": n, "n_replay": n,
        "live_detected": live_ok, "detection_rate": round(live_ok / n, 4),
        "detection_miss_upper": round(exact_binomial_upper(n - live_ok, n), 4),
        "replay_rejected": replay_ok, "false_accept_rate": round((n - replay_ok) / n, 4),
        "false_accept_upper": round(exact_binomial_upper(n - replay_ok, n), 4),
        "delta": 0.1, "live": live_rows, "replay": replay_rows,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (out / "freshness_trials.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps({k: payload[k] for k in ("detection_rate", "detection_miss_upper",
                                              "false_accept_rate", "false_accept_upper")}, indent=2))
    return 0


def cmd_synthetic(args) -> int:
    out = ROOT / "eval" / "results"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for sigma in args.sigmas:
        for trial in range(args.per):
            nonce = F.issue_nonce(rng=random.Random(1000 + trial))
            frames, ts, start = F.synthesize_frames(
                nonce, seed=trial, noise_sigma=sigma)
            r_true = F.verify_clip(frames, nonce, {"timestamps": ts})
            other = F.issue_nonce(rng=random.Random(5000 + trial))
            r_wrong = F.verify_clip(frames, other, {"timestamps": ts})
            rows.append({"sigma": sigma, "trial": trial,
                         "true_status": r_true["status"], "true_z": r_true["z"],
                         "true_bits": r_true["bits_matched"],
                         "true_lag_ms": r_true["lag_ms"],
                         "wrong_status": r_wrong["status"], "wrong_z": r_wrong["z"],
                         "wrong_bits": r_wrong["bits_matched"]})
    det = sum(1 for r in rows if r["true_status"] == "fresh")
    rej = sum(1 for r in rows if r["wrong_status"] == "stale")
    payload = {"rows": rows, "detection_rate": round(det / len(rows), 4),
               "replay_rejection_rate": round(rej / len(rows), 4),
               "n": len(rows)}
    (out / "freshness_synthetic.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps({k: payload[k] for k in ("n", "detection_rate",
                                              "replay_rejection_rate")}, indent=2))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("live")
    p.add_argument("--cam", type=int, default=0)
    p.add_argument("--seconds", type=float, default=6.0)
    p.set_defaults(fn=cmd_live)
    p = sub.add_parser("replay")
    p.add_argument("--clip", required=True)
    p.set_defaults(fn=cmd_replay)
    p = sub.add_parser("trials")
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--cam", type=int, default=0)
    p.add_argument("--seconds", type=float, default=6.0)
    p.set_defaults(fn=cmd_trials)
    p = sub.add_parser("synthetic")
    p.add_argument("--sigmas", type=float, nargs="+", default=[2.0, 5.0, 10.0, 20.0])
    p.add_argument("--per", type=int, default=5)
    p.set_defaults(fn=cmd_synthetic)
    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
