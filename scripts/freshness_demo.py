"""Freshness demo: live / replay / trials / synthetic.

  live       issue nonce, flash Pico, capture webcam, verify, print result.
             Prints the nonce_hash prefix only — never the nonce value.
  replay     verify a saved clip against a NEWLY issued nonce (-> stale).
  trials     N live + N replay trials; CIs; eval/results/freshness_trials.json.
             Requires a human to hold the Pico LED 30-50 cm in front of the
             webcam (see docs/reports/freshness.md). NOT MEASURED yet (N=0).
  synthetic  no hardware: null-tail calibration + decoder check per SNR level
             (>=5000 seeded wrong nonces x full lag search per level);
             writes eval/results/freshness_synthetic.json.

Note on nonce_hash: it is a SHA-256 COMMITMENT (binding), not a hiding one —
the 32-bit preimage is enumerable, so secrecy holds only inside the nonce
validity window. Clip files are named by nonce_hash[:16], never the value.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from jadal import freshness as F  # noqa: E402

RESKEYS = ("status", "lag_ms", "z", "bits_matched", "bits_total", "reason",
           "fps_measured", "fps_source")


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
    print(f"captured {rec['path']} @ {rec['fps']} fps")
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
            live_rows.append({"clip": Path(rec["path"]).name,
                              "nonce_hash": res["nonce_hash"],
                              **{k: res[k] for k in RESKEYS}})
            clips.append(rec["path"])
            print(f"live {i+1}/{args.n}: {res['status']} z={res['z']} "
                  f"bits={res['bits_matched']}/32 reason={res['reason']}")
        for i, clip in enumerate(clips[:args.n]):
            fresh_nonce = F.issue_nonce()  # replay: old clip vs new nonce
            res = F.verify_clip(clip, fresh_nonce)
            hit = res["status"] == "stale"
            replay_ok += hit
            replay_rows.append({"clip": Path(clip).name,
                                "nonce_hash": res["nonce_hash"],
                                **{k: res[k] for k in RESKEYS}})
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


LEVELS = [("A", 190.0, 5.0), ("B", 90.0, 15.0), ("C", 40.0, 20.0), ("D", 20.0, 20.0)]
NULL_PER_CLIP = 5000


def cmd_synthetic(args) -> int:
    from jadal.certify import exact_binomial_upper
    out = ROOT / "eval" / "results"
    out.mkdir(parents=True, exist_ok=True)
    thresh = F.DEFAULT_Z_THRESH
    level_rows, clip_rows, pooled_z = [], [], []
    for li, (name, swing, sigma) in enumerate(LEVELS):
        lvl_z, fa_n, fa_count = [], 0, 0
        true_z_min, true_bits_min = 1e9, 99
        for ci in range(args.per_clip):
            nonce = F.issue_nonce(rng=random.Random(100 + 10 * li + ci))
            frames, ts, _ = F.synthesize_frames(
                nonce, seed=ci, noise_sigma=sigma, on=30.0 + swing, off=30.0)
            roi, _ = F.find_led_roi(frames)
            M, starts = F.means_matrix(F.roi_series(frames, roi), ts)
            exp = np.array(F.bits_of(nonce), dtype=float)
            bc, br = F.maxcorr_over_lags(M, 2.0 * exp - 1.0)
            rng = np.random.RandomState(9000 + 100 * li + ci)
            P = rng.randint(0, 2, size=(NULL_PER_CLIP, F.BITS_TOTAL)).astype(float)
            maxs = np.empty(NULL_PER_CLIP)
            argb = np.empty(NULL_PER_CLIP, dtype=int)
            for k in range(NULL_PER_CLIP):
                c, b = F.maxcorr_over_lags(M, 2.0 * P[k] - 1.0)
                maxs[k], argb[k] = c, b
            mu, sd = float(maxs.mean()), float(maxs.std())
            znull = (maxs - mu) / sd
            lvl_z.append(znull)
            pooled_z.append(znull)
            # Full-rule false accepts: z>=thresh AND exact 32/32 at own best lag.
            for k in np.nonzero(znull >= thresh)[0]:
                _, bm = F.decode_bits(M[argb[k]], P[k])
                fa_count += int(bm == F.BITS_TOTAL)
            fa_n += NULL_PER_CLIP
            z_true = (bc - mu) / sd
            _, bm_true = F.decode_bits(M[br], exp)
            true_z_min = min(true_z_min, z_true)
            true_bits_min = min(true_bits_min, bm_true)
            clip_rows.append({"level": name, "clip": ci, "snr": round(swing / sigma, 2),
                               "true_z": round(float(z_true), 4), "true_bits": int(bm_true),
                               "null_mean": round(mu, 4), "null_sd": round(sd, 4)})
            print(f"L{name} clip{ci}: true z={z_true:.2f} bits={bm_true}/32 "
                  f"null p99={np.percentile(znull, 99):.2f}", flush=True)
        lvl_z = np.concatenate(lvl_z)
        fa_rate = fa_count / fa_n
        level_rows.append({"level": name, "swing": swing, "sigma": sigma,
                            "snr": round(swing / sigma, 2), "n_clips": args.per_clip,
                            "n_null": int(fa_n),
                            "null_z_p99": round(float(np.percentile(lvl_z, 99)), 4),
                            "null_z_p999": round(float(np.percentile(lvl_z, 99.9)), 4),
                            "null_z_max": round(float(lvl_z.max()), 4),
                            "fa_count_full_rule": int(fa_count), "fa_n": int(fa_n),
                            "fa_rate": round(float(fa_rate), 5),
                            "fa_upper_cp": round(exact_binomial_upper(fa_count, fa_n), 5),
                            "true_z_min": round(float(true_z_min), 4),
                            "true_bits_min": int(true_bits_min)})
    pooled = np.concatenate(pooled_z)
    payload = {"threshold": thresh,
               "threshold_provenance": (
                   "pooled null of %d seeded wrong nonces (%d SNR levels x "
                   "%d clips x %d, full lag search each); pooled p99=%.3f; "
                   "threshold %.1f set above it" % (
                       pooled.size, len(LEVELS), args.per_clip, NULL_PER_CLIP,
                       float(np.percentile(pooled, 99)), thresh)),
               "decision_rule": "fresh iff bits_matched==32 (exact, primary) "
                                "and z>=threshold (supporting)",
               "delta": 0.1, "levels": level_rows,
               "pooled": {"n_null": int(pooled.size),
                          "p99": round(float(np.percentile(pooled, 99)), 4),
                          "p999": round(float(np.percentile(pooled, 99.9)), 4),
                          "max": round(float(pooled.max()), 4)},
               "clips": clip_rows,
               "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    (out / "freshness_synthetic.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps({"threshold": thresh, "pooled_p99": payload["pooled"]["p99"],
                      "levels": [(r["level"], r["fa_count_full_rule"],
                                   r["fa_n"], r["fa_upper_cp"]) for r in level_rows]},
                     indent=2))
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
    p.add_argument("--per-clip", type=int, default=2)
    p.set_defaults(fn=cmd_synthetic)
    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
