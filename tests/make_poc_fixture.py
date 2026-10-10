"""Generate a demo fixture that follows prompts/poc-contract.md exactly.

Writes into tests/fixtures/poc/: summary.json, windows.jsonl,
stream_events.jsonl, thumbs/<window_id>_{a,b,c}.jpg (tiny JPEGs).

Usage: python3 tests/make_poc_fixture.py [--out DIR] [--n N] [--partial]
"""
import argparse
import hashlib
import json
import os
import random
import sys

try:
    from PIL import Image, ImageDraw
except ImportError:  # pragma: no cover
    sys.exit("PIL required to build the fixture")

REASONS = [
    ("TOO_DARK", "luma_mean", "<", 0.05),
    ("OVEREXPOSED", "luma_mean", ">", 0.92),
    ("BLURRY", "laplacian_var", "<", 18.0),
    ("OCCLUDED", "edge_density", "<", 0.02),
    ("FLICKER", "luma_std_t", ">", 0.12),
]

FACTORIES = ["factory_061", "factory_042", "factory_017"]
WORKERS = ["worker_003", "worker_007", "worker_011"]


def canonical_receipt(window_id, verdict, reasons, features_digest, versions):
    payload = {
        "window_id": window_id,
        "verdict": verdict,
        "reasons": reasons,
        "features_digest": features_digest,
        "versions": versions,
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


def make_thumb(path, label, hue):
    img = Image.new("RGB", (320, 180), (hue % 256, (hue * 2) % 256, (hue * 3) % 256))
    d = ImageDraw.Draw(img)
    d.rectangle([8, 8, 312, 172], outline=(255, 255, 255))
    d.text((16, 78), label, fill=(255, 255, 255))
    img.save(path, "JPEG", quality=80)


def build(out, n=60, partial=True, seed=7):
    rng = random.Random(seed)
    os.makedirs(os.path.join(out, "thumbs"), exist_ok=True)
    versions = {
        "poc_run": "fixture-0.1",
        "python": "3.x-fixture",
        "torch": "fixture",
        "ffmpeg": "fixture",
        "device": "cpu-fixture",
    }
    by_verdict = {"accept": 0, "reject": 0, "review": 0}
    by_reason = {}
    by_factory = {}
    windows = []
    for i in range(n):
        fac = FACTORIES[i % len(FACTORIES)]
        wor = WORKERS[(i // len(FACTORIES)) % len(WORKERS)]
        start = float((i * 10) % 1200)
        wid = f"f{fac[-3:]}_w{wor[-3:]}_c036_t{int(start):04d}"
        r = rng.random()
        if r < 0.55:
            verdict, stage = "accept", "S1"
            reasons = []
            score = round(rng.uniform(0.6, 0.97), 4)
        elif r < 0.85:
            verdict = "reject"
            code, feat, op, thr = REASONS[rng.randrange(len(REASONS))]
            stage = "S0" if rng.random() < 0.4 else "S1"
            val = round(thr * rng.uniform(0.2, 0.9) if op == "<" else thr * rng.uniform(1.05, 1.2), 4)
            reasons = [{"code": code, "evidence": {"feature": feat, "value": val, "op": op, "threshold": thr}}]
            by_reason[code] = by_reason.get(code, 0) + 1
            score = round(rng.uniform(0.05, 0.4), 4)
        else:
            verdict, stage = "review", "S2"
            reasons = []
            score = None
        by_verdict[verdict] += 1
        features = {
            "luma_mean": round(rng.uniform(0.05, 0.9), 4),
            "luma_std_t": round(rng.uniform(0.01, 0.15), 4),
            "laplacian_var": round(rng.uniform(5.0, 120.0), 4),
            "edge_density": round(rng.uniform(0.005, 0.2), 4),
        }
        fdigest = hashlib.sha256(
            json.dumps(features, sort_keys=True).encode()
        ).hexdigest()
        receipt = canonical_receipt(wid, verdict, reasons, fdigest, versions)
        thumbs = []
        for k, suf in enumerate("abc"):
            name = f"thumbs/{wid}_{suf}.jpg"
            make_thumb(os.path.join(out, name), f"{wid} {suf}", (i * 37 + k * 61) % 200 + 20)
            thumbs.append(name)
        w = {
            "window_id": wid,
            "factory": fac,
            "worker": wor,
            "clip": f"/clips/{fac}/{wor}/clip036.mp4",
            "start_s": start,
            "end_s": start + 10.0,
            "verdict": verdict,
            "stage": stage,
            "reasons": reasons,
            "score": score,
            "features": features,
            "features_digest": fdigest,
            "receipt_sha256": receipt,
            "versions": versions,
            "thumbs": thumbs,
            "latency_ms": {"decode": round(rng.uniform(20, 120), 1),
                           "features": round(rng.uniform(30, 200), 1),
                           "total": round(rng.uniform(60, 340), 1)},
            "label_source": "none",
        }
        windows.append(w)
        f = by_factory.setdefault(fac, {"n": 0, "accept": 0, "reject": 0, "review": 0})
        f["n"] += 1
        f[verdict] += 1

    with open(os.path.join(out, "windows.jsonl"), "w") as fh:
        for w in windows:
            fh.write(json.dumps(w) + "\n")
    hours = n * 10.0 / 3600.0
    summary = {
        "generated_utc": "2026-10-10T11:00:00Z",
        "n_windows": n,
        "hours": round(hours, 4),
        "by_verdict": by_verdict,
        "by_reason": by_reason,
        "by_factory": by_factory,
        "throughput": {
            "hours_of_video_per_wall_hour": 6.5,
            "rtf": 0.15,
            "gpu": "RTX 4070 Laptop (NOT datacenter, NOT a Pi)",
        },
        "partial": partial,
    }
    with open(os.path.join(out, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    # stream replay events fixture
    with open(os.path.join(out, "stream_events.jsonl"), "w") as fh:
        t = 0.0
        score = 0.5
        wid = windows[0]["window_id"] if windows else "demo"
        for chunk in range(12):
            t += 2.0
            score = max(0.02, min(0.99, score + rng.uniform(-0.18, 0.1)))
            if score < 0.15:
                decision = "early_reject"
            elif chunk == 11:
                decision = "accept"
            else:
                decision = "continue"
            fh.write(json.dumps({"t": t, "window_id": wid, "chunk": chunk,
                                 "running_score": round(score, 4),
                                 "decision": decision,
                                 "reason": "TOO_DARK" if decision == "early_reject" else None}) + "\n")
            if decision == "early_reject":
                break
    print(f"wrote {n} windows -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="tests/fixtures/poc")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--partial", action="store_true", default=True)
    ap.add_argument("--no-partial", dest="partial", action="store_false")
    args = ap.parse_args()
    build(args.out, args.n, args.partial)
