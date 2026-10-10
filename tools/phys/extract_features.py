"""Feature extraction for the deterministic physics model (clip -> feature record).

Decodes a window at 160x90 with ffmpeg (CPU, no GPU) and computes the two
feature families the rule table reads:

  * ``full.*``  -- 26 per-pixel spectral/temporal kernels (jadal/gpu/features.py,
    vendored verbatim into ``tools/phys/vendor/features_full.py``);
  * ``phys.*``  -- 16 physics/signal features (jadal/perpixel/physics.py,
    vendored into ``tools/phys/vendor/features_phys.py``).

Both are the exact kernels used to derive the frozen thresholds, so a clip
scored here is comparable to the cached gold features. All math is float64,
fixed-order, seed-free; the extraction is the only float stage.

CLI:
  python3 tools/phys/extract_features.py --clip VIDEO --start-s 0 --duration 10 \
      --out features.json [--window-id ID]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from tools.phys.vendor import features_full as F  # noqa: E402
from tools.phys.vendor import features_phys as P  # noqa: E402


def decode_window(path: str, start_s: float, dur_s: float, w: int = 160,
                  h: int = 90, timeout_s: int = 120):
    """ffmpeg decode of [start_s, start_s+dur_s) scaled to w,h -> (rgb, fps)."""
    vf = f"scale={w}:{h},format=rgb24"
    cmd = ["ffmpeg", "-hide_banner", "-v", "error", "-nostdin",
           "-ss", str(start_s), "-t", str(dur_s), "-i", path,
           "-vf", vf, "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                       stdin=subprocess.DEVNULL, timeout=timeout_s)
    if p.returncode != 0:
        raise RuntimeError(f"ffmpeg rc={p.returncode}: "
                           f"{p.stderr[-400:].decode('utf-8', 'replace')}")
    px = w * h * 3
    n = len(p.stdout) // px
    if n == 0:
        raise RuntimeError(f"no frames decoded from {path}@{start_s}")
    arr = np.frombuffer(p.stdout[:n * px], dtype=np.uint8).astype(np.float64) / 255.0
    return arr.reshape(n, h, w, 3), n / dur_s


def luma_of(rgb: np.ndarray) -> np.ndarray:
    return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]


def extract_window_features(path: str, start_s: float, duration_s: float) -> dict:
    """Return a flat feature record with ``full.*`` and ``phys.*`` keys."""
    rgb, fps = decode_window(path, start_s, duration_s)
    luma = luma_of(rgb)
    full = F.cpu_window_features(luma, rgb, fps=min(fps, 30.0))
    phys = P.extract_physics(rgb, fps, path, start_s, start_s + duration_s)
    feats = {f"full.{k}": float(v) for k, v in full.items()}
    feats.update({f"phys.{k}": float(v) for k, v in phys.items()})
    return feats


def main() -> int:
    ap = argparse.ArgumentParser(description="Extract physics features for a clip window.")
    ap.add_argument("--clip", required=True, help="video path")
    ap.add_argument("--start-s", type=float, default=0.0)
    ap.add_argument("--duration", type=float, default=10.0)
    ap.add_argument("--window-id", default=None)
    ap.add_argument("--out", default=None, help="write feature record JSON here")
    args = ap.parse_args()
    feats = extract_window_features(args.clip, args.start_s, args.duration)
    wid = args.window_id or os.path.basename(args.clip)
    rec = {"window_id": wid, "start_s": args.start_s, "duration_s": args.duration,
           "features": {k: feats[k] for k in sorted(feats)}}
    text = json.dumps(rec, indent=1, sort_keys=True)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text + "\n")
        print(f"wrote {args.out}", file=sys.stderr)
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
