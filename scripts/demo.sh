#!/usr/bin/env bash
# End-to-end REAL demo for baseline-v1 (GitHub issue #55, B2/B3/B5/B7).
#
# Pipeline under test (NO stubs): dock the 7 IndEgo clips -> real segmenter ->
# real L0 signals (mediapipe) -> real decider (EmbeddingGemma 2, GPU) ->
# funnel.route -> batches -> receipts -> dashboard-visible store. Then:
# re-verify every receipt via the API, prove a tampered receipt FAILS,
# assert 10 repeated runs + alone-vs-batch-of-5 give IDENTICAL receipt_sha256,
# and write eval/results/integration3_demo.json with the measured numbers.
#
# FAILS CLOSED (non-zero exit + clear message) when:
# - JADAL_USE_STUBS is set in the environment;
# - model files are absent (never stubs silently);
# - any receipt's config.versions names a stub or a silent fallback
#   (see jadal/pipeline.py::receipt_problems);
# - any receipt fails re-verify, or a tampered receipt passes.
#
# GPU: the lock is held only while the decider runs (service phase + the
# direct-pipeline determinism phase), never for the offline assembly.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY=/home/parshu/envs/jadal/bin/python
export PYTHONNOUSERSITE=1
GPU_LOCK=/home/parshu/projects/jadal-qa-wt/tools/gpu.lock

if [[ -n "${JADAL_USE_STUBS:-}" ]]; then
  echo "demo: REFUSING to run: JADAL_USE_STUBS is set (${JADAL_USE_STUBS})." >&2
  echo "demo: unset it: this demo proves the REAL pipeline, no stubs." >&2
  exit 1
fi
unset JADAL_USE_STUBS || true
export JADAL_BATCH_MAX_AGE_S=5
export JADAL_WATCH_BUCKET=1
export JADAL_WATCH_SITE=site-demo
export JADAL_WATCH_WORKER=w-demo

INDEGO=/home/parshu/projects/awing/data/indego
EG2=/home/parshu/projects/awing/data/eg2
STAGE="$ROOT/data_runtime/demo_clips"
RESULTS="$ROOT/eval/results/integration3_demo.json"

CLIPS=(
  "1_Assembly/mp4_480/User_15_410_0411_2_480.mp4"
  "1_Assembly/mp4_480/User_1_412_2010_3_480.mp4"
  "1_Assembly/03_Mechanical_Desk/mp4_480/User_1_User_15_411_2910_2_1_480.mp4"
  "3_Logistics_Organization/mp4_480/User_1_H_pal_transport2_Nar_480.mp4"
  "3_Logistics_Organization/mp4_480/User_1_H_pal_transport_Nar_480.mp4"
  "3_Logistics_Organization/mp4_480/User_1_2110_Organisation_480.mp4"
  "3_Logistics_Organization/mp4_480/User_1_User_16_451_3101_Logistics_4_480.mp4"
)

MODELS=(
  "models/blaze_face_short_range.tflite"
  "models/hand_landmarker.task"
  "models/decider_hands.npz"
  "models/decider_work.npz"
  "models/decider_step_prototypes.npz"
)

echo "== jadal integration3 demo (REAL modules, no stubs) =="
echo "root: $ROOT"

# -- preflight: models present, else a clear message (never stub silently) --
MISSING=0
for m in "${MODELS[@]}"; do
  if [[ ! -f "$ROOT/$m" ]]; then
    echo "demo: MISSING model file: $ROOT/$m" >&2
    MISSING=1
  fi
done
if [[ ! -f "$EG2/model.safetensors" ]]; then
  echo "demo: MISSING backbone weights: $EG2/model.safetensors" >&2
  MISSING=1
fi
if [[ "$MISSING" -ne 0 ]]; then
  echo "demo: REFUSING to run without models (see README quickstart + models/README.md" >&2
  echo "      for how to fetch them). Set JADAL_USE_STUBS=1 only for stub-mode smoke tests," >&2
  echo "      never for this baseline demo." >&2
  exit 1
fi
"$PY" -c "
import sys; sys.path.insert(0, '$ROOT')
from jadal import funnel
funnel.load_thresholds()  # raises ValueError on a tampered thresholds file
print('thresholds file: hash OK')
"

for rel in "${CLIPS[@]}"; do
  if [[ ! -f "$INDEGO/$rel" ]]; then
    echo "demo: MISSING clip: $INDEGO/$rel" >&2
    exit 1
  fi
done

# -- fresh runtime state (demo only; data_runtime/ is git-ignored) --
rm -rf "$ROOT/data_runtime/bucket" "$ROOT/data_runtime/jadal.db" \
       "$ROOT/data_runtime/segments" "$STAGE"
mkdir -p "$ROOT/data_runtime/bucket" "$STAGE"

for rel in "${CLIPS[@]}"; do
  ln -s "$INDEGO/$rel" "$STAGE/$(basename "$rel")"
done
echo "staged $(ls "$STAGE" | wc -l) clips"

# -- GPU section 1: live service (the decider runs here) --
echo "demo: acquiring GPU lock for the service phase (released right after)"
exec 9>"$GPU_LOCK"
flock -w 3600 9

PORT="$("$PY" -c "import socket; s=socket.socket(); s.bind(('127.0.0.1',0)); print(s.getsockname()[1])")"
URL="http://127.0.0.1:$PORT"
echo "service: $URL"

cd "$ROOT"
"$PY" -m uvicorn jadal.service:app --host 127.0.0.1 --port "$PORT" \
  > data_runtime/demo_server.log 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null || true' EXIT

for _ in $(seq 1 60); do
  if "$PY" -c "import urllib.request,sys; urllib.request.urlopen('$URL/health', timeout=2); sys.exit(0)" 2>/dev/null; then
    break
  fi
  sleep 0.5
done
"$PY" -c "import urllib.request; print('health:', urllib.request.urlopen('$URL/health', timeout=5).read().decode())"
"$PY" -c "import urllib.request; print('pico:', urllib.request.urlopen('$URL/pico/status', timeout=5).read().decode())"
"$PY" -c "
import urllib.request
body = urllib.request.urlopen('$URL/', timeout=5).read()
assert b'jadal' in body.lower() or b'html' in body.lower(), 'dashboard / did not serve web/'
print('dashboard /: served OK,', len(body), 'bytes')
"

T0=$(date +%s.%N)
JADAL_DEMO_URL="$URL" "$PY" sim/dock_sim.py --clips "$STAGE" --url "$URL" \
  --delay 0.5 --site site-demo --worker w-demo

# -- wait until the queue drains and batches stop appearing --
export JADAL_DEMO_URL="$URL"
"$PY" - "$URL" <<'EOF'
import sys, time, json, urllib.request
url = sys.argv[1]
last_n, calm = -1, 0
deadline = time.time() + 1800.0
while time.time() < deadline:
    h = json.load(urllib.request.urlopen(url + "/health", timeout=10))
    doc = json.load(urllib.request.urlopen(url + "/batches", timeout=10))
    n = len(doc["batches"])
    if h["pending_segments"] == 0 and n == last_n:
        calm += 1
        if calm >= 3 and n > 0:
            break
    else:
        calm = 0
    last_n = n
    time.sleep(5.0)
else:
    raise SystemExit("demo: batches never drained (timeout)")
print(f"batches closed: {n}")
EOF
T1=$(date +%s.%N)

# -- verdict table + save receipts + reverify every receipt + no-stub assert --
"$PY" - "$URL" "$ROOT" <<'EOF'
import json, sys, urllib.request
from pathlib import Path

url, root = sys.argv[1], Path(sys.argv[2])
sys.path.insert(0, str(root))
from jadal import pipeline as pipe

def get(path):
    return json.load(urllib.request.urlopen(url + path, timeout=60))

def post(path):
    req = urllib.request.Request(url + path, data=b"", method="POST")
    return json.load(urllib.request.urlopen(req, timeout=300))

batches = get("/batches")["batches"]
print(f"\n== verdict table ({len(batches)} batches) ==")
all_receipts, verdict_counts, batch_decisions = [], {}, {}
for b in sorted(batches, key=lambda d: d["created_utc"]):
    det = b["batch_decision"].upper()
    batch_decisions[det] = batch_decisions.get(det, 0) + 1
    print(f"batch {b['batch_id']} site={b['site_id']} decision={det} "
          f"rule='{b['rule']}' k={b['k']} n={b['n']}")
    detail = get(f"/batches/{b['batch_id']}")
    for v in detail["verdicts"]:
        verdict_counts[v["verdict"]] = verdict_counts.get(v["verdict"], 0) + 1
        reasons = ",".join(v["reasons"]) or "-"
        print(f"  {v['segment_id']} {v['verdict']:7s} conf={v['answer_conf']:.4f} "
              f"reasons=[{reasons}]")
    for rid in detail["receipt_ids"]:
        all_receipts.append(get(f"/receipts/{rid}"))
(root / "data_runtime" / "demo_receipts.json").write_text(json.dumps(all_receipts))
(root / "data_runtime" / "demo_batches.json").write_text(json.dumps(batches))
print(f"\nsaved {len(all_receipts)} receipts")

print(f"\n== no-stub assertion ({len(all_receipts)} receipts) ==")
bad = 0
for r in all_receipts:
    problems = pipe.receipt_problems(r)
    if problems:
        bad += 1
        print(f"  STUB/FALLBACK {r['receipt_id']}: {problems}")
if bad:
    raise SystemExit(f"demo: {bad} receipts name a stub or silent fallback "
                     "(refusing a stub receipt; see lines above)")
print("all receipts run on REAL modules (signals+decider+segmenter+funnel)")

print(f"\n== reverify ({len(all_receipts)} receipts) ==")
mismatch = 0
for r in all_receipts:
    out = post(f"/receipts/{r['receipt_id']}/reverify")
    flag = "MATCH" if out["match"] else "MISMATCH"
    if not out["match"]:
        mismatch += 1
    print(f"  {r['receipt_id']} {flag} versions_match={out.get('version_match')}")
assert mismatch == 0, f"{mismatch} receipts failed reverify"
print("all receipts reverify MATCH")

vc = {"verdict_counts": verdict_counts, "batch_decisions": batch_decisions,
      "n_receipts": len(all_receipts), "n_batches": len(batches)}
(root / "data_runtime" / "demo_verdicts.json").write_text(json.dumps(vc))
EOF

# -- tamper check: a receipt with one edited field must FAIL the offline check --
"$PY" - "$ROOT" <<'EOF'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
receipts = json.loads((root / "data_runtime" / "demo_receipts.json").read_text())
r = dict(receipts[0])
r["verdict"] = dict(r["verdict"], verdict="accept" if r["verdict"]["verdict"] != "accept" else "reject")
(root / "data_runtime" / "demo_receipt_tampered.json").write_text(json.dumps(r))
print("tampered receipt id:", r["receipt_id"])
EOF
if "$PY" scripts/verify_receipt.py data_runtime/demo_receipt_tampered.json; then
  echo "demo: TAMPERED receipt PASSED verification (must FAIL)" >&2
  exit 1
else
  echo "tampered receipt correctly FAILS verification"
fi
"$PY" - "$ROOT" <<'EOF'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
receipts = json.loads((root / "data_runtime" / "demo_receipts.json").read_text())
(root / "data_runtime" / "demo_receipt_clean.json").write_text(json.dumps(receipts[0]))
EOF
"$PY" scripts/verify_receipt.py data_runtime/demo_receipt_clean.json \
  || { echo "demo: CLEAN receipt failed offline verification" >&2; exit 1; }

echo "$T0 $T1" > data_runtime/demo_walltime.txt
kill $SRV 2>/dev/null || true
trap - EXIT
flock -u 9
echo "demo: GPU lock released after the service phase"

# -- GPU section 2: determinism on the direct pipeline (shortest clip) --
echo "demo: acquiring GPU lock for the determinism phase"
flock -w 3600 "$GPU_LOCK" "$PY" - "$ROOT" <<'EOF'
import json, sys, time
from pathlib import Path
root = Path(sys.argv[1])
sys.path.insert(0, str(root))
assert not __import__("os").environ.get("JADAL_USE_STUBS"), "JADAL_USE_STUBS must be unset"
from jadal.config import Config
from jadal import pipeline as pipe

cfg = Config()
bucket = sorted((root / "data_runtime" / "bucket").glob("*.mp4"))
assert bucket, "no docked clips found"
shortest = min(bucket, key=lambda p: p.stat().st_size)
print("determinism clip:", shortest.name)

# instrumented single run: per-stage latency (steady state, cache warm)
segs = pipe.process_clip(str(shortest), "w-demo", "site-demo", cfg)
assert len(segs) >= 2, "need >=2 segments for the batch-of-5 check"
t = time.time(); sigs = []
compute = pipe._compute_signals_fn()
for s in segs:
    t0 = time.time()
    sig = pipe._compute_signals_safe(compute, s["path"], cfg, int(cfg.float_decimals))
    sigs.append((time.time() - t0, sig))
signals_s = sum(x[0] for x in sigs)
decider = pipe._get_decider(pipe._decider_class(), cfg)
t = time.time(); decs = []
for s, (_, sig) in zip(segs, sigs):
    t0 = time.time()
    decs.append(pipe._round_floats(decider.decide(s["path"], sig), int(cfg.float_decimals)))
    decs[-1] = (time.time() - t0, decs[-1])
decider_s = sum(x[0] for x in decs)
t = time.time()
block = pipe.config_block(cfg)
thresholds_s = time.time() - t
t = time.time(); routed = [pipe.route_verdict(s, sig, dec, block["thresholds"]) for s, (_, sig), (_, dec) in zip(segs, sigs, decs)]
route_s = time.time() - t
per_seg = signals_s / len(segs) + decider_s / len(segs)
lat = {"segmenter_s_per_seg": None, "signals_s_per_seg": round(signals_s / len(segs), 3),
       "decider_s_per_seg": round(decider_s / len(segs), 3),
       "route_s_per_seg": round(route_s / len(segs), 3),
       "segments_per_hour": round(3600.0 / per_seg, 1) if per_seg > 0 else 0.0,
       "n_segments_timed": len(segs),
       "note": "steady-state per-segment cost (embedding cache warm); segmenter timed separately below"}
print("per-stage latency:", json.dumps(lat))

t = time.time()
segs2 = pipe.process_clip(str(shortest), "w-demo", "site-demo", cfg)
segmenter_s = time.time() - t
lat["segmenter_s_per_seg"] = round(segmenter_s / len(segs2), 3)

# 10 repeated FULL runs: identical receipt_sha256 per segment
runs = []
for _ in range(10):
    s = pipe.process_clip(str(shortest), "w-demo", "site-demo", cfg)
    _, receipts = pipe.run_batch(s, cfg)
    assert not any(pipe.receipt_problems(r) for r in receipts), "stub/fallback in determinism run"
    runs.append({r["verdict"]["segment_id"]: r["receipt_sha256"] for r in receipts})
first = runs[0]
assert all(r == first for r in runs), "10 runs diverged"
print(f"determinism: 10 runs x {len(first)} segments -> IDENTICAL receipt_sha256")

# alone (batch of 1) vs in a batch of 5: identical receipt_sha256
others = [s for p in sorted((root / "data_runtime" / "bucket").glob("*.mp4")) if p != shortest
          for s in pipe.process_clip(str(p), "w-demo", "site-demo", cfg)][:4]
assert len(others) == 4, "need 4 more segments for the batch-of-5 check"
target = pipe.process_clip(str(shortest), "w-demo", "site-demo", cfg)[0]
_, solo = pipe.run_batch([target], cfg)
_, batched = pipe.run_batch([target] + others, cfg)
by_id = {r["verdict"]["segment_id"]: r["receipt_sha256"] for r in batched}
assert solo[0]["receipt_sha256"] == by_id[target["segment_id"]], "solo vs batch-of-5 differ"
print("determinism: alone vs batch-of-5 -> IDENTICAL receipt_sha256")

out = {"ten_runs_identical": True, "ten_runs_n_segments": len(first),
       "ten_runs_hash_prefix": next(iter(first.values()))[:16],
       "alone_vs_batch5_identical": True, "latency": lat,
       "versions": pipe.impl_versions(cfg)}
(root / "data_runtime" / "demo_determinism.json").write_text(json.dumps(out, indent=2))
print("determinism + latency written")
EOF
echo "demo: GPU lock released after the determinism phase"

# -- offline assembly: eval/results/integration3_demo.json (no lock) --
"$PY" - "$ROOT" <<'EOF'
import json, sys
from datetime import datetime, timezone
from pathlib import Path
root = Path(sys.argv[1])
sys.path.insert(0, str(root))
from jadal import pipeline as pipe
from jadal.config import Config

verdicts = json.loads((root / "data_runtime" / "demo_verdicts.json").read_text())
determ = json.loads((root / "data_runtime" / "demo_determinism.json").read_text())
t0, t1 = (root / "data_runtime" / "demo_walltime.txt").read_text().split()
receipt0 = json.loads((root / "data_runtime" / "demo_receipts.json").read_text())[0]
payload = {
    "baseline": "B2/B3/B5/B7",
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "device_label": "x86 laptop, shared GPU, NOT a Pi",
    "commands": [
        "flock -w 3600 tools/gpu.lock bash scripts/demo.sh",
        "PYTHONNOUSERSITE=1 /home/parshu/envs/jadal/bin/python -m unittest discover -s tests",
    ],
    "n_clips": 7,
    "n_batches": verdicts["n_batches"],
    "n_receipts": verdicts["n_receipts"],
    "verdict_counts": verdicts["verdict_counts"],
    "batch_decisions": verdicts["batch_decisions"],
    "dock_to_batches_wall_s": round(float(t1) - float(t0), 1),
    "determinism": {k: v for k, v in determ.items() if k != "versions"},
    "latency": determ["latency"],
    "reverify": {"matched": verdicts["n_receipts"], "total": verdicts["n_receipts"]},
    "tamper": {"tampered_receipt_fails": True, "clean_receipt_passes": True},
    "versions": determ["versions"],
    "notes": [
        "Every receipt ran on real modules: receipt_problems() is empty for all receipts.",
        "Cross-architecture determinism is NOT claimed (arch recorded in versions).",
        "Thresholds are placeholders (configs/funnel_thresholds.json, status placeholder).",
    ],
}
out = root / "eval" / "results" / "integration3_demo.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(payload, indent=2, sort_keys=True))
print("wrote", out)
EOF

echo -e "\ndemo OK (REAL modules, no stubs)"
