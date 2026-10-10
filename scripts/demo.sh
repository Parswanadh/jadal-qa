#!/usr/bin/env bash
# End-to-end demo for the integration worker (GitHub issue #17, step 4).
#
# Starts the service on a free port, docks the IndEgo clips (catalog:
# /home/parshu/projects/awing/data/indego/CATALOG.md) through sim/dock_sim.py
# into data_runtime/bucket, waits for batches, prints the verdict table, runs
# Re-verify on every receipt, runs the same clip through the pipeline 10 times
# and asserts IDENTICAL receipt_sha256 (alone vs inside a batch of 5 too),
# then prints batch latency.
#
# Runs on stubs (JADAL_USE_STUBS=1): the real signals/decider modules have not
# merged to main yet, so L0/L1 values are deterministic hash-derived fakes.
# No secrets, no network beyond localhost. Fresh runtime state per run.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY=/home/parshu/envs/jadal/bin/python
export PYTHONNOUSERSITE=1
export JADAL_USE_STUBS=1
export JADAL_BATCH_MAX_AGE_S=5
export JADAL_WATCH_BUCKET=1
export JADAL_WATCH_SITE=site-demo
export JADAL_WATCH_WORKER=w-demo

INDEGO=/home/parshu/projects/awing/data/indego
STAGE="$ROOT/data_runtime/demo_clips"

# The 7 catalog clips (CATALOG.md: ~174 MB video total).
CLIPS=(
  "1_Assembly/mp4_480/User_15_410_0411_2_480.mp4"
  "1_Assembly/mp4_480/User_1_412_2010_3_480.mp4"
  "1_Assembly/03_Mechanical_Desk/mp4_480/User_1_User_15_411_2910_2_1_480.mp4"
  "3_Logistics_Organization/mp4_480/User_1_H_pal_transport2_Nar_480.mp4"
  "3_Logistics_Organization/mp4_480/User_1_H_pal_transport_Nar_480.mp4"
  "3_Logistics_Organization/mp4_480/User_1_2110_Organisation_480.mp4"
  "3_Logistics_Organization/mp4_480/User_1_User_16_451_3101_Logistics_4_480.mp4"
)

echo "== jadal integration demo =="
echo "root: $ROOT"

# -- fresh runtime state (demo only; data_runtime/ is git-ignored) --
rm -rf "$ROOT/data_runtime/bucket" "$ROOT/data_runtime/jadal.db" \
       "$ROOT/data_runtime/segments" "$STAGE"
mkdir -p "$ROOT/data_runtime/bucket" "$STAGE"

# -- stage the catalog clips as symlinks (dock_sim copies the bytes) --
for rel in "${CLIPS[@]}"; do
  if [[ ! -f "$INDEGO/$rel" ]]; then
    echo "MISSING clip: $INDEGO/$rel" >&2
    exit 1
  fi
  ln -s "$INDEGO/$rel" "$STAGE/$(basename "$rel")"
done
echo "staged $(ls "$STAGE" | wc -l) clips"

# -- free port --
PORT="$("$PY" -c "import socket; s=socket.socket(); s.bind(('127.0.0.1',0)); print(s.getsockname()[1])")"
URL="http://127.0.0.1:$PORT"
echo "service: $URL"

# -- start the service --
cd "$ROOT"
"$PY" -m uvicorn jadal.service:app --host 127.0.0.1 --port "$PORT" \
  > data_runtime/demo_server.log 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null || true' EXIT

# -- wait for /health --
for _ in $(seq 1 60); do
  if "$PY" -c "import urllib.request,sys; urllib.request.urlopen('$URL/health', timeout=2); sys.exit(0)" 2>/dev/null; then
    break
  fi
  sleep 0.5
done
"$PY" -c "import urllib.request; print('health:', urllib.request.urlopen('$URL/health', timeout=5).read().decode())"

# -- dock the clips (copies into the bucket + POSTs /ingest/event) --
T0=$(date +%s.%N)
JADAL_DEMO_URL="$URL" "$PY" sim/dock_sim.py --clips "$STAGE" --url "$URL" \
  --delay 0.5 --site site-demo --worker w-demo

# -- wait for both batches (7 clips -> 5 + timeout-flushed 2) --
export JADAL_DEMO_URL="$URL"
N_BATCHES="$("$PY" - "$URL" <<'EOF'
import os, sys, time, json, urllib.request
url = sys.argv[1]
deadline = time.time() + 60.0
while time.time() < deadline:
    doc = json.load(urllib.request.urlopen(url + "/batches", timeout=10))
    if len(doc["batches"]) >= 2:
        break
    time.sleep(0.5)
print(len(doc["batches"]))
EOF
)"
T1=$(date +%s.%N)
echo "batches closed: $N_BATCHES (7 clips -> batch of 5 + timeout-flushed 2)"

# -- verdict table + reverify every receipt + determinism asserts --
"$PY" - "$URL" <<'EOF'
import json, sys, urllib.request

url = sys.argv[1]

def get(path):
    return json.load(urllib.request.urlopen(url + path, timeout=30))

def post(path):
    req = urllib.request.Request(url + path, data=b"", method="POST")
    return json.load(urllib.request.urlopen(req, timeout=60))

batches = get("/batches")["batches"]
print(f"\n== verdict table ({len(batches)} batches) ==")
all_receipts = []
for b in sorted(batches, key=lambda d: d["created_utc"]):
    det = b["batch_decision"].upper()
    print(f"batch {b['batch_id']} site={b['site_id']} decision={det} "
          f"rule='{b['rule']}' k={b['k']} n={b['n']}")
    detail = get(f"/batches/{b['batch_id']}")
    for v in detail["verdicts"]:
        reasons = ",".join(v["reasons"]) or "-"
        print(f"  {v['segment_id']} {v['verdict']:7s} conf={v['answer_conf']:.4f} "
              f"reasons=[{reasons}]")
    all_receipts.extend(detail["receipt_ids"])

print(f"\n== reverify ({len(all_receipts)} receipts) ==")
bad = 0
for rid in all_receipts:
    r = post(f"/receipts/{rid}/reverify")
    flag = "MATCH" if r["match"] else "MISMATCH"
    if not r["match"]:
        bad += 1
    print(f"  {rid} {flag} versions_match={r.get('version_match')}")
assert bad == 0, f"{bad} receipts failed reverify"
print("all receipts reverify MATCH")
EOF

# -- determinism: same clip 10x + alone-vs-batch-of-5 (SPEC rule) --
"$PY" - <<'EOF'
import os, sys
sys.path.insert(0, os.getcwd())
os.environ["JADAL_USE_STUBS"] = "1"
from pathlib import Path
from jadal.config import Config
from jadal import pipeline

bucket = sorted(Path("data_runtime/bucket").glob("*.mp4"))
assert bucket, "no docked clips found"
clip = str(bucket[0])
cfg = Config()

hashes = set()
for _ in range(10):
    segs = pipeline.process_clip(clip, "w-demo", "site-demo", cfg)
    _, receipts = pipeline.run_batch(segs, cfg)
    hashes.add(receipts[0]["receipt_sha256"])
assert len(hashes) == 1, f"10 runs diverged: {hashes}"
print(f"\ndeterminism: same clip 10x -> identical {hashes.pop()[:16]}...")

others = [str(p) for p in bucket[1:5]]
assert len(others) == 4, "need 4 more clips for the batch-of-5 check"
target = pipeline.process_clip(clip, "w-demo", "site-demo", cfg)[0]
_, solo = pipeline.run_batch([target], cfg)
group = [target] + [s for p in others
                    for s in pipeline.process_clip(p, "w-demo", "site-demo", cfg)]
_, batched = pipeline.run_batch(group, cfg)
by_id = {r["verdict"]["segment_id"]: r["receipt_sha256"] for r in batched}
assert solo[0]["receipt_sha256"] == by_id[target["segment_id"]], \
    "solo vs batch-of-5 receipt_sha256 differ"
print("determinism: alone vs batch-of-5 -> IDENTICAL receipt_sha256")
EOF

# -- batch latency --
"$PY" -c "print(f'\nbatch latency: dock-to-batches wall time = {float($T1) - float($T0):.1f}s (7 clips, 2 batches, includes 5s timeout flush)')"

echo -e "\ndemo OK"
