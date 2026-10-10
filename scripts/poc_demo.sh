#!/usr/bin/env bash
# One-command PoC demo (issue #64): dashboard + scripted stream replay.
# Cleans up ONLY the PIDs it started (never kills by name).
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY=/home/parshu/envs/jadal/bin/python
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=2

DEFAULT_CLIP=/home/parshu/projects/awing/data/ego10k_scale/clips/factory_069/worker_004/factory069_worker004_00077.mp4
if [[ -n "${1:-}" ]]; then
  CLIP="$1"
elif [[ -f "$DEFAULT_CLIP" ]]; then
  CLIP="$DEFAULT_CLIP"  # pinned: DEMO.md expected scores were measured here
else
  CLIP="$(find /home/parshu/projects/awing/data/ego10k_scale/clips -name '*.mp4' | head -n 1)"
fi
EVENTS="${EVENTS:-/home/parshu/projects/jadal-qa-wt/poc_data/stream_events.jsonl}"
PORT=8840
SERVER_PID=""

cleanup() {
  if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
    echo "demo: stopped server pid ${SERVER_PID}"
  fi
}
trap cleanup EXIT INT TERM

echo "== jadal PoC demo =="
echo "clip:   $CLIP"
echo "events: $EVENTS"

# -- 1. dashboard server (poc-dashboard branch file, if present) --
if [[ -f "$ROOT/scripts/poc_server.py" ]]; then
  "$PY" "$ROOT/scripts/poc_server.py" --port "$PORT" &
  SERVER_PID="$!"
  sleep 2
  if kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "dashboard: http://localhost:$PORT/ (pid $SERVER_PID)"
  else
    echo "dashboard: FAILED to start (see above); replay continues headless"
    SERVER_PID=""
  fi
else
  echo "dashboard: stub -- scripts/poc_server.py is not on this branch yet"
  echo "dashboard: live verdicts stream to $EVENTS (tail -f it in another terminal)"
fi

# -- 2. scripted replay: clean start, then a blackout the funnel must catch --
echo "-- replay: 6 clean chunks, then blackout at t=12s --"
"$PY" "$ROOT/scripts/poc_stream_replay.py" --clip "$CLIP" \
  --speed 4 --max-chunks 16 --degrade blackout --degrade-at 12 \
  --out "$EVENTS" ${PICO:+--pico}

echo "-- last verdicts --"
tail -n 6 "$EVENTS"
echo "dashboard: ${SERVER_PID:+http://localhost:$PORT/ (pid $SERVER_PID)}"
echo "== demo done (server above stops on exit) =="
