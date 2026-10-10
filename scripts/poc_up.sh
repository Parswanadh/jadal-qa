#!/usr/bin/env bash
# poc_up.sh — one-command PoC: dashboard on :8840 (real poc_data) + optional live replay.
#
#   scripts/poc_up.sh [--replay] [--no-wait]
#
# (a) Serves the REAL poc_data on :8840 (never --demo-fixture). If :8840 is
#     already taken, reports the owning PID + cmdline (and whether it runs
#     --demo-fixture) and reuses it — never kills by name.
# (b) Prints the judge URLs.
# (c) --replay: runs the live 2-second-chunk replay with a scripted blackout
#     so the dashboard Live panel shows an early reject.
# (d) Cleans up ONLY the PIDs this script started (trap on INT/TERM).
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Environment overrides (documented defaults):
#   JADAL_PY         interpreter (default: python3 on PATH)
#   JADAL_POC_DATA   real PoC data dir
#   JADAL_CLIP_ROOT  Egocentric-10K clips for --replay
#   Defaults and the resolution order: scripts/poc_paths.py (never the demo fixture).
PY="${JADAL_PY:-python3}"
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=2

PORT=8840
DATA="$("$PY" "$ROOT/scripts/poc_paths.py" data)" || exit 1
EVENTS="$DATA/stream_events.jsonl"
REPLAY=0
NO_WAIT=0
for a in "$@"; do
  case "$a" in
    --replay) REPLAY=1 ;;
    --no-wait) NO_WAIT=1 ;;
    *) echo "usage: $0 [--replay] [--no-wait]" >&2; exit 2 ;;
  esac
done

SERVER_PID=""
REPLAY_PID=""

cleanup() {
  if [[ -n "${REPLAY_PID:-}" ]] && kill -0 "$REPLAY_PID" 2>/dev/null; then
    kill "$REPLAY_PID" 2>/dev/null || true
  fi
  if [[ -n "${SERVER_PID:-}" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
    echo "poc_up: stopped server pid $SERVER_PID (started by this script)"
  fi
}
trap cleanup INT TERM

port_pid() {
  # owning PID of TCP :$PORT via /proc scan (no lsof/ss dependency for parse)
  "$PY" -c "
import os, re, sys
port_hex = '%04X' % int(sys.argv[1])
inodes = set()
for f in ('/proc/net/tcp', '/proc/net/tcp6'):
    try: lines = open(f).readlines()[1:]
    except OSError: continue
    for ln in lines:
        p = ln.split()
        if len(p) < 10: continue
        local, state, inode = p[1], p[3], p[9]
        if state == '0A' and local.split(':')[1] == port_hex:
            inodes.add(inode)
for pid in filter(str.isdigit, os.listdir('/proc')):
    try: fds = os.listdir(f'/proc/{pid}/fd')
    except OSError: continue
    for fd in fds:
        try: t = os.readlink(f'/proc/{pid}/fd/{fd}')
        except OSError: continue
        if t.startswith('socket:[') and t[8:-1] in inodes:
            print(pid); raise SystemExit
" "$PORT" 2>/dev/null
}

echo "== jadal PoC =="
EXISTING="$(port_pid)"
if [[ -n "$EXISTING" ]]; then
  CMD="$(tr '\0' ' ' < /proc/$EXISTING/cmdline 2>/dev/null)"
  echo "port $PORT: already serving (pid $EXISTING): $CMD"
  case "$CMD" in
    *--demo-fixture*)
      echo "WARNING: existing server runs --demo-fixture (synthetic data), not the real run." ;;
    *)
      echo "existing server points at real data (no --demo-fixture flag). reusing it." ;;
  esac
else
  "$PY" "$ROOT/scripts/poc_server.py" --port "$PORT" --data-dir "$DATA" &
  SERVER_PID="$!"
  sleep 2
  if kill -0 "$SERVER_PID" 2>/dev/null && curl -sf -o /dev/null "http://127.0.0.1:$PORT/api/summary"; then
    echo "server started (pid $SERVER_PID) on :$PORT serving $DATA"
  else
    echo "server FAILED to start" >&2
    SERVER_PID=""
    exit 1
  fi
fi

BASE="http://localhost:$PORT"
echo "urls:"
echo "  overview: $BASE/#/overview"
echo "  gallery (rejects): $BASE/?verdict=reject#/gallery"
echo "  live:     $BASE/#/live"
echo "  about:    $BASE/#/about"

if [[ "$REPLAY" == "1" ]]; then
  CLIP_DIR="$("$PY" "$ROOT/scripts/poc_paths.py" clips)" || exit 1
  CLIP="${CLIP:-$(find "$CLIP_DIR" -name '*.mp4' | head -n 1)}"
  echo "-- replay: live 2 s chunks with scripted blackout (watch $BASE/#/live) --"
  echo "clip: $CLIP"
  "$PY" "$ROOT/scripts/poc_stream_replay.py" --clip "$CLIP" \
    --chunk-s 2 --speed 4 --max-chunks 16 --degrade blackout --degrade-at 12 \
    --out "$EVENTS" &
  REPLAY_PID="$!"
  wait "$REPLAY_PID"
  REPLAY_PID=""
  echo "-- last events --"
  tail -n 4 "$EVENTS"
fi

if [[ -n "$SERVER_PID" && "$NO_WAIT" == "0" ]]; then
  echo "serving $BASE (pid $SERVER_PID). Ctrl-C to stop (only this pid)."
  wait "$SERVER_PID"
fi
echo "== poc_up done =="
