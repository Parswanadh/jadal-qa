#!/usr/bin/env bash
# Parallel headless opencode workers, one git worktree + one private opencode server each.
#   launch_workers.sh start  <name> <agent> <prompt-file> [base-branch=origin/main] [--dry-run]
#   launch_workers.sh status
#   launch_workers.sh stop   <name>          (kills ONLY the PID recorded for that worker)
#   launch_workers.sh logs   <name>
# env: REPO=/workspace/repo WT_ROOT=/workspace/wt LOG_DIR=/workspace/logs STATE=/workspace/state JADAL_MODEL=<provider/model#variant> JADAL_MAX_WORKERS=4
# NOTE: opencode v2 `run` has no --dir/--variant: this script cd's into the worktree and uses '#variant' inside the model string.
set -u
REPO=${REPO:-/workspace/repo}; WT_ROOT=${WT_ROOT:-/workspace/wt}; LOG_DIR=${LOG_DIR:-/workspace/logs}; STATE=${STATE:-/workspace/state}; MAXW=${JADAL_MAX_WORKERS:-4}
mkdir -p "$WT_ROOT" "$LOG_DIR" "$STATE/pids"
alive() { [ -f "$STATE/pids/$1" ] && kill -0 "$(cat "$STATE/pids/$1")" 2>/dev/null; }
case "${1:-}" in
  start)
    name=${2:?name}; agent=${3:?agent}; pf=${4:?prompt file}; base=${5:-origin/main}; dry=0; [ "${*: -1}" = "--dry-run" ] && dry=1
    [ -f "$pf" ] || { echo "prompt file not found: $pf" >&2; exit 2; }
    n=0; for f in "$STATE"/pids/*; do [ -f "$f" ] && alive "$(basename "$f")" && n=$((n+1)); done
    [ "$n" -ge "$MAXW" ] && { echo "refused: $n workers already running (JADAL_MAX_WORKERS=$MAXW)" >&2; exit 3; }
    alive "$name" && { echo "worker $name already running" >&2; exit 4; }
    wt="$WT_ROOT/$name"; branch="${WORKER_BRANCH:-feat/$name}"
    cmd=(opencode run --auto --standalone --agent "$agent" ${JADAL_MODEL:+-m "$JADAL_MODEL"} --title "$name")
    if [ "$dry" = 1 ]; then echo "[dry-run] git -C $REPO worktree add -b $branch $wt $base"; echo "[dry-run] (cd $wt && ${cmd[*]} \"\$(cat $pf)\" > $LOG_DIR/$name.log 2>&1 &)"; exit 0; fi
    [ -d "$wt" ] || git -C "$REPO" worktree add -q -b "$branch" "$wt" "$base" || exit 5
    ( cd "$wt" && OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} OPENCODE_DISABLE_AUTOUPDATE=1 setsid nohup "${cmd[@]}" "$(cat "$pf")" > "$LOG_DIR/$name.log" 2>&1 < /dev/null & echo $! > "$STATE/pids/$name" )
    sleep 3; echo "started $name (agent $agent, branch $branch, pid $(cat "$STATE/pids/$name")); verify tool activity in $LOG_DIR/$name.log after ~60 s";;
  status)
    for f in "$STATE"/pids/*; do [ -f "$f" ] || continue; n=$(basename "$f"); if alive "$n"; then s=running; else s=stopped; fi
      echo "$n $s pid=$(cat "$f") log_bytes=$(stat -c %s "$LOG_DIR/$n.log" 2>/dev/null) last_write=$(stat -c %y "$LOG_DIR/$n.log" 2>/dev/null | cut -c12-19)"; done;;
  stop) name=${2:?name}; alive "$name" && { pid=$(cat "$STATE/pids/$name"); kill -- -"$pid" 2>/dev/null || kill "$pid"; echo "stopped $name (pid $pid)"; } || echo "$name not running";;
  logs) tail -n 60 "$LOG_DIR/${2:?name}.log" | tr -d '\033';;
  *) sed -n 2,9p "$0"; exit 1;;
esac
