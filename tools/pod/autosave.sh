#!/usr/bin/env bash
# Safety snapshots for a disposable pod: every INTERVAL seconds, for every worktree under WT_ROOT, commit all changes
# (respecting .gitignore) to a temporary index and push them to refs/heads/wip/<branch> (force allowed on wip/* only).
# The working branch and its history are NOT touched. Skips a worktree if secrets_guard finds a secret.
# usage: autosave.sh [--once]   env: WT_ROOT=/workspace/wt  REPO=/workspace/repo  INTERVAL=300
set -u
WT_ROOT=${WT_ROOT:-/workspace/wt}; REPO=${REPO:-/workspace/repo}; INTERVAL=${INTERVAL:-300}; GUARD="$(cd "$(dirname "$0")" && pwd)/secrets_guard.sh"
snap() {
  d=$1; cd "$d" || return; br=$(git rev-parse --abbrev-ref HEAD 2>/dev/null) || return; [ "$br" = HEAD ] && return
  idx=$(mktemp); trap 'rm -f "$idx"' RETURN
  export GIT_INDEX_FILE=$idx; git read-tree HEAD 2>/dev/null; git add -A 2>/dev/null
  if ! "$GUARD" >/dev/null 2>&1; then echo "$(date +%T) autosave: SECRET in $d, skipped" >&2; unset GIT_INDEX_FILE; return; fi
  tree=$(git write-tree); parent=$(git rev-parse HEAD)
  [ "$tree" = "$(git rev-parse HEAD^{tree})" ] && { unset GIT_INDEX_FILE; return; }
  c=$(git -c user.name="${GIT_AUTHOR_NAME:-jadal-pod}" -c user.email="${GIT_AUTHOR_EMAIL:-noreply@users.noreply.github.com}" commit-tree "$tree" -p "$parent" -m "wip snapshot $(date -u +%FT%TZ) of $br")
  unset GIT_INDEX_FILE; git push -q --force origin "$c:refs/heads/wip/$br" 2>&1 | tail -1; echo "$(date +%T) autosave: $br -> wip/$br ${c:0:8}"
}
loop() { for d in "$WT_ROOT"/*/ "$REPO"; do [ -e "$d/.git" ] && snap "${d%/}"; done; }
if [ "${1:-}" = "--once" ]; then loop; else while true; do loop; sleep "$INTERVAL"; done; fi
