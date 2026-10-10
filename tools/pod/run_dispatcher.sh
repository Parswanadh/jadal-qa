#!/usr/bin/env bash
# run_dispatcher.sh - sandboxed exec tier for PR-triggered agents (A6).
#
# Runs tests + the standard eval for a pull request ONLY after the maintainer
# has applied the `run-tests` label.  It never runs against the live tree:
# it makes a throwaway clone, checks out the PR head, drops to an unprivileged
# user and runs with an environment allowlist (tokens are absent), under a
# wall-clock timeout.  Results go to stdout for the watcher to post.
#
#   run_dispatcher.sh run <pr> [--repo owner/name] [--timeout SECONDS]
#   run_dispatcher.sh --check-gate <pr> [--repo owner/name]   # 0 ok, 3 no label
#   run_dispatcher.sh --print-allowlist
#   run_dispatcher.sh --dry-run <pr> [--repo ...] [--timeout ...]
#
# env (all optional):
#   PRWATCH_GH_BIN      gh executable (default: gh; tests point this at a fake)
#   PRWATCH_SRC_REPO    local repo to seed the clone (default: /workspace/repo)
#   PRWATCH_STATE_DIR   scratch dir (default: /workspace/state/prwatch)
#   PRRUNNER_USER       unprivileged user (default: nobody)
#   PRRUNNER_UID/GID    numeric ids (default: looked up from PRRUNNER_USER)
#   RUN_TESTS_CMD       command run inside the copy (default: unittest + standard eval)
#
# SECURITY: never echo a token; never pass PR text to a shell.
set -u

GH_BIN=${PRWATCH_GH_BIN:-gh}
SRC_REPO=${PRWATCH_SRC_REPO:-/workspace/repo}
STATE_DIR=${PRWATCH_STATE_DIR:-/workspace/state/prwatch}
PRRUN_USER=${PRRUNNER_USER:-nobody}
LABEL="run-tests"

ALLOWLIST=(PATH LANG LC_ALL TZ HOME TMPDIR PYTHONNOUSERSITE PYTHONDONTWRITEBYTECODE OMP_NUM_THREADS JADAL_BUNDLE JADAL_EXP_DIR)
DENYLIST=(HF_TOKEN GH_TOKEN AGENT_API_KEY GITHUB_TOKEN)
DEFAULT_TIMEOUT=900

usage() { sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'; }

uid_of() { id -u "$1" 2>/dev/null; }
gid_of() { id -g "$1" 2>/dev/null; }

print_allowlist() { printf '%s\n' "${ALLOWLIST[@]}"; }

# Exit 0 if the maintainer applied the run-tests label, 3 otherwise.
have_label() {
  local pr=$1 repo=$2
  "$GH_BIN" api "repos/${repo}/pulls/${pr}" --jq '.labels[].name' 2>/dev/null | grep -qx "$LABEL"
}

build_env() {
  # Emit `env -i` + allowlisted KEY=VALUE arguments (HOME points at scratch home).
  local home=$1; shift
  printf 'env\0-i\0'
  local v
  for v in "${ALLOWLIST[@]}"; do
    if [ "$v" = "HOME" ]; then printf 'HOME=%s\0' "$home"; continue; fi
    if [ -n "${!v:-}" ]; then printf '%s=%s\0' "$v" "${!v}"; fi
  done
  printf 'PATH=%s\0' "/usr/local/bin:/usr/bin:/bin"
  printf 'PYTHONNOUSERSITE=1\0'
}

cmd_check_gate() {
  local pr=$1 repo=$2
  if have_label "$pr" "$repo"; then
    echo "gate: label '$LABEL' present on PR #$pr"
    return 0
  fi
  echo "gate: REFUSED - label '$LABEL' absent on PR #$pr" >&2
  return 3
}

cmd_run() {
  local pr=$1 repo=$2 timeout=$3 dry=$4
  case "$timeout" in ''|*[!0-9]*) echo "bad --timeout: $timeout" >&2; return 2;; esac
  [ "$timeout" -ge 1 ] || { echo "bad --timeout: $timeout" >&2; return 2; }

  # Defence in depth: the watcher already gates; the script re-checks.
  if ! have_label "$pr" "$repo"; then
    echo "refused: PR #$pr does not carry the '$LABEL' label" >&2
    return 3
  fi

  local uid gid
  uid=$(uid_of "$PRRUN_USER"); gid=$(gid_of "$PRRUN_USER")
  [ -n "$uid" ] && [ -n "$gid" ] || { echo "unknown user: $PRRUN_USER" >&2; return 2; }

  if [ "$dry" = 1 ]; then
    echo "dry-run: pr=$pr repo=$repo timeout=${timeout}s user=$PRRUN_USER($uid:$gid)"
    echo "dry-run: allowlist: $(print_allowlist | tr '\n' ' ')"
    echo "dry-run: denylist (must be absent): ${DENYLIST[*]}"
    return 0
  fi

  mkdir -p "$STATE_DIR"
  local tmp; tmp=$(mktemp -d "${STATE_DIR}/exec.XXXXXX") || return 5
  local home="${tmp}/home"
  mkdir -p "$home"
  # shellcheck disable=SC2064
  trap "rm -rf -- '$tmp'" EXIT INT TERM

  # Throwaway clone. gh (with GH_TOKEN in this privileged process only) is the
  # git credential helper, so no token ever appears in a URL or argv.
  "$GH_BIN" auth setup-git >/dev/null 2>&1 || true
  if ! git clone --quiet --depth 1 "https://github.com/${repo}.git" "${tmp}/repo" 2>"${tmp}/clone.err"; then
    echo "clone failed:" >&2; tail -n 5 "${tmp}/clone.err" >&2; return 5
  fi
  if ! git -C "${tmp}/repo" fetch --quiet --depth 1 origin "pull/${pr}/head:prwatch-${pr}" 2>"${tmp}/fetch.err"; then
    echo "fetch of PR #$pr failed:" >&2; tail -n 5 "${tmp}/fetch.err" >&2; return 5
  fi
  git -C "${tmp}/repo" -c advice.detachedHead=false checkout --quiet "prwatch-${pr}" || return 5

  chown -R "$uid:$gid" "$tmp" 2>/dev/null || true
  chmod -R u+rwX,go-rwx "$tmp" 2>/dev/null || true

  local runtests=${RUN_TESTS_CMD:-'python3 -m unittest discover -s tests; python3 eval/standard/run_all.py'}

  echo "== exec tier: PR #$pr, user $PRRUN_USER($uid:$gid), timeout ${timeout}s, tree ${tmp}/repo =="
  # env -i via null-delimited args + setpriv drop + hard timeout.
  local envargs=()
  while IFS= read -r -d '' tok; do envargs+=("$tok"); done < <(build_env "$home")
  setpriv --reuid="$uid" --regid="$gid" --clear-groups --inh-caps=-all \
    "${envargs[@]}" \
    timeout -k 10 --signal=TERM "$timeout" \
    bash -c "cd \"\$1\" && { ${runtests}; rc=\$?; echo __EXEC_RC=\$rc; exit \$rc; }" _ "${tmp}/repo"
  local rc=$?
  if [ "$rc" = 124 ] || [ "$rc" = 137 ]; then
    echo "__EXEC_TIMEOUT__ after ${timeout}s" >&2
  fi
  return "$rc"
}

# ---- arg parsing ----------------------------------------------------------
case "${1:-}" in
  --print-allowlist) print_allowlist; exit 0;;
  --print-denylist) printf '%s\n' "${DENYLIST[@]}"; exit 0;;
  --check-gate)
    pr=${2:?pr number}; shift 2; repo="Parswanadh/jadal-qa"
    while [ $# -gt 0 ]; do case "$1" in --repo) repo=$2; shift 2;; *) shift;; esac; done
    cmd_check_gate "$pr" "$repo"; exit $?;;
  --dry-run)
    pr=${2:?pr number}; shift 2; repo="Parswanadh/jadal-qa"; timeout=$DEFAULT_TIMEOUT
    while [ $# -gt 0 ]; do case "$1" in --repo) repo=$2; shift 2;; --timeout) timeout=$2; shift 2;; *) shift;; esac; done
    cmd_run "$pr" "$repo" "$timeout" 1; exit $?;;
  run)
    pr=${2:?pr number}; shift 2; repo="Parswanadh/jadal-qa"; timeout=$DEFAULT_TIMEOUT
    while [ $# -gt 0 ]; do case "$1" in --repo) repo=$2; shift 2;; --timeout) timeout=$2; shift 2;; *) shift;; esac; done
    cmd_run "$pr" "$repo" "$timeout" 0; exit $?;;
  *) usage; exit 1;;
esac
