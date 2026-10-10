#!/usr/bin/env bash
# Run a command while holding one of N GPU slots on the pod's single A40.
# Prevents two heavy jobs from oversubscribing VRAM. Serialises politely, never kills.
# usage: bash tools/pod/gpu_run.sh <cmd> [args...]
# env: JADAL_GPU_SLOTS (default 2), JADAL_GPU_WAIT (default 2400 s)
set -u
N=${JADAL_GPU_SLOTS:-2}; WAIT=${JADAL_GPU_WAIT:-2400}
mkdir -p /workspace/state
for ((t=0; t<WAIT; t+=5)); do
  for ((s=1; s<=N; s++)); do
    exec {fd}>"/workspace/state/gpu.slot$s.lock" || continue
    if flock -n "$fd"; then
      "$@"; rc=$?
      flock -u "$fd" 2>/dev/null; exec {fd}>&- 2>/dev/null
      exit $rc
    fi
    exec {fd}>&- 2>/dev/null
  done
  sleep 5
done
echo "gpu_run: no free GPU slot after ${WAIT}s (N=$N)" >&2
exit 1
