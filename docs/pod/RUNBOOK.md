# Runbook (append new failures here)
| Symptom | Cause seen | Action |
|---|---|---|
| `opencode run` prints only the CLI help | v1 flags (`--dir`, `--variant`) on v2 | `cd <worktree>`; use `-m provider/model#variant` |
| Workers die with "Unexpected server error" at about the same time | opencode auto-updated (v1 -> v2) and the shared background service restarted | pin the version, `OPENCODE_DISABLE_AUTOUPDATE=1`, run workers with `--standalone` |
| Worker log stops growing | session ended or server error | check `launch_workers.sh status`; relaunch with a narrower brief and `--continue`/`-s <id>` |
| 403 on the Egocentric-10K download | HF token lacks gated-repo read permission | edit the token: "Read access to contents of all public gated repos you can access" |
| 403/404 on the bundle download | token not scoped to the private repo `Havoc1904/jadal-pod-data` | add the repo to the fine-grained token |
| Slow decode | NVDEC missing in apt ffmpeg | try another ffmpeg build or torchcodec; record the decode path |
| `cv2.phaseCorrelate` corrupts frames | OpenCV 5.0 mutates inputs when a window argument is passed | pass copies (`prev.copy(), cur.copy()`) |
| `python3 tools/exp.py render` rewrites `/workspace/repo/experiments/EXPERIMENTS.md` from a worktree | `/workspace/env.sh` exports `JADAL_EXP_DIR=/workspace/repo/experiments` | `JADAL_EXP_DIR="$PWD/experiments" python3 tools/exp.py render` (or `tools/pod/union_registry.py`); if it already happened, `git -C /workspace/repo checkout -- experiments/EXPERIMENTS.md` |
| `nproc` says 4 though the host has more | coreutils `nproc` honours `OMP_NUM_THREADS=4` from `/workspace/env.sh`; the real budget is the cgroup quota (`/sys/fs/cgroup/cpu.max` ≈ 7.65 cores, `nproc --all` = 96) | use `nproc --all` and `cpu.max` for sizing ([ENVIRONMENT.md](ENVIRONMENT.md)) |
| the `docs` agent gets "Permission denied: edit" for `tools/**` | `.opencode/agents/docs.md` allows `edit` on `docs/**`, `experiments/**`, `*.md` only | have the lead (or a code agent) add the file, or widen the docs agent's globs deliberately |
| a command that dumps the environment is rejected | the permission layer blocks `env`/`export` dumps (secret-exfiltration guard) | check single variables, e.g. `echo "${AGENT_API_KEY:+yes}"` |
| headless worker for an agent that needs `AGENT_API_KEY` fails | `AGENT_API_KEY` is not set on this pod | export it in the shell that starts the worker ([ENVIRONMENT.md](ENVIRONMENT.md)) |
| E1.03 / E3.02 second-batch re-runs stay blocked | `labels/ood_labels2.json` is absent from the bundle | wait for the second OOD label batch (n_bad ≈ 24) before re-running position/sweep tests |
| `unittest discover` errors on `test_signals`, `test_service`, `test_eval_splits`, `test_integration` | `imagehash` and `fastapi` are not installed and are missing from `tools/pod/requirements-pod.txt` (the rest of the suite runs: 467 tests, 4 import errors, 1 skip, 2026-10-10) | add the two packages to `tools/pod/requirements-pod.txt` and the pod image, or accept the four modules as un-runnable on the pod for now |
