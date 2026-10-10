# Runbook (append new failures here)
| Symptom | Cause seen | Action |
|---|---|---|
| `opencode run` prints only the CLI help | v1 flags (`--dir`, `--variant`) on v2 | `cd <worktree>`; use `-m provider/model#variant` |
| Workers die with "Unexpected server error" at about the same time | opencode auto-updated (v1 -> v2) and the shared background service restarted | pin the version, `OPENCODE_DISABLE_AUTOUPDATE=1`, run workers with `--standalone` |
| Every worker dies in ~1 s with `Agent not found: "experimenter"` | the worktree branched from `origin/main`, which does NOT contain the opencode kit (`.opencode/agents`, `opencode.json`, `AGENTS.md`, `tools/pod`) — the kit lives only on `pod/kit` | branch worktrees from a base that includes the kit (`pod/base`); fix: `git merge origin/pod/kit` into the base, then `git merge --ff-only pod/base` in each worktree and relaunch |
| `launch_workers.sh start` refuses with "N workers already running" although only a few are | `state/pids/*` also contains the lead and the opencode server PIDs (they count toward `JADAL_MAX_WORKERS`) | raise `JADAL_MAX_WORKERS`, or delete stale pid files for processes you did not start only after checking `ps` |
| A `task`-spawned `reviewer`/`explorer` cannot execute anything ("Permission denied: shell") | the subagent inherits the agent's `bash` allowlist, and `reviewer` is `bash: {"*": deny}` + an allowlist | reviews are then FILE-ONLY evidence; use the `tester` agent (no bash deny) for anything that must be executed, and say in the review which checks could not be run |
| `python3 -m unittest discover -s tests` reports `Ran N tests ... OK` but the new tests never ran | the test file uses bare module-level `test_*` functions (pytest style); `unittest` collects only `TestCase` subclasses | wrap in `unittest.TestCase` (or add `load_tests`); hit twice on this pod (regimes, mosaic) — check `unittest discover -p 'test_<name>*.py' -v` after adding a file |
| `nproc` prints 4 but the box has more | coreutils `nproc` honours `OMP_NUM_THREADS` from `/workspace/env.sh` | read the real budget from `/sys/fs/cgroup/cpu.max` (765000/100000 ≈ 7.65 cores on this pod) |
| A worker log stops growing for minutes | a long single command (model download / 1,500-window decode) prints nothing; not a stall | check for a live child `python` by `/proc/<pid>/cwd` before relaunching |
| Worker log stops growing | session ended or server error | check `launch_workers.sh status`; relaunch with a narrower brief and `--continue`/`-s <id>` |
| 403 on the Egocentric-10K download | HF token lacks gated-repo read permission | edit the token: "Read access to contents of all public gated repos you can access" |
| 403/404 on the bundle download | token not scoped to the private repo `Havoc1904/jadal-pod-data` | add the repo to the fine-grained token |
| Slow decode | NVDEC missing in apt ffmpeg | try another ffmpeg build or torchcodec; record the decode path |
| `cv2.phaseCorrelate` corrupts frames | OpenCV 5.0 mutates inputs when a window argument is passed | pass copies (`prev.copy(), cur.copy()`) |
