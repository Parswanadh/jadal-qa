# Environment (verified on the pod, 2026-10-10 ~15:16 UTC)

Every line below was produced by the command, or read from the file, shown with it, on this pod, on 2026-10-10.
The bootstrap-time copy of this report is `/workspace/state/ENVIRONMENT.md` (generated
2026-10-10T14:38:36Z by `tools/pod/bootstrap.sh`); where the two differ the difference is
called out. Nothing here is a performance claim.

## Hardware and host
| Fact | Value | Command / file |
|---|---|---|
| Host / volume | `ca-mtl-1-vast.runpod.net:/runpodfs/podvolumes/p45xfompxjl9/ztb434oig718bl` mounted at `/workspace` | `df -h /workspace` |
| GPU | NVIDIA A40, **46,068 MiB (46 GB)**, driver 580.173.02, 0 MiB used at check time | `nvidia-smi --query-gpu=name,memory.total,driver_version` |
| GPU decode (NVDEC) | `h264_cuvid`, `hevc_cuvid`, `av1_cuvid`, `mjpeg_cuvid`, `vp9_cuvid`, … present | `ffmpeg -hide_banner -decoders \| grep cuvid` |
| CPUs | `nproc` prints **4** with `/workspace/env.sh` sourced, but that is coreutils `nproc` honouring `OMP_NUM_THREADS=4` from that file, not the machine size: `nproc --all` = 96, `/proc/cpuinfo` = 96 processors, cgroup v2 `/sys/fs/cgroup/cpu.max` = `765000 100000` (≈ **7.65 cores** quota) | `nproc`, `nproc --all`, `grep -c ^processor /proc/cpuinfo`, `cat /sys/fs/cgroup/cpu.max` |
| Load at check time | load average 11–12 on the shared host (above the cgroup quota) | `uptime` |
| RAM | 503 GB total, 208 GB free, 424 GB available, no swap | `free -g` |
| Disk | `/workspace` 94 GB, 17 GB used, **77 GB free** (bootstrap measured 93 GB free at 14:38Z) | `df -h /workspace` |

**Correction to the bootstrap report:** it records `nproc: 96` (it ran before/without
`OMP_NUM_THREADS`); the CPU budget that matters is the cgroup quota of ≈7.65 cores, so
"CPU load ≤ 75 % of cores" (AGENTS.md rule 10) means ≈5.7 busy cores, not 3.

## Software
| Fact | Value | Command |
|---|---|---|
| opencode | v2.0.26 | `opencode --version` |
| GitHub CLI | 2.102.0 | `gh --version` |
| Python | 3.12.3 | `python3 --version` |
| torch | 2.8.0+cu128, CUDA 12.8, `cuda.is_available() == True` | `python3 -c "import torch; …"` |
| transformers | 5.19.0 | `python3 -c "import transformers; …"` |
| env file | `/workspace/env.sh` exports `OMP_NUM_THREADS=4`, `JADAL_BUNDLE`, `JADAL_DATA`, `JADAL_EXP_DIR=/workspace/repo/experiments`, `JADAL_MODEL='opencode-go/deepseek-v4.1-flash'`, `JADAL_MAX_WORKERS=6` | `cat /workspace/env.sh` |

Two gotchas that follow from `env.sh` (both observed today):
- `JADAL_EXP_DIR` points at `/workspace/repo/experiments`, so `python3 tools/exp.py render`
  run from any worktree rewrites `/workspace/repo/experiments/EXPERIMENTS.md`. In a
  worktree, use `JADAL_EXP_DIR="$PWD/experiments" python3 tools/exp.py render`, or
  `python3 tools/pod/union_registry.py` (see [EXPERIMENT_LOGGING.md](EXPERIMENT_LOGGING.md)).
- `JADAL_MAX_WORKERS=6` in `env.sh` overrides the `4` default inside
  `tools/pod/launch_workers.sh` for every shell that sources it.

## Agents, skills, commands, model
- Agents (`.opencode/agents/*.md`): `lead`, `experimenter`, `explorer`, `reviewer`, `tester`, `docs`, `infra`.
- Skills (`.opencode/skills/*/SKILL.md`): `jadal-context`, `jadal-data-map`, `jadal-experiment-protocol`, `jadal-loco-evaluation`, `jadal-git-workflow`, `jadal-pod-resources`, `jadal-honesty`, `jadal-review-format`, `jadal-parallel-agents`, `jadal-mosaic-spec`.
- Commands (`.opencode/commands/*.md`): `/exp-new`, `/exp-finish`, `/status`, `/review-pr`, `/sync-data`.
- Model in the pod shell: `JADAL_MODEL='opencode-go/deepseek-v4.1-flash'` (from `env.sh`).
- `AGENT_API_KEY` is **not set** in the pod shell (check: `echo "${AGENT_API_KEY:+yes}"`
  prints nothing):
  anything that expects a separate provider key for a headless worker will fail until the
  operator exports one; the interactive session above runs on the `opencode-go` provider.

## GPU slots and the autosave safety net
- `bash tools/pod/gpu_run.sh <cmd> [args…]` holds one of **2** slots (`JADAL_GPU_SLOTS`
  default 2, `JADAL_GPU_WAIT` default 2400 s), locks in `/workspace/state/gpu.slotN.lock`
  (`tools/pod/gpu_run.sh`). Keep total VRAM across jobs ≤ ~40 GB (AGENTS.md rule 10) — the
  card has 46,068 MiB.
- Autosave: tmux session **`autosave`** (1 window, created 2026-10-10 15:07:50 UTC) runs
  `tools/pod/autosave.sh`, which pushes a `wip/<branch>` snapshot of every worktree every
  300 s (`tmux ls`; `tools/pod/autosave.sh`). A second session `cf` is also open.

## Data bundle `/workspace/jadal_pod_bundle` (16 GB data)
`README.md`, `SHA256SUMS.txt`, `data/` (`indego/`, `ego10k_scale/`), `caches/`
(`gpu/features.json`, `ensemble/`, `frames10k/` incl. `segments.npz`, `perpixel/`),
`labels/` (`manifest_gold.json`, `ood_labels.json`), `docs/` (incl. `FACTS.md`),
`queues/`, `scripts/`, `experiments_local/`, `local_work/`, `opencode_kit/`, `prompts/`
(`find /workspace/jadal_pod_bundle -maxdepth 2`).
- `labels/ood_labels2.json` is **absent** (only `ood_labels.json`, 60 windows: 51 good /
  9 bad, human) — the second out-of-domain batch is not there, so the E1.03 and E3.02
  re-runs that wait for `n_bad ≈ 24` stay blocked (`ls /workspace/jadal_pod_bundle/labels`).

## Baselines reproduce (CPU, no GPU)
Run in this worktree on 2026-10-10; full outputs in the files below, both
`n = 150` human-labelled IndEgo windows from 7 clips, leave-one-clip-out.
- `OMP_NUM_THREADS=1 python3 tools/port/baseline_logistic_loco.py` →
  pooled LOCO AUROC **0.810** (good-vs-semantic-bad 0.797, good-vs-quality-bad 0.846;
  `pallet`/`pallet2` at 0.50). Output:
  `experiments/baselines/2026-10-10_baseline_logistic_loco.txt`.
- `OMP_NUM_THREADS=1 python3 tools/port/ensemble_views_exploratory.py` →
  EG2 mean-embedding logistic **0.848** [0.734, 0.909]; 26 per-pixel features 0.810
  [0.694, 0.889]; probability average **0.853** [0.731, 0.922]; rank average 0.856
  [0.722, 0.925]. Output:
  `experiments/baselines/2026-10-10_ensemble_views_exploratory.txt`.
- `docs/pod/SETUP.md` used to point at `scripts/baseline_logistic_loco.py`; the script
  lives at `tools/port/baseline_logistic_loco.py` (corrected there today).

## Not measured / not claimed on this pod
- Raspberry Pi: no Pi hardware; anything about Pi speed is a laptop-core proxy (the pod
  bundle's `/workspace/jadal_pod_bundle/docs/FACTS.md`, which is **not** in this repo, and
  the in-repo `docs/reports/edge.md`, which says "NOT DONE: never run on a Pi"). `not measured`.
- Hyperspectral: RGB cameras only (in-repo: `README.md` line 18, `docs/ARCHITECTURE.md` line 26; pod bundle FACTS: `/workspace/jadal_pod_bundle/docs/FACTS.md`).
- Signed receipts: receipts are SHA-256 of canonical JSON, **not** signed
  (`README.md` line 39, `jadal/receipts.py`).
- End-to-end throughput of the pipeline on this pod with a locked GPU: `not measured` here
  (the descriptive numbers live in the pod bundle's `/workspace/jadal_pod_bundle/docs/FACTS.md`,
  which is not part of this repo).
