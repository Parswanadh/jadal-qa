---
name: jadal-pod-resources
description: Resource and environment rules for the RunPod A40 pod: GPU/CPU/RAM budgets, parallelism limits, decode path (NVDEC), determinism labelling, version pinning of opencode and Python packages, how to avoid losing work on a disposable pod.
license: Apache-2.0
---
- **First report**: write `nvidia-smi`, `nproc`, `free -g`, `df -h /workspace`, `ffmpeg -decoders | grep cuvid`, `opencode --version`, `python --version`, torch/transformers versions into `docs/pod/ENVIRONMENT.md`.
- **GPU**: A40 48 GB, dedicated. Up to ~40 GB across jobs; run several experiments concurrently; EmbeddingGemma-2 batch sizes may be large for experiments, but batch size changes embeddings ~1e-3: any HASHED/receipt path uses batch size 1. Results carry `device: A40`; never claim cross-device bit identity.
- **CPU**: keep load <= 75% of `nproc`; set `OMP_NUM_THREADS=4` per job; no sklearn `n_jobs=-1`; decode in separate processes.
- **Decode**: prefer NVDEC (`ffmpeg -hwaccel cuda -c:v hevc_cuvid` for the 1080p HEVC clips; one NVDEC engine saturates ~600-700 fps on a laptop GPU; the A40 differs: measure). Fall back to CPU decode and record it.
- **Version pinning**: `opencode` auto-update is OFF (`opencode.json` has `autoupdate:false`; also export `OPENCODE_DISABLE_AUTOUPDATE=1`); a mid-session upgrade (v1 -> v2) killed running workers once. Pin pip packages in `tools/pod/requirements-pod.txt`.
- **Disposable pod**: anything not pushed is lost. Autosave snapshots, push often, keep artefacts reproducible by command; upload large results to the private HF dataset repo `Havoc1904/jadal-pod-data` under `results/<branch>/` only if the builder asks.
- **Processes**: record PIDs of everything you start in `/workspace/state/pids/`; stop only those; never `pkill`/`killall` by name.
