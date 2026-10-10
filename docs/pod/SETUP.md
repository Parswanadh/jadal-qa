# Setup on the pod (A40 46 GB; verified facts in [ENVIRONMENT.md](ENVIRONMENT.md))
1. In the pod terminal set (never write them to files): `export HF_TOKEN=...` (pod-only, read-only; must include the gated dataset `builddotai/Egocentric-10K` and the private dataset `Havoc1904/jadal-pod-data`), `export GH_TOKEN=...` (fine-grained, this repo only: contents + pull requests write, expiry <= 24 h), `export AGENT_API_KEY=...` (model provider key for opencode; configure with `opencode auth` — **not set on this pod**, see ENVIRONMENT.md), `export JADAL_MODEL='<provider/model#variant>'` (`/workspace/env.sh` already sets `opencode-go/deepseek-v4.1-flash`; see `opencode models`).
2. Bootstrap: `bash tools/pod/bootstrap.sh` (on the initial setup, download the script from the HF dataset repo: `hf download Havoc1904/jadal-pod-data --repo-type dataset --include 'opencode_kit/tools/pod/*' --local-dir /workspace/jadal_pod_bundle`). It installs packages, downloads the bundle, remaps paths, clones the repo, records the environment (`/workspace/state/ENVIRONMENT.md`).
3. Install the kit: `source /workspace/env.sh && bash /workspace/jadal_pod_bundle/opencode_kit/tools/pod/install_kit.sh` (copies agents/skills/config into the repo on branch `pod/kit`, verifies `opencode debug agents`, pushes).
4. Pin opencode: `opencode --version` (tested on v2.0.26); `OPENCODE_DISABLE_AUTOUPDATE=1` and `"autoupdate": false`. Do not upgrade mid-run (an automatic v1 -> v2 switch killed workers once).
5. Start autosave: `tmux new -d -s autosave 'bash /workspace/repo/tools/pod/autosave.sh'`.
6. Start the lead: `cd /workspace/repo && opencode run --auto --standalone --agent lead --title lead "$(cat /workspace/jadal_pod_bundle/prompts/PROMPT_4_OPENCODE_POD.md)"` (or open the TUI with `opencode --auto`, switch to the `lead` agent and paste the prompt).
Smoke tests (CPU only, n = 150 human-labelled IndEgo windows, leave-one-clip-out; full outputs in `experiments/baselines/`):
`OMP_NUM_THREADS=1 python3 tools/port/baseline_logistic_loco.py` → pooled AUROC 0.810
(`experiments/baselines/2026-10-10_baseline_logistic_loco.txt`) and
`OMP_NUM_THREADS=1 python3 tools/port/ensemble_views_exploratory.py` → EG2 view 0.848
[0.734, 0.909], probability average 0.853 [0.731, 0.922]
(`experiments/baselines/2026-10-10_ensemble_views_exploratory.txt`).
