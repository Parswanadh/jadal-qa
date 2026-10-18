#!/usr/bin/env bash
# One-shot, idempotent pod bootstrap. Secrets come ONLY from env: HF_TOKEN (read access to the gated dataset
# builddotai/Egocentric-10K and the private dataset Havoc1904/jadal-pod-data), GH_TOKEN (fine-grained, this repo only).
# usage: bash bootstrap.sh            (run from anywhere; logs to /workspace/logs/bootstrap.log)
set -u
W=/workspace; REPO=$W/repo; BUNDLE=$W/jadal_pod_bundle; mkdir -p $W/{logs,state/pids,wt,artifacts}; LOG=$W/logs/bootstrap.log
say() { echo "[$(date +%T)] $*" | tee -a "$LOG"; }
need() { [ -n "${!1:-}" ] || { say "MISSING env $1 (set it in the terminal, never in a file)"; return 1; }; say "env $1: set (hidden)"; }
miss=0; need HF_TOKEN || miss=1; need GH_TOKEN || miss=1; [ $miss = 1 ] && { say "stopping: set the variables above"; exit 1; }
export HF_HUB_DISABLE_XET=1 OPENCODE_DISABLE_AUTOUPDATE=1 JADAL_BUNDLE=$BUNDLE JADAL_DATA=$BUNDLE/data PYTHONNOUSERSITE=1
say "system packages"; (apt-get update -qq && apt-get install -y -qq tmux git git-lfs ffmpeg jq rsync >/dev/null) 2>>"$LOG" || say "apt failed (not root? continue)"
say "GitHub CLI (needed for PRs; uses GH_TOKEN from the environment)"
command -v gh >/dev/null 2>&1 || { mkdir -p /usr/share/keyrings && curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg -o /usr/share/keyrings/githubcli-archive-keyring.gpg 2>>"$LOG" && chmod go+r /usr/share/keyrings/githubcli-archive-keyring.gpg && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" > /etc/apt/sources.list.d/github-cli.list && apt-get update -qq 2>>"$LOG" && apt-get install -y -qq gh >/dev/null 2>>"$LOG"; } || say "gh install failed (PR creation will need the REST API)"
command -v gh >/dev/null 2>&1 && say "gh $(gh --version | head -1)"
say "python deps"; python3 -m pip install -q -r "$(dirname "$0")/requirements-pod.txt" 2>>"$LOG" || say "pip install had errors, see log"
say "data bundle (private HF dataset Havoc1904/jadal-pod-data)"
python3 - <<PY 2>>"$LOG" || say "bundle download FAILED (does the HF token include this private repo?)"
from huggingface_hub import snapshot_download
snapshot_download('Havoc1904/jadal-pod-data', repo_type='dataset', local_dir='$BUNDLE')
print('bundle ok')
PY
[ -f "$BUNDLE/scripts/remap_paths.py" ] && python3 "$BUNDLE/scripts/remap_paths.py" --bundle "$BUNDLE" --data "$JADAL_DATA" | tee -a "$LOG"
say "clone repo (token never printed or stored in the remote URL after clone)"
if [ ! -d "$REPO/.git" ]; then git clone -q "https://x-access-token:${GH_TOKEN}@github.com/Parswanadh/jadal-qa.git" "$REPO" 2>>"$LOG" && git -C "$REPO" remote set-url origin https://github.com/Parswanadh/jadal-qa.git; fi
git -C "$REPO" config credential.helper '!f() { echo username=x-access-token; echo "password=$GH_TOKEN"; }; f'
git -C "$REPO" config user.name "${GIT_AUTHOR_NAME:-jadal-pod}"; git -C "$REPO" config user.email "${GIT_AUTHOR_EMAIL:-123052774+Parswanadh@users.noreply.github.com}"
git -C "$REPO" fetch -q origin 2>>"$LOG"
say "environment report -> docs/pod/ENVIRONMENT.md (template filled)"
{ echo "# Environment (generated $(date -u +%FT%TZ))"; echo '```'; nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv 2>&1; echo "nproc: $(nproc)"; free -g | sed -n 1,2p; df -h /workspace | tail -1; echo "python: $(python3 --version)"; python3 -c "import torch;print('torch',torch.__version__,'cuda',torch.version.cuda,torch.cuda.is_available())" 2>&1; python3 -c "import transformers;print('transformers',transformers.__version__)" 2>&1; ffmpeg -hide_banner -decoders 2>/dev/null | grep -E "cuvid" | head -4; echo "opencode: $(opencode --version 2>&1 | head -1)"; echo '```'; } > "$W/state/ENVIRONMENT.md"; cat "$W/state/ENVIRONMENT.md" | tee -a "$LOG" | head -20
say "default model: set JADAL_MODEL (provider/model#variant). Available deepseek models:"; opencode models 2>/dev/null | grep -i deepseek | head -8 | tee -a "$LOG"
cat > $W/env.sh <<ENV
export HF_HUB_DISABLE_XET=1 OPENCODE_DISABLE_AUTOUPDATE=1 PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4
export JADAL_BUNDLE=$BUNDLE JADAL_DATA=$BUNDLE/data JADAL_EXP_DIR=$REPO/experiments
# export JADAL_MODEL=<provider/model#variant>   # set in your shell; see 'opencode models'
ENV
say "done. Next: source $W/env.sh ; bash $BUNDLE/opencode_kit/tools/pod/install_kit.sh"
