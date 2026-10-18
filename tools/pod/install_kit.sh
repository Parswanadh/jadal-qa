#!/usr/bin/env bash
# Install the opencode kit (AGENTS.md, opencode.json, .opencode/{agents,skills,commands}, tools/pod, docs/pod) into the repo on branch pod/kit,
# verify that opencode v2 sees the agents/config, commit and push. Idempotent.
set -u
BUNDLE=${JADAL_BUNDLE:-/workspace/jadal_pod_bundle}; REPO=${REPO:-/workspace/repo}; KIT=$BUNDLE/opencode_kit
[ -d "$KIT" ] || { echo "kit not found at $KIT (download the bundle first)"; exit 1; }
cd "$REPO" || exit 1; git fetch -q origin; git checkout -q -B pod/kit origin/main 2>/dev/null || git checkout -q -B pod/kit
rsync -a "$KIT"/ "$REPO"/ --exclude 'README_KIT.md'
mkdir -p experiments; [ -f experiments/registry.jsonl ] || : > experiments/registry.jsonl
[ -f tools/exp.py ] || cp "$BUNDLE/scripts/exp.py" tools/exp.py
chmod +x tools/pod/*.sh 2>/dev/null
grep -q '^artifacts/' .gitignore 2>/dev/null || printf '\n# pod\nartifacts/\n*.mp4\n*.npz\n*.npy\n.scratch/\n' >> .gitignore
echo "--- verify opencode (v2):"; opencode --version; opencode debug agents 2>&1 | grep -E "lead|experimenter|explorer|reviewer|tester|docs|infra" | head -8; opencode debug config 2>&1 | head -12
bash tools/pod/secrets_guard.sh --tree || { echo "secrets found: not committing"; exit 2; }
git add -A && git commit -q -m "pod: install opencode kit (agents, skills, commands, scripts, docs)" && git push -q -u origin pod/kit && echo "pushed pod/kit" || echo "nothing to commit or push failed"
