# Workflow rules (binding; also in AGENTS.md and skill jadal-git-workflow)
- One worktree + one branch per concern under /workspace/wt; commit small and often (`<track>:` prefix); push at every milestone and at least every 15 min.
- Safety net: `tools/pod/autosave.sh` pushes snapshots to `wip/<branch>` every 5 min (secrets-guarded). Real history = your commits.
- Never push to main, never force-push PR branches, never merge your own PR, never touch `jadal/config.py`, `jadal/__init__.py`, `docs/SPEC.md`, `docs/BASELINE.md`.
- Definition of done: code + tests (< 30 s, no network) + results JSON (n, label source, protocol, CI, device, command) + registry entry + doc page + commit + push (+ PR with an honest body).
- Documentation duties: feature doc, per-experiment README (automatic), `experiments/EXPERIMENTS.md` (`python3 tools/exp.py render`), `STATUS.md` every 15 min, this docs/pod index.
