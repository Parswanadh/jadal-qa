---
name: jadal-git-workflow
description: Branching, worktrees, continuous commits, pushes, PR rules and documentation duties for this repo on the pod (including the autosave snapshot refs). Load before creating a branch, committing or opening a PR.
license: Apache-2.0
---
1. Clone once: `/workspace/repo` (GitHub `Parswanadh/jadal-qa`, token from `GH_TOKEN`, never printed). One worktree per concern: `git worktree add -b <branch> /workspace/wt/<name> origin/main` (branches: `pod/kit`, `feat/exp-mosaic`, `feat/exp-fps-pod`, `feat/exp-motion-cue`, `feat/exp-regimes-pod`, ...).
2. Commit small and often, message prefix `<track>:`; push after every milestone and at least every 15 minutes (`git push -u origin <branch>`). `tools/pod/autosave.sh` additionally pushes a safety snapshot of every worktree to `wip/<branch>` every 5 minutes (those refs may be force-pushed; PR branches never).
3. Before each commit: `tools/pod/secrets_guard.sh` (blocks tokens/keys), `git status` shows no data/video/npz artefacts (they live in `/workspace/artifacts`, git-ignored).
4. Tests before a PR: `PYTHONNOUSERSITE=1 python -m unittest discover -s tests` (`pico_status` example test fails without a Pico: known).
5. PR: `gh pr create --base main --head <branch> --title "<track>: <what> (<headline number with CI>)" --body "<what changed, how measured (n, label source, protocol, CI, device), limits, files, registry ids>"`; never merge; never push to main; protected files untouched.
6. Documentation duties for every PR: a doc page for the feature (`docs/pod/` or `docs/reports/<name>.md`), the registry entries (`experiments/`), `STATUS.md` current, `docs/pod/README.md` index updated, claims in docs backed by files.
7. After PR feedback: new commits on the same branch (no force-push).
