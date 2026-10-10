# Experiment logging
Tool: `python3 tools/exp.py new|finish|list|render`. Registry: `experiments/registry.jsonl` (append-only; latest line per id wins); per-experiment folder `experiments/<id>/README.md`; rendered log `experiments/EXPERIMENTS.md`. Fields: id, track, title, **why**, hypothesis, data + label source + n, **data regime**, method, owner, verdict (adopt|maybe|drop|inconclusive), summary with numbers + CI, results path, decision, cost. Negative results are logged. Results JSON: n, label_source, protocol, ci95, device, cost, command, git_sha. Tracks: E1 fps sweep, E2 mosaic, E3 temporal policy, E4 regimes/profiles, E5 domain adaptation, E6 motion cue, E7 cost model, E8 big-GPU search/re-scoring.

**One log from many worktrees**: `python3 tools/pod/union_registry.py` merges
`experiments/registry_snapshot.jsonl` (the historical laptop log) with the live
`experiments/registry.jsonl` (later source wins per id, same rule as `exp.py`), optionally
renders `experiments/EXPERIMENTS.md`, and with `--seed` appends the snapshot lines whose id
the live registry does not have yet (`--seed` is append-only and idempotent). It is the
supported way to keep the log complete when several worktrees each write their own
registry; the rendered file is byte-identical to what `tools/exp.py render` produces from
the same merged registry. Tests: `tests/test_union_registry.py`.

**Gotcha**: `JADAL_EXP_DIR` in `/workspace/env.sh` is `/workspace/repo/experiments`, so
`python3 tools/exp.py render` from a worktree rewrites the *main* checkout's
`EXPERIMENTS.md`. Inside a worktree use `JADAL_EXP_DIR="$PWD/experiments" python3 tools/exp.py render`
(or the union tool) and check `git status` afterwards ([ENVIRONMENT.md](ENVIRONMENT.md)).
