## What

<!-- One sentence. One concern per PR. -->

## Why

<!-- The problem, or the question this answers. Link the issue. For a protected file, the issue is required. -->

Issue: #

## How measured

<!-- For any number in this PR: the results file (path), n, label source, protocol, CI, device. Write "no numbers" if there are none. -->

- Results file:
- n:
- Label source (human / generated / synthetic, and by whom):
- Protocol (split, what was fitted on which data, threshold selection):
- Confidence interval:
- Device (CPU / GPU, cores, RAM):

## Data regime

<!-- Which data this applies to: IndEgo, Egocentric-10K, Awign footage, synthetic, or none. Regime-specific models or profiles must be named. -->

## Experiment

- [ ] Not an experiment PR, or the experiment is registered with `tools/exp.py` (`experiments/registry.jsonl` and a folder under `experiments/`).

## Honesty checklist

- [ ] Every number in this PR is backed by a results file committed on this branch.
- [ ] Negative or inconclusive results are kept and stated, not removed.
- [ ] Limits and what is not measured are written down.
- [ ] No dataset frames, clips or thumbnails are committed.
- [ ] Any change to the verdict path is deterministic (fixed seeds, no wall-clock time, stable ordering).

## Protected files

- [ ] This PR does not touch `jadal/config.py`, `jadal/__init__.py`, `docs/SPEC.md` or `docs/BASELINE.md`.
- [ ] It does touch one of them, and the linked issue approves it: #

## Secrets

- [ ] No keys, tokens, passwords or private keys in the diff, the commit messages or the history.
- [ ] `tools/oss_audit.sh` was run and reports no secret pattern.
