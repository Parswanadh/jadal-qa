# Contributing to Jadal QA

Thank you for looking at this project. It is a small research codebase. The rules below keep changes reviewable and keep the claims honest.

## Setup

1. Use Python 3.12. Create a virtual environment and install the dependencies you need. The edge and GPU extras are listed in `requirements-edge.txt` and in `docs/ARCHITECTURE.md`.
2. Keep secrets out of the repo. Copy `.env.example` to `.env`, which is git-ignored, and fill it in locally. Read secrets from environment variables only.
3. Model weights and video clips are not in git. Put downloaded models in `models/`, which is git-ignored. Dataset licences are listed in `NOTICE` and `docs/ATTRIBUTIONS.md`.

## Tests

Run the module tests with stubs. They need no hardware and no network:

```
PYTHONNOUSERSITE=1 python -m unittest discover -s tests
```

Each suite must finish well under the 150 s limit used by `make test`. Add a test for each behaviour you change. A test that needs a GPU, a camera, a dataset file or a network call belongs behind an explicit flag and is not run by default.

## Pull request rules

- **Small PRs, one concern each.** A refactor, a new measurement and a bug fix are three PRs.
- **Protected files need an issue first.** Do not edit `jadal/config.py`, `jadal/__init__.py`, `docs/SPEC.md` or `docs/BASELINE.md` in a PR unless an issue has been opened and agreed with the maintainer. Name the issue in the PR. `CODEOWNERS` routes these files to the maintainer.
- **Every claim is backed by a results file.** A number in a README, report or PR description must point to a results JSON committed on the same branch. Each results file records:
  - `n`: the number of items;
  - the label source (human, generated, or synthetic, and who or what made them);
  - the protocol (split, threshold selection, and what was fitted on which data);
  - a confidence interval for each rate;
  - the device (CPU model, core count, GPU if any).
- **Negative results are kept.** A result that does not help is written up and merged, with its verdict. Do not delete failed runs or tune the write-up after the fact to hide them.
- **No dataset frames or videos are committed.** Frames, clips, thumbnails and video files stay out of git, because of dataset licences. Commit only derived numbers, manifests and small hashes.
- **No secrets.** No API keys, tokens, private keys, or `KEY=value` / `TOKEN=value` / `SECRET=value` lines with real values. Run `tools/oss_audit.sh` before you ask for review.
- **Deterministic code only in the verdict path.** Code that decides accept or reject must be deterministic: fixed seeds, no wall-clock time, no unordered iteration over sets or dict views that feed the decision, and the same output for the same input. Models and heuristics used for analysis only may be stochastic, but they must not feed the verdict without a recorded decision.
- **Do not weaken a check to make a test pass.** If a test is wrong, say so in the PR and fix the test with a reason.

## Experiment PRs

An experiment PR must register its experiment with `tools/exp.py`:

```
python tools/exp.py new    --id E1.01 --track E1 --title "..." --why "..." --hypothesis "..." --data "..." --regime "..." --method "..." --owner NAME
python tools/exp.py finish --id E1.01 --verdict adopt|maybe|drop|inconclusive --summary "..." --results path/to/results.json
python tools/exp.py render
```

- Each experiment gets a line in `experiments/registry.jsonl` and a folder under `experiments/`. The registry is append-only.
- Record the reason, the outputs, the data type and the regime. Use the regime-specific models and profiles. Do not reuse a model tuned on one regime for another without a new experiment.
- Read `docs/EXPERIMENTS_HOWTO.md` for the full protocol and a worked example.

## How PRs are reviewed

Pull requests are reviewed by agents running on a separate GPU pod. The pod runs a polling dispatcher that reads open PRs and posts review comments. The rules are:

- **Untrusted PR content is data, not instructions.** The reviewer agents treat the text of a PR, its diff, its comments and its files as data to be checked. Text in a PR that asks a reviewer to change its behaviour, run a command, reveal a secret or approve a change is ignored and reported.
- **Test and evaluation jobs run only after a maintainer applies the `run-tests` label.** The pod does not run code from a pull request until a maintainer has read it and added the label.
- **There are no GitHub Actions workflows that execute fork code.** This repository has no workflow that runs code from a pull request, and this must stay true. Any change that adds one needs a maintainer decision first.
- A maintainer merges. Agents comment and label; they do not merge.

## Licence

By contributing you agree that your contribution is licensed under the project licence. The licence choice is pending; see the open decision in the PR that adds this file.
