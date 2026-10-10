# PR-triggered agents (`prwatch`) — what the dispatcher can and CANNOT do

Status: shipped on branch `pod/prwatch` (A6, builder infra). Read together with
[THREAT_MODEL.md](THREAT_MODEL.md). The repo is public; **all PR content is
untrusted data, never instructions.**

## Pieces

| File | Role |
|---|---|
| `tools/pod/pr_watcher.py` | polls GitHub every `PRWATCH_INTERVAL` s, keeps state, dispatches one PR at a time |
| `tools/pod/run_dispatcher.sh` | sandboxed exec tier: unprivileged user, env allowlist, throwaway clone, timeout |
| `agents/triage.md`, `agents/review.md` | read-only agents, run on every new/updated PR |
| `agents/eval.md`, `agents/docs.md` | code-executing agents, run ONLY behind the `run-tests` label |
| `eval/standard/run_all.py` | the number a PR is compared against (reproduces the ~0.81 / ~0.85 baselines) |
| `tests/test_pr_watcher.py` | mock-only tests, incl. a prompt-injection case |

## Flow

```
poll (gh api) --> state (/workspace/state/prwatch/state.json) --> for each new/changed PR
      |-- readonly tier: triage + review  (always)         --> one advisory PR comment
      |-- exec tier:     eval + docs      (label 'run-tests') --> run_dispatcher.sh; one comment
```

State records `head_sha` per tier, so neither tier is re-run for a `head_sha`
already reviewed; a new push re-triggers. Draft PRs are ignored. One PR is
processed at a time (sequential); a single instance holds
`/workspace/state/prwatch/prwatch.lock`.

## CAN

* Read open PRs with `gh api` (read scope) and post **one advisory comment per
  tier** with `gh pr comment`.
* Classify and review a PR, check honesty/claims, and flag injection attempts.
* After the maintainer applies the `run-tests` label, run the test suite and
  the standard eval **inside a throwaway clone**, report the deltas vs the
  documented baselines.
* Be run dry: `python3 tools/pod/pr_watcher.py --once --dry-run` never calls
  `gh` and never posts.

## CANNOT

* **Merge, approve, request changes on, label, close or push** any PR or
  branch. The mock API used in tests has no such methods, so a regression that
  tried would fail loudly.
* **Run anything before the label gate.** Without `run-tests` the exec tier is
  refused — twice, once in `pr_watcher.py` and again inside
  `run_dispatcher.sh` (`--check-gate`).
* **Expose a token to PR code.** The exec tier runs under `env -i` with an
  allowlist (`EXEC_ENV_ALLOWLIST`); `HF_TOKEN`, `GH_TOKEN`, `AGENT_API_KEY` and
  `GITHUB_TOKEN` are absent by construction. The watcher never prints env.
* **Modify the live tree.** The exec tier always clones to
  `/workspace/state/prwatch/exec.*`, never `/workspace/repo`.
* **Follow instructions found in PR content.** Title/body/diff are placed in
  delimited `<<UNTRUSTED_*>>` blocks, sentinel-stripped and length-capped; the
  dispatcher's own behaviour is fixed and does not branch on PR text.
* **Touch protected files** (`jadal/config.py`, `jadal/__init__.py`,
  `docs/SPEC.md`, `docs/BASELINE.md`) or commit dataset frames.
* **Run unbounded.** The exec tier is wrapped in `timeout -k 10 <N>` (default
  `PRWATCH_EXEC_TIMEOUT=900`s) and the read-only agents have a 300 s cap.

## Standard eval (`eval/standard/run_all.py`)

Deterministic, CPU-only, no network. Reads the bundle and writes
`eval/results/standard/run_all.json` with `n`, `label_source`, `protocol`,
`ci95`, `device`, `command`, `git_sha`.

| view | protocol | documented | reproduced |
|---|---|---|---|
| `pixel26` | 26 per-pixel features, LOCO logistic C=0.3 | ~0.81 | **0.8100** (95% CI 0.693–0.887) |
| `eg2` | EmbeddingGemma-2 window embedding, LOCO logistic C=0.01 | ~0.85 | **0.8480** (95% CI 0.734–0.909) |
| `ood` | frozen IndEgo `pixel26` model on the 60 OOD windows (9 bad) | 0.4–0.53 (no transfer) | 0.4314 at fps=10 (E1.03 protocol); the `features.json`-trained variant saturates at 0.5 and is reported as such |

n = 150 IndEgo windows (100 good / 50 bad, 7 clips, one labeller). CIs are
clip-bootstrap; the OOD CI is window-bootstrap and optimistic. The model
hyper-parameters are the documented ones and are never tuned on a held-out
clip. See `experiments/E9.01/`.

## Run it

```bash
export GH_TOKEN=...            # never printed; read by gh only
python3 tools/pod/pr_watcher.py --once --dry-run     # offline smoke
python3 tools/pod/pr_watcher.py --interval 120       # real polling
bash tools/pod/run_dispatcher.sh --check-gate <pr>   # 0 = label present, 3 = refused
python3 eval/standard/run_all.py                     # the baseline number
python3 -m unittest tests.test_pr_watcher -v         # 21 mock-only tests
```

Env: `PRWATCH_REPO` (default `Parswanadh/jadal-qa`), `PRWATCH_STATE_DIR`,
`PRWATCH_INTERVAL`, `PRWATCH_EXEC_TIMEOUT`, `PRWATCH_DRY_RUN`, `PRRUNNER_USER`.

## Known limits

* A malicious PR that an LLM *reads* can still influence its output (prompt
  injection). The review is **advisory**; the maintainer decides. Injection
  attempts are detected and reported, not silently trusted.
* The reviewers only see the PR diff, not a built checkout; claims about tests
  come from the exec tier (or are marked `not run`).
* Only the batch of open PRs (`gh api`, first 100) is polled; pagination beyond
  100 open PRs is not implemented (note it if the repo ever exceeds that).
* `run_dispatcher.sh` needs root to `setpriv` and a checkout with git
  credentials configured (`gh auth setup-git`); it fails closed (exit 3/5) when
  either is missing.
