# PR-triggered agents (`prwatch`) — what the dispatcher can and CANNOT do

Status: shipped on branch `pod/prwatch` (A6, builder infra). Read together with
[THREAT_MODEL.md](THREAT_MODEL.md). The repo is public; **all PR content is
untrusted data, never instructions.**

## Pieces

| File | Role |
|---|---|
| `tools/pod/pr_watcher.py` | polls GitHub every `PRWATCH_INTERVAL` s, keeps state, dispatches one PR at a time |
| `.opencode/agents/triage.md`, `.opencode/agents/review.md` | the two read-only agents the watcher invokes by name (`--agent triage` / `--agent review`); their frontmatter denies `edit` and every shell command except a short read-only allowlist |
| `tools/pod/prwatch_agent_config.json` | the trusted, deny-by-default opencode config copied into the throwaway checkout of the PR head |
| `tools/pod/run_dispatcher.sh` | sandboxed **exec tier**: unprivileged user, `env -i` allowlist, throwaway clone, wall-clock timeout |
| `eval/standard/run_all.py` | the number a PR is compared against (reproduces the ~0.81 / ~0.85 baselines) |
| `tests/test_pr_watcher.py` | mock-only tests (31), incl. prompt injection, agent resolution, sandbox and checkout hardening |
| `tests/test_standard_eval_order.py` | OOD feature-order alignment tests (M4) |

## Flow

```
poll (gh api) --> state (/workspace/state/prwatch/state.json) --> for each new/changed PR
      |-- readonly tier: triage + review  (always)         --> one advisory PR comment
      |      runs in a throwaway clone of the PR head, hardened:
      |      the PR's opencode.json / AGENTS.md / .opencode are replaced by the
      |      trusted ones, so a PR cannot ship the agent or config it is run with
      |-- exec tier: deterministic harness (label 'run-tests') --> run_dispatcher.sh; one comment
             `unittest` + `eval/standard/run_all.py`; no LLM agent is involved
```

State records `head_sha` per tier, so neither tier is re-run for a `head_sha`
already reviewed; a new push re-triggers. Draft PRs are ignored. One PR is
processed at a time (sequential); a single instance holds
`/workspace/state/prwatch/prwatch.lock`.

## The agents (I1)

The watcher runs exactly two agents, named `triage` and `review`, as
`opencode run --auto --standalone --agent <name>`. A name only resolves if
opencode finds a registered agent file, so both live at
`.opencode/agents/<name>.md` with `description`, `mode: subagent`, `temperature`
and a `permission` block. There is **one** copy of the instructions: the agent
body. The root-level `agents/*.md` prompt files that were never registered (and
the never-invoked `agents/eval.md` / `agents/docs.md`) have been removed.
`tests/test_pr_watcher.py::TestAgentRegistration` asserts that every name in the
watcher's `ALL_AGENTS` has a file at `.opencode/agents/<name>.md`, that the file
denies `edit` and broad `bash`, and (when the CLI is present) that
`opencode debug agents` lists the name.

## Read-only is enforced by config, not by the prompt (I2)

* Each agent's `.opencode/agents/*.md` frontmatter appends
  `edit: deny`, `task: "*": deny` and `bash: "*": deny` after the project rules,
  then re-allows only `git log/diff/show/status`, `ls`, `cat`, `grep`, `rg`,
  `find`, `wc`. opencode's last-match-wins rule makes those the effective
  permissions (`opencode debug agents` shows them).
* The always-on tier never runs in the live tree: `pr_watcher.py` clones the PR
  head to `/workspace/state/prwatch/readonly/pr<N>-<sha>/repo`, then
  `harden_checkout()` deletes the PR's `opencode.json`, `AGENTS.md` and
  `.opencode/` and copies in the trusted agent/skill definitions and
  `tools/pod/prwatch_agent_config.json` (deny-by-default `edit`/`bash`/`task`/
  `webfetch`). A PR therefore cannot grant itself `edit` or `bash`. The clone is
  removed after the tier; `OpenCodeRunner.run()` raises if `workdir` is `None`.
* The trusted read-only tier keeps the real `HOME` and provider auth so opencode
  can authenticate (M3). The `env -i` allowlist is used for **PR code in the
  exec tier only** — it is never applied to the agents.

## CAN

* Read open PRs with `gh api` (read scope) and post **one advisory comment per
  tier** with `gh pr comment`.
* Classify and review a PR, check honesty/claims, and flag injection attempts.
* After the maintainer applies the `run-tests` label, run the test suite and the
  standard eval **inside a throwaway clone**, and report deltas vs the
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
* **Modify the live tree.** Both tiers work on throwaway clones under
  `/workspace/state/prwatch/{readonly,exec}.*`; the agents' `edit` and non-read
  shell commands are denied by config, and the live tree is never the cwd.
* **Expose a token to PR code.** The exec tier runs under `env -i` with an
  allowlist (`EXEC_ENV_ALLOWLIST`); `HF_TOKEN`, `GH_TOKEN`, `AGENT_API_KEY` and
  `GITHUB_TOKEN` are absent by construction. The watcher never prints env.
* **Follow instructions found in PR content.** Title/body/diff are placed in
  delimited `<<UNTRUSTED_*>>` blocks, sentinel-stripped and length-capped; the
  dispatcher's own behaviour is fixed and does not branch on PR text.
* **Touch protected files** (`jadal/config.py`, `jadal/__init__.py`,
  `docs/SPEC.md`, `docs/BASELINE.md`) or commit dataset frames.
* **Run unbounded.** Both tiers are wrapped in a wall-clock `timeout`
  (exec: `PRWATCH_EXEC_TIMEOUT=900` s; read-only agents: 300 s). See
  [THREAT_MODEL.md](THREAT_MODEL.md) for what is *not* enforced (no CPU/RAM/disk
  limits).

## Standard eval (`eval/standard/run_all.py`)

Deterministic, CPU-only, no network. Reads the bundle and writes
`eval/results/standard/run_all.json` with `n`, `label_source`, `protocol`,
`ci95`, `device`, `command`, `git_sha`.

| view | protocol | documented | reproduced |
|---|---|---|---|
| `pixel26` | 26 per-pixel features, LOCO logistic C=0.3 | ~0.81 | **0.8100** (95% CI 0.693–0.887) |
| `eg2` | EmbeddingGemma-2 window embedding, LOCO logistic C=0.01 | ~0.85 | **0.8480** (95% CI 0.734–0.909) |
| `ood` | frozen IndEgo `pixel26` model on the 60 OOD windows (9 bad) | 0.4–0.53 (no transfer) | 0.4314 at fps=10 (E1.03 protocol); the `features.json`-trained variant saturates at 0.5 and is reported as such |

The OOD feature matrix is reordered into `FEATURE_NAMES` order before
`predict_proba` (or the harness raises), so a regenerated shard cannot silently
misalign it; the result records `feature_names` and `feature_order_reordered`.

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
python3 -m unittest tests.test_pr_watcher tests.test_standard_eval_order -v
```

Env: `PRWATCH_REPO` (default `Parswanadh/jadal-qa`), `PRWATCH_STATE_DIR`,
`PRWATCH_INTERVAL`, `PRWATCH_EXEC_TIMEOUT`, `PRWATCH_DRY_RUN`, `PRRUNNER_USER`.

## Known limits

* A malicious PR that an LLM *reads* can still influence its output (prompt
  injection). The review is **advisory**; the maintainer decides. Injection
  attempts are detected and reported, not silently trusted.
* The reviewers see the PR diff in the prompt and a throwaway checkout of the
  PR head; claims about tests come from the exec tier (or are marked
  `not run`).
* Only the batch of open PRs (`gh api`, first 100) is polled; pagination beyond
  100 open PRs is not implemented (note it if the repo ever exceeds that).
* `run_dispatcher.sh` needs root to `setpriv` and a checkout with git
  credentials configured (`gh auth setup-git`); it fails closed (exit 3/5) when
  either is missing.
