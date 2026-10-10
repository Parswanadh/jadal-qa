# Threat model: PR-triggered agents on a public repo

Assets: tokens (HF, GH, model key), the pod, the labels, the repo. Attackers: PR authors (code, text, filenames), issue authors, poisoned dependencies.

Controls: (1) polling dispatcher (no inbound ports, no self-hosted Actions runner); (2) the two read-only agents (`triage`, `review`) are registered at `.opencode/agents/<name>.md` with an **enforced** `edit: deny` + narrow-shell permission block, run in a throwaway, hardened clone of the PR head (never the live tree) and treat PR content as data; (3) the **exec tier is the deterministic harness** (`unittest` + `eval/standard/run_all.py`, no LLM agent) and runs only after the maintainer applies the `run-tests` label, as an unprivileged user (default `nobody`, override `PRRUNNER_USER`) with an `env -i` allowlist and a wall-clock `timeout`; (4) results are posted as comments with a token limited to PR/issue write; agents never merge or push to PR branches; (5) the agent instructions are the registered `.opencode/agents/*.md` files and are reviewed like code; (6) `secrets_guard.sh` on every commit; (7) pod tokens are pod-only, short-lived and revoked afterwards.

**What is actually enforced (M1).** The exec tier enforces: a wall-clock limit
(`timeout -k 10 <N>`, default 900 s), the `env -i` allowlist (no token), the
unprivileged user (`setpriv`, default `nobody`, requires root), and a throwaway
clone that is deleted on exit. It does **not** enforce CPU, RAM or disk quotas:
there is no cgroup, `ulimit`, `cpulimit` or filesystem quota. A malicious test
can therefore burn CPU/RAM/disk for up to the timeout. Read-only agents have a
300 s wall-clock cap only. The default unprivileged user is `nobody`
(`PRRUNNER_USER` overrides it); the doc's older `prrunner` name was never the
default in the implementation.

Not protected: a malicious PR that is read by an LLM can still try to influence its output (prompt injection): the verdict is advisory, the maintainer decides; a compromised model provider; GPU side channels.

## PR content is untrusted data (binding)

The repository is public, so **any** of these can be attacker-controlled and must
be handled as data: PR title, body, author login, branch name, commit messages,
diff and file contents, file names/paths, review comments, issue text, and any
linked page. None of them is an instruction to any agent or to the dispatcher.

Implementation (`tools/pod/pr_watcher.py`):

1. **Fixed behaviour.** The dispatcher never branches on PR text. It always
   runs the same agent list (`triage`, `review`) and always builds the command
   from fixed strings. PR fields are never passed to a shell and never become
   argv entries. The exec tier is the fixed harness, not an agent list.
2. **Registered, enforced agents.** Each agent is a file at
   `.opencode/agents/<name>.md`; its body is the instruction and its
   `permission` block appends `edit: deny` and `bash: "*": deny` (plus a short
   read-only shell allowlist) after the project rules. opencode's last-match-wins
   ordering makes those effective, so "read-only" no longer depends on the model
   obeying the prompt. A test asserts both agents resolve and deny `edit`.
3. **Hardened throwaway checkout.** The always-on tier clones the PR head to
   `/workspace/state/prwatch/readonly/pr<N>-<sha>/repo` and then replaces the
   PR's `opencode.json`, `AGENTS.md` and `.opencode/` with the trusted copies
   from the live tree (`harden_checkout()`, `tools/pod/prwatch_agent_config.json`,
   deny-by-default). A PR cannot ship the agent, skill or config it is run with,
   and the live tree is never the cwd. The clone is deleted after the tier.
4. **Delimited, sanitized data.** Each field is placed in a
   `<<UNTRUSTED_<FIELD>_BEGIN>> … <<UNTRUSTED_<FIELD>_END>>` block after
   `sanitize_untrusted()`: NUL/control characters removed, any forged
   `<<UNTRUSTED_…>>` sentinel replaced with `[sentinel-removed]`, length capped.
   The system role text precedes the blocks and states they are data.
5. **Injection detection (report-only).** `detect_injection()` flags
   instruction-like phrases ("ignore your instructions", "print the env / run
   curl", prompt-leak and role-rewrite patterns). Matches are reported as a
   finding; they never change control flow.
6. **No exfiltration path.** The read-only tier spawns no subprocess beyond the
   fixed `git` clone and the single fixed `opencode run`; the exec tier sees only
   the allowlisted environment. `redact_secrets()` runs immediately before
   posting every comment, stripping token-shaped strings and the exact values of
   any sensitive env var, so a model that quotes a token cannot leak it.
7. **Trusted tier keeps real auth (M3).** The read-only agents inherit the real
   `HOME` and provider credentials so opencode can authenticate; the `env -i`
   allowlist that strips every token applies to **PR code in the exec tier
   only**.
8. **Least action.** The only GitHub write is `pr comment`. The mock API used
   in tests exposes no merge/approve/label/push methods, so any future call to
   one fails the test suite.

## Threat → control

| Threat | Control |
|---|---|
| PR text instructs the agent to leak secrets / run commands | fixed behaviour + delimited data + `redact_secrets` + injection flag |
| PR body/diff tries shell injection into the watcher | PR fields never reach a shell or argv |
| PR ships its own `opencode.json` / `.opencode/agents/*.md` with `edit: allow` | `harden_checkout()` deletes them and copies the trusted agent/config files; `edit`/broad `bash` denied by config |
| Malicious PR code runs in CI on the pod | exec tier only after `run-tests`; throwaway clone; `setpriv` drop; `env -i` allowlist; wall-clock timeout |
| PR code reads HF/GH/model tokens | tokens absent from the exec env by construction; `--print-allowlist` proves it |
| Read-only agent tries to edit or run a shell command | agent `permission` block denies `edit` and `bash: "*"`, allows a short read-only list; `opencode debug agents` shows it |
| PR code writes to the live tree or the labels | clone lives under `/workspace/state/prwatch/{readonly,exec}.*`, removed on exit; live tree never the cwd |
| Agent auto-merges or pushes | no merge/approve/push method exists in the API wrapper; tests assert the allowed call set |
| Forged data sentinel closes the data block | `sanitize_untrusted` strips all `<<UNTRUSTED_…>>` sentinels |
| Resource exhaustion / hang | `timeout -k 10` on the exec tier; 300 s cap on read-only agents; one PR at a time; instance lock. **No CPU/RAM/disk quota is enforced** (see above) |

## Residual risk (not protected)

* Prompt injection can still bias a *review*; the review is advisory and the
  maintainer decides. This is why the exec tier is gated by a human label.
* No CPU/RAM/disk limits: a labelled PR can consume the pod's CPU/RAM/disk for
  up to the timeout.
* A compromised model provider or a poisoned dependency.
* GPU side channels.
* The watcher trusts `gh`'s authentication and the filesystem permissions of
  `/workspace/state/prwatch`; an attacker with root on the pod is out of scope.
