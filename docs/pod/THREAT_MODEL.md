# Threat model: PR-triggered agents on a public repo

Assets: tokens (HF, GH, model key), the pod, the labels, the repo. Attackers: PR authors (code, text, filenames), issue authors, poisoned dependencies.
Controls: (1) polling dispatcher (no inbound ports, no self-hosted Actions runner); (2) triage + review agents are read-only and treat PR content as data; (3) test/eval jobs run only after the maintainer applies the `run-tests` label, as an unprivileged user (`prrunner`) with an env allowlist (`env -i`), time/CPU/RAM/disk limits, no access to `/workspace/secrets`; (4) results are posted as comments with a token limited to PR/issue write; agents never merge or push to PR branches; (5) agent prompts live in the repo under `agents/` and are reviewed like code; (6) `secrets_guard.sh` on every commit; (7) pod tokens are pod-only, short-lived and revoked afterwards.
Not protected: a malicious PR that is read by an LLM can still try to influence its output (prompt injection): the verdict is advisory, the maintainer decides; a compromised model provider; GPU side channels.

## PR content is untrusted data (binding)

The repository is public, so **any** of these can be attacker-controlled and must
be handled as data: PR title, body, author login, branch name, commit messages,
diff and file contents, file names/paths, review comments, issue text, and any
linked page. None of them is an instruction to any agent or to the dispatcher.

Implementation (`tools/pod/pr_watcher.py`):

1. **Fixed behaviour.** The dispatcher never branches on PR text. It always
   runs the same agent list (`triage`, `review`; `eval`, `docs` behind the
   label) and always builds the command from fixed strings. PR fields are never
   passed to a shell and never become argv entries.
2. **Delimited, sanitized data.** Each field is placed in a
   `<<UNTRUSTED_<FIELD>_BEGIN>> … <<UNTRUSTED_<FIELD>_END>>` block after
   `sanitize_untrusted()`: NUL/control characters removed, any forged
   `<<UNTRUSTED_…>>` sentinel replaced with `[sentinel-removed]`, length capped.
   The system role text precedes the blocks and states they are data.
3. **Injection detection (report-only).** `detect_injection()` flags
   instruction-like phrases ("ignore your instructions", "print the env / run
   curl", prompt-leak and role-rewrite patterns). Matches are reported as a
   finding; they never change control flow.
4. **No exfiltration path.** The read-only tier spawns no subprocess at all
   beyond the single fixed `opencode run`; the exec tier sees only the
   allowlisted environment. `redact_secrets()` runs immediately before posting
   every comment, stripping token-shaped strings and the exact values of any
   sensitive env var, so a model that quotes a token cannot leak it.
5. **Least action.** The only GitHub write is `pr comment`. The mock API used
   in tests exposes no merge/approve/label/push methods, so any future call to
   one fails the test suite.

## Threat → control

| Threat | Control |
|---|---|
| PR text instructs the agent to leak secrets / run commands | fixed behaviour + delimited data + `redact_secrets` + injection flag |
| PR body/diff tries shell injection into the watcher | PR fields never reach a shell or argv |
| Malicious PR code runs in CI on the pod | exec tier only after `run-tests`; throwaway clone; `setpriv` drop; `env -i` allowlist; timeout |
| PR code reads HF/GH/model tokens | tokens absent from the exec env by construction; `--print-allowlist` proves it |
| PR code writes to the live tree or the labels | clone lives under `/workspace/state/prwatch/exec.*`, removed on exit; live tree never mounted into the exec tier |
| Agent auto-merges or pushes | no merge/approve/push method exists in the API wrapper; tests assert the allowed call set |
| Forged data sentinel closes the data block | `sanitize_untrusted` strips all `<<UNTRUSTED_…>>` sentinels |
| Resource exhaustion / hang | `timeout -k 10` on the exec tier; 300 s cap on read-only agents; one PR at a time; instance lock |

## Residual risk (not protected)

* Prompt injection can still bias a *review*; the review is advisory and the
  maintainer decides. This is why the exec tier is gated by a human label.
* A compromised model provider or a poisoned dependency.
* GPU side channels.
* The watcher trusts `gh`'s authentication and the filesystem permissions of
  `/workspace/state/prwatch`; an attacker with root on the pod is out of scope.
