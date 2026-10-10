# Fact-check of the LB5 branch-triage report — B1-verify (independent)

**Auditor:** lead-b-verify, branch `b/verify-triage` (scratch worktree `/workspace/wt/b-verify-triage`),
read-only: no branch was rebased, rewritten, pushed to or deleted; `git merge-tree` and `git ls-tree`
write no refs.

**Artifact under test:** `docs/reports/branch_triage.md` on branch `b/triage` (PR #94), tip `7de5055`
(payload `056c68f`).
**Results JSON:** `eval/results/pod_verify/triage_factcheck.json`.
**Live count taken at 16:15:13Z** (report snapshot: 16:00Z-ish, base `origin/main = 65f37dc`).

## Counts

| Verdict | Count |
|---|---|
| TRUE | 26 |
| FALSE | 2 primary entries (**3 false statements** counting one embedded listing error) |
| UNVERIFIABLE | 4 |
| entries checked | 32 |

## FALSE claims, verbatim

1. `docs/reports/exp_regimes.md` is added by five (#81, #82, #83, #87, #88) — **eight** snapshot heads
   carry it (#81, #82, #83, #87, #88, #89, #90, #91); the report misses #89/#90/#91.
2. "eight add `registry_snapshot.jsonl`, `experiments/EXPERIMENTS.md` and `docs/pod/*`" — the
   `experiments/EXPERIMENTS.md` part is **seven** of the nine; `#91 b/verify` has no `EXPERIMENTS.md`
   at any path in its snapshot tree.
3. (embedded in an otherwise TRUE claim) "… and `docs/PUBLIC_SNAPSHOT_EXCLUDE.txt` for #76/#91" — that
   file is a conflict for **#76 only**; #91's 11 conflicts are the 6 shared files plus the five
   `eval/results/exp_temporal/E3.0*.json`.

None of the three is decision-changing (they are count errors in section 4-B/4-A prose, and the
counts are undercounts, not overcounts).

## Most consequential drift (not report errors — the queue moved)

* Open PRs: **28** at 16:01:12–16:02:36Z (report's count, verified) → **30** already at the report
  commit 16:05:09Z → **31** at 16:15:13Z (timestamped count; PRs #92–#99 opened since).
* `origin/main` advanced `65f37dc → f18ca43` when **#76 landed at 16:08:46Z**; **#57, #63, #67**
  merged 16:08:47–48Z and **#84 at 16:14:47Z**. So the report's "0/28 branches are ancestors of main"
  was TRUE at snapshot and is **FALSE now: 5/28 heads are ancestors of main**. Wave 1 and the
  "rebase #76 first / close #57/#63/#67 superseded" advice is now moot (all three had already been
  merged; since they are content-subsets of #76 no work was lost).
* Several open branches were force-pushed by other workers after the report (e.g. `pod/prwatch`,
  `feat/exp-motion-cue`, `feat/exp-cost-model`, `feat/exp-fps`, `docs/pod-docs`, `b/verify`), which is
  why a few live counts no longer match the 16:00Z snapshot. The report's own warning ("the queue is
  live … re-run §1 before acting") is therefore essential, not boilerplate.

## Verification highlights (exact commands in the JSON)

| Claim (short) | Verdict | Evidence |
|---|---|---|
| `handover/BRANCH_TRIAGE_REPORT.md` exists nowhere | TRUE | `git log --all -- 'handover/…'` empty; no ref has a `handover/` tree; `find /` nothing |
| 25 open PRs at 15:57Z / 28 at 16:00Z | TRUE | reconstructed from `createdAt/mergedAt/closedAt` of all 69 PRs: 25 at 15:57Z, 28 during 16:01:12–16:02:36Z (literal 16:00:00Z = 25, label ~1 min early) |
| all base=main, no drafts, no reviews, no reviewers | TRUE | `gh pr list --json isDraft,baseRefName,reviewDecision,reviewRequests` |
| 0/28 branches are ancestors of main | TRUE at snapshot / **FALSE now** | `git merge-base --is-ancestor origin/<b> 65f37dc` → 0; against live `origin/main` → 5 |
| 28/28 branch-only, 1–76 commits ahead | TRUE | `git rev-list --left-right --count 65f37dc...<ref>`: min 1, max 76 (#76 at `7e88bbe^1`) |
| 1,337 branch-only files total | TRUE | reproduced exactly for 22 rows; table sums to 1337; ±3 drift on 5 rewritten branches |
| GitHub 18 MERGEABLE / 10 CONFLICTING | UNVERIFIABLE | GitHub recomputes live (all 31 now UNKNOWN; #68 returns null) |
| local merge-tree 19 clean / 9 conflicting | TRUE | 18 clean + 8 conflicts reproduced; #76 at `1d4c8d5` = 1 conflict → 9 |
| all 18 tabulated tree OIDs | TRUE | byte-identical reproduction via `git merge-tree --write-tree 65f37dc <snapshot-ref>` |
| all "behind main by N" values | TRUE | `git rev-list --left-right --count` reproduced for all 27 stated rows |
| #84 row (16 files, tree `442ce43c8366`) | UNVERIFIABLE | `feat/exp-fps` was rewritten+merged after the report; report-era tip is in no ref (live: 127 files, tree `cb2ce4cd8aef`) |
| §4-A conflicted paths all exist on main; keep main's version is loss-free | TRUE | every conflict path `git cat-file -e 65f37dc:<path>` = YES; branch blobs are the identical stale pod/base port (`4c418d4`) across all 8 branches vs main's #79 review fix (`ff88cb1`) |
| #68 criss-cross (multiple merge bases) | TRUE | `git merge-base --all 65f37dc origin/feat/frames-spectral` → 2 bases |
| #76 contains #57/#63/#67 | TRUE | `git merge-base --is-ancestor` → CONTAINED ×3 |
| no `.github/workflows`, only CodeRabbit check | TRUE | `.github` = 0 files on main; only check name is `CodeRabbit` |
| `make test` = unittest discover tests+web | TRUE | `Makefile:13-15` verbatim |
| pod interpreter absent; only `requirements-edge.txt` | TRUE | `ls -l /home/parshu/envs/jadal/bin/python` → No such file |
| gate run 3 = 493 passed, 0 failed, 3 skipped, 4 errors | TRUE | reproduced from a clean `git archive 65f37dc` outside /tmp: **493 passed, 3 skipped, 4 errors, 90 subtests** in 18.03s |
| 4 collection errors = missing `fastapi`/`imagehash` | TRUE | both modules absent; `test_eval_splits.py:26`/`test_signals.py:23` → `ModuleNotFoundError: imagehash` |
| runs 1–2 numbers (interrupted collection, `/tmp` exit-3 false positive) | UNVERIFIABLE | not re-run; consistent with run 3, no contradiction |
| #94: 132-file diff, CONFLICTING, payload = `056c68f` (2 files) | TRUE | `gh api pulls/94`: changed_files=132, mergeable=false/dirty; `git show --stat 056c68f` = 2 files |
| "no other branch was pushed to / rebased / force-pushed" | UNVERIFIABLE | no audit trail; `b/triage` itself is clean (2 files vs `pod/base`), but other workers did rewrite/push several branches during 16:04–16:15Z |

## Flags on recommendations that touch other people's work

* **"close #57/#63/#67 as superseded once #76 lands"** — evidence *supports* it
  (`git merge-base --is-ancestor` true for all three). In the event they were **merged** at
  16:08:47–48Z rather than closed; because their content is a subset of #76 nothing was lost, and all
  three heads are now ancestors of main. No deletion of work occurred.
* **"keep main's version" for the 9 conflicting PRs** — supported, and stronger than the report
  argued: the branch-side blobs for the 6 shared conflicted files are byte-identical across all 8 open
  branches and equal the pre-review pod/base port, so no branch-unique content is dropped.
* **No recommendation in the report deletes, and none closes a branch without ancestry evidence.**
  The report's refusal to copy other workers' report files onto `b/triage` (and its cherry-pick,
  not merge-wholesale, advice for PR #94) is correct: the branch's diff against main is 132 files
  while its payload is 2.

## Method limits

Historical GitHub `mergeable` values and the section-6 runs 1–2 were not recoverable/re-run; branch
snapshot state was reconstructed from `origin/wip/*` snapshot commits (15:47–16:10Z) and from pre-merge
parents of merge commits (`7e88bbe^1` for #76) where a branch was force-pushed after the report. Where
neither exists (#84 `feat/exp-fps`), the row is marked UNVERIFIABLE rather than FALSE.
