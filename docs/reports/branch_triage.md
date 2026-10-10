# Branch triage & merge queue — LB5 (claim LB5, track B, owner lead-b)

**Snapshot (refreshed):** 2026-10-10T16:23Z UTC · base ref `origin/main` = **`e320c58`**
("Merge pull request #96 from Parswanadh/docs/pod-sync"). The first pass was taken at 16:00Z against
`65f37dc` from `/workspace/wt/b-triage` (cut from `pod/base`); **this refreshed pass is
`/workspace/wt/b-triage-main` (`b/triage-main`, branched from `origin/main`) and supersedes PR #94**.
The queue table below is a **point-in-time snapshot, not a static fact** — the survey saw the open-PR
count move 25 (15:57Z) → 28 (16:00Z) → 33 (16:23Z) and `main` move `65f37dc` → `e320c58`.
No branch was pushed to, rebased, force-pushed, or deleted by this worker.

## 0. Missing input — finding (not a blocker)

The expected input **`handover/BRANCH_TRIAGE_REPORT.md` does not exist**, on disk or in any git ref:

| Check | Command | Result |
|---|---|---|
| filesystem | `find / -name 'BRANCH_TRIAGE*' -not -path '*/proc/*'` | no matches |
| directory | `ls -la /workspace/repo/handover` | `No such file or directory` |
| history | `git log --all --oneline -- 'handover/BRANCH_TRIAGE_REPORT.md'` | empty |
| every ref | `for r in $(git for-each-ref ...); do git cat-file -e "$r:handover/BRANCH_TRIAGE_REPORT.md"; done` | not found in any ref |
| any `handover/` tree | `git ls-tree -d --name-only <ref> \| grep -qx handover` over all `refs/remotes/origin` | no ref has a `handover/` directory |

Treated as a finding: the triage below was **reconstructed from git + live PR state**. This file
(`docs/reports/branch_triage.md`) is the replacement artifact for the missing handover report.

## 1. Method (reproducible)

```bash
git -C /workspace/repo branch -a
git -C /workspace/repo log --oneline --all -40
GH_TOKEN=$(gh auth token) gh pr list -R Parswanadh/jadal-qa --state open --limit 100 \
  --json number,title,headRefName,createdAt,mergeable,reviewDecision
git fetch --all --prune                      # refs verified current (origin/main unchanged)
# per PR head branch <b>:
git merge-base --is-ancestor origin/<b> origin/main
git rev-list --left-right --count origin/main...origin/<b>
git diff --name-status origin/main...origin/<b>
git merge-tree --write-tree origin/main origin/<b>     # exit 0 = conflict-free merge
git ls-tree -r --name-only origin/main                # to separate true branch-only files
GH_TOKEN=$(gh auth token) gh pr list ... --json number,mergeable,statusCheckRollup
```

**The queue is live.** The inventory command returned **25 open PRs at 15:57Z and 28 at 16:00Z**
(#89 `feat/exp-motion-cue`, #90 `feat/exp-cost-model`, #91 `b/verify` were opened mid-survey).
All counts below are the 33-PR snapshot at 16:23Z; re-run §1 before acting on it — the second
(refresh) pass re-ran exactly these commands and the numbers had already moved (see §2/§3 drift note).

## 2. Inventory counts (33 open PRs @16:23Z)

| Metric | Value | Source |
|---|---|---|
| open PRs | **33** (all `base=main`, none draft, **none with a review decision**, no requested reviewers) | `gh pr list` @16:23Z |
| branches that are **ancestors of main** (nothing left to merge) | **0 / 33** | `git merge-base --is-ancestor` |
| **branch-only** PRs (carry work not on main) | **33 / 33** — 1 to 15 commits ahead, **1,601** branch-only files total | `git rev-list --left-right --count`, `git ls-tree` |
| local `git merge-tree --write-tree` vs `origin/main` (git 2.43, fresh fetch) | **18 clean · 15 conflicting** (#69, #75, #81, #87, #88, #91, #92, #93, #94, #95, #97, #98, #99, #100, #101) | `git merge-tree` |
| GitHub `mergeable` (lazily computed) | 2 MERGEABLE · 3 CONFLICTING · **28 UNKNOWN** — GitHub was still recomputing after main moved | `gh pr list --json mergeable` |
| CI status rollup | **only check is `CodeRabbit` on all 33**; `origin/main` has **no `.github/workflows`** → **no automated test CI**; the merge gate is manual | `gh pr list --json statusCheckRollup`, `git ls-tree -r origin/main -- .github` |
| branch-only docs/results paths | 75 distinct paths | §5 |

Because 0/33 branches are ancestors of main, **there are no `close` (already-merged) candidates**.

Because 0/28 branches are ancestors of main, **there are no `close` (already-merged) candidates**.

## 3. Queue table — every open PR

Legend: *ancestor of main* = `git merge-base --is-ancestor`; *branch-only* = files present on the
head branch and absent from `origin/main` (`git ls-tree` set difference); blockers are the union of
the GitHub `mergeable` field, the local `git merge-tree` result, staleness, and duplicate-file
collisions with other open PRs; actions follow the rules in §4.

| PR | head branch | title | ancestor of main? | branch-only? | what blocks it | recommended action |
|---|---|---|---|---|---|---|
| #25 | `feat/freshness` | freshness: nonce-LED liveness check | No | Yes — 6 files (1 docs/results) | merge-tree clean (4a4667f57a6d); behind main by 230 | merge |
| #28 | `feat/hub` | hub: video hub over EmbeddingGemma 2 (search,… | No | Yes — 16 files (1 docs/results) | merge-tree clean (ce27f41f717d); behind main by 230 | merge |
| #30 | `feat/baseline-report` | Baseline scoreboard: one-command B1-B8 report | No | Yes — 5 files (2 docs/results) | merge-tree clean (a09d79f8624f); behind main by 230 | merge |
| #45 | `feat/picoidle` | Pico attract mode: standalone idle animation,… | No | Yes — 4 files (1 docs/results) | merge-tree clean (c135e2e0c1f7); behind main by 140; 2 files diverge from newer main copies | merge |
| #47 | `feat/gold-video` | gold: video player for label tool | No | Yes — 2 files (1 docs/results) | merge-tree clean (42c09b20727d); behind main by 224; 1 files diverge from newer main copies | merge |
| #61 | `feat/spectral-harness` | Spectral harness: eval harness, arms A+E, sco… | No | Yes — 13 files (1 docs/results) | merge-tree clean (eef2540b895e); behind main by 194; 1 files diverge from newer main copies | merge |
| #68 | `feat/frames-spectral` | Frames-spectral: classifier on 10k AI-labelle… | No | Yes — 21 files (1 docs/results) | merge-tree clean (6fb8d4429186) · gh=CONFLICTING/DIRTY (criss-cross bases); behind main by 138; 14 files diverge from newer main copies | hold+rebase |
| #69 | `feat/scale-data` | Scale-data: real Ego10K subset (4.4h, 47 fact… | No | Yes — 20 files (1 docs/results) | **gh=UNKNOWN**; merge-tree: 1 conflicts; behind main by 142; 1 files diverge from newer main copies | rebase |
| #70 | `feat/frames10k` | frames10k: EG2 frame heads + human-label tran… | No | Yes — 30 files (1 docs/results) | merge-tree clean (0f3483838e7c); behind main by 142 | merge |
| #71 | `feat/gpu-heads` | GPU multi-head training: batched heads, neste… | No | Yes — 23 files (1 docs/results) | merge-tree clean (d060fc20684f); behind main by 140 | merge |
| #72 | `feat/perpixel-accuracy` | Per-pixel accuracy: whole-video features, rul… | No | Yes — 17 files (0 docs/results) | merge-tree clean (cbb7b55f6afd); behind main by 100; 46 files diverge from newer main copies | merge |
| #73 | `feat/breakthrough-lab` | Breakthrough lab: 7 math prototypes + lit rev… | No | Yes — 65 files (1 docs/results) | merge-tree clean (c4d7334273c6); behind main by 140 | merge |
| #75 | `feat/poc-run` | PoC run: Egocentric-10K through S0/S1 funnel… | No | Yes — 0 files (0 docs/results) | **gh=UNKNOWN**; merge-tree: 2 conflicts; behind main by 118; 2 files diverge from newer main copies | rebase |
| #77 | `feat/custom-algo-rhythm` | rhythm: purpose-built hand-region residual-mo… | No | Yes — 40 files (1 docs/results) | merge-tree clean (8d12c1f0b4f6); behind main by 133 | merge |
| #80 | `oss/kit` | OSS kit: licence proposal, contributing, temp… | No | Yes — 16 files (3 docs/results) | merge-tree clean (c8378f8c60e1); behind main by 106; 1 files diverge from newer main copies; dupe adds: `experiments/registry.jsonl` ×19 | merge |
| #81 | `pod/prwatch` | prwatch: PR-triggered review agents + standar… | No | Yes — 80 files (14 docs/results) | **gh=UNKNOWN**; merge-tree: 6 conflicts; behind main by 11; 39 files diverge from newer main copies; dupe adds: `experiments/registry.jsonl` ×19, `docs/pod/AGENTS_AND_SKILLS.md` ×17 | rebase |
| #82 | `docs/pod-docs` | docs: pod environment page, union registry to… | No | Yes — 74 files (16 docs/results) | merge-tree clean (885b1fe05517); behind main by 0; 46 files diverge from newer main copies; dupe adds: `experiments/registry.jsonl` ×19, `docs/pod/AGENTS_AND_SKILLS.md` ×17 | merge |
| #83 | `feat/exp-regimes-pod` | E4: regimes + profiles + router (E4.01-E4.04,… | No | Yes — 85 files (17 docs/results) | merge-tree clean (44d2505b3096); behind main by 5; 32 files diverge from newer main copies; dupe adds: `experiments/registry.jsonl` ×19, `docs/pod/AGENTS_AND_SKILLS.md` ×17 | merge |
| #87 | `feat/exp-domain-adapt` | E5 domain adaptation: in-domain heads, few-sh… | No | Yes — 87 files (18 docs/results) | **gh=UNKNOWN**; merge-tree: 25 conflicts; behind main by 106; 64 files diverge from newer main copies; dupe adds: `experiments/registry.jsonl` ×19, `docs/pod/AGENTS_AND_SKILLS.md` ×17 | rebase |
| #88 | `feat/exp-fps-pod` | fps: E1.03 OOD fps sweep + E1.04 frame select… | No | Yes — 79 files (16 docs/results) | **gh=UNKNOWN**; merge-tree: 29 conflicts; behind main by 106; 67 files diverge from newer main copies; dupe adds: `experiments/registry.jsonl` ×19, `docs/pod/AGENTS_AND_SKILLS.md` ×17 | rebase |
| #89 | `feat/exp-motion-cue` | E6: motion cue is ego-motion magnitude; label… | No | Yes — 81 files (17 docs/results) | merge-tree clean (bb2f380d92ae); behind main by 5; 32 files diverge from newer main copies; dupe adds: `experiments/registry.jsonl` ×19, `docs/pod/AGENTS_AND_SKILLS.md` ×17 | merge |
| #90 | `feat/exp-cost-model` | cost: E7.01 — hours of video per wall-hour on… | No | Yes — 74 files (14 docs/results) | merge-tree clean (ca2aa0d3f64a); behind main by 5; 32 files diverge from newer main copies; dupe adds: `experiments/registry.jsonl` ×19, `docs/pod/AGENTS_AND_SKILLS.md` ×17 | merge |
| #91 | `b/verify` | [B1] Independent verification: baselines, E1.… | No | Yes — 83 files (12 docs/results) | **gh=CONFLICTING**; merge-tree: 34 conflicts; behind main by 106; 73 files diverge from newer main copies; dupe adds: `experiments/registry.jsonl` ×19, `docs/pod/AGENTS_AND_SKILLS.md` ×17; superseded by #102 b/verify-main | superseded by #102 |
| #92 | `b/poc-integrate` | [B2] PoC integration: adopted sampling + earl… | No | Yes — 76 files (12 docs/results) | **gh=UNKNOWN**; merge-tree: 25 conflicts; behind main by 106; 64 files diverge from newer main copies; dupe adds: `experiments/registry.jsonl` ×19, `docs/pod/AGENTS_AND_SKILLS.md` ×17 | rebase |
| #93 | `b/submission-docs` | [B3] Regenerate explainer/deck/submission pac… | No | Yes — 73 files (17 docs/results) | **gh=UNKNOWN**; merge-tree: 26 conflicts; behind main by 106; 65 files diverge from newer main copies; dupe adds: `experiments/registry.jsonl` ×19, `docs/pod/AGENTS_AND_SKILLS.md` ×17 | rebase |
| #94 | `b/triage` | [B5] Branch triage + merge queue report | No | Yes — 68 files (12 docs/results) | **gh=UNKNOWN**; merge-tree: 25 conflicts; behind main by 106; 64 files diverge from newer main copies; dupe adds: `experiments/registry.jsonl` ×19, `docs/pod/AGENTS_AND_SKILLS.md` ×17; superseded by #102-era re-land b/triage-main (this PR) | superseded by this PR |
| #95 | `feat/exp-mosaic` | E2 mosaic (builder experiment): one tiled ima… | No | Yes — 89 files (20 docs/results) | **gh=UNKNOWN**; merge-tree: 25 conflicts; behind main by 106; 64 files diverge from newer main copies; dupe adds: `experiments/registry.jsonl` ×19, `docs/pod/AGENTS_AND_SKILLS.md` ×17 | rebase |
| #97 | `b/ood-ready` | [B4] Stage frozen pre-registered OOD run (one… | No | Yes — 76 files (12 docs/results) | **gh=UNKNOWN**; merge-tree: 25 conflicts; behind main by 106; 64 files diverge from newer main copies; dupe adds: `experiments/registry.jsonl` ×19, `docs/pod/AGENTS_AND_SKILLS.md` ×17 | rebase |
| #98 | `b/verify-lb2` | [B1-verify] LB2 PoC integration re-check (C s… | No | Yes — 69 files (11 docs/results) | **gh=UNKNOWN**; merge-tree: 25 conflicts; behind main by 106; 64 files diverge from newer main copies; dupe adds: `experiments/registry.jsonl` ×19, `docs/pod/AGENTS_AND_SKILLS.md` ×17 | rebase |
| #99 | `b/verify-gate` | [B1-verify] Merge gate: full test suite statu… | No | Yes — 73 files (12 docs/results) | **gh=UNKNOWN**; merge-tree: 25 conflicts; behind main by 106; 64 files diverge from newer main copies; dupe adds: `experiments/registry.jsonl` ×19, `docs/pod/AGENTS_AND_SKILLS.md` ×17 | rebase |
| #100 | `b/verify-ood` | [B1-verify] OOD 0.529 does not reproduce from… | No | Yes — 70 files (12 docs/results) | **gh=UNKNOWN**; merge-tree: 25 conflicts; behind main by 106; 64 files diverge from newer main copies; dupe adds: `experiments/registry.jsonl` ×19, `docs/pod/AGENTS_AND_SKILLS.md` ×17 | rebase |
| #101 | `b/verify-audit` | [B1-verify] Submission number audit vs FACTS.… | No | Yes — 69 files (12 docs/results) | **gh=CONFLICTING**; merge-tree: 25 conflicts; behind main by 106; 64 files diverge from newer main copies; dupe adds: `experiments/registry.jsonl` ×19, `docs/pod/AGENTS_AND_SKILLS.md` ×17 | rebase |
| #102 | `b/verify-main` | [B1] Independent verification, rebased onto m… | No | Yes — 17 files (2 docs/results) | merge-tree clean (9a051905de5a); behind main by 0; dupe adds: `experiments/registry.jsonl` ×19, `docs/reports/pod_verification.md` ×2 | merge |

**Drift since the 16:00Z pass** (why the table is stamped, not static): `main` moved
`65f37dc → e320c58` (PR #96 `docs/pod-sync`, which itself brought in a large tree incl. what
`feat/exp-fps` and `integration/poc` carried); PRs **#57, #63, #67, #76 are no longer open**
(landed/closed), the queue grew to **33** (`#95 feat/exp-mosaic`, `#98–#101` B1-verify follow-ups,
`#102 b/verify-main`, plus the re-landed `b/*` branches). **#91 is superseded by #102**
(`b/verify-main`, merge-tree clean) and **#94 is superseded by this PR**. Some PRs that conflicted at
16:00Z are clean now (`#82` at 0 behind, `#83`, `#89`, `#90`); others became conflicted as main moved
(`#69`, `#75`, `#87`, `#88`, `#91–#94`, `#97–#101`).

## 4. Blocker classes (refreshed)

**A. All 15 conflicts are mechanically resolvable — no content has to be hand-merged.** For **every**
conflicting PR, **every** conflicted path already exists on `origin/main` (checked per path against
`git ls-tree -r origin/main`; 100% in all 15 cases). The conflicts are `add/add` (the branch adds a
file main already has, usually an older pre-review copy) or edits to files main rewrote. Sizes:
`#69` 1 file (`.gitignore`); `#75` 2 (`.gitignore`, `docs/reports/poc_run.md`); `#81` 6
(`eval/exp_fps/*`, `jadal/ensemble/scorer.py`, …); the pod-lineage/`b/*` family
(`#87, #88, #91, #92, #93, #94, #95, #97, #98, #99, #100, #101`) 25–34 files each, all containing the
same core set — `.gitignore`, `configs/ensemble_v1.json`, `docs/reports/exp_temporal.md`,
`eval/ensemble/eval_ood.py`, `eval/exp_temporal/run_e3.py`, `eval/exp_temporal/write_e305.py`,
`jadal/ensemble/scorer.py`, `eval/results/exp_temporal/E3.0*.json`, `docs/POC_STATUS.md`. Recipe
(applied by the branch owner, **never** by force-pushing someone else's branch):

```bash
git rebase --onto origin/main "$(git merge-base origin/main <branch>)" <branch>
git checkout --ours <conflicted-path> && git add <conflicted-path> && git rebase --continue
```

**B. Duplicate additions of the same new files — now the dominant hazard.** `origin/main` still has
**no `experiments/` directory**. `experiments/registry.jsonl` is added by **19** open PRs,
`docs/pod/*` and `registry_snapshot.jsonl` by **17**, `docs/reports/exp_regimes.md` by **17**,
`experiments/EXPERIMENTS.md` by **9**, `docs/reports/pod_verification.md` by 2. Whoever merges second
must rebase or union-merge; landing several of these in one batch guarantees conflicts.

**C. Staleness.** Behind main by 0–230 commits (median ≈106). No branch is a fast-forward.

**D. No review decision on any of the 33 PRs, and no test CI** (only `CodeRabbit`; no
`.github/workflows`). Review plus the manual gate in §6 are the only quality guards.

**E. Criss-cross merge bases (#68).** Local `git merge-tree` exits 0 while GitHub reports
`CONFLICTING/DIRTY` — `git merge-base --all` shows multiple merge bases. Treat as needs-rebase; do not
merge on the local check alone.

**F. Supersessions.** `#91 → #102`, `#94 → this PR (b/triage-main)`; `#57/#63/#67/#76` already landed.

## 5. Branch-only reports — rebase check (in my own worktree only)

All 33 branches are branch-only, so all 75 distinct branch-only `docs/`+`experiments/` paths are on a
branch but not on main. Mechanical check, run without touching any branch:

* **18 PRs are conflict-free as-is** — `git merge-tree --write-tree origin/main origin/<head>` exits 0
  and returns a tree (OIDs in the table). Example: `git merge-tree --write-tree origin/main
  origin/b/verify-main` → clean, matching PR #102.
* **15 PRs need the §4-A recipe**; verified mechanically that *all* their conflicted paths already
  exist on main, so keeping main's copy loses no branch content.
* **Nothing was copied onto this branch.** Every branch-only report path already belongs to an open
  PR and 75 paths collide between PRs (e.g. `experiments/registry.jsonl` ×19), so copying one here
  would manufacture the duplicate-add conflicts this report flags. Supersessions are recorded, not
  materialised. No branch was rebased/rewritten/force-pushed; only `git merge-tree` was used (it
  writes no refs).

## 6. Merge gate — the full test suite (re-run on the new main)

Canonical command is `make test`:
`timeout 150 env PYTHONNOUSERSITE=1 /home/parshu/envs/jadal/bin/python -m unittest discover -s tests`
(plus `-s web`). Its interpreter **does not exist on the pod** (`ls` → No such file or directory) and
there is no service/edge requirements file, so the pytest proxy is used.

Run on `origin/main` = **`e320c58`**, in a clean export at
`/workspace/wt/b-triage-main/.lb5-gate` (non-`/tmp`, so path guards behave as on a real checkout),
`OMP_NUM_THREADS=2`:

| Run | Command | Result |
|---|---|---|
| A (literal) | `timeout 900 python3 -m pytest tests/ -q -p no:cacheprovider` | **interrupted at collection**: `1 warning, 4 errors in 4.18s`, exit 2 |
| B (same + continue) | `... pytest tests/ -q -p no:cacheprovider --continue-on-collection-errors` | **594 passed, 0 failed, 4 skipped, 4 errors, 90 subtests passed in 48.42s**, exit 1 |

* **0 real failures.** The 4 collection errors are pod environment gaps:
  `tests/test_eval_splits.py`, `tests/test_signals.py` → `ModuleNotFoundError: imagehash`;
  `tests/test_integration.py`, `tests/test_service.py` → `ModuleNotFoundError: fastapi`
  (`pandas`, `pydantic`, `uvicorn` also absent; `numpy`, `cv2`, `torch`, `scipy`, `PIL`, `sklearn`
  present).
* For the record, the previous main `65f37dc` measured **493 passed / 0 failed / 3 skipped / 4
  collection errors** under the same command (run 3 in the first pass).
* **Verdict: main is green — 594 passed, 0 failed — modulo 4 modules missing from the pod Python.**
  Reproduce with:

```bash
cd <clean checkout of main outside /tmp> && export OMP_NUM_THREADS=2 &&   timeout 900 python -m pytest tests/ -q --continue-on-collection-errors
```

## 7. Recommended merge order (refreshed)

1. **Wave 1 — merge now** (merge-tree clean; de-duplicate first): `#25, #28, #30, #45, #47, #61,
   #68*, #70, #71, #72, #73, #77, #80, #82, #83, #89, #90, #102` (*#68 only after a real rebase, see
   §4-E). `#80` should drop its 0-byte `experiments/registry.jsonl`.
2. **Wave 2 — mechanical rebase then merge** (apply §4-A): smallest first — `#69` (1 file), `#75`
   (2), `#81` (6) — then the 25–34-file family `#87, #88, #91, #92, #93, #95, #97, #98, #99, #100,
   #101`. `#94` is superseded by this PR; `#91` by `#102`.
3. **Registry/doc de-duplication:** 19 PRs add `experiments/registry.jsonl` and 17 add `docs/pod/*`.
   Pick one canonical owner per file (this PR ships the 36-id union, same as `#102`) and have the rest
   drop or union-merge their copy — otherwise each subsequent merge conflicts.
4. **Gate:** no approving reviews and no test CI → run §6 manually before each merge.

## 8. What was landed here (and what was deliberately not done)

Landed in **this PR** (`b/triage-main`, branched from `origin/main` = `e320c58`; **supersedes #94**,
which was cut from `pod/base` and conflicted on 25 files):
* `docs/reports/branch_triage.md` — the replacement for the missing
  `handover/BRANCH_TRIAGE_REPORT.md`, refreshed at 16:23Z;
* `experiments/registry.jsonl` — the 36-id union copied from `/workspace/wt/b-verify/experiments/
  registry.jsonl`, verified to contain `LB5`. `origin/main` has no registry file; 19 open PRs add one,
  so this is the shared union, not a competing stub. **Supersedes the 1-line stub in PR #94.**

Deliberately **not** done: no other branch pushed to, rebased, rewritten or force-pushed (#94 is left
open and untouched — it is superseded by this PR, not rewritten); no branch created or deleted beyond
`b/triage-main`; no other worktree touched; no PR merged or closed; no branch-only report copied here
(§5); `/workspace/artifacts` untouched; no secrets read into or written out of any file. The scratch
merge-gate export (`.lb5-gate/`) was removed before the commit.
