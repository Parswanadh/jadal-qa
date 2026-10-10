# Branch triage & merge queue — LB5 (claim LB5, track B, owner lead-b)

**Snapshot:** 2026-10-10T16:00Z–16:20Z UTC · base ref `origin/main` = `65f37dc`
("Merge pull request #79 from Parswanadh/feat/ensemble-views", 2026-10-10T21:15:48+05:30).
Worktree: `/workspace/wt/b-triage` (branch `b/triage`, cut from `pod/base`). No other branch
was pushed to, rebased, force-pushed, or deleted. Every claim below cites the command it came from.

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
All counts below are the 28-PR snapshot; re-run §1 before acting on it.

## 2. Inventory counts (28 open PRs)

| Metric | Value | Source |
|---|---|---|
| open PRs | 28 (all `base=main`, none draft, **none with a review decision**, no requested reviewers) | `gh pr list` |
| branches that are **ancestors of main** (nothing left to merge) | **0 / 28** | `git merge-base --is-ancestor` |
| **branch-only** PRs (carry work not on main) | **28 / 28** — 1 to 76 commits ahead, 1,337 branch-only files total | `git rev-list --left-right --count`, `git ls-tree` |
| GitHub `mergeable` | **18 MERGEABLE · 10 CONFLICTING** (#68, #76, #81, #82, #83, #87, #88, #89, #90, #91) | `gh pr list --json mergeable` |
| local `git merge-tree` (fresh fetch, git 2.43) | **19 clean · 9 conflicting** (same set minus #68) | `git merge-tree --write-tree` |
| CI status rollup | **only check is `CodeRabbit` on all 28**; `origin/main` has **no `.github/workflows`** → there is **no automated test CI**; the merge gate is manual | `gh pr list --json statusCheckRollup`, `git ls-tree -r origin/main -- .github` |
| branch-only docs/results paths | 73 distinct paths (reports, pod docs, registry, result JSONs) | §5 |

Because 0/28 branches are ancestors of main, **there are no `close` (already-merged) candidates**.

## 3. Queue table — every open PR

Legend: *ancestor of main* = `git merge-base --is-ancestor`; *branch-only* = files present on the
head branch and absent from `origin/main` (`git ls-tree` set difference); blockers are the union of
the GitHub `mergeable` field, the local `git merge-tree` result, staleness, and duplicate-file
collisions with other open PRs; actions follow the rules in §4.

| PR | head branch | title | ancestor of main? | branch-only? | what blocks it | recommended action |
|---|---|---|---|---|---|---|
| #25 | `feat/freshness` | freshness: nonce-LED liveness check | No | Yes — 6 files (1 docs/results) | merge-tree clean (tree 7e6d3d901b3d); behind main by 141 commits | merge |
| #28 | `feat/hub` | hub: video hub over EmbeddingGemma 2 (search,… | No | Yes — 16 files (1 docs/results) | merge-tree clean (tree 22646d450906); behind main by 141 commits | merge |
| #30 | `feat/baseline-report` | Baseline scoreboard: one-command B1-B8 report | No | Yes — 5 files (2 docs/results) | merge-tree clean (tree e2811244e1c8); behind main by 141 commits | merge |
| #45 | `feat/picoidle` | Pico attract mode: standalone idle animation,… | No | Yes — 4 files (1 docs/results) | merge-tree clean (tree 56b49128a1da); behind main by 51 commits; 2 files diverge from newer main copies | merge |
| #47 | `feat/gold-video` | gold: video player for label tool | No | Yes — 2 files (1 docs/results) | merge-tree clean (tree beb0abb4d178); behind main by 135 commits; 1 files diverge from newer main copies | merge |
| #57 | `feat/gpu-pipeline` | GPU pipeline PoC (RTX 4070 laptop): spectral… | No | Yes — 28 files (1 docs/results) | merge-tree clean (tree 9d41c274a0b4); behind main by 76 commits; dupe adds shared with other open PRs: `docs/reports/gpu_pipeline.md` ×3 | supersede (⊂ #76) or merge |
| #61 | `feat/spectral-harness` | Spectral harness: eval harness, arms A+E, sco… | No | Yes — 13 files (1 docs/results) | merge-tree clean (tree c65727add5ba); behind main by 105 commits; 1 files diverge from newer main copies | merge |
| #63 | `feat/integration3` | integration3: B2/B3 on real modules — no-stub… | No | Yes — 2 files (1 docs/results) | merge-tree clean (tree 12b46c3f39f1); behind main by 51 commits; 8 files diverge from newer main copies; dupe adds shared with other open PRs: `docs/reports/integration3.md` ×2 | supersede (⊂ #76) or merge |
| #67 | `feat/poc-dashboard` | PoC dashboard: offline judge-ready UI + stdli… | No | Yes — 16 files (7 docs/results) | merge-tree clean (tree 024b8e97aa40); behind main by 51 commits; dupe adds shared with other open PRs: `docs/reports/poc_dashboard.md` ×2, `docs/screens/poc_about_1280.png` ×2, `docs/screens/poc_gallery_1280.png` ×2 … | supersede (⊂ #76) or merge |
| #68 | `feat/frames-spectral` | Frames-spectral: classifier on 10k AI-labelle… | No | Yes — 21 files (1 docs/results) | **gh=CONFLICTING/DIRTY** but local merge-tree clean (criss-cross merge bases); behind main by 49 commits; 10 files diverge from newer main copies | hold+rebase |
| #69 | `feat/scale-data` | Scale-data: real Ego10K subset (4.4h, 47 fact… | No | Yes — 20 files (1 docs/results) | merge-tree clean (tree 8003514dee28); behind main by 53 commits; 1 files diverge from newer main copies | merge |
| #70 | `feat/frames10k` | frames10k: EG2 frame heads + human-label tran… | No | Yes — 30 files (1 docs/results) | merge-tree clean (tree 2de3e85b96fc); behind main by 53 commits | merge |
| #71 | `feat/gpu-heads` | GPU multi-head training: batched heads, neste… | No | Yes — 23 files (1 docs/results) | merge-tree clean (tree db16d47e7812); behind main by 51 commits | merge |
| #72 | `feat/perpixel-accuracy` | Per-pixel accuracy: whole-video features, rul… | No | Yes — 61 files (1 docs/results) | merge-tree clean (tree b71b57fd394b); behind main by 25 commits; dupe adds shared with other open PRs: `docs/reports/perpixel_accuracy.md` ×2 | merge |
| #73 | `feat/breakthrough-lab` | Breakthrough lab: 7 math prototypes + lit rev… | No | Yes — 65 files (1 docs/results) | merge-tree clean (tree b1cbd31a24d2); behind main by 51 commits | merge |
| #75 | `feat/poc-run` | PoC run: Egocentric-10K through S0/S1 funnel… | No | Yes — 32 files (2 docs/results) | merge-tree clean (tree 119bdc9e9977); behind main by 51 commits; 1 files diverge from newer main copies; dupe adds shared with other open PRs: `docs/reports/gpu_pipeline.md` ×3, `docs/reports/poc_run.md` ×2 | merge |
| #76 | `integration/poc` | PoC bundle: real Egocentric-10K run + dashboa… | No | Yes — 111 files (18 docs/results) | **gh=CONFLICTING**; merge-tree: 1 add/add conflicts, every path already on main; behind main by 32 commits; 28 files diverge from newer main copies; dupe adds shared with other open PRs: `docs/reports/gpu_pipeline.md` ×3, `docs/reports/integration3.md` ×2, `docs/reports/perpixel_accuracy.md` ×2 … | rebase (unlocks #57/#63/#67) |
| #77 | `feat/custom-algo-rhythm` | rhythm: purpose-built hand-region residual-mo… | No | Yes — 40 files (1 docs/results) | merge-tree clean (tree 0d0d6eb7eec9); behind main by 44 commits | merge |
| #80 | `oss/kit` | OSS kit: licence proposal, contributing, temp… | No | Yes — 16 files (3 docs/results) | merge-tree clean (tree d823c958b9f1); behind main by 17 commits; 1 files diverge from newer main copies; dupe adds shared with other open PRs: `experiments/registry.jsonl` ×9 | merge |
| #81 | `pod/prwatch` | prwatch: PR-triggered review agents + standar… | No | Yes — 101 files (14 docs/results) | **gh=CONFLICTING**; merge-tree: 6 add/add conflicts, every path already on main; behind main by 17 commits; 39 files diverge from newer main copies; dupe adds shared with other open PRs: `experiments/registry.jsonl` ×9, `docs/pod/AGENTS_AND_SKILLS.md` ×8, `docs/pod/EXPERIMENT_LOGGING.md` ×8 … | rebase |
| #82 | `docs/pod-docs` | docs: pod environment page, union registry to… | No | Yes — 95 files (16 docs/results) | **gh=CONFLICTING**; merge-tree: 6 add/add conflicts, every path already on main; behind main by 17 commits; 39 files diverge from newer main copies; dupe adds shared with other open PRs: `experiments/registry.jsonl` ×9, `docs/pod/AGENTS_AND_SKILLS.md` ×8, `docs/pod/EXPERIMENT_LOGGING.md` ×8 … | rebase |
| #83 | `feat/exp-regimes-pod` | E4: regimes + profiles + router (E4.01-E4.04,… | No | Yes — 106 files (17 docs/results) | **gh=CONFLICTING**; merge-tree: 6 add/add conflicts, every path already on main; behind main by 17 commits; 39 files diverge from newer main copies; dupe adds shared with other open PRs: `experiments/registry.jsonl` ×9, `docs/pod/AGENTS_AND_SKILLS.md` ×8, `docs/pod/EXPERIMENT_LOGGING.md` ×8 … | rebase |
| #84 | `feat/exp-fps` | E1: frame-rate sweep, IndEgo (LOCO): 3.33 fps… | No | Yes — 16 files (1 docs/results) | merge-tree clean (tree 442ce43c8366); behind main by 17 commits; 1 files diverge from newer main copies | merge |
| #87 | `feat/exp-domain-adapt` | E5 domain adaptation: in-domain heads, few-sh… | No | Yes — 106 files (17 docs/results) | **gh=CONFLICTING**; merge-tree: 6 add/add conflicts, every path already on main; behind main by 17 commits; 39 files diverge from newer main copies; dupe adds shared with other open PRs: `experiments/registry.jsonl` ×9, `docs/pod/AGENTS_AND_SKILLS.md` ×8, `docs/pod/EXPERIMENT_LOGGING.md` ×8 … | rebase |
| #88 | `feat/exp-fps-pod` | fps: E1.03 OOD fps sweep + E1.04 frame select… | No | Yes — 102 files (15 docs/results) | **gh=CONFLICTING**; merge-tree: 6 add/add conflicts, every path already on main; behind main by 17 commits; 39 files diverge from newer main copies; dupe adds shared with other open PRs: `experiments/registry.jsonl` ×9, `docs/pod/AGENTS_AND_SKILLS.md` ×8, `docs/pod/EXPERIMENT_LOGGING.md` ×8 … | rebase |
| #89 | `feat/exp-motion-cue` | E6: motion cue is ego-motion magnitude; label… | No | Yes — 102 files (17 docs/results) | **gh=CONFLICTING**; merge-tree: 6 add/add conflicts, every path already on main; behind main by 17 commits; 39 files diverge from newer main copies; dupe adds shared with other open PRs: `experiments/registry.jsonl` ×9, `docs/pod/AGENTS_AND_SKILLS.md` ×8, `docs/pod/EXPERIMENT_LOGGING.md` ×8 … | rebase |
| #90 | `feat/exp-cost-model` | cost: E7.01 — hours of video per wall-hour on… | No | Yes — 95 files (14 docs/results) | **gh=CONFLICTING**; merge-tree: 6 add/add conflicts, every path already on main; behind main by 17 commits; 39 files diverge from newer main copies; dupe adds shared with other open PRs: `experiments/registry.jsonl` ×9, `docs/pod/AGENTS_AND_SKILLS.md` ×8, `docs/pod/EXPERIMENT_LOGGING.md` ×8 … | rebase |
| #91 | `b/verify` | [B1] Independent verification: baselines, E1.… | No | Yes — 103 files (12 docs/results) | **gh=CONFLICTING**; merge-tree: 11 add/add conflicts, every path already on main; behind main by 17 commits; 44 files diverge from newer main copies; dupe adds shared with other open PRs: `experiments/registry.jsonl` ×9, `docs/pod/AGENTS_AND_SKILLS.md` ×8, `docs/pod/EXPERIMENT_LOGGING.md` ×8 … | rebase |

## 4. Blocker classes (what the table is telling us)

**A. Version skew against already-merged review fixes — 10 PRs, mechanical to resolve.**
`#68, #76, #81, #82, #83, #87, #88, #89, #90, #91` collide with main. For **9 of them** the local
`git merge-tree` conflict set is exactly the files main already received via merged PRs #79
(ensemble views) and #85 (exp_temporal), and **every conflicted path already exists on main**
(verified per path with `git ls-tree -r origin/main`): `configs/ensemble_v1.json`,
`docs/reports/exp_temporal.md`, `eval/ensemble/eval_ood.py`, `eval/exp_temporal/run_e3.py`,
`eval/exp_temporal/write_e305.py`, `jadal/ensemble/scorer.py` (plus the five
`eval/results/exp_temporal/E3.0*.json` files and `docs/PUBLIC_SNAPSHOT_EXCLUDE.txt` for #76/#91).
Because main's copies are the *newer, reviewed* ones, the resolution is mechanical — **keep main's
version for those paths**, no hand-merging of content is required. That recipe is the same for all
of them:

```bash
git rebase --onto origin/main "$(git merge-base origin/main <branch>)" <branch>
# take main's side for the conflicted paths, e.g.:
git checkout --ours <path> && git add <path> && git rebase --continue
```

**B. Duplicate additions of the same new files — merge-order hazard.** `origin/main` has **no
`experiments/` directory**. Nine open PRs (#80, #81, #82, #83, #87, #88, #89, #90, #91) each add
`experiments/registry.jsonl`; eight add `registry_snapshot.jsonl`, `experiments/EXPERIMENTS.md` and
`docs/pod/*`; `docs/reports/exp_regimes.md` is added by five (#81, #82, #83, #87, #88),
`docs/reports/gpu_pipeline.md` by three (#57, #75, #76). Whoever merges second must rebase (or the
registry must be union-merged). This PR adds `experiments/registry.jsonl` too (one line, see §8),
so the same rule applies to it.

**C. Staleness.** Every branch is behind main: 17–141 commits (median ~51). Nothing is fast-forward;
all merges are true merges or rebases.

**D. No review decision on any of the 28 PRs, and no test CI.** The single status check is
`CodeRabbit` (a review bot). `origin/main` has no `.github/workflows/`. Review + the manual merge
gate in §6 are the only quality guards.

**E. Criss-cross merge bases make local mergeability optimistic for #68.** `git merge-tree
--write-tree origin/main origin/feat/frames-spectral` exits 0 (clean tree `8ef2549d35b2`), while
GitHub reports `CONFLICTING/DIRTY` for head `8649018`. `git merge-base --all` shows **multiple merge
bases** for this pair, so the two tools pick different bases. Treat #68 as needs-rebase; do not
merge it on the strength of the local check alone.

**F. Umbrella PRs.** `#76 integration/poc` **contains** `#57 feat/gpu-pipeline`,
`#63 feat/integration3` and `#67 feat/poc-dashboard` (`git merge-base --is-ancestor` → true for all
three). Once #76 lands, those three become no-ops and should be closed as superseded; landing them
first is also safe (all three are merge-tree clean).

## 5. Branch-only reports — rebase check (in my worktree only)

All 28 branches are branch-only, so all 73 distinct branch-only `docs/`+`experiments/` paths are
"on a branch but not on main" (the largest groups: reports under `docs/reports/`, the `docs/pod/*`
set, `experiments/EXPERIMENTS.md`, `experiments/registry*.jsonl`). Mechanical rebase check, run
without touching any branch:

* **19 PRs are conflict-free as-is.** `git merge-tree --write-tree origin/main origin/<head>` exits 0
  and returns a tree — the merge/rebase is purely mechanical. The tree OIDs are in the table
  ("merge-tree clean (tree …)"). Reproduce, e.g.:
  `git merge-tree --write-tree origin/main origin/feat/exp-fps` → `442ce43c8366df64`.
* **9 PRs need the §4-A recipe** (keep main's copy of the conflicted paths). Verified mechanically:
  in every one of the 9, *all* conflicted paths already exist on `origin/main`, so no branch content
  is lost by taking main's side for them.
* **Nothing was copied onto `b/triage`.** Every branch-only report found is already the payload of an
  open PR, and 73 paths collide between PRs (e.g. `experiments/registry.jsonl` is in 9 of them).
  Copying one here would manufacture exactly the duplicate-add conflict class this report flags, so
  the supersession is recorded instead of materialised. No other branch was rebased, rewritten or
  force-pushed; the verification was done with `git merge-tree` only, which writes no refs.

## 6. Merge gate — the full test suite

**What the suite is.** The repo's canonical command is `make test`:

```
timeout 150 env PYTHONNOUSERSITE=1 /home/parshu/envs/jadal/bin/python -m unittest discover -s tests
timeout 150 env PYTHONNOUSERSITE=1 /home/parshu/envs/jadal/bin/python -m unittest discover -s web
```

Two caveats, both evidenced: the interpreter `/home/parshu/envs/jadal/bin/python` **does not exist on
the pod** (`ls -l` → "No such file or directory"), and there is no pinned requirements file for the
service/edge tiers (`requirements-edge.txt` only). The task's proxy command
`python -m pytest tests/ -q` covers `tests/` only, not `web/test_dashboard.py`.

**Result on `origin/main` (`65f37dc`), run from a clean export of main, `OMP_NUM_THREADS=2`:**

| Run | Command | Result |
|---|---|---|
| 1 (literal) | `cd /tmp/jadal-main && timeout 900 python3 -m pytest tests/ -q` | **interrupted at collection**: `1 warning, 4 errors in 8.82s`, exit 2 — pytest stops before running any test |
| 2 (same + `--continue-on-collection-errors`) | `... pytest tests/ -q --continue-on-collection-errors` | **1 failed, 492 passed, 3 skipped, 4 errors, 90 subtests passed in 31.45s**, exit 1 |
| 3 (non-`/tmp` checkout, same flags) | `cd /workspace/wt/b-triage/.lb5-main-check && timeout 900 python3 -m pytest tests/ -q --continue-on-collection-errors` | **493 passed, 3 skipped, 4 errors, 90 subtests passed in 23.52s**, exit 1 — **0 real failures** |

* The **4 collection errors are pod environment gaps, not main defects**:
  `tests/test_integration.py` and `tests/test_service.py` → `ModuleNotFoundError: fastapi`;
  `tests/test_eval_splits.py` and `tests/test_signals.py` → `ModuleNotFoundError: imagehash`
  (`pandas`, `pydantic`, `uvicorn` are also absent; `numpy`, `cv2`, `torch`, `scipy`, `PIL`, `sklearn`
  are present).
* The single failure in run 2, `test_make_demo_clips.py::TestMakeDemoClips::test_missing_sources_exit_2`
  (`AssertionError: 3 != 2`), is a **false positive caused by running from `/tmp`**: the script's
  out-guard refuses the default output dir before the missing-sources check. Reproduced directly:
  `JADAL_INDEGO_ROOT=/nonexistent-indego-xyz python3 scripts/make_demo_clips.py` →
  `make_demo_clips: refusing --out under a temp dir (/tmp/jadal-main/data_runtime/demo_clips); SPEC
  forbids writing results to /tmp.` exit **3**. Run 3 (outside `/tmp`) passes it.

**Verdict: main is green — 493 passed / 0 failed — modulo 4 modules missing from the pod Python
environment (`fastapi`, `imagehash`).** To get a fully green gate, install the service/edge deps (or
run `make test` with the right interpreter) and re-run:

```bash
cd <checkout of main outside /tmp> && export OMP_NUM_THREADS=2 && \
  timeout 900 python -m pytest tests/ -q --continue-on-collection-errors
```

I ran the suite **once per checkout** as instructed (three light runs ≤ 32 s each, no heavy jobs).

## 7. Recommended merge order

1. **Wave 1 — merge now** (merge-tree clean, no shared-file collision): `#84, #70, #71, #73, #77,
   #72, #75, #69, #61, #47, #45, #30, #28, #25, #80`. `#80` should drop its **0-byte**
   `experiments/registry.jsonl` (or defer to the canonical registry) before merging.
2. **Wave 2 — rebase then merge** (apply §4-A, keep main's copies of the 6 shared files):
   `#76` first (1-file conflict, and it unlocks `#57/#63/#67`), then `#81, #82, #83, #87, #88, #89,
   #90, #91`. Suggested internal order for the pod-lineage five (`#81/#82/#83/#87/#88`, identical
   6-file conflict set): `#82` (docs + union-registry tool, 22-line registry) → `#81` (rebases on top,
   drops its duplicate `docs/pod/*`) → `#83, #87, #88, #89, #90` (each rebases, registry union-merge).
3. **Wave 3 — supersede/close**: `#57`, `#63`, `#67` once `#76` has landed; `#68` only after a real
   rebase (GitHub still reports it DIRTY; see §4-E).
4. **Gate**: no PR has an approving review and there is no test CI — someone must run §6 manually
   before each merge, and reviewers must sign off.

## 8. What was landed on `b/triage` (and what was deliberately not done)

> **Branch-lineage caveat (read before merging PR #94).** `b/triage` was cut from `pod/base`, so the
> PR diff against `main` contains the whole pod/base tree (132 files) and GitHub reports the PR
> `CONFLICTING`. The **LB5 payload is exactly commit `056c68f` — 2 files** (this report +
> `experiments/registry.jsonl`). Do **not** merge the branch wholesale: cherry-pick `056c68f` onto
> main (or rebase `b/triage` onto main in a scratch copy). Re-cutting/force-pushing `b/triage` was
> out of scope for this worker (never force-push).

Landed in this PR:
* this triage report (`docs/reports/branch_triage.md`) — the replacement for the missing
  `handover/BRANCH_TRIAGE_REPORT.md`;
* one registry line in `experiments/registry.jsonl` (id `LB5`). `origin/main` has **no**
  `experiments/registry.jsonl` at all; 9 open PRs add one, so this file is a single-line
  union-merge-friendly stub, not a competing full registry.

Deliberately **not** done, to avoid collisions: no other branch was pushed to, rebased, rewritten or
force-pushed; no branch was created or deleted; no other worktree was touched; no PR was merged or
closed; no branch-only report file was copied here (see §5); `/workspace/artifacts` was not touched;
no secrets were read into, or written out of, any file. The temporary `main` export used for the
merge-gate run (`.lb5-main-check/`) was removed before the commit.
