# docs/pod — running Jadal experiments on a GPU pod with opencode v2
Index (keep this page true; update it in every PR that changes the pod workflow):
| Page | What it covers |
|---|---|
| [SETUP.md](SETUP.md) | env vars, bootstrap, data bundle, kit installation, version pinning |
| [WORKFLOW.md](WORKFLOW.md) | branch/worktree/commit/PR rules, autosave, definition of done |
| [AGENTS_AND_SKILLS.md](AGENTS_AND_SKILLS.md) | the agent roster, the skills, the commands, how to add one |
| [PARALLELISM.md](PARALLELISM.md) | subagents vs headless workers, limits, the launcher |
| [EXPERIMENT_LOGGING.md](EXPERIMENT_LOGGING.md) | the registry, fields, verdicts, rendering |
| [THREAT_MODEL.md](THREAT_MODEL.md) | PR-triggered agents on a public repo: risks and controls |
| [RUNBOOK.md](RUNBOOK.md) | failures seen so far and what to do |
| [ENVIRONMENT.md](ENVIRONMENT.md) | verified pod facts (GPU/CPU/RAM/disk, versions, agents, bundle, baselines), every line with the command that produced it; last checked 2026-10-10 |

Related: [../reports/pod_experiments_summary.md](../reports/pod_experiments_summary.md)
(the wave-by-wave result summary the lead fills from the merged registry),
`/workspace/state/ENVIRONMENT.md` (bootstrap-time environment dump).
