# Running many agents in parallel
1. **Inside one session**: the lead issues several `task` tool calls in ONE message (independent tasks, disjoint output folders, self-contained briefs). Subagents cannot spawn subagents.
2. **Headless workers**: `bash tools/pod/launch_workers.sh start <name> <agent> <prompt-file>` (worktree + branch + private server `--standalone` + log + PID); `status`, `stop <name>`, `logs <name>`; cap `JADAL_MAX_WORKERS` (default 4). Verify after 60 s that the log shows tool activity (a log with only CLI help text means invalid flags for this version).
3. **v2 CLI facts (checked on v2.0.26)**: `opencode run [flags] [message...]` with `--agent`, `-m provider/model#variant`, `--session/-s`, `--continue`, `--fork`, `--format json`, `--file`, `--title`, `--auto` (auto-approve what is not explicitly denied), `--standalone` (private server), `--server`; no `--dir` and no `--variant`. Sessions: `opencode session list`. Background service: `opencode service start|stop|restart|status`.
4. **Budgets**: A40 48 GB (~40 GB total across jobs), CPU load <= 75% of cores, RAM <= 8 GB per process.
5. **Failure handling**: relaunch narrower; log the failure in RUNBOOK.md.
