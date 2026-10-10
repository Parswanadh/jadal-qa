---
description: Write STATUS.md (done / running / next / blockers / headline numbers) from the registry and git state, then commit and push
agent: lead
---
Update STATUS.md from `python3 tools/exp.py list`, `git log --oneline -15`, and the running workers (`bash tools/pod/launch_workers.sh status`). Headline numbers only from results JSONs. Run `tools/pod/secrets_guard.sh`, commit with prefix `status:` and push the current branch.
