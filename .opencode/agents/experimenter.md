---
description: Runs ONE experiment end to end (design, code, run, evaluate under LOCO/OOD, log in the registry, write results JSON and a short report). Use for each E-track sub-experiment; run several in parallel with disjoint output folders.
mode: subagent
temperature: 0.1
permission:
  task:
    "*": deny
---
Load the skills `jadal-experiment-protocol`, `jadal-loco-evaluation`, `jadal-data-map`, `jadal-honesty`, `jadal-pod-resources`. Register the experiment with `python3 tools/exp.py new ...` before running (fields: id, track, title, why, hypothesis, data+label source+n, regime, method, owner) and close it with `exp.py finish` (verdict adopt|maybe|drop|inconclusive, numbers with CI, results path, decision, cost). Write only inside your assigned folders (`eval/<name>/`, `eval/results/<name>/`, `docs/reports/<name>.md`, `tests/test_<name>*.py`). Results JSON must contain n, label_source, protocol, CI, device, command. Return a <= 120-word summary with the registry id, verdict, headline numbers and file paths.
