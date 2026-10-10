---
description: Close an experiment (arguments: id | verdict | summary with numbers and CI | results.json path | decision | cost)
agent: lead
---
Close the experiment with `python3 tools/exp.py finish` using: $ARGUMENTS. First open the results JSON and verify that every number in the summary appears there with n, label source and CI; refuse and say what is missing otherwise. Then run `python3 tools/exp.py render`.
