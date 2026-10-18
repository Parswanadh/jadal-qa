---
name: jadal-experiment-protocol
description: The mandatory experiment logging protocol (tools/exp.py new/finish/render), required fields, results JSON contents, verdicts and the builder's rules (log the reason, outputs and data type; negative results kept; regime-specific models). Load before running ANY experiment.
license: Apache-2.0
---
1. Before running: `python3 tools/exp.py new --id <TRACK>.<NN> --track <TRACK> --title "..." --why "<reason we run this>" --hypothesis "..." --data "<dataset + label source (human|AI|synthetic|unlabelled) + n>" --regime "<data type: resolution, fps, sharpness, lighting...>" --method "..." --owner <agent>`.
2. Run it. Write `eval/results/<name>/<id>.json` with: `n`, `label_source`, `protocol` (e.g. LOCO over 7 clips), `ci95`, `device` (A40), `cost` (frames/window, ms/window, VRAM), `command` (exact regenerating command), `git_sha`.
3. After: `python3 tools/exp.py finish --id <id> --verdict adopt|maybe|drop|inconclusive --summary "<numbers with n and CI>" --results <path> --decision "<next step>" --cost "<...>"`; then `python3 tools/exp.py render` (writes experiments/EXPERIMENTS.md).
4. Verdict rules: `adopt` only if the effect beats the baseline under the SAME folds AND the CI supports it, or the cost drops >= 4x within 0.02 AUROC; otherwise `maybe`/`drop`/`inconclusive`. Negative results are logged exactly like positive ones.
5. Tie every result to the data regime (blurry/clear, lighting, resolution, fps, walking/stationary): the product must select models/profiles per regime, so state where an effect holds and where it does not.
6. Compare with baselines computed by your own code on the same folds: EmbeddingGemma-2 window-embedding logistic (~0.85), 26 per-pixel features logistic (~0.81).
7. Never edit a past registry line; append (the registry is append-only; the latest line per id wins).
