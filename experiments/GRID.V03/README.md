# GRID.V03 — Ablations: timestamps, tile order, single frame, tile count, reasoning step, grey null

**Why:** V02 could be explained by any single frame, by the labels, or by the reasoning step rather than by the multi-frame sheet.

**Hypothesis:** Removing timestamps, shuffling tile order, or dropping to one frame changes the verdict materially; the grey null sheet is not read as usable.

**Data / label source:** The V03 stratified 20 windows (10 bad / 10 good, frozen ids in PREREG)

**Data regime:** Six arms x 20 windows: no timestamps, shuffled order, one middle frame, 6 tiles, 24 tiles, JSON-only prompt; plus a flat grey null sheet

**Method:** Per-arm AUROC and paired differences against the V02 primary arm on the same windows; the null must read cannot_tell or usable=false.

**Owner:** dsh  **Started:** 2026-10-18 08:29

## Results
(pending — `exp.py finish` fills the registry; put figures/JSON in this folder)

**Verdict (2026-10-18 09:04): inconclusive.** Six arms x 20 stratified windows (10 bad / 10 good, all 10 bad NO_ACTIVE_WORK, 14 factories) + 2 contentless-sheet controls = 122 calls, exactly the cap. NEGATIVE CONTROL PASSES: grey sheet through quality -> usable=false, HANDS_NOT_VISIBLE, cannot_tell ('All twelve tiles are blank gray...'); grey sheet through read_tiles -> 12 lines, all timestamps '--:--.-' (no hallucinated times). So the model reads the image. Arm AUROC (usable; binary so equal to balanced accuracy at 0.5): c_single_middle_frame 0.600 [0.379,0.800]; d24 0.650 [0.470,0.846]; a_no_timestamps 0.700 [0.500,0.896]; e_json_only (REFERENCE) 0.700 [0.500,0.896]; d6 0.750 [0.542,0.944]; b_shuffled 0.800 [0.586,1.000]. Every paired factory-clustered difference vs the reference INCLUDES ZERO (max +0.100 [0.000,+0.278]); the critical 12-frame vs single-frame comparison is -0.100 [-0.250,+0.111] and is additionally confounded (arm c uses the quality prompt, the reference uses quality_json_only). NOTHING IS ESTABLISHED. Read against V-A's V04 noise floor (per-run AUROC spread 0.0714, Fleiss kappa 0.52), no difference <=0.10 on 20 windows is interpretable. Notable: arms a and e returned identical verdicts on 20/20 windows (identical usable AND task_progress) on genuinely different sheets (0/20 shared PNG sha256), so within this subset neither the timestamps nor the reasoning step changes the verdict.

Results file: eval/results/exp_vlm/GRID.V03.json

Decision: INCONCLUSIVE, not a drop: the design cannot resolve arm differences at this budget. The sheet variants are not shown to matter, and the single-frame question is NOT answered. To read this table, repeat each arm several times per cell (V04-style) before comparing; do not quote any arm ordering from this run.

Cost: 122 image API calls (cap 122 = 120 arms + 2 controls); sub-stream ledger total 142 incl. V01's 20
