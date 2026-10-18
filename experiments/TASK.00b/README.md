# TASK.00b — IndEgo VIA step annotations as free L1 (per-10s-slice) / L2 (per-clip completion) ground truth

**Why:** The brief asks whether the IndEgo keystep annotations that already ship with the dataset can serve as free L1/L2 ground truth instead of paid labels. If the derived L1 slice label agrees with the human gold window labels and with the shipped score direction, the annotations are a usable zero-cost eval; if it does not, they are not.

**Hypothesis:** Slices inside an annotated step agree with human gold good windows, and the shipped 26-feature LOCO score is higher on advancing slices than on no_relevant_work slices.

**Data / label source:** IndEgo VIA temporal-segment annotations (annotated_keysteps_*/AS_*.json) for the 4 gold clips that have one: ecu_full, ecu_late, desk (assembly) and organise (logistics); the other 3 logistics gold clips (pallet, pallet2, logistics4) have no annotation file. Human gold window labels + cached 26 features from manifest_gold.json.

**Data regime:** in-domain IndEgo (LOCO over the 7 clips); the L1 measure uses only the 4 annotated clips (4 groups), which makes the CIs weak

**Method:** Parse the VIA JSON (metadata[*].z = [start_s,end_s] seconds, av['1'] = step option id, attribute['1'].options = step names). L1: a gold 10-s window is advancing if an annotated step covers the majority of it, else no_relevant_work. L2: per clip, completed iff the annotation reaches the last step of that clip's task list in keysteps_*.txt. Then without any training: (a) cross-tab agreement between derived L1 and the human gold window label (rate, Cohen kappa), overall and restricted to reasons about work activity; (b) AUROC of the shipped 26-feature LOCO out-of-fold score for advancing vs no_relevant_work with clip-grouped bootstrap CI, plus the full-fit shipped-head score as a sensitivity check.

**Owner:** S5-task  **Started:** 2026-10-18 06:38

## Results
(pending — `exp.py finish` fills the registry; put figures/JSON in this folder)

**Verdict (2026-10-18 06:42): drop.** The VIA keystep annotations parse cleanly (180/180 files, 0 unparseable) and cover 4 of the 7 gold clips (ecu_full, ecu_late, desk, organise); pallet, pallet2 and logistics4 have no annotation file. Format: attribute['1'].options maps option-id -> step label, metadata[*].z = [start_s,end_s] seconds, metadata[*].av['1'] = step option-id; cv2 durations match the manifest to 0.01 s, confirming the unit. L2 is degenerate: all 4 clips reach the last step of their matched task list (4/4 completed) and each annotation omits task-list steps (desk 7 of 10, ecu_late 10 of 15, ecu_full 1 of 15, organise 1 of 3), so 'reaches the last step' is not 'did the task'. L1 is near-constant: with the majority-overlap rule 105 of 110 gold windows are advancing and only 5 are no_relevant_work (gaps are ~5% of annotated time). Agreement with human gold labels: 75.45% (always-advancing baseline 70.91%), Cohen kappa 0.208, balanced accuracy 0.578; all 5 gap windows are gold-bad (precision 1.0) but that is 5 of 32 gold-bad windows. Restricted to gold reasons about work activity (n=94): agreement 87.23%, kappa 0.356. The two label sources disagree on 12 of the 16 human NO_ACTIVE_WORK windows on these 4 clips -- the annotation places them inside a step. Shipped 26-feature LOCO score, advancing vs no_relevant_work: AUROC 0.6476 [0.4565, 0.8305] clip-grouped bootstrap (2000 draws, 4 clips), n=110, n_bad=5 -- nominally in the predicted direction but the interval includes 0.5; per-clip 0.4815 (desk, n=29), 0.8587 (ecu_full, n=48), 0.6923 (ecu_late, n=14), undefined (organise, all 19 advancing). Decision: not usable as free L1/L2 ground truth in this form on these clips -- the gap class is too rare and the L2 flag has no variance -- but the per-step segmentation itself is parseable and free, so it remains a candidate auxiliary signal if a denser gap definition or more annotated clips appear.

Results file: eval/results/task/TASK.00b.json

Decision: Do NOT use the derived L1/L2 labels as a substitute for human labels in threshold tuning or headline metrics. IndEgo steps are demo-style assembly/logistics tasks; nothing here transfers to factory work without testing it there.

Cost: 0 GPU, cached features + sklearn fits on 150x26, ~10 s CPU
