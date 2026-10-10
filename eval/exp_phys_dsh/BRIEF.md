# DSH-PHYS BRIEF (read first; everything here is verified fact)

WORKTREE: `/workspace/wt/dsh-phys`  BRANCH: `dsh/phys-v2` (base = origin/main).
Other worktrees/branches belong to OTHER agents: never touch them, never `git checkout` a branch,
never `git worktree` anything. Only commit on `dsh/phys-v2`. Commit prefix `dsh:`.
Hermetic rule: sandbox blocks `multiprocessing` (SemLock PermissionError) — use
`concurrent.futures.ThreadPoolExecutor`, never `mp.Pool`.

## Mission
Raise the accuracy of the DETERMINISTIC PHYSICS RULE MODEL (fixed constants, integer Q=4
comparisons, no learned weights in the decision path, byte-identical receipts).
In-domain first (IndEgo), then survive the domain shift WITHOUT labels.

## DATA (all under BUNDLE=/workspace/jadal_pod_bundle, read-only)
- `caches/perpixel/gold.npz`: IndEgo gold 150 windows. keys: `X` (150,121) float64,
  `names` (121,) = feature names prefixed `B_` (per-frame basic), `G_` (GPU window aggregate),
  `P_` (perpixel series); `y` (1=good), `ids`, `clips` (7 clips), `T`. THE in-domain pool.
  Map from rule-table names: `full.luma_mean`=`G_luma_mean`, `full.hf_ratio`=`G_hf_ratio`,
  `full.dct_energy`=`G_dct_energy`, `full.temp_b0`=`G_temp_b0`, `full.frozen_frac`=`G_frozen_frac`,
  `full.motion_energy`=`G_motion_energy`, `full.skin_frac`=`G_skin_frac`,
  `full.bright_frac`=`G_bright_frac`, `full.dark_frac`=`G_dark_frac`, `full.luma_p05`=`G_luma_p05`.
- `caches/perpixel/theory_feats.npz`: `F` (150,18) physics/theory feats, `ids`, `meta` (json str).
- `caches/perpixel/series_gold.npz`: `S` (150,100,8) per-frame series, `ids`.
- `caches/gpu/features.json`: 26 GPU per-pixel features (window-level).
- `labels/manifest_gold.json`: 150 items {id, clip, start_s, end_s, label (1=good,0=bad), reason}.
- `labels/ood160_labels.json`: 160 human-labelled Egocentric-10K windows, n_good=138 n_bad=22
  (reasons NO_ACTIVE_WORK 19, SHAKY 1, HANDS_NOT_VISIBLE 2). Videos:
  `/workspace/jadal_pod_bundle/data/ego10k_scale/label_windows{,_2}/<window>_s_<hash>.mp4`, 10 s each.
- `labels/round3/ood_labels3_108_with_factory.json`: CONFIRMATORY HOLD-OUT 108 windows.
  **DO NOT TUNE, FIT, THRESHOLD OR SELECT ON IT.** It is scored exactly once, at the end, by the lead.

## FROZEN FEATURE MATRICES (built with the frozen vendored kernels, use these)
- `eval/results/exp_phys_dsh/inputs/ood160_full.npz` — OOD 160, keys `ids,y,names,X,clips,meta,errors`.
  X columns = `full.*` + `phys.*` (same names as the rule table). Built by
  `eval/exp_phys_dsh/build_matrix.py --set ood160` (threads).
- IndEgo equivalent in the SAME name space: `eval/results/exp_phys_dsh/inputs/indego_full.npz`
  (built by the lead from the IndEgo mp4s under `data/indego/*/mp4_480/*.mp4`, 6 files available;
  if absent, use gold.npz + the name map above and say so in your JSON).

## EXISTING MODEL (read-only, cite it)
- `eval/results/exp_phys/rule_table.json` (Q1, PR #141) — quantile thresholds p01/p05/p10/p25
  (mirrored for high-is-bad), score = count of crossed thresholds, REJECT if score>=14 or a
  QUALITY rule crosses its extreme threshold.
- `tools/phys/{rules.json,model.py,score_window.py,README.md}` (Q4, PR #144) — packaged scorer.
- `docs/reports/exp_phys_determinism.md` (Q3, PR #140) — determinism proof.
- Current: in-domain IndEgo AUROC 0.822 [0.715,0.861], BA 0.808 decided, REVIEW 0.60 @5% false-reject,
  decision 2 ms. OOD FAILS (0.475-0.493; 99.55% reject on the corpus).
- Per-rule LOCO AUROC: BLUR 0.695, CODING_NOISE 0.695, NO_ACTIVE_WORK 0.695, DUPLICATE 0.675,
  HANDS_NOT_VISIBLE 0.610, OVEREXPOSED 0.565, TOO_DARK 0.550, FROZEN_STATIC 0.500.

## PROTOCOL (binding)
- In-domain metric: leave-one-clip-out (LOCO) over the 7 IndEgo clips, clip-bootstrap 95% CI.
- OOD metric: window-bootstrap 95% CI (optimistic; state it), n=160, human labels, tiny n_bad=22.
- Tune ONLY on training clips. Never on the 108 hold-out. Never on OOD labels for fitting
  (OOD labels may be used to *report* transfer, never to pick a threshold).
- DETERMINISM: every threshold integer at Q=4 (i.e. `value_q < threshold_q` with ints), fixed
  constants, no learned weights, no RNG. Decision emits `{rule, feature, value, op, threshold}`.
  Receipt = sha256 of canonical (sorted-key, fixed-format) JSON; must be byte-identical across runs.
- Every result JSON needs: n, label source, protocol, the exact command, device.
- Register experiments: `python3 tools/exp.py new ...` before running, `finish` after
  (registry `experiments/registry.jsonl`). If `tools/exp.py` is unusable, append a JSONL line
  yourself with the same fields and say so.
- Honesty: negative results reported. No "novel"/"first"/"SOTA". n_bad=22 on OOD is small — say so.

## DELIVERABLES (lead owns final assembly)
- `configs/phys_rules_v2.json` (rule table + integer weights + per-threshold derivation)
- updated `tools/phys/` scorer
- results JSONs in `eval/results/exp_phys_dsh/*.json`
- report `docs/reports/exp_phys_dsh.md`
