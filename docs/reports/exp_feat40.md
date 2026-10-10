# F40 — per-pixel features 26 → 40: accuracy gain (in-domain LOCO + OOD grouped CI)

Branch `m/feat40` (off `pod/base`) · experiment **F40.01** (registry
`experiments/registry.jsonl`) · results `eval/results/exp_feat40/` · figure
`docs/experiments/exp_feat40/images/feat40.png` · device: CPU on the A40 pod host.

## Goal

Expand the per-pixel feature vector from **26 to 40** by adding **14** documented
features and measure the accuracy gain in-domain (LOCO) and out-of-domain
(factory-grouped CI on the 160 human-labelled Egocentric-10K windows), plus a
one-shot score of the 108 round-3 hold-out.

## Baselines to beat (documented, frozen)

| view | protocol | AUROC | source |
|---|---|---|---|
| 26 per-pixel | in-domain LOCO | **0.810** | `tools/port/baseline_logistic_loco.py` |
| frozen ensemble | OOD-160 (grouped, optimistic) | **0.566** | `docs/FACTS.md` |
| O4 adapted recipe (stable-8 + Z3-6) | OOD-160 pooled | **0.781** [0.634, 0.895] | `m/acc-adapt` N1.01 |
| O4 adapted recipe | hold-out 108 | **0.676** [0.525, 0.827] | `m/confirm` C1.01 |

## The 14 added features

See `docs/experiments/exp_feat40/README.md` for definitions and per-window cost.
Six are the **Z3 transferable set** (`ct_fsize_std`, `mo_hand_motion_excess`,
`inv_clip_frac`, `inv_laplacian_var`, `inv_hist_entropy`, `inv_dct_hf_lf`),
reused verbatim from the frozen caches. Eight are **fresh**:
three finer spectral radial bands (`spec_b1/b2/b3`), two colour-channel ratios
(`col_rg_ratio`, `col_bg_ratio`), two temporal-lag differences
(`temp_lag2`, `temp_lag3`) and a per-pixel noise floor (`noise_floor`).

## Results

(filled from `eval/results/exp_feat40/eval_feat40.json`)

## Ablation

(filled)

## Cost

(filled)

## Honesty / limits

(filled)
