# eval/exp_adapt — Z1 OOD error reduction

Track Z1 (domain adaptation follow-up). Question: can the out-of-domain error of the
frozen PoC be cut *without* touching the human OOD labels? Methods: sign-stable feature
selection, unsupervised alignment (per-domain standardisation / ZCA whitening, CORAL
source↔target), per-clip normalisation, a factory-grouped few-shot curve, and honest
self-training on the unlabelled Egocentric-10K corpus.

## Run

```bash
export JADAL_BUNDLE=/workspace/jadal_pod_bundle
export JADAL_E5_INPUTS=/workspace/wt/exp-domain-adapt/eval/results/exp_domain/inputs
OMP_NUM_THREADS=2 python3 eval/exp_adapt/run_adapt.py
```

CPU only (numpy + sklearn + scipy). No GPU, no network. Fixed seeds.

## Inputs

| input | source | label source |
|---|---|---|
| IndEgo 150 windows, 26 per-pixel feats | `$JADAL_BUNDLE/caches/gpu/features.json` + `labels/manifest_gold.json` | human (single labeller) |
| IndEgo EG2 window means (768-d) | `$JADAL_BUNDLE/caches/frames10k/segments.npz` | human window labels |
| OOD-60 26 feats (native ~30fps + 10fps-effective) + EG2 means | `$JADAL_E5_INPUTS/ood_pixel_selfdec.npz`, `ood_eg2_means.npz` (regenerate with E5 `prep_inputs.py`) | human (`ood_labels.json`, 51 good / 9 bad) |
| frozen ensemble scores | `$JADAL_BUNDLE/caches/ensemble/ood_scores.json` | human (for eval) |
| unlabelled Egocentric-10K corpus (1546 windows, 26 feats, native ~30fps) | `$JADAL_E8_JSONL` (`/workspace/artifacts/windows_rescored.jsonl`, E8) | **none** (unlabelled) |

The E5 OOD `*.npz` are git-ignored in E5; the path is configurable via
`JADAL_E5_INPUTS` and they are the frozen E5 decode of the same transcodes, so E5
and Z1 share exactly the same OOD features.

## Protocol

* **In-domain** = leave-one-clip-out over the 7 IndEgo clips, clip-bootstrap CI;
  C chosen by inner LOCO on the training clips only.
* **Out-of-domain** = head fit on IndEgo, scored on the 60 human-labelled
  Egocentric-10K windows; window-bootstrap CI (**optimistic**, ~1 window/factory).
  The OOD set has **n_bad = 9**; every OOD number is **exploratory**.
* **Classification of transfer** follows the E5.03 rule (chance iff both |AUROC−0.5|<0.05).

Results: `eval/results/exp_adapt/Z1.01.json` … `Z1.03.json`. Report:
`docs/reports/exp_domain_adapt_pod.md`.
