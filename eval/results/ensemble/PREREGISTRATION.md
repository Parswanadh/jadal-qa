# Pre-registration: ensemble_v1 out-of-domain scoring

- Frozen config: `configs/ensemble_v1.json`
- sha256: `4da1068a78d5d15021f4e167d6aa24fd2e0bb08aa88f83c83c2f167cc086fe31`
- Frozen at: 2026-10-10 (before any OOD label was read; `ood_labels/` was empty).
- OOD set: 60 Egocentric-10K windows from
  `/home/parshu/projects/jadal-qa-wt/scale-data/eval/results/scale/label_queue_random.json`
  (480p transcodes at each item's `path`; all 60 files exist).
- OOD clips are factories/workers, not IndEgo clips: CIs in `eval_ood.py` are
  by window (bootstrap) and optimistic — stated in the script output.
- Rules after this point: no tuning, no threshold changes; `eval/ensemble/eval_ood.py
  --labels <labels.json>` only reads labels and prints metrics; refuses with
  < 10 labelled windows per class; works on partial labels.
- Cached frozen scores: `eval/results/ensemble/ood_scores.json`
  (produced by `eval/ensemble_score_windows.py`; command in `docs/reports/ensemble_views.md`).
