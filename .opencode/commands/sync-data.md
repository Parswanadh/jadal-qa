---
description: Re-download the data bundle from the private HF dataset repo and remap paths
agent: lead
---
Run `HF_HUB_DISABLE_XET=1 hf download Havoc1904/jadal-pod-data --repo-type dataset --local-dir "$JADAL_BUNDLE"` then `python3 "$JADAL_BUNDLE/scripts/remap_paths.py" --bundle "$JADAL_BUNDLE" --data "$JADAL_DATA"`. Report new or changed files (for example labels/ood_labels2.json) and re-run the baselines if labels changed.
