"""Egocentric-10K OOD scoring protocol (fix round 1 / I6).

The frozen artefact (configs/perpixel_v1.json + perpixel_conv.json) is
PRE-REGISTERED for ONE scoring on a new human-labelled Egocentric-10K
sample (different factories/workers). That sample does not exist yet:
this script defines the protocol and writes an empty placeholder so the
holdout story is machine-checkable.

Holdout hygiene: eval/perpixel/audit.py's AI-label probe reads Ego10K
*embeddings + AI labels* (frames10k shards) for noise diagnosis only.
The frozen scorer's weights, thresholds, and decisions never saw Ego10K
pixels; no Ego10K human label exists or was used. When the human Ego10K
sample lands: (1) freeze nothing new, (2) run this script, (3) report
once, (4) do not tune afterwards.

Usage: PYTHONNOUSERSITE=1 .../python eval/perpixel/ego10k_score.py \\
           --manifest /path/to/ego10k_manifest.json
Manifest schema: {items:[{id, clip, path, start_s, end_s, label, reason}]}.
Output: eval/results/perpixel/ego10k.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(os.path.dirname(HERE), "results", "perpixel")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=None)
    args = ap.parse_args()
    if args.manifest is None:
        ph = {"status": "placeholder",
              "note": ("no human-labelled Egocentric-10K sample available; "
                       "frozen scorer untouched. Run with --manifest when it "
                       "lands; report once; do not tune.") }
        with open(os.path.join(RES, "ego10k.json"), "w") as f:
            json.dump(ph, f, indent=1)
        print("wrote placeholder ego10k.json", flush=True)
        return
    from jadal.perpixel import scorer as SC
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score
    gm = json.load(open(args.manifest))
    ys, ss = [], []
    for it in gm["items"]:
        o = SC.score(it["path"], it["start_s"], it["end_s"])
        ss.append(o["usable_prob"])
        ys.append(it["label"])
    pred = [1 if s >= 0.5 else 0 for s in ss]
    out = {"n": len(ys),
           "auroc": float(roc_auc_score(ys, ss)),
           "bal_acc": float(balanced_accuracy_score(ys, pred))}
    with open(os.path.join(RES, "ego10k.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("wrote ego10k.json", out, flush=True)


if __name__ == "__main__":
    main()
