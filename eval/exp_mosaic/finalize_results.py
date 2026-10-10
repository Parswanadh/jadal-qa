"""Post-process E2 result JSONs: add top-level `ci95` (+ `headline`) so every
file carries n, label_source, protocol, ci95, device, cost, command, git_sha.

Also prints the markdown table used in docs/reports/exp_mosaic.md.
Regen: python3 eval/exp_mosaic/finalize_results.py
"""
import json
import os

R = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "results", "exp_mosaic")
REF = "N9"

for eid in ("E2.01", "E2.02", "E2.03", "E2.04", "E2.05", "E2.06", "E2.07"):
    p = os.path.join(R, f"{eid}.json")
    d = json.load(open(p))
    if "results" in d:  # E2.01 / E2.02 grid sweeps
        d["ci95"] = d["results"][REF]["indego"]["ci95"]
        d["headline"] = {"indego": d["results"][REF]["indego"],
                         "ood": d["results"][REF]["ood"]}
    else:
        d["ci95"] = d["indego"]["ci95"]
        d["headline"] = {"indego": d["indego"], "ood": d["ood"]}
    json.dump(d, open(p, "w"), indent=1)

# report table
p1 = json.load(open(os.path.join(R, "E2.01.json")))
p2 = json.load(open(os.path.join(R, "E2.02.json")))
print("| variant | frames/window (fps_used) | IndEgo AUROC [CI95] | OOD AUROC [CI95] | d IndEgo vs mean |")
print("|---|---|---|---|---|")
for k in p1["results"]:
    m = p1["results"][k]
    f = p2["results"][k]
    print(f"| mosaic {k} | {m['n_frames']} ({m['fps_used']:.1f}) | "
          f"{m['indego']['auroc']:.3f} [{m['indego']['ci95'][0]:.2f},{m['indego']['ci95'][1]:.2f}] | "
          f"{m['ood']['auroc']:.3f} | {m['delta_vs_meanframes']:+.3f} |")
    print(f"| mean-of-frames {k} | {f['n_frames']} ({f['fps_used']:.1f}) | "
          f"{f['indego']['auroc']:.3f} [{f['indego']['ci95'][0]:.2f},{f['indego']['ci95'][1]:.2f}] | "
          f"{f['ood']['auroc']:.3f} | 0.000 |")
