"""Video-level determinism: 10 repeated score() of 5 gold windows (GPU, under lock).

Writes eval/results/ensemble/determinism.json {id: {identical, digests, probs}}.
Same machine + same device -> identical digests expected. EG2 numerics across
devices are NOT claimed identical.
"""
import json
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "2")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MANIFEST = "/home/parshu/projects/jadal-qa-wt/spectral_data/manifest_gold.json"
PICK = ["g000", "g029", "g100", "g120", "g140"]  # spread across clips
REPS = 10


def main():
    from jadal.ensemble.scorer import score
    M = {i["id"]: i for i in json.load(open(MANIFEST))["items"]}
    out = {}
    ok = True
    for wid in PICK:
        it = M[wid]
        ds, ps, vs = [], [], []
        for _ in range(REPS):
            r = score(it["path"], it["start_s"], it["end_s"])
            ds.append(r["features_digest"])
            ps.append(r["usable_prob"])
            vs.append(r["verdict"])
        same = len(set(ds)) == 1 and len(set(ps)) == 1
        ok &= same
        out[wid] = {"clip": it["clip"], "identical": bool(same),
                    "n_digests": len(set(ds)), "usable_prob": ps[0], "verdict": vs[0],
                    "digest": ds[0]}
        print(f"{wid} {it['clip']:12s} identical={same} p={ps[0]} {vs[0]} {ds[0][:12]}", flush=True)
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results",
                     "ensemble", "determinism.json")
    json.dump(out, open(p, "w"), indent=1)
    print("PASS" if ok else "FAIL", "->", p)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
