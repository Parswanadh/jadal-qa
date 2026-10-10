"""Arm B duplicate evaluation: spectral signatures + threshold on held-out half.

Writes spectral_data/arms/B/dups.json = {"pairs": {"a|b": {"similarity",
"latency_s"}}} and prints P/R/F1 overall + per perturbation kind.
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from jadal.spectral.dup import signature, similarity  # noqa: E402

DATA = "/home/parshu/projects/jadal-qa-wt/spectral_data"


def main():
    m = json.load(open(os.path.join(DATA, "manifest_dups.json")))
    files = m["files"]
    pairs = m["pairs"]
    sigs, lat = {}, {}
    for fid, path in sorted(files.items()):
        t0 = time.time()
        dur = 12.0
        sigs[fid] = signature(path, 0, None)
        lat[fid] = time.time() - t0
    sims = {}
    for p in pairs:
        key = f"{p['a']}|{p['b']}"
        t0 = time.time()
        s = similarity(sigs[p["a"]], sigs[p["b"]])
        sims[key] = (s, p["is_duplicate"],
                     p.get("kind", p.get("negative_kind", "?")))
    # stratified threshold split: first half of EACH (label, kind) group
    # trains the threshold, second half evaluates
    from collections import defaultdict
    groups: dict = defaultdict(list)
    for p in pairs:
        groups[(p["is_duplicate"],
                p.get("kind", p.get("negative_kind", "?")))].append(p)
    tr_pairs, te_pairs = [], []
    for g in sorted(groups):
        g = groups[g]
        h = len(g) // 2
        tr_pairs += g[:h]
        te_pairs += g[h:]
    tr = [(sims[f"{p['a']}|{p['b']}"][0], p["is_duplicate"]) for p in tr_pairs]
    te = [(sims[f"{p['a']}|{p['b']}"][0], p["is_duplicate"],
           p.get("kind", p.get("negative_kind", "?"))) for p in te_pairs]
    best_t, best_f1 = 0.5, -1.0
    for t in [i / 100 for i in range(5, 100)]:
        tp = sum(1 for s, y in tr if y and s >= t)
        fp = sum(1 for s, y in tr if not y and s >= t)
        fn = sum(1 for s, y in tr if y and s < t)
        f1 = 2 * tp / (2 * tp + fp + fn + 1e-12)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    tp = sum(1 for s, y, _ in te if y and s >= best_t)
    fp = sum(1 for s, y, _ in te if not y and s >= best_t)
    fn = sum(1 for s, y, _ in te if y and s < best_t)
    tn = sum(1 for s, y, _ in te if not y and s < best_t)
    prec = tp / (tp + fp + 1e-12)
    rec = tp / (tp + fn + 1e-12)
    print(f"threshold={best_t:.2f} (train F1={best_f1:.3f})")
    print(f"TEST: P={prec:.3f} R={rec:.3f} "
          f"F1={2 * prec * rec / (prec + rec + 1e-12):.3f} "
          f"(tp={tp} fp={fp} fn={fn} tn={tn} n={len(te)})")
    bykind: dict = {}
    for s, yv, k in te:
        d = bykind.setdefault(k, {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "n": 0})
        d["n"] += 1
        d["tp" if yv and s >= best_t else
          "fp" if s >= best_t else
          "fn" if yv else "tn"] += 1
    for k, d in sorted(bykind.items()):
        p_ = d["tp"] / (d["tp"] + d["fp"] + 1e-12)
        r_ = d["tp"] / (d["tp"] + d["fn"] + 1e-12)
        print(f"  {k}: n={d['n']} P={p_:.3f} R={r_:.3f} "
              f"fp={d['fp']} fn={d['fn']}")
    out = {"pairs": {f"{p['a']}|{p['b']}":
                     {"similarity": float(sims[f"{p['a']}|{p['b']}"][0]),
                      "latency_s": float(lat[p["a"]] + lat[p["b"]])}
                     for p in pairs},
           "threshold": best_t,
           "test": {"precision": prec, "recall": rec, "per_kind": bykind}}
    os.makedirs(os.path.join(DATA, "arms", "B"), exist_ok=True)
    json.dump(out, open(os.path.join(DATA, "arms", "B", "dups.json"), "w"))
    print("wrote spectral_data/arms/B/dups.json")


if __name__ == "__main__":
    main()
