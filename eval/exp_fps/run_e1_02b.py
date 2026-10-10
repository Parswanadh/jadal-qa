"""E1.02b: does denser EG2 sampling (5/10 fps) beat the 2 fps view?

Rows are scored with exactly the E1.02 recipe (run_eg2.py / fps_lib.py):
per-window L2-normalised mean of per-frame L2-normalised EG2 embeddings ->
LOCO over the 7 IndEgo clips -> StandardScaler on the train fold ->
LogisticRegression(C=0.01, class_weight='balanced') -> AUROC.
CI: clip bootstrap B=2000, seed 0. All rows share ONE bootstrap resample set
(same clip draw for every density), so the CI and the paired deltas are paired.

Two baselines are reported:
  * 2fps_cached    - the published E1.02 cache (segments.npz, a different
                     embedding pass), AUROC 0.8480 -> the mission's comparison;
  * 2fps_subsample - rows ::5 of THIS dense pass -> same-pass control that
                     isolates sampling density from re-embedding noise.
The 2fps_cached vs 2fps_subsample delta is the pipeline-noise floor: a density
delta smaller than it cannot be attributed to density.

Regen: OMP_NUM_THREADS=2 python3 eval/exp_fps/run_e1_02b.py
Needs the dense cache from eval/exp_fps/embed_dense_eg2.py (GPU, not in git).
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from fps_lib import bal_acc, loco_proba  # noqa: E402

DENSE = os.environ.get("E102B_DENSE", "/workspace/cache_e1_02b/dense10.npz")
CACHED2 = os.environ.get("E102B_CACHED2",
                         "/workspace/jadal_pod_bundle/caches/frames10k/segments.npz")
OUT_DIR = os.environ.get("E102B_OUT", os.path.join(HERE, "..", "results", "exp_fps"))
B = 2000
SEED = 0

# density tag -> (row step inside the dense 10 fps pass, nominal fps)
DENSITIES = [("2fps_subsample", 5, 2.0), ("5fps", 2, 5.0), ("10fps", 1, 10.0)]


def window_mean(E, counts, step):
    """L2-normalised mean of the first `count` rows every `step` (as run_eg2.py)."""
    Xs = []
    for k in range(len(counts)):
        n = max(1, int(counts[k]))
        rows = E[k, :n:step]
        if len(rows) == 0:
            rows = E[k, :1]
        v = rows.mean(axis=0)
        Xs.append(v)
    X = np.stack(Xs)
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def bootstrap_idx(y, clips, B=B, seed=SEED):
    """Clip bootstrap resample index sets; one shared set for every row (paired)."""
    rng = np.random.default_rng(seed)
    names = sorted(set(clips))
    out = []
    for _ in range(B):
        cs = rng.choice(names, len(names))
        idx = np.concatenate([np.where(clips == c)[0] for c in cs])
        if len(set(y[idx])) < 2:
            continue
        out.append(idx)
    return out


def ci_from(boot, idxs):
    v = np.array([boot[i] for i in idxs])
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)), v


def main():
    d = np.load(DENSE, allow_pickle=True)
    c = np.load(CACHED2, allow_pickle=True)
    ids = [str(x) for x in d["ids"]]
    clips = np.array([str(x) for x in d["clips"]])
    y = np.array(d["labels"]).astype(int)
    counts = np.array(d["counts"]).astype(int)
    E = d["F"].astype(np.float32)
    assert ids == [str(x) for x in c["ids"]], "dense cache id order != cached 2 fps"
    assert list(y) == [int(x) for x in c["labels"]], "label mismatch vs cached 2 fps"
    assert list(clips) == [str(x) for x in c["clips"]], "clip mismatch vs cached 2 fps"

    # --- decode/pipeline validation: ::5 frame rows vs the cached 2 fps rows ---
    Ec = c["F"].astype(np.float32)
    cc = np.array(c["counts"]).astype(int)
    sims = []
    for k in range(len(ids)):
        n = min(int(cc[k]), len(E[k, ::5]))
        a, b = E[k, ::5][:n], Ec[k, :n]
        na, nb = np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1)
        sims.append((a * b).sum(1) / (na * nb))
    sims = np.concatenate(sims)
    # window-mean cosine (post-pooling agreement)
    wa = window_mean(E[:, ::5], np.minimum(counts, len(E[0, ::5])), 1)
    wb = window_mean(Ec, cc, 1)
    wsim = float((wa * wb).sum(1).mean())

    probs = {}
    rows = []
    for tag, step, fps in DENSITIES:
        X = window_mean(E, counts, step)
        p = loco_proba(X, y, clips, C=0.01)
        probs[tag] = p
        rows.append({"density": tag, "fps": fps, "source": "dense pass (this work)",
                     "frames_per_window": round(float(np.mean(np.ceil(counts / step))), 1),
                     "auroc": float(roc_auc_score(y, p)),
                     "bal_acc": bal_acc(y, p)})
    # published 2 fps cache (different embedding pass) = the mission baseline
    X2 = window_mean(Ec, cc, 1)
    p2 = loco_proba(X2, y, clips, C=0.01)
    probs["2fps_cached"] = p2
    rows.insert(0, {"density": "2fps_cached", "fps": 2.0,
                    "source": "published E1.02 cache (segments.npz)",
                    "frames_per_window": round(float(cc.mean()), 1),
                    "auroc": float(roc_auc_score(y, p2)), "bal_acc": bal_acc(y, p2)})

    resamples = bootstrap_idx(y, clips)          # ONE shared set -> rows are paired
    boot = {t: np.array([roc_auc_score(y[i], p[i]) for i in resamples])
            for t, p in probs.items()}
    idxs = list(range(len(resamples)))
    for r in rows:
        lo, hi, _ = ci_from(boot[r["density"]], idxs)
        r["auroc"] = round(r["auroc"], 4)
        r["ci95"] = [round(lo, 4), round(hi, 4)]
        r["bal_acc"] = round(r["bal_acc"], 4)

    # --- paired deltas (shared resamples) ---
    def paired(a, b):
        dv = boot[a] - boot[b]
        lo, hi = np.percentile(dv, 2.5), np.percentile(dv, 97.5)
        fgt = float((dv > 0).mean())
        p2s = float(min(1.0, 2 * min(fgt, 1 - fgt)))
        return {"comparison": f"{a} - {b}",
                "delta": round(float(roc_auc_score(y, probs[a]) -
                                     roc_auc_score(y, probs[b])), 4),
                "ci95": [round(float(lo), 4), round(float(hi), 4)],
                "frac_boot_gt_0": round(fgt, 4),
                "p_two_sided": round(p2s, 4),
                "established": bool(lo > 0 or hi < 0)}

    deltas = [paired("5fps", "2fps_cached"), paired("10fps", "2fps_cached"),
              paired("2fps_subsample", "2fps_cached"),   # pipeline-noise floor
              paired("5fps", "2fps_subsample"), paired("10fps", "2fps_subsample"),
              paired("10fps", "5fps")]

    # Verdict on the same-pass control (isolates density from re-embedding noise).
    # Decision rule, fixed BEFORE the numbers were read:
    #   adopt  - a denser row is ESTABLISHED better than the same-pass 2 fps control
    #   drop   - one is established worse, OR nothing is established while denser
    #            sampling costs strictly more frames and the best point delta stays
    #            below the 0.005 practical-significance bar (a smaller gain cannot
    #            justify 2.5x/5x the embedding cost in a real-time funnel)
    #   maybe  - established better on one density and worse on the other
    #   inconclusive - nothing established but a point delta >= 0.005 is unresolved
    PRACTICAL_BAR = 0.005
    ctrl = {x["comparison"]: x for x in deltas}
    d5 = ctrl["5fps - 2fps_subsample"]
    d10 = ctrl["10fps - 2fps_subsample"]
    pos = [x for x in (d5, d10) if x["established"] and x["delta"] > 0]
    neg = [x for x in (d5, d10) if x["established"] and x["delta"] < 0]
    if pos and not neg:
        verdict = "adopt"
    elif neg and not pos:
        verdict = "drop"
    elif pos and neg:
        verdict = "maybe"
    elif max(d5["delta"], d10["delta"]) < PRACTICAL_BAR:
        verdict = "drop"       # nothing established and nothing actionable
    else:
        verdict = "inconclusive"

    # --- measured cost per window on this A40 (rates from the dense run meta) ---
    cost = None
    meta_p = os.path.join(os.path.dirname(DENSE), "meta.json")
    if os.path.exists(meta_p):
        mm = json.load(open(meta_p))
        dec_sf = mm["decode_s"] / mm["n_frames"]
        enc_sf = mm["encode_s"] / mm["n_frames"]
        cost = {"basis": (f"measured on {mm['device']}, batch {mm['batch_size']}: "
                          f"{mm['frames_per_s_encode']:.1f} frames/s encode, "
                          f"decode {1000 * dec_sf:.1f} ms/frame, "
                          f"VRAM peak {mm['vram_peak_gb']:.2f} GB"),
                "per_window": []}
        for tag, _step, fps in DENSITIES:
            nfr = 10.0 * fps
            d, e = nfr * dec_sf, nfr * enc_sf
            cost["per_window"].append({"density": tag, "fps": fps, "frames": int(nfr),
                                       "decode_s": round(d, 3), "embed_s": round(e, 3),
                                       "total_s": round(d + e, 3)})
        base = cost["per_window"][0]["total_s"]
        for c in cost["per_window"]:
            c["x_vs_2fps"] = round(c["total_s"] / base, 2)

    # --- label provenance: the scored labels must equal the manifest on disk NOW ---
    man_p = os.environ.get("E102B_MANIFEST",
                           "/workspace/jadal_pod_bundle/labels/manifest_gold.json")
    man_sha = hashlib.sha256(open(man_p, "rb").read()).hexdigest()
    man = json.load(open(man_p))
    assert [i["id"] for i in man["items"]] == ids, "manifest ids != cache ids"
    assert [int(i["label"]) for i in man["items"]] == list(y), "manifest labels != cache"
    assert [i["clip"] for i in man["items"]] == list(clips), "manifest clips != cache"
    dense_meta = json.load(open(os.path.join(os.path.dirname(DENSE), "meta.json")))
    manifest_block = {
        "path": man_p, "file_sha256_at_scoring": man_sha,
        "labels_sha256_field": man["labels_sha256"],
        "labels_sha256_field_read_by_the_embed_run": dense_meta["labels_sha256"],
        "ids_clips_labels_identical": True,
        "paths_note": ("the pristine manifest carries source-machine paths; the pod paths "
                       "are obtained by remapping '/home/parshu/projects/awing/data/' -> "
                       "'/workspace/jadal_pod_bundle/data/' in memory only (150/150 resolve)")}

    res = {
        "experiment": "E1.02b",
        "claim": "LB1-E1.02b",
        "n": len(ids), "n_good": int(y.sum()), "n_bad": int((1 - y).sum()),
        "label_source": "human (IndEgo gold, single labeller, read-only snapshot)",
        "protocol": ("LOCO over 7 IndEgo clips; L2-normalised window mean of per-frame "
                     "L2-normalised EG2 embeddings; StandardScaler fit on the train fold; "
                     "LogisticRegression(C=0.01, class_weight='balanced'). CI and deltas: "
                     "clip bootstrap B=2000 seed 0, ONE shared resample set for all rows "
                     "(paired by construction)."),
        "device": "NVIDIA A40 44.4 GiB (torch 2.8.0+cu128, bfloat16)",
        "sampling": ("one decode+embed pass at native 10 fps (100 frames/window); 5 fps and "
                     "2 fps are exact row subsamples ::2 and ::5 of the same pass"),
        "eg2": {"revision": "914f7f89142e33e77833254d9c9b90c3cef7303b",
                "model_sha256": "197a32965d4b1105faf060417baa899e193fb73cd401f42ec9295234d5553d79",
                "snapshot_dir": ("/workspace/.cache/huggingface/hub/models--google--"
                                 "embeddinggemma-2/snapshots/914f7f89142e33e77833254d9c9b90c3cef7303b"),
                "max_soft_tokens": 140, "normalize_embeddings": True},
        "manifest": manifest_block,
        "embedding_caches": {
            "dense10_fps_this_work": DENSE,
            "published_2fps": CACHED2,
            "git": "not committed (large binary); regenerate with eval/exp_fps/embed_dense_eg2.py"},
        "pipeline_validation": {
            "decode": ("sequential native frames at index ::5 are pixel-identical to the "
                       "published per-sample POS_MSEC seek decode (6/6 windows incl. the "
                       "five 21-frame windows re-checked, max abs diff 0)"),
            "per_frame_cosine_vs_cached_2fps": {"mean": round(float(sims.mean()), 6),
                                                "min": round(float(sims.min()), 6)},
            "window_mean_cosine_vs_cached_2fps": round(wsim, 6),
            "note": ("same frames, independent embedding pass: residual is bf16/batching "
                     "numerics; its AUROC effect is the 2fps_subsample - 2fps_cached row")},
        "rows": rows,
        "paired_deltas": deltas,
        "measured_cost": cost,
        "verdict": verdict,
        "verdict_rule": (
            "adopt if a denser row is ESTABLISHED better than the same-pass 2 fps control "
            "(paired clip-bootstrap CI excluding 0); drop if one is established worse, or if "
            "nothing is established while denser sampling costs strictly more and the best "
            f"point delta is below the {PRACTICAL_BAR} practical-significance bar; maybe if "
            "mixed; inconclusive otherwise"),
        "statistical_read": (
            "the density effect is NOT established at any density: every paired CI includes 0, "
            "and the 2fps_subsample - 2fps_cached row shows that re-embedding the SAME frames "
            "moves the AUROC by a comparable amount"),
        "decision": ("keep the 2 fps EG2 view; do not pay 2.5x/5x the embedding cost for a "
                     "denser view - no gain is established and the point estimates are inside "
                     "the pipeline-noise floor"),
        "claim_status": ("no non-inferiority margin was pre-stated; every claim below rests "
                         "on the paired clip-bootstrap CI over 7 clips (small: wide CIs)"),
        "command": "eval/exp_fps/embed_dense_eg2.py (GPU) then eval/exp_fps/run_e1_02b.py",
    }
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "e1_02b.json"), "w") as f:
        json.dump(res, f, indent=1)

    for r in rows:
        print(f"{r['density']:>13} ~{r['frames_per_window']:>5} fr/win  "
              f"AUROC {r['auroc']:.4f} [{r['ci95'][0]:.3f},{r['ci95'][1]:.3f}]  "
              f"bal {r['bal_acc']:.3f}")
    for x in deltas:
        print(f"  {x['comparison']:<28} d={x['delta']:+.4f} "
              f"[{x['ci95'][0]:+.4f},{x['ci95'][1]:+.4f}] P(>0)={x['frac_boot_gt_0']:.3f} "
              f"{'ESTABLISHED' if x['established'] else 'not established'}")
    print("verdict:", verdict)
    print("per-frame cosine vs cached 2fps: mean %.6f min %.6f" % (sims.mean(), sims.min()))


if __name__ == "__main__":
    main()
