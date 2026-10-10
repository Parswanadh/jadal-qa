"""Arm B CLI: extract spectral features, score with a standalone LOCO
logistic baseline, ablate feature families, and write the arm output file.

Reads the harness manifest when present, else builds items read-only from
funnel/eval/gold/segments.json + labels.json (never writes there).

Outputs:
  spectral_data/arms/B/<manifest>.json  (gold now; synth/dups when ready)
  eval/results/spectral/features_B.json  (ablation + LOCO metrics + RTF)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from jadal.spectral.features import (  # noqa: E402
    ALL_FEATURES, FAMILIES, TIERS, compressed_stats, decode_pipe,
    FrameAccumulator,
)
from jadal.spectral.stream import SpectralStream  # noqa: E402

DATA = "/home/parshu/projects/jadal-qa-wt/spectral_data"
GOLD_SEG = "/home/parshu/projects/jadal-qa-wt/funnel/eval/gold/segments.json"
GOLD_LAB = "/home/parshu/projects/jadal-qa-wt/funnel/eval/gold/labels.json"


def probe_duration(path):
    cap = cv2.VideoCapture(path)
    n = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
    cap.release()
    return float(n / fps) if n and fps else 10.0


def load_items(manifest_name="manifest_gold.json"):
    mp = os.path.join(DATA, manifest_name)
    if os.path.exists(mp):
        m = json.load(open(mp))
        if "items" in m and m["items"] and "path" in m["items"][0]:
            items = m["items"]
            if manifest_name == "manifest_synth.json":
                gm = json.load(open(os.path.join(DATA, "manifest_gold.json")))
                gclip = {i["id"]: i["clip"] for i in gm["items"]}
                for it in items:
                    it.setdefault("clip", gclip.get(it.get("source_id", ""), "synth"))
                    if "start_s" not in it:
                        it["start_s"] = 0.0
                        it["end_s"] = probe_duration(it["path"])
            return items, {"source": manifest_name,
                           "snapshot": m.get("snapshot_utc")}
    segs = json.load(open(GOLD_SEG))["segments"]
    labs = json.load(open(GOLD_LAB))["latest"]
    items = []
    for s in segs:
        sid = s["segment_id"]
        if sid not in labs:
            continue
        items.append({"id": sid, "clip": s["clip"], "path": s["path"],
                      "start_s": s["start_s"], "end_s": s["end_s"],
                      "label": 1 if labs[sid]["verdict"] == "good" else 0,
                      "reason": labs[sid].get("reason")})
    sha = hashlib.sha256(open(GOLD_LAB, "rb").read()).hexdigest()[:16]
    return items, {"source": f"direct-gold labels_sha={sha} n={len(items)}"}


def extract_timed(path, s, e, tier):
    t0 = time.time()
    frames, mv = decode_pipe(path, s, e, TIERS[tier]["size"])
    t_dec = time.time() - t0
    t0 = time.time()
    acc = FrameAccumulator(TIERS[tier]["size"], window=frames.shape[0] + 1)
    for i in range(frames.shape[0]):
        acc.update(frames[i])
    fps_h = frames.shape[0] / max(e - s, 1e-9)
    feats = acc.running_features(fps_hint=fps_h)
    t_math = time.time() - t0
    t0 = time.time()
    feats["mv_per_frame"] = mv
    feats.update(compressed_stats(path, s, e, fps_hint=fps_h))
    t_cs = time.time() - t0
    lat = t_dec + t_math + t_cs
    return {k: float(feats[k]) for k in ALL_FEATURES}, lat, t_dec, t_math


def chunk_scores_stream(path, s, e, tier, clf, mu, sd):
    """Per-2s running scores via SpectralStream (cv2 chunk decode)."""
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(s * fps))
    st = SpectralStream(tier=tier, fps_hint=fps)
    scores, lat = [], []
    chunk, t = [], s
    idx = int(s * fps)
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        tt = idx / fps
        if tt >= e:
            break
        if tt >= t + 2.0 and chunk:
            t0 = time.time()
            fv = st.feed(chunk)
            lat.append(time.time() - t0)
            scores.append(float(score_one(fv, clf, mu, sd)))
            chunk, t = [], t + 2.0
        if tt >= s:
            chunk.append(fr)
        idx += 1
    cap.release()
    if chunk:
        fv = st.feed(chunk)
        scores.append(float(score_one(fv, clf, mu, sd)))
    return scores, lat


def design_matrix(items_feats, keys):
    X = np.array([[f[k] for k in keys] for _, f in items_feats],
                 dtype=np.float64)
    return np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)


def score_one(feats, clf, mu, sd, keys=ALL_FEATURES):
    from sklearn.linear_model import LogisticRegression  # noqa
    x = np.array([feats.get(k, 0.0) for k in keys], dtype=np.float64)
    x = np.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)
    z = (x - mu) / sd
    return float(clf.predict_proba(z.reshape(1, -1))[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="manifest_gold.json")
    ap.add_argument("--tiers", default="T0,T1")
    ap.add_argument("--max-items", type=int, default=0)
    ap.add_argument("--out-suffix", default="")
    ap.add_argument("--no-stream", action="store_true")
    ap.add_argument("--train-manifest", default="",
                    help="if set, train head on this manifest (pooled) and "
                         "score --manifest items (transfer mode for synth)")
    args = ap.parse_args()

    # Owner-operated CLI, but keep user-supplied strings out of paths:
    # manifests must be plain basenames inside DATA, tiers must be known,
    # and the suffix may not contain path separators or parent refs.
    for m in [args.manifest, args.train_manifest]:
        if m and ("/" in m or "\\" in m or ".." in m):
            ap.error("--manifest/--train-manifest must be a plain basename")
    tiers = [t.strip() for t in args.tiers.split(",") if t.strip()]
    for t in tiers:
        if t not in TIERS:
            ap.error(f"--tiers must be a subset of {sorted(TIERS)}")
    if "/" in args.out_suffix or "\\" in args.out_suffix or ".." in args.out_suffix:
        ap.error("--out-suffix may not contain path separators or '..'")
    args.tiers = ",".join(tiers)

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler

    items, meta = load_items(args.manifest)
    if args.max_items:
        items = items[:args.max_items]
    print(f"[armB] items={len(items)} source={meta['source']}", flush=True)
    goods = sum(1 for i in items if i["label"] == 1)
    print(f"[armB] n_good={goods} n_bad={len(items) - goods}", flush=True)

    for tier in args.tiers.split(","):
        t_all = time.time()
        feats, lats, decs, maths = {}, [], [], []
        for it in items:
            f, l, d, m = extract_timed(it["path"], it["start_s"], it["end_s"], tier)
            feats[it["id"]] = f
            lats.append(l)
            decs.append(d)
            maths.append(m)
        print(f"[armB] {tier} extracted {len(feats)} segs in "
              f"{time.time() - t_all:.1f}s", flush=True)

        data = [(it["clip"], it["id"], feats[it["id"]]) for it in items]
        y = {it["id"]: it["label"] for it in items}
        clips = sorted(set(it["clip"] for it in items))
        transfer: dict = {}

        if args.train_manifest:
            titems, tmeta = load_items(args.train_manifest)
            print(f"[armB] transfer: training head on {len(titems)} "
                  f"items from {tmeta['source']}", flush=True)
            tfeats = {}
            for it in titems:
                f, _, _, _ = extract_timed(it["path"], it["start_s"],
                                           it["end_s"], tier)
                tfeats[it["id"]] = f
            ty = {it["id"]: it["label"] for it in titems}
            tdata = [(it["clip"], it["id"], tfeats[it["id"]]) for it in titems]
            tclips = sorted(set(it["clip"] for it in titems))

            def fit_head(dd, yy, keys):
                X = design_matrix([(i, f) for i, f in dd], keys)
                sc = StandardScaler().fit(X)
                clf = LogisticRegression(max_iter=2000, C=1.0).fit(
                    sc.transform(X), np.array([yy[i] for i, _ in dd]))
                return clf, sc.mean_, sc.scale_

            # LOCO-OOF on train for operating threshold (5% FAR on goods)
            toof = {}
            for c in tclips:
                tr = [(cc, i, f) for cc, i, f in tdata if cc != c]
                te = [(cc, i, f) for cc, i, f in tdata if cc == c]
                Xtr = design_matrix([(i, f) for _, i, f in tr], ALL_FEATURES)
                ytr = np.array([ty[i] for _, i, _ in tr])
                if len(set(ytr)) < 2:
                    for _, i, _ in te:
                        toof[i] = 0.5
                    continue
                sc = StandardScaler().fit(Xtr)
                clf = LogisticRegression(max_iter=2000, C=1.0).fit(
                    sc.transform(Xtr), ytr)
                for _, i, f in te:
                    toof[i] = score_one(f, clf, sc.mean_, sc.scale_)
            tgoods = [toof[i] for i in toof if ty[i] == 1]
            tbads = [toof[i] for i in toof if ty[i] == 0]
            far5_thr = float(np.percentile(tgoods, 5)) if tgoods else 0.5
            # Youden-J threshold on train OOF (separates gold good/bad)
            cands = sorted(set(toof.values()))
            you_thr, you_j = 0.5, -1.0
            for t_ in cands:
                tpr = np.mean([s >= t_ for s in tgoods]) if tgoods else 0.0
                tnr = np.mean([s < t_ for s in tbads]) if tbads else 0.0
                if tpr + tnr - 1 > you_j:
                    you_j, you_thr = tpr + tnr - 1, float(t_)
            thr = float(you_thr)
            transfer = {"threshold_youden": thr, "threshold_far5": far5_thr,
                        "train_source": tmeta["source"]}
            clf, mu, sd = fit_head([(i, tfeats[i]) for i in tfeats],
                                   ty, ALL_FEATURES)
            oof = {it["id"]: score_one(feats[it["id"]], clf, mu, sd)
                   for it in items}
            fold_clf = {c: (clf, mu, sd) for c in clips}
            # per-degradation detection rate at Youden threshold + AUROC
            bydeg: dict = {}
            for it in items:
                bydeg.setdefault(it.get("degradation", "?"), []).append(
                    (oof[it["id"]], it["label"]))
            from sklearn.metrics import roc_auc_score as _auc
            perdeg = {}
            for k, v in sorted(bydeg.items()):
                det = float(np.mean([s < thr for s, _ in v]))
                perdeg[k] = {"n": len(v), "detect_rate": det}
            clean_s = [s for s, _ in bydeg.get("clean", [])]
            for k, v in sorted(bydeg.items()):
                if k == "clean":
                    continue
                try:
                    a = float(_auc([1] * len(clean_s) + [0] * len(v),
                                   clean_s + [s for s, _ in v]))
                except Exception:
                    a = float("nan")
                perdeg[k]["auroc_vs_clean"] = a
            transfer["per_degradation"] = perdeg
            print(f"[armB] {tier} youden_thr={thr:.3f} far5_thr={far5_thr:.3f} " +
                  ", ".join(f"{k}:det={v['detect_rate']:.2f}/auc={v.get('auroc_vs_clean', float('nan')):.2f}"
                             for k, v in sorted(perdeg.items()) if k != "clean"),
                  flush=True)
            # family-only transfer ablation (head per family on gold)
            transfer["family_only_auroc"] = {}
            for fam, keys in FAMILIES.items():
                c2, mu2, sd2 = fit_head([(i, tfeats[i]) for i in tfeats],
                                        ty, keys)
                s = [score_one(feats[it["id"]], c2, mu2, sd2, keys=keys)
                     for it in items]
                try:
                    a = float(roc_auc_score([y[it["id"]] for it in items], s))
                except Exception:
                    a = float("nan")
                transfer["family_only_auroc"][fam] = a
                print(f"[armB] {tier} transfer family-only {fam}: AUROC={a:.3f}",
                      flush=True)

        # LOCO with per-fold standardization (fit on train clips only).
        # Transfer mode skips this: oof/fold_clf already hold the pooled
        # gold-head scores, so `auc` below is the transfer AUROC.
        if not args.train_manifest:
            oof, fold_clf = {}, {}
            for c in clips:
                tr = [(i, f) for cc, i, f in data if cc != c]
                te = [(i, f) for cc, i, f in data if cc == c]
                if not tr or not te:
                    continue
                Xtr = design_matrix(tr, ALL_FEATURES)
                ytr = np.array([y[i] for i, _ in tr])
                if len(set(ytr)) < 2:
                    for i, _ in te:
                        oof[i] = 0.5
                    continue
                sc = StandardScaler().fit(Xtr)
                clf = LogisticRegression(max_iter=2000, C=1.0).fit(sc.transform(Xtr), ytr)
                fold_clf[c] = (clf, sc.mean_, sc.scale_)
                for i, f in te:
                    oof[i] = score_one(f, clf, sc.mean_, sc.scale_)

        yv = np.array([y[i] for i in feats])
        sv = np.array([oof[i] for i in feats])
        try:
            auc = float(roc_auc_score(yv, sv)) if len(set(yv)) > 1 else float("nan")
        except Exception:
            auc = float("nan")
        # bootstrap CI, clip-level resampling
        rng = np.random.default_rng(0)
        ids = list(feats)
        boots = []
        for _ in range(2000):
            samp_clips = rng.choice(clips, size=len(clips), replace=True)
            idx = [i for c in samp_clips for i in
                   [it["id"] for it in items if it["clip"] == c]]
            if len(set(y[i] for i in idx)) < 2:
                continue
            try:
                boots.append(roc_auc_score([y[i] for i in idx],
                                           [oof[i] for i in idx]))
            except Exception:
                pass
        if boots:
            lo, hi = float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
        else:
            lo, hi = float("nan"), float("nan")
        tag = "TRANSFER-AUROC" if args.train_manifest else "LOCO AUROC"
        print(f"[armB] {tier} {tag}={auc:.3f} 95%CI=({lo:.3f},{hi:.3f})",
              flush=True)

        # streaming chunk scores (score each chunk with its own clip's fold clf)
        arm_items = {}
        for j, it in enumerate(items):
            entry = {"score": float(oof[it["id"]]), "features": feats[it["id"]],
                     "latency_s": float(lats[j])}
            if not args.no_stream and it["clip"] in fold_clf:
                clf, mu, sd = fold_clf[it["clip"]]
                cs, cl = chunk_scores_stream(it["path"], it["start_s"],
                                             it["end_s"], tier, clf, mu, sd)
                entry["chunk_scores"] = cs
                entry["chunk_latency_s"] = cl
                if args.train_manifest:
                    # time-to-decision: first chunk crossing the operating
                    # (Youden) threshold. Gold LOCO mode has no single
                    # operating point, so TTD is transfer-mode only.
                    thr = transfer["threshold_youden"]
                    ttd = next((k + 1 for k, s_ in enumerate(cs)
                                if s_ < thr), None)
                    entry["time_to_decision_chunks"] = ttd
            arm_items[it["id"]] = entry

        mname = args.manifest.replace("manifest_", "").replace(".json", "")
        out = {"arm": "B", "tier": tier, "version": "spectral-features",
               "label_source": meta["source"],
               "protocol": ("transfer: pooled head trained on " +
                            args.train_manifest if args.train_manifest
                            else "LOCO clip-level"),
               "loco_auroc": auc, "loco_ci": [lo, hi],
               "transfer": transfer,
               "items": arm_items}
        os.makedirs(os.path.join(DATA, "arms", "B"), exist_ok=True)
        op = os.path.join(DATA, "arms", "B", f"{mname}_{tier}{args.out_suffix}.json")
        json.dump(out, open(op, "w"))
        print(f"[armB] wrote {op}", flush=True)

        # ablation: leave-one-family-out + family-only (gold mode only;
        # transfer mode reports family_only_auroc on synth instead)
        abl = {"loco_auroc_full": auc, "ci": [lo, hi], "leave_one_out": {},
               "family_only": {}, "transfer": transfer}
        if not args.train_manifest:
            for fam, keys in FAMILIES.items():
                for mode, kk in (("loo", [k for k in ALL_FEATURES if k not in keys]),
                                 ("only", keys)):
                    soof = {}
                    for c in clips:
                        tr = [(i, f) for cc, i, f in data if cc != c]
                        te = [(i, f) for cc, i, f in data if cc == c]
                        Xtr = design_matrix(tr, kk)
                        ytr = np.array([y[i] for i, _ in tr])
                        if len(set(ytr)) < 2 or not te:
                            for i, _ in te:
                                soof[i] = 0.5
                            continue
                        sc = StandardScaler().fit(Xtr)
                        clf = LogisticRegression(max_iter=2000, C=1.0).fit(
                            sc.transform(Xtr), ytr)
                        for i, f in te:
                            soof[i] = score_one(f, clf, sc.mean_, sc.scale_, keys=kk)
                    try:
                        a = float(roc_auc_score([y[i] for i in feats],
                                                [soof[i] for i in feats]))
                    except Exception:
                        a = float("nan")
                    abl["leave_one_out" if mode == "loo" else "family_only"][fam] = a
                    print(f"[armB] {tier} ablation {mode} {fam}: AUROC={a:.3f}", flush=True)

        # RTF (this run: wall; 1-core numbers measured separately w/ taskset)
        seg_s = float(np.mean([it["end_s"] - it["start_s"] for it in items]))
        abl["rtf"] = {
            "seg_seconds": seg_s,
            "mean_latency_total_s": float(np.mean(lats)),
            "mean_latency_decode_s": float(np.mean(decs)),
            "mean_latency_math_s": float(np.mean(maths)),
            "rtf_total": float(seg_s / np.mean(lats)),
            "rtf_no_decode": float(seg_s / max(np.mean(maths), 1e-9)),
        }
        abl["n"] = {"items": len(items), "n_good": goods,
                    "n_bad": len(items) - goods, "clips": clips,
                    "protocol": "LOCO clip-level, bootstrap CI clip-resampled"}
        os.makedirs("eval/results/spectral", exist_ok=True)
        ap2 = f"eval/results/spectral/features_B_{tier}{args.out_suffix}.json"
        json.dump(abl, open(ap2, "w"), indent=1)
        print(f"[armB] wrote {ap2}", flush=True)


if __name__ == "__main__":
    main()
