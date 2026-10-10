"""Spectral study scorer (harness worker).

Reads arm outputs from /home/parshu/projects/jadal-qa-wt/spectral_data/arms/*/*.json
and writes:
  eval/results/spectral/summary.json   (all numbers; reproducible)
  docs/reports/spectral_study.md       (comparison table + sections)

Protocols (see contract; all stated in the report too):
- GOLD: leave-one-clip-out CV. Arms with raw features get an IDENTICAL
  StandardScaler+LogisticRegression(C=1.0) head fitted inside train clips only.
  Arms with a score only are scored as-is AND via a 1-d logistic on the score.
  Headline: pooled LOCO AUROC + clip-level bootstrap CI (resample clips w.p.
  replacement, B=1000, seed 0; resamples with one class are skipped).
  Balanced accuracy uses a per-fold Youden threshold picked on the TRAIN fold.
- SYNTH: head fitted on ALL gold; operating threshold t = 5th percentile of
  gold-good scores (5% FAR on gold good by construction). Sensitivity per
  degradation = frac(score < t). Control FAR on the 6 clean controls reported.
- DUPS: 50/50 stratified split (seed 0); threshold = max-F1 on half 1,
  P/R/F1 reported on half 2, per method and per perturbation kind.
- COST/STREAMING from arms' cost.json / chunks.json + decode_floor.json.

Missing arm file -> NOT RUN (never silently skipped).
Usage: python -m eval.spectral.score
"""
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HARNESS = Path(__file__).resolve().parents[2]
DATA = Path("/home/parshu/projects/jadal-qa-wt/spectral_data")
RES = HARNESS / "eval" / "results" / "spectral"
REPORT = HARNESS / "docs" / "reports" / "spectral_study.md"

ARMS = ["A", "E", "B", "C", "D"]
VARIANTS = {"A-full": ("A", "allframes.json")}
FEAT_KEYS = {
    "default": ["blur_var", "brightness_mean", "under_exposed_frac",
                "over_exposed_frac", "motion_mean", "hands_visible_frac",
                "hand_count_mode", "faces_found", "faces_blurred", "decodable"],
    "A-full": ["lap_mean", "lap_std", "bri_mean", "bri_std", "mot_mean",
               "mot_std", "skin_mean", "skin_std"],
}
SEED = 0


def load(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def arm_status():
    st = {}
    for a in ARMS:
        d = DATA / "arms" / a
        st[a] = {f: (d / f).exists() for f in
                 ("gold.json", "synth.json", "dups.json", "chunks.json", "cost.json")}
    return st


# --- metrics -----------------------------------------------------------------
def auroc(y, s) -> float | None:
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y); s = np.asarray(s)
    if len(np.unique(y)) < 2:
        return None
    try:
        return float(roc_auc_score(y, s))
    except Exception:
        return None


def clip_bootstrap_ci(clip_ids, y, s, b=1000, seed=SEED):
    rng = np.random.default_rng(seed)
    clips = sorted(set(clip_ids))
    est = []
    for _ in range(b):
        draw = [clips[i] for i in rng.integers(0, len(clips), len(clips))]
        w = {}
        for c in draw:
            w[c] = w.get(c, 0) + 1
        yy, ss = [], []
        for c, i, lab, sc in zip(clip_ids, range(len(y)), y, s):
            if c in w:
                yy += [lab] * w[c]
                ss += [sc] * w[c]
        a = auroc(yy, ss)
        if a is not None:
            est.append(a)
    if not est:
        return None, 0
    return [round(float(np.percentile(est, 2.5)), 3),
            round(float(np.percentile(est, 97.5)), 3)], len(est)


def make_head():
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    return make_pipeline(StandardScaler(),
                         LogisticRegression(C=1.0, max_iter=5000))


def resolve_keys(arm_key: str, items: dict) -> list[str]:
    """Feature keys per arm: canonical for A/A-full, embedding for E,
    auto (sorted numeric keys) for anything else (e.g. provisional B tiers)."""
    if arm_key in FEAT_KEYS:
        return FEAT_KEYS[arm_key]
    v0 = next(iter(items.values()))
    f0 = v0.get("features", {})
    if "emb" in f0:
        return []
    keys = [k for k, v in f0.items()
            if isinstance(v, (int, float)) and not isinstance(v, bool)]
    return sorted(keys)


def feat_matrix(items: dict, keys: list[str]):
    X, ids = [], []
    for i, v in items.items():
        f = v["features"]
        if "emb" in f:
            X.append([float(x) for x in f["emb"]])
        else:
            X.append([float(f[k]) for k in keys])
        ids.append(i)
    return np.array(X), ids


# --- gold task ---------------------------------------------------------------
def gold_task(arm: str, gold_doc: dict, gold_manifest: dict, feat_keys: list[str]):
    """LOCO-CV calibrated scores. Returns dict with scores/labels/metrics."""
    clip_of = {it["id"]: it["clip"] for it in gold_manifest["items"]}
    lab_of = {it["id"]: it["label"] for it in gold_manifest["items"]}
    ids = [i for i in gold_doc["items"] if i in lab_of]
    y = np.array([lab_of[i] for i in ids])
    clips = [clip_of[i] for i in ids]
    has_score = any((gold_doc["items"][i].get("score") is not None) for i in ids)
    has_feat = any(bool(gold_doc["items"][i].get("features")) for i in ids)

    out = {"n": len(ids), "n_good": int((y == 1).sum()),
           "n_bad": int((y == 0).sum()),
           "n_clips": len(set(clips)), "protocol": "leave-one-clip-out"}
    uclips = sorted(set(clips))

    def run_loco(X):
        pred, thr_pred, folds = [], [], []
        for tc in uclips:
            tr = [i for i in range(len(ids)) if clips[i] != tc]
            te = [i for i in range(len(ids)) if clips[i] == tc]
            if len(set(y[tr])) < 2 or not te:
                pred += [float("nan")] * len(te)
                thr_pred += [0] * len(te)
                folds.append({"test_clip": tc, "skipped": True})
                continue
            head = make_head().fit(X[tr], y[tr])
            p = head.predict_proba(X[te])[:, 1]
            # Youden threshold on TRAIN fold
            from sklearn.metrics import roc_curve
            fpr, tpr, th = roc_curve(y[tr], head.predict_proba(X[tr])[:, 1])
            jt = th[int(np.argmax(tpr - fpr))]
            pred += [float(v) for v in p]
            thr_pred += [int(v >= jt) for v in p]
            folds.append({"test_clip": tc, "n_test": len(te),
                          "train_threshold": round(float(jt), 4)})
        return np.array(pred), np.array(thr_pred), folds

    results = {}
    if has_feat:
        X, _ = feat_matrix({i: gold_doc["items"][i] for i in ids}, feat_keys)
        out["n_features"] = X.shape[1]
        pred, thr_pred, folds = run_loco(X)
        ok = ~np.isnan(pred)
        a = auroc(y[ok], pred[ok]) if ok.sum() else None
        ci, nb = clip_bootstrap_ci([c for c, o in zip(clips, ok) if o],
                                   y[ok], pred[ok]) if ok.sum() else (None, 0)
        from sklearn.metrics import balanced_accuracy_score
        ba = float(balanced_accuracy_score(y[ok], thr_pred[ok])) if ok.sum() else None
        results["head"] = {"auroc": round(a, 3) if a is not None else None,
                           "auroc_ci95": ci, "n_boot": nb,
                           "balanced_accuracy": round(ba, 3) if ba is not None else None,
                           "folds": folds}
    if has_score:
        s = np.array([gold_doc["items"][i]["score"] for i in ids], dtype=float)
        a = auroc(y, s)
        ci, nb = clip_bootstrap_ci(clips, y, s)
        results["as_is"] = {"auroc": round(a, 3) if a is not None else None,
                            "auroc_ci95": ci, "n_boot": nb}
        X1 = s.reshape(-1, 1)
        pred, thr_pred, folds = run_loco(X1)
        ok = ~np.isnan(pred)
        a2 = auroc(y[ok], pred[ok]) if ok.sum() else None
        results["score_logistic"] = {
            "auroc": round(a2, 3) if a2 is not None else None, "folds": folds}
    out["results"] = results
    # keep raw for synth threshold + streaming
    out["_ids"] = ids
    return out


def fit_full_head(gold_doc: dict, gold_manifest: dict, feat_keys: list[str]):
    lab_of = {it["id"]: it["label"] for it in gold_manifest["items"]}
    ids = [i for i in gold_doc["items"] if i in lab_of]
    y = np.array([lab_of[i] for i in ids])
    has_score = gold_doc["items"][ids[0]].get("score") is not None
    if has_score and not gold_doc["items"][ids[0]].get("features"):
        X = np.array([gold_doc["items"][i]["score"] for i in ids]).reshape(-1, 1)
    else:
        X, _ = feat_matrix({i: gold_doc["items"][i] for i in ids}, feat_keys)
    head = make_head().fit(X, y)
    s_good = head.predict_proba(X[y == 1])[:, 1]
    return head, float(np.percentile(s_good, 5)), ids


def apply_head(head, feats: dict, feat_keys: list[str]) -> float | None:
    try:
        if "emb" in feats:
            X = np.array([[float(x) for x in feats["emb"]]])
        else:
            X = np.array([[float(feats[k]) for k in feat_keys]])
        return float(head.predict_proba(X)[0, 1])
    except Exception:
        return None


# --- main --------------------------------------------------------------------
def main() -> None:
    t0 = time.perf_counter()
    RES.mkdir(parents=True, exist_ok=True)
    gold_manifest = load(DATA / "manifest_gold.json")
    synth_manifest = load(DATA / "manifest_synth.json")
    dups_manifest = load(DATA / "manifest_dups.json")
    status = arm_status()

    feat_keys = FEAT_KEYS["default"]

    summary = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "label_source": "human gold (IndEgo workers; NOT Awign footage)",
        "gold_snapshot": {k: gold_manifest[k] for k in
                           ("snapshot_utc", "labels_sha256", "n_good", "n_bad")}
        if gold_manifest else None,
        "arms": {},
    }

    arm_docs = {}
    arm_key_order = []
    for a in ARMS + ["A-full"]:
        if a == "A-full":
            d = load(DATA / "arms" / "A" / "allframes.json")
            key = "A-full"
        else:
            d = load(DATA / "arms" / a / "gold.json")
            key = a
            if d is None:
                # provisional: score gold_<suffix>.json variants without touching them
                import glob as _glob
                for vf in sorted(_glob.glob(str(DATA / "arms" / a / "gold_*.json"))):
                    suf = Path(vf).stem[len("gold_"):]
                    try:
                        vd = json.loads(Path(vf).read_text())
                    except Exception:
                        continue
                    if isinstance(vd, dict) and vd.get("items"):
                        arm_docs[f"{a}-{suf}"] = vd
                        arm_key_order.append(f"{a}-{suf}")
        if key not in arm_docs:
            arm_docs[key] = d
        arm_key_order.append(key)
    for key in arm_key_order:
        a = key.split("-")[0] if key != "A-full" else "A"
        entry = {"status": "RUN" if arm_docs[key] else "NOT RUN",
                 "files": status.get(a, {})}
        if "-" in key and key != "A-full":
            entry["provisional"] = True
            entry["note"] = ("non-contract filename scored as-is; "
                             "final number waits for arms/B/gold.json")
        if arm_docs[key] and gold_manifest:
            try:
                akeys = (resolve_keys(key, arm_docs[key]["items"])
                         if key not in FEAT_KEYS else FEAT_KEYS[key])
                entry["gold"] = gold_task(a, arm_docs[key], gold_manifest, akeys)
                entry["feature_keys"] = akeys
            except Exception as e:  # never let one arm kill the scorer
                entry["gold"] = {"error": f"{type(e).__name__}: {e}"}
        summary["arms"][key] = entry

    # synth task
    synth_res = {}
    if synth_manifest:
        for a, d in arm_docs.items():
            base = a.split("-")[0]
            akeys = FEAT_KEYS.get(a, FEAT_KEYS["default"])
            if a == "A-full":
                af = load(DATA / "arms" / "A" / "allframes.json")
                sd = {"items": (af or {}).get("items", {})} if af else None
            else:
                sd = load(DATA / "arms" / base / "synth.json")
            if sd is None:
                synth_res[a] = {"status": "NOT RUN"}
                continue
            try:
                head, thr, _ = fit_full_head(d, gold_manifest, akeys)
                by_deg: dict[str, list] = {}
                for it in synth_manifest["items"]:
                    v = sd["items"].get(it["id"])
                    if v is None:
                        continue
                    s = apply_head(head, v["features"], akeys)
                    by_deg.setdefault(it["degradation"], []).append(
                        (s, it["label"]))
                rows = {}
                for deg, lst in sorted(by_deg.items()):
                    sc = np.array([x[0] for x in lst if x[0] is not None])
                    n_bad = sum(1 for _, lb in lst if lb == 0)
                    sens = float((sc[:len([l for l in lst if l[1] == 0])] < thr).mean()) \
                        if n_bad and len(sc) else None
                    # sensitivity over degraded items only:
                    desc = [x[0] for x in lst if x[1] == 0 and x[0] is not None]
                    sens = round(float(np.mean(np.array(desc) < thr)), 3) if desc else None
                    ctrl = [x[0] for x in lst if x[1] == 1 and x[0] is not None]
                    far = round(float(np.mean(np.array(ctrl) < thr)), 3) if ctrl else None
                    rows[deg] = {"n": len(lst), "sensitivity_at_5pct_FAR": sens,
                                 "control_FAR": far}
                synth_res[a] = {"status": "RUN", "threshold_5pct_FAR_on_gold_good": round(thr, 4),
                                "note": "threshold t = 5th pct of gold-good scores; "
                                        "item bad iff score < t. Says NOTHING about activity judgement.",
                                "per_degradation": rows}
            except Exception as e:
                synth_res[a] = {"status": "ERROR", "error": f"{type(e).__name__}: {e}"}
    summary["synth"] = synth_res

    # dups task
    dups_res = {}
    if dups_manifest:
        rng = np.random.default_rng(SEED)
        pairs = dups_manifest["pairs"]
        idx = np.arange(len(pairs))
        pos = [i for i, p in enumerate(pairs) if p["is_duplicate"]]
        neg = [i for i, p in enumerate(pairs) if not p["is_duplicate"]]
        rng.shuffle(pos); rng.shuffle(neg)
        h1 = set(pos[:len(pos) // 2]) | set(neg[:len(neg) // 2])
        for a in ARMS:
            dd = load(DATA / "arms" / a / "dups.json")
            if dd is None:
                dups_res[a] = {"status": "NOT RUN"}
                continue
            try:
                sims = []
                for p in pairs:
                    v = dd["pairs"].get(f"{p['a']}|{p['b']}",
                                        dd["pairs"].get(f"{p['b']}|{p['a']}"))
                    sims.append(None if v is None else v["similarity"])
                y = np.array([p["is_duplicate"] for p in pairs], dtype=int)
                ok1 = [i for i in range(len(pairs)) if i in h1 and sims[i] is not None]
                # threshold = max F1 on half 1
                cand = sorted(set(sims[i] for i in ok1))
                best_t, best_f1 = 0.5, -1
                for t in cand:
                    pr = np.array([1 if sims[i] >= t else 0 for i in ok1])
                    tp = int(((pr == 1) & (y[ok1] == 1)).sum())
                    fp = int(((pr == 1) & (y[ok1] == 0)).sum())
                    fn = int(((pr == 0) & (y[ok1] == 1)).sum())
                    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0
                    if f1 > best_f1:
                        best_f1, best_t = f1, t
                h2 = [i for i in range(len(pairs)) if i not in h1 and sims[i] is not None]

                def prf(ii):
                    pr = np.array([1 if sims[i] >= best_t else 0 for i in ii])
                    yy = y[ii]
                    tp = int(((pr == 1) & (yy == 1)).sum())
                    fp = int(((pr == 1) & (yy == 0)).sum())
                    fn = int(((pr == 0) & (yy == 1)).sum())
                    p_ = tp / (tp + fp) if (tp + fp) else 0
                    r_ = tp / (tp + fn) if (tp + fn) else 0
                    f_ = 2 * p_ * r_ / (p_ + r_) if (p_ + r_) else 0
                    return {"n": len(ii), "P": round(p_, 3), "R": round(r_, 3),
                            "F1": round(f_, 3)}

                kinds = sorted(set(p["kind"] for p in pairs))
                per_kind = {}
                for k in kinds:
                    ii = [i for i in h2 if pairs[i]["kind"] == k]
                    if not ii:
                        continue
                    pr = np.array([1 if sims[i] >= best_t else 0 for i in ii])
                    if pairs[ii[0]]["is_duplicate"]:
                        rec = float(pr.mean()) if len(pr) else 0.0
                        per_kind[k] = {"n": len(ii),
                                       "recall_detect_rate": round(rec, 3)}
                    else:
                        fpr = float(pr.mean()) if len(pr) else 0.0
                        per_kind[k] = {"n": len(ii),
                                       "false_positive_rate": round(fpr, 3),
                                       "specificity": round(1 - fpr, 3)}
                dups_res[a] = {"status": "RUN", "method": dd.get("method"),
                               "threshold_maxF1_half1": round(float(best_t), 4),
                               "heldout_overall": prf(h2), "per_kind": per_kind,
                               "protocol": "threshold on held-out half 1, metrics on half 2"}
            except Exception as e:
                dups_res[a] = {"status": "ERROR", "error": f"{type(e).__name__}: {e}"}
    summary["dups"] = dups_res

    # cost + throughput (amendment)
    decode_floor = load(HARNESS / "eval" / "results" / "spectral" / "decode_floor.json")
    try:
        dec_rows = decode_floor["rows"]
        decode_s = float(np.mean([r["full_decode_cv2"]["wall_s"] for r in dec_rows]))
    except Exception:
        decode_s = None
    cost = {"decode_floor": decode_floor,
            "decode_s_per_segment_mean": decode_s,
            "arms": {}}
    for a in ARMS + ["A-full"]:
        base = a.split("-")[0]
        cd = load(DATA / "arms" / base / "cost.json")
        if a == "A-full":
            af = load(DATA / "arms" / "A" / "allframes.json")
            rtf_sc = (af or {}).get("mean_rtf_single_core_taskset")
            lat = round(10.0 / rtf_sc, 3) if rtf_sc else None
            cd = {"cost": {"mean_latency_s": lat,
                           "peak_ram_mb_self": 271.4,
                           "note": "all-frames 160px single-pass, taskset -c 0 "
                                   f"(RTF {rtf_sc}); features identical to allframes.json"}}
        if cd is None:
            cost["arms"][a] = {"status": "NOT RUN"}
            continue
        c = cd.get("cost", cd)
        lat = c.get("mean_latency_s") or c.get("mean_latency_s_gpu")
        gpu_only = "mean_latency_s" not in c
        rtf = round(10.0 / lat, 2) if lat else None
        row = {"status": "RUN",
               ("latency_s_per_segment_GPU" if gpu_only else "latency_s_per_segment_1core"): lat,
               "gpu_only": gpu_only,
               "rtf_incl_decode": rtf, "peak_ram_mb": c.get("peak_ram_mb_self"),
               "backend": c.get("backend"),
               "note": c.get("note", "")}
        if rtf:
            row["cores_10kh_in_2h"] = int(np.ceil(5000 / rtf))
            row["cores_10kh_in_3h"] = int(np.ceil(3333 / rtf))
            row["edge_50x_feasible"] = bool(rtf >= 50)
            row["live_1x_feasible"] = bool(rtf >= 1)
            if decode_s and not gpu_only and lat > decode_s:
                row["rtf_excl_decode"] = round(10.0 / (lat - decode_s), 2)
            else:
                row["rtf_excl_decode"] = None
        if base == "E" and c.get("segments_per_hour"):
            sph = c["segments_per_hour"]
            hrs = 3_600_000 / sph
            row["eg2_segments_per_hour"] = sph
            row["eg2_hours_for_10kh"] = round(hrs, 1)
            row["eg2_years_for_10kh"] = round(hrs / 8760, 2)
            row["peak_vram_mb"] = c.get("peak_vram_mb")
        cost["arms"][a] = row
    summary["cost"] = cost

    # streaming: time-to-decision per degradation
    stream = {}
    for a, d in arm_docs.items():
        if d is None:
            stream[a] = {"status": "NOT RUN"}
            continue
        base = a.split("-")[0]
        ch = load(DATA / "arms" / base / "chunks.json")
        if ch is None or not synth_manifest or a == "A-full":
            stream[a] = {"status": "NOT STREAMABLE" if ch is None else "NOT RUN",
                         "reason": ("no chunk scores exposed" if ch is None
                                    else "chunk features only for A/E base variants")}
            continue
        try:
            sd = load(DATA / "arms" / base / "synth.json")
            head, thr, _ = fit_full_head(d, gold_manifest, feat_keys)
            ttd: dict[str, list] = {}
            lat_all = []
            for it in synth_manifest["items"]:
                ce = ch["items"].get(it["id"])
                if ce is None:
                    continue
                if isinstance(ce, dict) and "running_mean" in ce:  # E-style
                    seq = [{"emb": e} for e in ce["running_mean"]]
                    lat_all.append(ce.get("latency_s", 0) / 5)
                else:  # A-style list of prefix feature dicts
                    seq = [p["features"] for p in ce]
                    lat_all += [p.get("latency_s", 0) for p in ce]
                decided = None
                for k, feats in enumerate(seq):
                    s = apply_head(head, feats, feat_keys)
                    if s is not None and s < thr:
                        decided = (k + 1) * 2
                        break
                if it["label"] == 0:
                    ttd.setdefault(synth_deg(it), []).append(
                        decided if decided else float("inf"))
            rows = {deg: {"n": len(v),
                          "median_ttd_s": round(float(np.median([x for x in v if x != float("inf")])),
                                                1) if any(x != float("inf") for x in v) else None,
                          "never_decided": sum(1 for x in v if x == float("inf"))}
                    for deg, v in sorted(ttd.items())}
            stream[a] = {"status": "RUN",
                         "operating_threshold": round(thr, 4),
                         "per_chunk_latency_s_mean": round(float(np.mean(lat_all)), 3) if lat_all else None,
                         "realtime_capable": bool(lat_all and float(np.mean(lat_all)) < 2.0),
                         "ttd_per_degradation": rows,
                         "note": "prefix-recompute latency is an UPPER bound; true incremental is cheaper"}
        except Exception as e:
            stream[a] = {"status": "ERROR", "error": f"{type(e).__name__}: {e}"}
    summary["streaming"] = stream

    # hyperspectral + VLM verdicts (other workers)
    summary["hyperspectral_verdict"] = find_verdict(
        ["hyperspectral", "arm C", "ICVL"], "C")
    summary["vlm_verdict"] = find_verdict(["VLM", "arm D", "SmolVLM", "Qwen"], "D")

    summary["scorer_wall_s"] = round(time.perf_counter() - t0, 1)
    (RES / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    write_report(summary)
    print(f"scorer done in {summary['scorer_wall_s']}s", flush=True)


def synth_deg(it):
    return it.get("degradation", "unknown")


def find_verdict(keywords, arm):
    """Look for the other workers' verdict reports; else NOT RUN."""
    cands = [
        HARNESS / "docs" / "reports" / "spectral_study.md",
        DATA / f"verdict_arm{arm}.md",
        DATA / "arms" / arm / "report.md",
        Path(f"/home/parshu/projects/jadal-qa-wt/spectral-features/docs/reports/spectral_study.md"),
        Path(f"/home/parshu/projects/jadal-qa-wt/spectral-vlm/docs/reports/spectral_study.md"),
    ]
    for c in cands:
        if c.exists() and c != REPORT:
            try:
                txt = c.read_text()
                if any(k.lower() in txt.lower() for k in keywords):
                    return {"status": "FOUND", "path": str(c),
                            "excerpt": txt[:1500]}
            except Exception:
                pass
    dd = load(DATA / "arms" / arm / "gold.json")
    if dd is not None:
        return {"status": "ARM RUN, report pending", "path": f"spectral_data/arms/{arm}/"}
    return {"status": "NOT RUN"}


def write_report(s: dict) -> None:
    L = []
    A = s["arms"].get
    L.append("# Spectral study — harness-scored comparison\n")
    L.append(f"_Generated {s['created_utc']} by eval/spectral/score.py. "
             "Every number below comes from eval/results/spectral/summary.json._\n")
    L.append("## Question\n")
    L.append("Does per-pixel frequency/temporal-spectral analysis of RGB video (arm B), "
             "a bounded hyperspectral check (arm C), a generic VLM (arm D), or the "
             "EmbeddingGemma-2 head (arm E) beat the current OpenCV+MediaPipe L0 "
             "signals (arm A) at (1) near-duplicate detection and (2) deciding whether "
             "a 10-s segment is USABLE work footage — within a 10,000 h in 2–3 h "
             "throughput budget, whole-video single-pass? Sampling 4–8 frames per "
             "segment is NOT acceptable as the main method.\n")
    L.append("## Data & protocol\n")
    g = s.get("gold_snapshot") or {}
    L.append(f"- HUMAN GOLD (live IndEgo labels, snapshot {g.get('snapshot_utc')}, "
             f"sha256 {str(g.get('labels_sha256'))[:12]}…, "
             f"n_good={g.get('n_good')}, n_bad={g.get('n_bad')}). "
             "CV: leave-one-clip-out; AUROC with clip-level bootstrap CI (B=1000). "
             "A CI spanning 0.5 means 'no evidence'. Awign data differs: nothing here "
             "is evidence on Awign footage.\n")
    L.append("- SYNTHETIC (by construction, label says nothing about activity): "
             "good-labelled segments + 9 degradations; "
             "sensitivity at 5% FAR threshold set on gold-good scores.\n")
    L.append("- DUPLICATES (by construction): threshold on held-out half 1, "
             "P/R/F1 on half 2, per method and per perturbation. "
             "Horizontal-flip excluded (NOT a duplicate).\n")
    L.append("## Results per task\n")
    L.append("| arm | gold AUROC (95% CI) | bal.acc | synth sens. (median over degs) | dups F1 | status |")
    L.append("|---|---|---|---|---|---|")
    table_arms = ["A", "A-full", "E", "B", "D"] + sorted(
        k for k in s["arms"] if k.startswith("B-") or k.startswith("D-"))
    for a in table_arms:
        e = s["arms"].get(a, {})
        g = e.get("gold") or {}
        if e.get("status") != "RUN" or "results" not in g:
            L.append(f"| {a} | — | — | — | — | {e.get('status', 'NOT RUN')} |")
            continue
        gr = (g.get("results", {}).get("head") or
              g.get("results", {}).get("as_is") or {})
        ci = gr.get("auroc_ci95")
        ci_s = f"({ci[0]}–{ci[1]})" if ci else "(n/a)"
        au = gr.get("auroc")
        ba = gr.get("balanced_accuracy", "—")
        syn = s["synth"].get(a, {})
        pd = (syn.get("per_degradation") or {})
        sens = [v["sensitivity_at_5pct_FAR"] for v in pd.values()
                if v.get("sensitivity_at_5pct_FAR") is not None]
        sens_s = round(float(np.median(sens)), 2) if sens else "—"
        f1 = ((s["dups"].get(a, {}) or {}).get("heldout_overall", {}) or {}).get("F1", "—")
        L.append(f"| {a} | {au} {ci_s} | {ba} | {sens_s} | {f1} | RUN |")
    L.append("\n### Gold detail (per arm, LOCO folds)")
    for a in table_arms:
        e = s["arms"].get(a, {})
        g = e.get("gold") or {}
        if e.get("status") == "RUN" and "results" in g:
            L.append(f"\n**{a}** n={g.get('n')} "
                     f"(good={g.get('n_good')}, bad={g.get('n_bad')}, "
                     f"clips={g.get('n_clips')}): "
                     f"`{json.dumps(g['results'])}`")
        elif e.get("status") == "RUN":
            L.append(f"\n**{a}** `{json.dumps(g)}`")
    L.append("\n### Synthetic detail (sensitivity at 5% FAR)")
    for a, v in (s.get("synth") or {}).items():
        L.append(f"\n**{a}** `{json.dumps(v)}`")
    L.append("\n### Duplicates detail (held-out P/R/F1)")
    for a, v in (s.get("dups") or {}).items():
        L.append(f"\n**{a}** `{json.dumps(v)}`")
    L.append("\n## Cost & Pi-class feasibility\n")
    L.append("_Single-core wall time normalised by a CPU benchmark "
             "(matmul-512 reps/s; see decode_floor.json). NOT measured on a Pi — "
             "numbers scale by benchmark ratio,/clock, and memory bandwidth._\n")
    L.append("| arm | s/segment (1 core) | RTF | cores for 10kh/2h | /3h | edge 50x | live 1x |")
    L.append("|---|---|---|---|---|---|---|")
    for a, v in (s.get("cost", {}).get("arms") or {}).items():
        lat = v.get('latency_s_per_segment_1core',
                    v.get('latency_s_per_segment_GPU'))
        if v.get('gpu_only'):
            lat = f"{lat} (GPU)"
        L.append(f"| {a} | {lat} | "
                 f"{v.get('rtf_incl_decode')} | {v.get('cores_10kh_in_2h')} | "
                 f"{v.get('cores_10kh_in_3h')} | {v.get('edge_50x_feasible')} | "
                 f"{v.get('live_1x_feasible')} | {v.get('status')} |")
    L.append(f"\n_Decode share: full cv2 decode ≈ {s.get('cost', {}).get('decode_s_per_segment_mean')} s/seg "
             "(single-core; RTF excl decode in summary.json per arm)._")
    L.append("\n### Accuracy-vs-cost Pareto (gold AUROC vs cores for 10kh/2h)")
    pareto = []
    for a in table_arms:
        e = s["arms"].get(a, {})
        g = (e.get("gold") or {}).get("results", {}).get("head", {})
        c = (s.get("cost", {}).get("arms") or {}).get(a, {})
        if g.get("auroc") is not None and c.get("cores_10kh_in_2h"):
            pareto.append((a, g["auroc"], c["cores_10kh_in_2h"],
                           c.get("rtf_incl_decode")))
    pareto.sort(key=lambda r: r[2])
    for a, au, cores, rtf in pareto:
        L.append(f"- {a}: AUROC {au}, {cores} cores (RTF {rtf})")
    L.append("_Pareto read: cheapest (fewest cores) first; an arm is dominated if "
             "another is both cheaper and more accurate._")
    eg2 = ((s.get("cost", {}).get("arms") or {}).get("E") or {})
    if eg2.get("eg2_segments_per_hour"):
        L.append(f"\nEG2: {eg2['eg2_segments_per_hour']} seg/h/GPU → "
                 f"{eg2['eg2_hours_for_10kh']} GPU-h ({eg2['eg2_years_for_10kh']} GPU-years) "
                 f"for 10,000 h = 3.6M segments. VRAM peak {eg2.get('peak_vram_mb')} MB.")
    L.append("\n## Streaming\n")
    for a, v in (s.get("streaming") or {}).items():
        L.append(f"\n**{a}** `{json.dumps(v, default=str)}`")
    L.append("\n## Hyperspectral verdict\n")
    L.append(f"`{json.dumps(s.get('hyperspectral_verdict'), default=str)}`\n")
    L.append("## VLM verdict\n")
    L.append(f"`{json.dumps(s.get('vlm_verdict'), default=str)}`\n")
    L.append("## Recommendation for Jadal\n")
    L.append("_TBD once arms B and D report; current partial read below._\n")
    L.append(partial_recommendation(s))
    L.append("\n## Limitations\n")
    L.append("- Only 3 IndEgo clips labelled so far; 14 bad labels → CIs are wide; "
             "any CI spanning 0.5 = no evidence.\n")
    L.append("- Synthetic degradations measure capture-quality sensitivity only, "
             "NOT activity judgement.\n")
    L.append("- Cost numbers are workstation single-core, not a Pi; scaled by benchmark.\n")
    L.append("- Human labels are IndEgo workers; Awign footage differs.\n")
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(L) + "\n")


def partial_recommendation(s: dict) -> str:
    bits = []
    for a, e in s["arms"].items():
        g = (e.get("gold") or {}).get("results", {}).get("head", {})
        if g.get("auroc") is not None:
            flag = " (PROVISIONAL filename)" if e.get("provisional") else ""
            bits.append(f"{a}: AUROC {g['auroc']} CI {g.get('auroc_ci95')}{flag}")
    return ("Partial (arms B-final/D pending): " + "; ".join(bits) + "." if bits
            else "No arms scored yet.")


if __name__ == "__main__":
    main()
