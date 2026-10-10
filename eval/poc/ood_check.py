"""Out-of-domain check: PoC verdicts vs human labels on 60 Egocentric-10K windows.

Joins each human-labelled segment to its PoC window by source clip basename and
the nearest window start (the PoC windows sit on 10-s boundaries). Computes the
PoC score AUROC (good = positive class, higher score = more usable), verdict
counts, balanced accuracy and a window-level bootstrap CI. Also copies two
numbers that come from other branches' scorers, labelled "reported, not
recomputed here", plus the matching lines from the FACTS file.

Environment (all optional, defaults are for the sibling-worktree layout):
  JADAL_POC_DATA       dir with windows.jsonl        (default <repo>/../poc_data)
  JADAL_OOD_LABELS     {latest:{segment_id:{verdict}}} (default <repo>/../ood_labels/labels.json)
  JADAL_OOD_QUEUE      label queue with segment list  (default
                       <repo>/../scale-data/eval/results/scale/label_queue_random.json)
  JADAL_OOD_ENSEMBLE   ensemble handover text         (default
                       <repo>/../handover/ood_eval_ensemble_v1_exploratory.txt)
  JADAL_OOD_FACTS      explainer FACTS.md             (default
                       <repo>/../../jadal-ops/explainer/FACTS.md)
  JADAL_OOD_OUT        output JSON                    (default
                       <repo>/eval/results/poc/ood_check_60.json)

Usage: python3 eval/poc/ood_check.py
"""
import json
import os
import random

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
WS = os.path.join(ROOT, "..")


def _env(name, default):
    return os.environ.get(name, default)


POC_DATA = _env("JADAL_POC_DATA", os.path.join(WS, "poc_data"))
LABELS = _env("JADAL_OOD_LABELS", os.path.join(WS, "ood_labels", "labels.json"))
QUEUE = _env("JADAL_OOD_QUEUE", os.path.join(
    WS, "scale-data", "eval", "results", "scale", "label_queue_random.json"))
ENSEMBLE = _env("JADAL_OOD_ENSEMBLE", os.path.join(
    WS, "handover", "ood_eval_ensemble_v1_exploratory.txt"))
FACTS = _env("JADAL_OOD_FACTS", os.path.join(WS, "..", "jadal-ops", "explainer", "FACTS.md"))
OUT = _env("JADAL_OOD_OUT", os.path.join(ROOT, "eval", "results", "poc", "ood_check_60.json"))

N_BOOT = 2000
SEED = 0
MAX_START_GAP_S = 10.0  # a window must start within one window length of the label


def auroc(pos, neg):
    """Mann-Whitney AUROC with average ranks for ties; pos = good class."""
    if not pos or not neg:
        return None
    scored = sorted([(s, 1) for s in pos] + [(s, 0) for s in neg])
    rank_sum, i = 0.0, 0
    while i < len(scored):
        j = i
        while j + 1 < len(scored) and scored[j + 1][0] == scored[i][0]:
            j += 1
        avg = (i + j) / 2.0 + 1
        rank_sum += avg * sum(1 for k in range(i, j + 1) if scored[k][1] == 1)
        i = j + 1
    n_pos, n_neg = len(pos), len(neg)
    return (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def load_windows(path):
    by_key = {}
    latest = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                w = json.loads(line)
            except ValueError:
                continue
            latest[w["window_id"]] = w  # keep last occurrence (dedupe)
    for w in latest.values():
        key = os.path.basename(w.get("clip", ""))
        by_key.setdefault(key, []).append(w)
    return by_key


def join(queue_segments, labels, by_key):
    rows, unmatched = [], []
    for seg in queue_segments:
        lab = labels.get(seg["segment_id"])
        if lab is None:
            unmatched.append((seg["segment_id"], "no label"))
            continue
        key = os.path.basename(seg["source_path"])
        cands = by_key.get(key, [])
        if not cands:
            unmatched.append((seg["segment_id"], "no PoC window for clip"))
            continue
        best = min(cands, key=lambda w: abs(w["start_s"] - seg["source_start_s"]))
        gap = abs(best["start_s"] - seg["source_start_s"])
        if gap > MAX_START_GAP_S:
            unmatched.append((seg["segment_id"], f"nearest start {gap:.1f}s away"))
            continue
        rows.append({
            "segment_id": seg["segment_id"],
            "clip": key,
            "source_start_s": seg["source_start_s"],
            "poc_window_id": best["window_id"],
            "poc_start_s": best["start_s"],
            "label": lab["verdict"],
            "poc_verdict": best["verdict"],
            "poc_score": best.get("score"),
        })
    return rows, unmatched


def bootstrap_ci(rows, rng):
    good = [r for r in rows if r["label"] == "good"]
    bad = [r for r in rows if r["label"] == "bad"]
    vals = []
    for _ in range(N_BOOT):
        g = [good[rng.randrange(len(good))] for _ in good]
        b = [bad[rng.randrange(len(bad))] for _ in bad]
        a = auroc([s(r) for r in g], [s(r) for r in b])
        if a is not None:
            vals.append(a)
    vals.sort()
    lo = vals[int(0.025 * (len(vals) - 1))]
    hi = vals[int(0.975 * (len(vals) - 1))]
    return round(lo, 3), round(hi, 3)


def s(r):
    return r["poc_score"] if r["poc_score"] is not None else 0.0  # S0 rejects have no score


def grep_line(path, needle):
    with open(path, encoding="utf-8") as fh:
        for no, line in enumerate(fh, 1):
            if needle in line:
                return {"text": line.strip()[:400],
                        "source": f"{os.path.relpath(path, WS)}:{no}"}
    return None


def main():
    with open(LABELS, encoding="utf-8") as fh:
        labels = json.load(fh)["latest"]
    with open(QUEUE, encoding="utf-8") as fh:
        segments = json.load(fh)["segments"]
    by_key = load_windows(os.path.join(POC_DATA, "windows.jsonl"))
    rows, unmatched = join(segments, labels, by_key)
    if unmatched:
        raise SystemExit(f"unmatched labelled segments: {unmatched}")

    good = [r for r in rows if r["label"] == "good"]
    bad = [r for r in rows if r["label"] == "bad"]
    # "good" is the positive class; a higher PoC score means more usable
    poc_auc = auroc([s(r) for r in good], [s(r) for r in bad])
    rng = random.Random(SEED)
    ci = bootstrap_ci(rows, rng)

    def counts(group):
        c = {"accept": 0, "reject": 0, "review": 0}
        for r in group:
            c[r["poc_verdict"]] += 1
        return c

    cg, cb = counts(good), counts(bad)
    bal_acc = round((cg["accept"] / len(good) + cb["reject"] / len(bad)) / 2, 2)

    ensemble = grep_line(ENSEMBLE, "ensemble_v1 OOD AUROC")
    perpixel = grep_line(FACTS, "Frozen per-pixel scorer (configs/perpixel_v1.json")
    facts = [grep_line(FACTS, needle) for needle in (
        "OUT-OF-DOMAIN RESULT",
        "Frozen per-pixel scorer (configs/perpixel_v1.json",
        "Frozen two-view ensemble ensemble_v1",
    )]

    out = {
        "n": len(rows),
        "n_good": len(good),
        "n_bad": len(bad),
        "label_source": {
            "human_labels": os.path.relpath(LABELS, WS),
            "queue": os.path.relpath(QUEUE, WS),
            "join": "source clip basename + nearest window start (within 10 s)",
        },
        "poc_score_auroc": round(poc_auc, 3),
        "auroc_ci95_window_bootstrap": list(ci),
        "bootstrap": {"unit": "window", "n_boot": N_BOOT, "seed": SEED,
                      "stratified": True},
        "verdicts": {"good": cg, "bad": cb},
        "good_accepted": cg["accept"], "good_rejected": cg["reject"],
        "good_review": cg["review"],
        "bad_accepted": cb["accept"], "bad_rejected": cb["reject"],
        "bad_review": cb["review"],
        "balanced_accuracy": bal_acc,
        "caveat": ("n_bad=9 < 10 (pre-registered minimum): exploratory, not the "
                   "pre-registered test. The window-bootstrap CI is optimistic: "
                   "windows from the same clip are not independent."),
        "reported_not_recomputed": {
            "frozen_ensemble_v1_auroc": {
                "value": 0.415,
                "source": ensemble,
                "note": "reported, not recomputed here (ensemble scorer lives on another branch)",
            },
            "perpixel_v1_auroc": {
                "value": 0.515,
                "source": perpixel,
                "note": "reported, not recomputed here (per-pixel scorer lives on another branch)",
            },
        },
        "facts": [f for f in facts if f],
        "windows": rows,
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
        fh.write("\n")
    print(json.dumps({k: out[k] for k in ("n", "n_good", "n_bad", "poc_score_auroc",
                                           "auroc_ci95_window_bootstrap",
                                           "balanced_accuracy")}))
    print("good", cg, "bad", cb)


if __name__ == "__main__":
    main()
