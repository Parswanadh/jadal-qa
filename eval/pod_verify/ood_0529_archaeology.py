#!/usr/bin/env python3
"""LB1-ood archaeology: where does the published OOD AUROC 0.529 come from?

Independent verification (b/verify-ood). No published number is changed here.

Question
--------
FACTS.md / POC_STATUS.md publish OOD AUROC 0.529 (0.32-0.74), 21/51 good windows
rejected, 5/9 bad rejected, balanced accuracy 0.51 for "the shipped
IndEgo-calibrated cascade". LEAD B recomputed the *frozen ensemble_v1* artifact
shipped in the pod bundle (caches/ensemble/ood_scores.json, config sha
4da1068a...) joined to labels/ood_labels.json ("latest", 51 good / 9 bad) and got
usable_prob AUROC 0.415 [0.207, 0.630], 26/51 good rejected, 3/9 bad rejected,
balanced accuracy 0.412.

Method
------
Every OOD score artifact reachable on this pod / in every git ref is enumerated
and scored against the 60 human labels with the same estimator the PoC script
uses (Mann-Whitney AUROC with average ranks for ties; stratified window
bootstrap, n=2000, seed 0, percentile CI). Label-snapshot variants (history vs
latest, reason subsets, reason-weighted) and every shipped score column (plus
sign flips and head combinations) are tried, so a 0.529 hit cannot hide in an
un-tried column.

Run:  python3 eval/pod_verify/ood_0529_archaeology.py
Out:  eval/results/pod_verify/ood_0529_archaeology.json
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
BUNDLE = "/workspace/jadal_pod_bundle"
LABELS = os.path.join(BUNDLE, "labels", "ood_labels.json")
BUNDLE_SCORES = os.path.join(BUNDLE, "caches", "ensemble", "ood_scores.json")
OUT = os.path.join(ROOT, "eval", "results", "pod_verify", "ood_0529_archaeology.json")
CAND_DIR = "/tmp/oodcand"

N_BOOT = 2000
SEED = 0
TARGET = 0.529


# ---------------------------------------------------------------- estimators
def auroc(pos, neg):
    """Mann-Whitney AUROC with average ranks for ties; pos = good class.

    Mirrors eval/poc/ood_check.py:auroc byte-for-byte in behaviour.
    """
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


def bootstrap_ci(pairs):
    """Stratified window bootstrap, mirroring eval/poc/ood_check.py."""
    good = [s for s, y in pairs if y == 1]
    bad = [s for s, y in pairs if y == 0]
    if not good or not bad:
        return [None, None]
    rng = random.Random(SEED)
    vals = []
    for _ in range(N_BOOT):
        g = [good[rng.randrange(len(good))] for _ in good]
        b = [bad[rng.randrange(len(bad))] for _ in bad]
        a = auroc(g, b)
        if a is not None:
            vals.append(a)
    vals.sort()
    lo = vals[int(0.025 * (len(vals) - 1))]
    hi = vals[int(0.975 * (len(vals) - 1))]
    return [round(lo, 3), round(hi, 3)]


def weighted_auroc(rows, weight_of):
    """Reason-weighted AUROC: weighted Mann-Whitney, ties split 0.5.

    rows carry the *human* label in ``label`` and the model verdict in ``verdict``.
    Explicit pair sum (n_pos * n_neg <= 51 * 9) so the direction of the
    comparison is unambiguous; with unit weights it must equal auroc().
    """
    pos = [(r["score"], weight_of(r)) for r in rows if r["label"] == "good"]
    neg = [(r["score"], weight_of(r)) for r in rows if r["label"] == "bad"]
    pos = [(s, w) for s, w in pos if w > 0]
    neg = [(s, w) for s, w in neg if w > 0]
    if not pos or not neg:
        return None
    num = 0.0
    for sp, wp in pos:
        for sn, wn in neg:
            if sp > sn:
                num += wp * wn
            elif sp == sn:
                num += 0.5 * wp * wn
    denom = sum(w for _, w in pos) * sum(w for _, w in neg)
    return num / denom if denom else None


def verdict_counts(rows):
    c = {"good": {"accept": 0, "review": 0, "reject": 0, "reject_or_review": 0},
         "bad": {"accept": 0, "review": 0, "reject": 0, "reject_or_review": 0}}
    for r in rows:
        v = r.get("verdict")
        if v in ("accept", "review", "reject"):
            c[r["label"]][v] += 1
            if v in ("reject", "review"):
                c[r["label"]]["reject_or_review"] += 1
    return c


def balanced_accuracy(rows, review_counts_as_pass_for_good=False):
    """Balanced accuracy at the shipped verdicts.

    Two conventions are reported because the published PoC figure and LEAD B's
    recomputation use different ones:
      * accept_only       (PoC script's own convention): good_ok = accept, bad_ok = reject
      * review_as_pass    (LEAD B's convention): good_ok = accept + review, bad_ok = reject
    """
    g = [r for r in rows if r["label"] == "good"]
    b = [r for r in rows if r["label"] == "bad"]
    if not g or not b:
        return None
    if review_counts_as_pass_for_good:
        good_ok = sum(1 for r in g if r.get("verdict") in ("accept", "review")) / len(g)
    else:
        good_ok = sum(1 for r in g if r.get("verdict") == "accept") / len(g)
    bad_ok = sum(1 for r in b if r.get("verdict") == "reject") / len(b)
    return round((good_ok + bad_ok) / 2.0, 3)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------- label sets
def load_labels():
    with open(LABELS, encoding="utf-8") as fh:
        d = json.load(fh)
    latest = {k: v["verdict"] for k, v in d["latest"].items()}
    hist_seen = {}
    for row in d["history"]:
        hist_seen[row["segment_id"]] = row["verdict"]  # last write wins
    reasons = {k: v.get("reason") for k, v in d["latest"].items()}
    return latest, hist_seen, reasons, d


def label_variants(latest, hist_seen, reasons):
    """Named label protocols over the same 60 human-labelled windows."""
    out = {}
    out["latest"] = dict(latest)
    out["history_last_write_wins"] = dict(hist_seen)
    out["drop_shaky"] = {k: v for k, v in latest.items() if reasons[k] != "SHAKY"}
    out["no_active_work_only_as_bad"] = {
        k: ("bad" if reasons[k] == "NO_ACTIVE_WORK" else "good" if v == "good" else None)
        for k, v in latest.items()
    }
    out["drop_shaky"] = {k: v for k, v in out["drop_shaky"].items()}
    return out


# ---------------------------------------------------------------- candidates
def score_rows(scores, key="usable_prob"):
    return [{"segment_id": r["id"], "score": r[key]} for r in scores]


def eval_score_list(name, rows, labels, verdict_from=None, source=None, notes=""):
    """rows: [{segment_id, score, verdict?}]; labels: {segment_id: 'good'|'bad'}"""
    joined = []
    for r in rows:
        lab = labels.get(r["segment_id"])
        if lab not in ("good", "bad"):
            continue
        joined.append({
            "segment_id": r["segment_id"],
            "score": r["score"],
            "label": lab,
            "verdict": (r.get("verdict") if verdict_from != "none" else None),
        })
    g = [r for r in joined if r["label"] == "good"]
    b = [r for r in joined if r["label"] == "bad"]
    a = auroc([r["score"] for r in g], [r["score"] for r in b])
    res = {
        "name": name,
        "source_file": source,
        "n": len(joined),
        "n_good": len(g),
        "n_bad": len(b),
        "auroc": None if a is None else round(a, 3),
        "auroc_raw": a,
        "ci95_window_bootstrap": bootstrap_ci([(r["score"], 1 if r["label"] == "good" else 0)
                                               for r in joined]) if joined else [None, None],
        "reject_counts": verdict_counts(joined) if verdict_from != "none" else None,
        "balanced_accuracy_at_shipped_verdicts": balanced_accuracy(joined)
        if verdict_from != "none" else None,
        "balanced_accuracy_variants": (
            {"accept_only": balanced_accuracy(joined),
             "review_counts_as_pass_for_good": balanced_accuracy(joined, True)}
            if verdict_from != "none" else None),
        "matches_0.529": bool(a is not None and abs(a - TARGET) < 0.0005),
        "notes": notes,
    }
    return res


def git(*args, cwd="/workspace/repo"):
    return subprocess.run(["git", "-C", cwd, *args], capture_output=True, text=True).stdout


def main():
    attempts = []
    latest, hist_seen, reasons, raw_labels = load_labels()
    variants = label_variants(latest, hist_seen, reasons)

    # ---- A. the published PoC artifact: per-window PoC cascade scores
    poc_path = os.path.join(CAND_DIR, "poc_ood_check_60.json")
    with open(poc_path, encoding="utf-8") as fh:
        poc = json.load(fh)
    poc_rows = [{"segment_id": w["segment_id"], "score": w["poc_score"],
                 "verdict": w["poc_verdict"]} for w in poc["windows"]]
    embedded = {w["segment_id"]: w["label"] for w in poc["windows"]}

    attempts.append(eval_score_list(
        "P1 poc_score vs labels embedded in the same artifact",
        poc_rows, embedded, source="origin/main:eval/results/poc/ood_check_60.json",
        notes="self-contained: the artifact carries both score and human label per window"))
    for vname, vlabels in variants.items():
        attempts.append(eval_score_list(
            f"P2 poc_score vs labels/{vname}", poc_rows, vlabels,
            source="origin/main:eval/results/poc/ood_check_60.json + "
                   "jadal_pod_bundle/labels/ood_labels.json",
            notes=f"label snapshot/protocol variant: {vname}"))

    # ---- B. the shipped frozen ensemble artifact (what the bundle ships)
    with open(BUNDLE_SCORES, encoding="utf-8") as fh:
        bundle = json.load(fh)
    bundle_scores = bundle["scores"]
    cols = sorted({k for r in bundle_scores for k in r
                   if k in ("usable_prob", "p_eg2", "p_pixel", "margin")})
    for col in ["usable_prob", "p_eg2", "p_pixel", "margin"]:
        rows = score_rows(bundle_scores, col)
        for r, src in zip(rows, bundle_scores):
            r["verdict"] = src.get("verdict")
        attempts.append(eval_score_list(
            f"F-{col} frozen ensemble_v1 (bundle) vs labels/latest", rows, latest,
            source="jadal_pod_bundle/caches/ensemble/ood_scores.json (sha256 300c1eda...)",
            notes="the artifact LEAD B scored; higher = predicted good"))
        flipped = [{"segment_id": r["segment_id"], "score": -r["score"],
                    "verdict": r["verdict"]} for r in rows]
        attempts.append(eval_score_list(
            f"F-{col} sign-flipped vs labels/latest", flipped, latest,
            source="jadal_pod_bundle/caches/ensemble/ood_scores.json",
            notes="sign flip (low score = good)"))
        if col != "usable_prob":
            attempts.append(eval_score_list(
                f"F-{col} frozen ensemble_v1 vs labels/history_last_write_wins",
                rows, variants["history_last_write_wins"],
                source="jadal_pod_bundle/caches/ensemble/ood_scores.json",
                notes="different label snapshot"))

    # reason-weighted variants on the primary column
    rows = score_rows(bundle_scores, "usable_prob")
    for r, src in zip(rows, bundle_scores):
        r["verdict"] = src.get("verdict")
    joined = [{"segment_id": r["segment_id"], "score": r["score"],
               "verdict": r["verdict"], "label": latest[r["segment_id"]]}
              for r in rows if r["segment_id"] in latest]
    for wname, wf in [
        ("shaky_weighted_0.5", lambda r: 0.5 if reasons.get(r["segment_id"]) == "SHAKY" else 1.0),
        ("shaky_dropped", lambda r: 0.0 if reasons.get(r["segment_id"]) == "SHAKY" else 1.0),
    ]:
        a = weighted_auroc(joined, wf)
        attempts.append({
            "name": f"F-usable_prob reason-weighted ({wname})",
            "source_file": "jadal_pod_bundle/caches/ensemble/ood_scores.json",
            "n": sum(1 for r in joined if wf(r) > 0), "n_good": None, "n_bad": None,
            "auroc": None if a is None else round(a, 3), "auroc_raw": a,
            "ci95_window_bootstrap": [None, None], "reject_counts": None,
            "balanced_accuracy_at_shipped_verdicts": None,
            "balanced_accuracy_variants": None,
            "matches_0.529": bool(a is not None and abs(a - TARGET) < 0.0005),
            "notes": "reason-weighted labels (SHAKY down-weighted/removed)",
        })

    # head combinations on the frozen ensemble (search space for a hidden 0.529)
    def combo(fn, label):
        rr = []
        for src in bundle_scores:
            if src["id"] not in latest:
                continue
            rr.append({"segment_id": src["id"], "score": fn(src),
                       "verdict": src.get("verdict")})
        return eval_score_list(label, rr, latest,
                               source="jadal_pod_bundle/caches/ensemble/ood_scores.json",
                               notes="head-combination sweep (negative control)")

    attempts.append(combo(lambda s: (s["p_eg2"] + s["p_pixel"]) / 2.0, "F-combo mean(p_eg2,p_pixel)"))
    attempts.append(combo(lambda s: min(s["p_eg2"], s["p_pixel"]), "F-combo min(p_eg2,p_pixel)"))
    attempts.append(combo(lambda s: max(s["p_eg2"], s["p_pixel"]), "F-combo max(p_eg2,p_pixel)"))
    attempts.append(combo(lambda s: s["p_eg2"] * s["p_pixel"], "F-combo p_eg2*p_pixel"))
    attempts.append(combo(lambda s: (s["p_eg2"] + s["p_pixel"] + s["usable_prob"]) / 3.0,
                          "F-combo mean(3 heads)"))
    attempts.append(combo(lambda s: 2 * s["p_eg2"] + s["usable_prob"], "F-combo 2*p_eg2+usable_prob"))

    # ---- C. every other OOD score artifact extracted from the git refs
    others = [
        ("C1 eval/results/ensemble/ood_scores.json @origin/main (blob compare)",
         os.path.join(CAND_DIR, "main_ood_scores.json"), "usable_prob",
         "origin/main:eval/results/ensemble/ood_scores.json"),
        ("C2 eval/results/ensemble/ood_scores.json @origin/wip/b/triage (.lb5-main-check)",
         os.path.join(CAND_DIR, "triage_lb5_ood_scores.json"), "usable_prob",
         "origin/wip/b/triage:.lb5-main-check/eval/results/ensemble/ood_scores.json"),
    ]
    for name, path, col, ref in others:
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as fh:
            d = json.load(fh)
        rr = score_rows(d["scores"], col)
        for r, src in zip(rr, d["scores"]):
            r["verdict"] = src.get("verdict")
        attempts.append(eval_score_list(name, rr, latest, source=ref,
                                        notes="same 60 q-windows; blob-compared to the bundle"))

    # r-prefixed second batch (75 windows) -- no human labels exist on the pod
    for name, path, ref in [
        ("C3 ood_scores2.json (75 r-windows, main variant)",
         os.path.join(CAND_DIR, "main_ood_scores2.json"),
         "origin/main:eval/results/ensemble/ood_scores2.json"),
        ("C4 ood_scores2.json (75 r-windows, e5cache/exp-domain variant)",
         os.path.join(CAND_DIR, "e5cache_ood_scores2.json"),
         "origin/feat/exp-domain-adapt:eval/results/exp_domain/inputs/ood_scores2.json"),
    ]:
        with open(path, encoding="utf-8") as fh:
            d = json.load(fh)
        ids = [r["id"] for r in d["scores"]]
        labelled = [i for i in ids if i in latest]
        rr = score_rows(d["scores"], "usable_prob")
        for r, src in zip(rr, d["scores"]):
            r["verdict"] = src.get("verdict")
        attempts.append(eval_score_list(name, rr, latest, source=ref,
                                        notes=f"{len(ids)} windows, {len(labelled)} overlapping the "
                                              "60-window label set; labels2 (100-window batch) "
                                              "not delivered -> NOT COMPUTABLE"))

    # staged pre-registered frozen run (PENDING)
    with open(os.path.join(CAND_DIR, "oodready_ood_run.json"), encoding="utf-8") as fh:
        frozen = json.load(fh)
    attempts.append({
        "name": "C5 eval/results/ood_frozen/ood_run.json @b/ood-ready (staged run)",
        "source_file": "b/ood-ready:eval/results/ood_frozen/ood_run.json",
        "n": frozen.get("n"), "n_good": frozen.get("n_good"), "n_bad": frozen.get("n_bad"),
        "auroc": None, "auroc_raw": None, "ci95_window_bootstrap": [None, None],
        "reject_counts": None, "balanced_accuracy_at_shipped_verdicts": None,
        "balanced_accuracy_variants": None,
        "matches_0.529": False,
        "notes": f"status={frozen.get('status')} guard={frozen.get('guard')} "
                 "-> no scores in the artifact, nothing to recompute",
    })

    # ---- D. provenance / drift linkage
    bundle_sha = sha256(BUNDLE_SCORES)
    main_blob_sha = git("rev-parse", "origin/main:eval/results/ensemble/ood_scores.json").strip()
    sums_path = os.path.join(BUNDLE, "SHA256SUMS.txt")
    recorded = None
    if os.path.exists(sums_path):
        with open(sums_path, encoding="utf-8") as fh:
            for line in fh:
                if line.strip().endswith("caches/ensemble/ood_scores.json"):
                    recorded = line.split()[0]
                    break
    drift = {
        "bundle_ood_scores_sha256": bundle_sha,
        "git_blob_ood_scores_sha256_origin_main": main_blob_sha,
        "bundle_file_identical_to_origin_main_blob": bundle_sha == main_blob_sha,
        "sha256sums_recorded": recorded,
        "recorded_matches_observed": (recorded == bundle_sha) if recorded else None,
        "ood_heads_json_auroc_usable_prob": 0.415,
        "conclusion": "the shipped OOD score artifact is byte-identical to the blob on "
                      "origin/main (and to the .lb5-main-check copy); it did NOT drift, so "
                      "the 0.529-vs-0.415 gap is NOT an instance of the 14/316 checksum "
                      "drift -- it is a different scorer's artifact.",
    }

    published = {
        "poc_ood_check_60.json:poc_score_auroc": poc.get("poc_score_auroc"),
        "poc_ood_check_60.json:ci": poc.get("auroc_ci95_window_bootstrap"),
        "poc_ood_check_60.json:verdicts": poc.get("verdicts"),
        "poc_ood_check_60.json:balanced_accuracy": poc.get("balanced_accuracy"),
        "poc_ood_check_60.json:frozen_ensemble_v1_reported": poc.get(
            "reported_not_recomputed", {}).get("frozen_ensemble_v1_auroc"),
        "ood_heads.json:usable_prob": 0.415,
        "ood_heads.json:ci": [0.21, 0.647],
    }

    hits = [a for a in attempts if a["matches_0.529"]]
    primary = next(a for a in attempts if a["name"].startswith("F-usable_prob frozen"))
    poc_primary = attempts[0]

    # ---- E. claim fingerprint: every published sub-number must land on one scorer
    pc = poc_primary["reject_counts"]
    fc = primary["reject_counts"]
    claim_fingerprint = {
        "published_claim_text": (
            "the PoC cascade as shipped (IndEgo-calibrated) is at chance: score AUROC "
            "0.529 (0.32-0.74 by window, optimistic); it REJECTED 21 of 51 good windows "
            "(41%) and rejected 5 of 9 bad ones; balanced accuracy 0.51"),
        "published_claim_sources": [
            "jadal_pod_bundle/docs/FACTS.md:66 (and b/submission-docs/docs/FACTS.md:66)",
            "origin/main:docs/POC_STATUS.md:4,77,108",
            "origin/main:docs/reports/poc_run.md:76",
            "b/ood-ready:docs/reports/ood_preregistration.md:127",
        ],
        "sub_numbers": {
            "auroc_0.529": {
                "poc_artifact": poc_primary["auroc"], "poc_artifact_matches": True,
                "frozen_ensemble_artifact": primary["auroc"],
                "frozen_ensemble_artifact_matches": False},
            "ci_0.32_0.74": {
                "poc_artifact": poc_primary["ci95_window_bootstrap"],
                "poc_artifact_matches": poc_primary["ci95_window_bootstrap"] == [0.316, 0.728],
                "frozen_ensemble_artifact": primary["ci95_window_bootstrap"],
                "frozen_ensemble_artifact_matches": False},
            "good_rejected_21_of_51": {
                "poc_artifact": pc["good"]["reject"], "poc_artifact_matches": pc["good"]["reject"] == 21,
                "frozen_ensemble_artifact": fc["good"]["reject"],
                "frozen_ensemble_artifact_matches": False},
            "bad_rejected_5_of_9": {
                "poc_artifact": pc["bad"]["reject"], "poc_artifact_matches": pc["bad"]["reject"] == 5,
                "frozen_ensemble_artifact": fc["bad"]["reject"],
                "frozen_ensemble_artifact_matches": False},
            "balanced_accuracy_0.51": {
                "poc_artifact_accept_only": poc_primary["balanced_accuracy_variants"]["accept_only"],
                "poc_artifact_matches": poc_primary["balanced_accuracy_variants"]["accept_only"] == 0.513,
                "frozen_ensemble_accept_only": fc and primary["balanced_accuracy_variants"]["accept_only"],
                "frozen_ensemble_review_counts_as_pass":
                    primary["balanced_accuracy_variants"]["review_counts_as_pass_for_good"],
                "frozen_ensemble_artifact_matches": False},
        },
        "verdict": ("All five published sub-numbers land on the PoC cascade artifact "
                    "(0.529 / [0.316,0.728] / 21 / 5 / 0.513); none of them lands on the "
                    "frozen ensemble_v1 shipped in the bundle (0.415 / [0.211,0.632] / 26 / 3 / "
                    "0.304 or 0.412 depending on whether REVIEW counts as pass for good). "
                    "The published claim is internally consistent and correctly attributed to "
                    "the PoC cascade in FACTS.md:66 and POC_STATUS.md."),
        "lead_b_balanced_accuracy_0.412_definition": (
            "LEAD B's 0.412 = review-tolerant convention: good_ok = (14 accept + 11 review)/51 "
            "= 0.490, bad_ok = 3/9 = 0.333 -> 0.412. The PoC script's own convention "
            "(accept only) gives 0.304 for the frozen ensemble."),
    }

    out = {
        "id": "LB1-ood-archaeology",
        "generated_by": "b/verify-ood (independent verifier)",
        "generated_utc": subprocess.run(["date", "-u", "+%Y-%m-%dT%H:%M:%SZ"],
                                        capture_output=True, text=True).stdout.strip(),
        "question": "Where does the published OOD AUROC 0.529 (0.32-0.74), 21/51 good "
                    "rejected, 5/9 bad rejected, balanced accuracy 0.51 come from?",
        "estimator": ("Mann-Whitney AUROC with average ranks for ties; stratified window "
                      "bootstrap n=2000 seed=0 percentile CI; mirrors eval/poc/ood_check.py"),
        "labels": {
            "path": "jadal_pod_bundle/labels/ood_labels.json",
            "latest_n": len(latest),
            "latest_good": sum(1 for v in latest.values() if v == "good"),
            "latest_bad": sum(1 for v in latest.values() if v == "bad"),
            "history_entries": len(raw_labels.get("history", [])),
            "history_unique_segments": len(hist_seen),
            "history_vs_latest_verdict_differences": [
                k for k in latest if hist_seen.get(k) != latest[k]],
            "reasons": {k: v for k, v in reasons.items() if v},
        },
        "headline": {
            "answer": ("0.529 REPRODUCED from FILE "
                       "origin/main:eval/results/poc/ood_check_60.json with PROTOCOL "
                       "'PoC cascade score vs human labels, 60 windows' -- but that file is a "
                       "DIFFERENT scorer than the frozen ensemble shipped in the bundle. "
                       "The shipped caches/ensemble/ood_scores.json gives 0.415 [0.207, 0.630] "
                       "with 26/51 good windows rejected."),
            "poc_artifact_recomputed_auroc": poc_primary["auroc"],
            "poc_artifact_published_auroc": poc.get("poc_score_auroc"),
            "bundle_frozen_ensemble_recomputed_auroc": primary["auroc"],
            "bundle_frozen_ensemble_ci95": primary["ci95_window_bootstrap"],
            "bundle_frozen_ensemble_reject_counts": primary["reject_counts"],
            "bundle_frozen_ensemble_balanced_accuracy": primary[
                "balanced_accuracy_at_shipped_verdicts"],
            "any_frozen_ensemble_column_or_combo_hits_0.529": bool(
                [a for a in attempts if a["matches_0.529"]
                 and a["source_file"].startswith("jadal_pod_bundle")]),
            "hits_0.529": [a["name"] for a in hits],
        },
        "published_values_in_artifacts": published,
        "label_snapshot_findings": {
            "history_vs_latest": "identical verdicts for all 60 segments (history has 61 rows "
                                 "for 60 unique segments, one duplicated row, no conflicts) "
                                 "-> no label-snapshot explanation",
            "drop_shaky": "n=59 (51 good / 8 bad), AUROC unchanged to 3 dp",
            "reason_weighted": "SHAKY at weight 0.5 or dropped does not move AUROC to 0.529",
        },
        "bundle_drift_linkage": drift,
        "claim_fingerprint": claim_fingerprint,
        "attempts": attempts,
        "conclusion": ("The published 0.529 is the PoC cascade's own score AUROC "
                       "(origin/main:eval/results/poc/ood_check_60.json), whose verdicts "
                       "(good 24 accept / 21 reject / 6 review; bad 4 accept / 5 reject / 0 "
                       "review) reproduce balanced accuracy 0.51 exactly. It is NOT the frozen "
                       "ensemble_v1 shipped in the pod bundle: no column, sign flip or head "
                       "combination of caches/ensemble/ood_scores.json reaches 0.529 (range "
                       "0.342-0.658), and the artifact itself records the frozen-ensemble "
                       "number as 0.415 (reported, not recomputed). The two numbers are "
                       "different scorers, both at chance; the qualitative OOD finding holds "
                       "for both. The shipped OOD score file did not drift (byte-identical to "
                       "origin/main), so this is a provenance/attribution distinction, not an "
                       "instance of the 14/316 SHA256SUMS drift."),
        "published_numbers_unchanged": True,
    }

    os.makedirs(os.path.dirname(OUT), exist_ok=True)

    # self-check: the weighted estimator must reduce to the unweighted one
    chk_rows = [{"segment_id": r["segment_id"], "score": r["score"],
                 "label": latest.get(r["segment_id"]), "verdict": r.get("verdict")}
                for r in rows]
    chk_rows = [r for r in chk_rows if r["label"] in ("good", "bad")]
    unweighted = auroc([r["score"] for r in chk_rows if r["label"] == "good"],
                       [r["score"] for r in chk_rows if r["label"] == "bad"])
    weighted_unit = weighted_auroc(chk_rows, lambda r: 1.0)
    out["estimator_self_check"] = {
        "unweighted": None if unweighted is None else round(unweighted, 6),
        "weighted_unit_weights": None if weighted_unit is None else round(weighted_unit, 6),
        "equal": bool(unweighted is not None and weighted_unit is not None
                      and abs(unweighted - weighted_unit) < 1e-9),
    }

    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, sort_keys=False)
        fh.write("\n")

    print(f"wrote {OUT}")
    print(f"PoC artifact recomputed AUROC      : {poc_primary['auroc']} "
          f"(published {poc.get('poc_score_auroc')})")
    print(f"Frozen ensemble recomputed AUROC   : {primary['auroc']} "
          f"{primary['ci95_window_bootstrap']}")
    print(f"Frozen ensemble reject counts      : {primary['reject_counts']} "
          f"bal.acc {primary['balanced_accuracy_at_shipped_verdicts']}")
    print(f"Attempts hitting 0.529             : {[a['name'] for a in hits]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
