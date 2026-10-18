#!/usr/bin/env python
"""TASK.00(b) -- IndEgo VIA keystep annotations as free L1 / L2 ground truth.

Zero training.  For the 4 IndEgo gold clips that actually have a VIA temporal
annotation, we derive

  L1 (per 10-s slice) : advancing        -- the slice is covered by an annotated step
                        no_relevant_work -- the slice falls in a gap between steps
  L2 (per clip)       : completed iff the annotation reaches the LAST step of that
                        clip's task list in the dataset's keysteps_*.txt

and then measure, with no model fitting at all:
  (a) agreement between the derived L1 slice label and the human gold window label
      (manifest_gold.json): confusion matrix, agreement rate, Cohen's kappa,
      overall and restricted to gold reasons that are about work activity;
  (b) AUROC of the shipped 26-feature LOCO score for advancing vs no_relevant_work,
      clip-grouped bootstrap CI, n printed.

Run:
  JADAL_BUNDLE=<bundle> /home/parshu/envs/jadal/bin/python tools/task/task00b.py
"""
from __future__ import annotations

import glob
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from dsh import common  # noqa: E402

ID = "TASK.00b"
OUT = os.path.join(common.REPO, "eval", "results", "task", f"{ID}.json")
INDEGO = "/home/parshu/projects/awing/data/indego"


# ------------------------------------------------------------------ parsing ---
def parse_keysteps_txt(path):
    """keysteps_*.txt -> {section_name: [step_label, ...]}.

    Format (observed): a section header line, then comma-separated step labels
    one per line, e.g.
        Assembly_ECU
        00_preparation,
        01_inspect_and_clean,
        ...
    """
    sections, cur = {}, None
    with open(path, encoding="utf-8", errors="replace") as fh:
        lines = fh.readlines()
    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#"):
            cur = line.lstrip("#").strip()
            sections.setdefault(cur, [])
            continue
        looks_like_step = bool(re.match(r"^\d{2}_", line))
        if not looks_like_step:
            cur = line.rstrip(":").strip()
            sections.setdefault(cur, [])
            continue
        if cur is None:
            continue
        sections[cur].append(line.rstrip(",").strip())
    return {k: v for k, v in sections.items() if v}


def parse_via(path):
    """VGG Image Annotator temporal-segment JSON -> dict.

    Observed structure: attribute["1"].aname is the attribute name ("steps" or
    "TEMPORAL-SEGMENTS"), attribute["1"].options maps option-id -> step label;
    file["1"].fname is the video file; metadata[*].z = [start_s, end_s] in
    seconds and metadata[*].av["1"] = the chosen option-id.  Checked against the
    real clip durations with cv2 (see `duration_check` in the results).
    """
    with open(path) as fh:
        d = json.load(fh)
    attrs = d.get("attribute", {})
    fid = sorted(d.get("file", {}).keys())
    if not attrs or not fid:
        return None
    a = attrs[sorted(attrs.keys())[0]]
    opts = {str(k): str(v) for k, v in (a.get("options") or {}).items()}
    fname = d["file"][fid[0]].get("fname")
    segs = []
    for _, m in (d.get("metadata") or {}).items():
        z = m.get("z")
        if not z or len(z) < 2:
            continue
        av = m.get("av") or {}
        opt = av.get(sorted(av.keys())[0]) if av else None
        segs.append({"start_s": float(z[0]), "end_s": float(z[1]),
                     "option_id": str(opt) if opt is not None else None,
                     "step": opts.get(str(opt), None)})
    segs.sort(key=lambda s: s["start_s"])
    return {"aname": a.get("aname"), "fname": fname, "options": opts,
            "segments": segs, "path": path,
            "has_xy": any(m.get("xy") for m in (d.get("metadata") or {}).values())}


def duration_check(fname, clip_dir_candidates):
    """cv2 duration for the annotated video, to prove z is in seconds."""
    import cv2
    for root in clip_dir_candidates:
        for p in glob.glob(os.path.join(root, "**", fname), recursive=True):
            cap = cv2.VideoCapture(p)
            fps = cap.get(cv2.CAP_PROP_FPS)
            nf = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            cap.release()
            return {"path": p, "fps": round(float(fps), 4),
                    "n_frames": int(nf),
                    "duration_s": round(float(nf) / float(fps), 3) if fps else None,
                    "wh": [int(w), int(h)]}
    return None


# ------------------------------------------------------------------ metrics ---
def gap_list(segments, duration_s):
    """Un-annotated stretches between the first and last annotated step."""
    gaps = []
    prev_end = None
    for s in segments:
        if prev_end is None:
            gaps.append({"start_s": 0.0, "end_s": round(s["start_s"], 3)})
        elif s["start_s"] - prev_end > 1e-9:
            gaps.append({"start_s": round(prev_end, 3), "end_s": round(s["start_s"], 3)})
        prev_end = s["end_s"]
    if duration_s is not None and prev_end is not None and duration_s - prev_end > 1e-9:
        gaps.append({"start_s": round(prev_end, 3), "end_s": round(float(duration_s), 3)})
    return [g for g in gaps if g["end_s"] - g["start_s"] > 1e-9]


def cohen_kappa(a, b):
    a, b = np.asarray(a, int), np.asarray(b, int)
    n = len(a)
    po = float((a == b).mean())
    pe = float((a.sum() / n) * (b.sum() / n) + ((n - a.sum()) / n) * ((n - b.sum()) / n))
    return float("nan") if pe == 1 else (po - pe) / (1 - pe)


def crosstab(pred, gold):
    """pred/gold in {0,1}; returns counts + agreement statistics."""
    pred, gold = np.asarray(pred, int), np.asarray(gold, int)
    tp = int(((pred == 1) & (gold == 1)).sum())
    fp = int(((pred == 1) & (gold == 0)).sum())
    fn = int(((pred == 0) & (gold == 1)).sum())
    tn = int(((pred == 0) & (gold == 0)).sum())
    n = tp + fp + fn + tn
    out = {"n": n, "pred_advancing_gold_good": tp, "pred_advancing_gold_bad": fp,
           "pred_no_relevant_work_gold_good": fn, "pred_no_relevant_work_gold_bad": tn,
           "agreement_rate": round((tp + tn) / n, 4) if n else float("nan"),
           "cohen_kappa": round(cohen_kappa(pred, gold), 4) if n else float("nan")}
    if (tp + fn) and (tn + fp):
        tpr = tp / (tp + fn)
        tnr = tn / (tn + fp)
        out.update({"sensitivity_advancing_vs_good": round(tpr, 4),
                    "specificity_no_relevant_work_vs_bad": round(tnr, 4),
                    "balanced_accuracy": round(0.5 * (tpr + tnr), 4)})
    return out


def auroc_block(y, s, groups, ci_type="clip-grouped bootstrap (2000 draws, seed 0)"):
    y = np.asarray(y, int)
    s = np.asarray(s, float)
    a = common.fast_auc(y, s)
    lo, hi, nb = common.group_boot_ci(y, s, groups, n_boot=2000, seed=0)
    wlo, whi, wnb = common.window_boot_ci(y, s, n_boot=2000, seed=0)
    return {"auroc": round(float(a), 4), "n": int(len(y)),
            "n_advancing": int((y == 1).sum()), "n_no_relevant_work": int((y == 0).sum()),
            "n_groups": int(len(set(np.asarray(groups).tolist()))),
            "ci95": [round(lo, 4), round(hi, 4)], "ci95_type": ci_type,
            "n_boot_valid": nb,
            "ci95_window_optimistic": [round(wlo, 4), round(whi, 4)],
            "ci95_type_window_optimistic": "window bootstrap (2000 draws, seed 0) -- optimistic"}


def main():
    # ---- annotations -------------------------------------------------------
    ann_files = sorted(glob.glob(os.path.join(INDEGO, "**", "annotated_keysteps_*", "*.json"),
                                 recursive=True))
    ann = {}
    unparseable = []
    for f in ann_files:
        try:
            v = parse_via(f)
        except Exception as e:  # never swallow silently
            unparseable.append({"path": f, "error": f"{type(e).__name__}: {e}"})
            continue
        if v and v["fname"]:
            ann[v["fname"]] = v
    if not ann:
        raise SystemExit("no parsable VIA annotation files -- TASK.00b cannot run (documented, skipped)")

    # ---- gold windows ------------------------------------------------------
    g = common.load_gold()
    man = json.load(open(os.path.join(common.BUNDLE, "labels", "manifest_gold.json")))
    end_by_id = {it["id"]: it["end_s"] for it in man["items"]}
    clips = sorted(set(g["clip"].tolist()))
    clip_of_video = {}
    for i, p in enumerate(g["path"]):
        clip_of_video.setdefault(os.path.basename(p), g["clip"][i])

    annotated_clips, missing_clips = {}, {}
    for fname, v in ann.items():
        c = clip_of_video.get(fname)
        if c:
            annotated_clips[c] = v
    for c in clips:
        vids = {os.path.basename(p) for i, p in enumerate(g["path"]) if g["clip"][i] == c}
        if not (vids & set(ann.keys())):
            missing_clips[c] = sorted(vids)

    # ---- task lists --------------------------------------------------------
    txt_files = sorted(glob.glob(os.path.join(INDEGO, "**", "keysteps_*.txt"), recursive=True))
    txt_files += sorted(glob.glob(os.path.join(INDEGO, "*.txt")))
    sections = {}
    for t in txt_files:
        for k, v in parse_keysteps_txt(t).items():
            sections.setdefault(k, {"steps": v, "file": os.path.relpath(t, INDEGO)})

    def find_task_list(labels):
        """The unique task list containing every annotated step label."""
        want = set(labels)
        hits = {k: v for k, v in sections.items() if want <= set(v["steps"])}
        if len(hits) == 1:
            k = next(iter(hits))
            return k, hits[k]
        if hits:  # ambiguous: prefer the smallest superset
            k = min(hits, key=lambda k: len(hits[k]["steps"]))
            return k, hits[k]
        return None, None

    dur_roots = [os.path.join(INDEGO, d) for d in ("1_Assembly", "3_Logistics_Organization")]

    # ---- per-clip L1 + L2 --------------------------------------------------
    per_clip, l1_rows = {}, []
    for c in sorted(annotated_clips):
        v = annotated_clips[c]
        labels = [s["step"] for s in v["segments"] if s["step"]]
        task_name, task = find_task_list(labels)
        last_step = task["steps"][-1] if task else None
        ann_labels = set(labels)
        l2 = "completed" if (last_step is not None and last_step in ann_labels) else "not_completed"
        cov = sum(s["end_s"] - s["start_s"] for s in v["segments"])
        first, last = v["segments"][0]["start_s"], v["segments"][-1]["end_s"]
        dchk = duration_check(v["fname"], dur_roots)
        per_clip[c] = {
            "clip": c, "video": v["fname"], "annotation": os.path.relpath(v["path"], INDEGO),
            "attribute_name": v["aname"],
            "n_annotated_segments": len(v["segments"]),
            "n_distinct_steps_annotated": len(ann_labels),
            "n_options_in_annotation": len(v["options"]),
            "n_steps_in_matched_task_list": len(task["steps"]) if task else None,
            "task_list_matched": task_name,
            "task_list_file": task["file"] if task else None,
            "first_step_s": round(first, 3), "last_step_end_s": round(last, 3),
            "annotated_span_s": round(last - first, 3),
            "annotated_coverage_s": round(cov, 3),
            "last_step_of_task_list": last_step,
            "L2": l2,
            "duration_check_cv2": dchk,
            "steps_present": sorted(ann_labels),
            "steps_missing_from_task_list": sorted(ann_labels - set(task["steps"])) if task else None,
            "task_list_steps_not_annotated": (sorted(set(task["steps"]) - ann_labels) if task else None),
            "n_task_list_steps_not_annotated": (len(set(task["steps"]) - ann_labels) if task else None),
            "gaps_s": gap_list(v["segments"], (dchk or {}).get("duration_s")),
        }
        l1_rows.extend(rows_for_clip(g, c, v, end_by_id))

    # ---- L1 label variants -------------------------------------------------
    def variant(name):
        return np.asarray([r["labels"][name] for r in l1_rows], int)

    sel_4 = np.asarray([r["clip"] for r in l1_rows])
    gold_y = np.asarray([r["gold_label"] for r in l1_rows], int)
    # manifest_gold uses JSON null for "no bad reason"; normalise to ""
    gold_reason = np.asarray([str(r["gold_reason"] or "") for r in l1_rows], dtype=object)
    clip_arr = np.asarray([r["clip"] for r in l1_rows], dtype=object)
    l1_major = variant("majority_overlap")
    WORK_ACTIVITY_REASONS = ("", "NO_ACTIVE_WORK")
    work_only = np.asarray([r in WORK_ACTIVITY_REASONS for r in gold_reason])

    agree = {
        "primary_majority_overlap": crosstab(l1_major, gold_y),
        "primary_majority_overlap_gold_reason_work_only": crosstab(l1_major[work_only], gold_y[work_only]),
        "sensitivity_any_overlap": crosstab(variant("any_overlap"), gold_y),
        "sensitivity_midpoint_in_step": crosstab(variant("midpoint_in_step"), gold_y),
        "sensitivity_fully_inside_step": crosstab(variant("fully_inside_step"), gold_y),
        "constant_baselines": {
            "always_advancing_agreement": round(float((gold_y == 1).mean()), 4),
            "always_gold_majority_agreement": round(float(max((gold_y == 1).mean(), (gold_y == 0).mean())), 4),
            "note": ("the agreement rate must be read against the 0.709 always-advancing baseline: "
                     "the derived L1 label is 105/110 advancing"),
        },
        "coverage_fraction_histogram": {
            "n_windows_with_zero_step_overlap": int(sum(1 for r in l1_rows if r["step_overlap_frac"] == 0)),
            "n_windows_fully_inside_a_step": int(sum(1 for r in l1_rows if r["step_overlap_frac"] >= 0.999)),
            "n_windows_partial": int(sum(1 for r in l1_rows if 0 < r["step_overlap_frac"] < 0.999)),
        },
    }
    # direct source-vs-source disagreement: human said NO_ACTIVE_WORK, annotation says inside a step
    gold_na = np.asarray([r == "NO_ACTIVE_WORK" for r in gold_reason])
    source_disagreement = {
        "n_gold_no_active_work_windows_on_these_4_clips": int(gold_na.sum()),
        "n_of_those_annotation_says_advancing": int((gold_na & (l1_major == 1)).sum()),
        "n_of_those_annotation_says_gap": int((gold_na & (l1_major == 0)).sum()),
        "n_gold_good_windows_annotation_says_gap": int(((gold_y == 1) & (l1_major == 0)).sum()),
        "ids_annotation_says_gap": [r["id"] for k, r in enumerate(l1_rows) if l1_major[k] == 0],
        "gold_bad_by_reason": {k: int(v) for k, v in
                               zip(*np.unique(gold_reason[gold_y == 0], return_counts=True))},
    }
    per_clip_agree = {}
    for c in sorted(set(sel_4.tolist())):
        m = sel_4 == c
        blk = crosstab(l1_major[m], gold_y[m])
        blk["n_gap_windows"] = int((l1_major[m] == 0).sum())
        blk["always_advancing_agreement"] = round(float((gold_y[m] == 1).mean()), 4)
        per_clip_agree[c] = blk

    # ---- (b) shipped score direction ---------------------------------------
    # S1: harness canonical LOCO score (GRID.00 reproduces the pod's 0.810 with it)
    p_loco = common.loco_proba(g["X"], g["y"], g["clip"])
    # S2: the portal head's own hyperparameters, still out-of-fold
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    p_portal = np.zeros(len(g["y"]), float)
    for c in sorted(set(g["clip"].tolist())):
        tr, te = g["clip"] != c, g["clip"] == c
        sc = StandardScaler().fit(g["X"][tr])
        m = LogisticRegression(max_iter=2000).fit(sc.transform(g["X"][tr]), g["y"][tr])
        p_portal[te] = m.predict_proba(sc.transform(g["X"][te]))[:, 1]
    # S3: full fit over all 150 (what the portal scores a NEW window with) -- optimistic in-domain
    sc_all = StandardScaler().fit(g["X"])
    m_all = LogisticRegression(max_iter=2000).fit(sc_all.transform(g["X"]), g["y"])
    p_full = m_all.predict_proba(sc_all.transform(g["X"]))[:, 1]

    idx = np.asarray([g["ids"].index(r["id"]) for r in l1_rows])
    y_l1 = l1_major
    scores = {"loco_harness_C0.3_balanced": p_loco[idx],
              "loco_portal_C1.0_maxiter2000": p_portal[idx],
              "fullfit_portal_not_out_of_fold": p_full[idx]}
    aurocs = {k: auroc_block(y_l1, s, clip_arr) for k, s in scores.items()}
    aurocs_rev = {k: round(1.0 - v["auroc"], 4) for k, v in aurocs.items()}
    per_clip_auroc = {}
    for c in sorted(set(clip_arr.tolist())):
        m = clip_arr == c
        if len(set(y_l1[m].tolist())) == 2:
            per_clip_auroc[c] = {"auroc": round(common.fast_auc(y_l1[m], scores["loco_harness_C0.3_balanced"][m]), 4),
                                 "n": int(m.sum()),
                                 "n_advancing": int((y_l1[m] == 1).sum())}
        else:
            per_clip_auroc[c] = {"auroc": None, "n": int(m.sum()),
                                 "n_advancing": int((y_l1[m] == 1).sum()),
                                 "note": "one class only -- AUROC undefined"}

    # gold-label AUROC with the same score, for context on the same 4 clips
    auroc_gold = auroc_block(gold_y, scores["loco_harness_C0.3_balanced"], clip_arr)

    cfg = {"id": ID,
           "l1_rule": "advancing iff annotated step intervals cover >50% of the 10-s window, else no_relevant_work",
           "l2_rule": "completed iff the annotation reaches the last step of the clip's matched task list",
           "score_primary": "common.loco_proba (StandardScaler+LogisticRegression C=0.3 class_weight=balanced, leave-one-clip-out)",
           "annotation_glob": "data/indego/**/annotated_keysteps_*/*.json (VIA temporal segments)",
           "n_boot": 2000, "seed": 0}

    payload = {
        "id": ID, "status": "done",
        "headline": "AUROC of the shipped 26-feature LOCO score for advancing (1) vs no_relevant_work (0), 4 annotated clips",
        "auroc": aurocs["loco_harness_C0.3_balanced"]["auroc"],
        "ci95": aurocs["loco_harness_C0.3_balanced"]["ci95"],
        "ci95_type": aurocs["loco_harness_C0.3_balanced"]["ci95_type"],
        "n": aurocs["loco_harness_C0.3_balanced"]["n"],
        "n_bad": aurocs["loco_harness_C0.3_balanced"]["n_no_relevant_work"],
        "n_good": aurocs["loco_harness_C0.3_balanced"]["n_advancing"],
        "n_groups": aurocs["loco_harness_C0.3_balanced"]["n_groups"],
        "label_source": ("L1 derived from IndEgo VIA keystep annotations (dataset-shipped, no human "
                         "QA labeller); agreement is measured against " + g["label_source"]),
        "protocol": ("zero-training: VIA temporal segments -> per-window L1 label; agreement with human "
                     "gold labels; AUROC of the shipped 26-feature LOCO score for the derived L1 label; "
                     "clip-grouped bootstrap CIs; 4 annotated clips only"),
        "segments_engine": {
            "n_annotation_files_on_disk": len(ann_files),
            "n_parsed": len(ann),
            "n_unparseable": len(unparseable),
            "unparseable": unparseable,
            "n_gold_clips": len(clips),
            "clips_with_annotation": sorted(annotated_clips),
            "clips_without_annotation": missing_clips,
            "task_list_sections_parsed": {k: {"file": v["file"], "n_steps": len(v["steps"])}
                                          for k, v in sections.items()},
        },
        "per_clip": per_clip,
        "l2_summary": {c: per_clip[c]["L2"] for c in per_clip},
        "l2_variance": len(set(per_clip[c]["L2"] for c in per_clip)),
        "l1_label_counts": {
            "n_advancing": int((l1_major == 1).sum()),
            "n_no_relevant_work": int((l1_major == 0).sum()),
            "n_gold_good": int((gold_y == 1).sum()),
            "n_gold_bad": int((gold_y == 0).sum()),
        },
        "agreement_with_human_gold": agree,
        "agreement_per_clip": per_clip_agree,
        "source_vs_source_disagreement": source_disagreement,
        "score_direction": {
            "primary": "higher score -> more likely advancing",
            "auroc_advancing_vs_no_relevant_work": aurocs,
            "auroc_flipped": aurocs_rev,
            "per_clip_auroc_primary_score": per_clip_auroc,
            "auroc_gold_labels_same_windows_same_score": auroc_gold,
        },
        "caveat": ("IndEgo keysteps are demo-style assembly/logistics tasks recorded for a research "
                   "dataset; the 4 annotated clips are one worker's one task each, so nothing here "
                   "transfers to factory work without testing it there. L2 has no variance on this "
                   "subset and 4 groups make the grouped CI weak. These are descriptive rates on "
                   "dataset-shipped annotations, not accuracy of a product on unlabelled video."),
        "config": cfg, "config_sha256": common.config_sha256(cfg),
        "device": common.DEVICE_CPU,
        "cost": "0 GPU; cached features + sklearn fits on 150x26; ~5 s CPU",
        "command": "JADAL_BUNDLE=<bundle> /home/parshu/envs/jadal/bin/python tools/task/task00b.py",
        "git_sha": common.git_sha(), "seed": 0,
        "l1_rows": l1_rows,
    }
    common.save_results(OUT, payload)
    print("annotated clips:", sorted(annotated_clips))
    print("clips without annotation:", missing_clips)
    print("L2:", payload["l2_summary"], "distinct:", payload["l2_variance"])
    print("L1 counts:", payload["l1_label_counts"])
    print("agreement (primary):", json.dumps(agree["primary_majority_overlap"]))
    print("agreement work-only:", json.dumps(agree["primary_majority_overlap_gold_reason_work_only"]))
    print("constant baseline:", json.dumps(agree["constant_baselines"]))
    print("source disagreement:", json.dumps({k: v for k, v in source_disagreement.items()
                                              if not k.startswith("ids_")}))
    print("per-clip agreement:", json.dumps(per_clip_agree, indent=1))
    print("AUROC advancing vs no_relevant_work:",
          json.dumps({k: (v["auroc"], v["ci95"], v["n"], v["n_no_relevant_work"])
                      for k, v in aurocs.items()}, indent=1))
    print("per-clip:", json.dumps(per_clip_auroc))
    print("wrote", OUT)


def rows_for_clip(g, clip, v, end_by_id):
    """One row per gold window of `clip`, with every L1 variant."""
    segs = [(s["start_s"], s["end_s"]) for s in v["segments"]]
    out = []
    for i, c in enumerate(g["clip"]):
        if c != clip:
            continue
        a = float(g["start_s"][i])
        b = float(end_by_id.get(g["ids"][i], a + 10.0))
        out.append(_row(g, i, clip, a, b, segs, v))
    return out


def _row(g, i, clip, a, b, segs, v):
    dur = b - a
    ov = sum(max(0.0, min(b, e) - max(a, s)) for s, e in segs)
    frac = ov / dur if dur else 0.0
    mid = 0.5 * (a + b)
    in_mid = any(s <= mid < e for s, e in segs)
    inside = any(s <= a and b <= e for s, e in segs)
    steps_hit = sorted({v["segments"][k]["step"] for k, (s, e) in enumerate(segs)
                        if min(b, e) - max(a, s) > 0 and v["segments"][k]["step"]})
    return {"id": g["ids"][i], "clip": clip, "start_s": round(a, 3), "end_s": round(b, 3),
            "step_overlap_frac": round(frac, 4), "steps_hit": steps_hit,
            "gold_label": int(g["y"][i]), "gold_reason": str(g["reason"][i] or ""),
            "labels": {"majority_overlap": int(frac > 0.5),
                       "any_overlap": int(ov > 0),
                       "midpoint_in_step": int(in_mid),
                       "fully_inside_step": int(inside)}}


if __name__ == "__main__":
    main()
